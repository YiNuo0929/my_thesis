import argparse, os
import math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from math import sqrt
from itertools import cycle
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Reproducibility ----------
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Utils ----------
def load_csvs(path_like: str) -> pd.DataFrame:
    p = Path(path_like)
    if p.is_dir():
        files = sorted([*p.glob("*.csv")])
        if not files:
            raise FileNotFoundError(f"No CSVs under: {p}")
        dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(p, encoding="utf-8-sig")
    return df

def get_feature_cols(df: pd.DataFrame, col):
    all_cols = set(df.columns)
    ap_cols = [f"ap{i}" for i in range(col)]
    missing = [c for c in ap_cols if c not in all_cols]
    if missing:
        raise ValueError(f"資料集中缺少欄位（ap0~ap{col-1}）：{missing[:10]} ...")
    return ap_cols

def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    mask_valid = (train_ap != missing_val)
    n_feats = train_ap.shape[1]
    mins = np.zeros(n_feats, dtype=np.float32)
    maxs = np.ones(n_feats, dtype=np.float32)

    for j in range(n_feats):
        col = train_ap[:, j]
        m = mask_valid[:, j]
        if m.any():
            v = col[m]
            vmin = v.min()
            vmax = v.max()
            if abs(vmax - vmin) < 1e-6:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        mins[j] = vmin
        maxs[j] = vmax
    return mins, maxs

def apply_scaler(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray, missing_val: float = -110.0):
    x = x.copy().astype(np.float32)
    miss_mask = (x == missing_val)
    denom = (maxs - mins)
    denom[denom == 0.0] = 1.0
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)
    x[miss_mask] = -1.0
    return x

class RSSISourceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, label_col="rp_id",
                 mins=None, maxs=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.y_raw = df[label_col].values.astype(np.int64)
        self.X = df[ap_cols].values.astype(np.float32)
        self.missing_val = missing_val
        self.mins = mins
        self.maxs = maxs
        if (mins is not None) and (maxs is not None):
            self.X = apply_scaler(self.X, self.mins, self.maxs, self.missing_val)
        self.X = np.expand_dims(self.X, axis=1)
        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

class RSSITargetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols,
                 mins=None, maxs=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.X = df[ap_cols].values.astype(np.float32)
        self.missing_val = missing_val
        self.mins = mins
        self.maxs = maxs
        if (mins is not None) and (maxs is not None):
            self.X = apply_scaler(self.X, self.mins, self.maxs, self.missing_val)
        self.X = np.expand_dims(self.X, axis=1)

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i])

# ---------- Model Blocks ----------

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, p_drop=0.2):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop)
        )
    def forward(self, x): return self.seq(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return x

# =========================================================================
# Layer 1: Gated Transformer Encoder Layer (With Weight Logging)
# =========================================================================
class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.last_attn_weights = None 

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn_weights = attn_weights.detach().cpu()
        
        gate_score = torch.sigmoid(self.gate_linear(src)) 
        gated_output = attn_output * gate_score
        
        src = src + self.dropout1(gated_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# =========================================================================
# Layer 2: Standard Transformer Layer (With Weight Logging)
# =========================================================================
class StandardTransformerEncoderLayerWithWeights(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.last_attn_weights = None 

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask,
            need_weights=True, 
            average_attn_weights=False
        )
        self.last_attn_weights = attn_weights.detach().cpu()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# -------- Transformer-based Extractor (Updated) --------
class TransformerExtractor(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool,
        mask_value: float,
        z_dim: int,
        use_mask: bool,
        bottleneck_hidden: int = None,
        use_gating: bool = True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.z_dim = z_dim
        self.use_mask = use_mask
        self.mask_value = mask_value
        self.use_gating = use_gating

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"num_tokens={num_tokens} 不能剛好 reshape 成正方形")
        self.side = side
        self.num_tokens = num_tokens
        self.patch_size = 2
        if self.side % self.patch_size != 0:
            raise ValueError(f"side={self.side} 不能被 patch_size={self.patch_size} 整除")
        self.num_patches_per_side = self.side // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        patch_dim = 1 * self.patch_size * self.patch_size

        self.patch_proj = nn.Linear(patch_dim, d_model)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            max_len = self.num_patches + 1
        else:
            self.cls_token = None
            max_len = self.num_patches

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        if self.use_gating:
            layers = [
                GatedTransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        else:
            layers = [
                StandardTransformerEncoderLayerWithWeights(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        self.encoder = nn.ModuleList(layers)

        if bottleneck_hidden is None:
            bottleneck_hidden = d_model

        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_hidden, z_dim)
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 3: x = x.squeeze(1)
        B, L = x.shape
        img = x.view(B, 1, self.side, self.side)
        patches_raw = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches_raw.transpose(1, 2)

        key_padding_mask = None
        if self.use_mask:
            eq_mask = (patches_raw == self.mask_value)
            key_padding_mask = eq_mask.all(dim=1)

        h = self.patch_proj(patches)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            h = torch.cat([cls, h], dim=1)

        h = self.pos_encoding(h)

        src_key_padding_mask = None
        if key_padding_mask is not None:
            if self.use_cls_token:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
                src_key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)
            else:
                src_key_padding_mask = key_padding_mask

        for layer in self.encoder:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        if self.use_cls_token:
            cls_feat = h[:, 0, :]
        else:
            cls_feat = h.mean(dim=1)

        z = self.bottleneck(cls_feat)
        return z

# -------- Predictor / Reconstruction --------
class PredictorMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=[256, 256], p_drop=0.2):
        super().__init__()
        dims = [in_dim] + hidden
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks.append(MLPBlock(a, b, p_drop=p_drop))
        self.feat = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(dims[-1], n_classes)
    def forward(self, x):
        x = self.feat(x)
        return self.head(x)

class ReconstructionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=[256, 256], p_drop=0.2):
        super().__init__()
        dims = [in_dim] + hidden
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks.append(MLPBlock(a, b, p_drop=p_drop))
        self.feat = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(dims[-1], out_dim)
    def forward(self, x):
        x = self.feat(x)
        return self.head(x)

# -------- TransClassifier --------
class TransClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vit_tokens: int,
        n_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        z_dim: int,
        pred_hidden,
        recon_hidden,
        p_drop: float,
        use_mask: bool,
        mask_value: float,
        use_gating: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.vit_tokens = vit_tokens

        if input_dim != vit_tokens:
            self.pre_linear = nn.Linear(input_dim, vit_tokens)
        else:
            self.pre_linear = None

        self.extractor = TransformerExtractor(
            num_tokens=vit_tokens,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_cls_token=True,
            z_dim=z_dim,
            bottleneck_hidden=None,
            use_mask=use_mask,
            mask_value=mask_value,
            use_gating=use_gating
        )
        self.predictor = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=pred_hidden,
            p_drop=p_drop,
        )
        self.reconstructor = ReconstructionMLP(
            in_dim=z_dim,
            out_dim=input_dim,
            hidden=recon_hidden,
            p_drop=p_drop,
        )

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        x_flat = x
        if self.pre_linear is not None:
            x_vit = self.pre_linear(x_flat)
        else:
            x_vit = x_flat
        x_vit = x_vit.unsqueeze(1)
        z = self.extractor(x_vit)
        logits = self.predictor(z)
        recon = self.reconstructor(z)
        return logits, recon

# ---------- Metrics / Maps ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
    if not need.issubset(df.columns):
        raise ValueError("rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp

def reconstruction_loss(pred, target, mask_value=-1.0):
    mask = (target != mask_value)
    if mask.sum() == 0:
        return (pred * 0.0).sum()
    diff = pred - target
    diff = diff[mask]
    return torch.mean(diff * diff)

# =========================================================================
# VIZ FUNCTION: Compare Specific RPs (Merged)
# =========================================================================
def compare_specific_rps(model, X_te, y_te, target_rps, device, output_dir):
    model.eval()
    
    # 搜尋 Target RPs 的樣本索引
    samples = {}
    # 對於每個 Target RP，我們試著找出測試集中的一個樣本
    for target in target_rps:
        # y_te 可能有沒對應到的 -1，所以要過濾
        indices = np.where(y_te == target)[0]
        if len(indices) > 0:
            # 取該類別的第一個樣本
            samples[target] = X_te[indices[0]] 
        else:
            print(f"Warning: RP {target} not found in test data.")

    if not samples:
        print("No matching RPs found for visualization.")
        return

    # 先跑一次第一個樣本來確定 Head 數量
    first_rp = list(samples.keys())[0]
    x_tmp = torch.from_numpy(samples[first_rp]).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(x_tmp)
    
    last_layer = model.extractor.encoder[-1]
    if hasattr(last_layer, 'last_attn_weights') and last_layer.last_attn_weights is not None:
        # Shape: [Batch=1, Heads, Seq, Seq]
        num_heads = last_layer.last_attn_weights.shape[1]
    else:
        print("Error: No attention weights found.")
        return

    num_rps = len(samples)
    
    # 準備畫布: Rows = RPs, Cols = Heads
    fig, axes = plt.subplots(num_rps, num_heads, figsize=(4 * num_heads, 3.5 * num_rps), squeeze=False)
    
    print(f"Generating comparison for RPs: {list(samples.keys())} across {num_heads} heads...")

    for i, (rp_id, x_np) in enumerate(samples.items()):
        # Forward pass
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device) # [1, 1, L]
        with torch.no_grad():
            _ = model(x_tensor)
        
        # 抓取最後一層的 Weights: [Heads, Seq, Seq]
        attn_weights = last_layer.last_attn_weights[0].cpu().numpy()
        
        for h in range(num_heads):
            ax = axes[i, h]
            attn_map = attn_weights[h]
            
            # 繪圖 (只畫前 50x50 避免太擠，若想看全部可移除 slicing)
            sns.heatmap(attn_map[:50, :50], ax=ax, cmap="viridis", cbar=True)
            
            if i == 0: ax.set_title(f"Head {h}")
            if h == 0: ax.set_ylabel(f"RP {rp_id}\nQuery Token")
            else: ax.set_ylabel("")
            if i == num_rps - 1: ax.set_xlabel("Key Token")
            else: ax.set_xlabel("")
            
    plt.tight_layout()
    save_path = os.path.join(output_dir, "rp_heads_comparison.png")
    plt.savefig(save_path)
    print(f"Comparison saved to: {save_path}")
    plt.close()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_path", type=str, required=True)
    parser.add_argument("--target_train_path", type=str, required=True)
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, default=r"rp_id.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./rssi_trans_recon_ckpt")
    parser.add_argument("--column", type=int, default=256)

    parser.add_argument("--pred_hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--recon_hidden", type=int, nargs="+", default=[256])
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--lambda_recon", type=float, default=0.1)
    parser.add_argument("--use_mask", action="store_true", help="啟用 key_padding_mask")
    parser.add_argument("--no_gating", action="store_true", help="關閉 Gated Attention")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vit_tokens", type=int, default=256)
    
    # [NEW ARG] 指定要視覺化的 RP ID 列表
    parser.add_argument("--target_rps", type=int, nargs="+", default=[0, 10, 20, 30], 
                        help="List of RP IDs to visualize specific attention heads")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Load Data ---
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src, args.column)
    mins, maxs = fit_scaler(df_src[ap_cols].values.astype(np.float32), missing_val=args.missing_val)

    ds_src = RSSISourceDataset(df_src, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_src.id2idx
    idx2id = ds_src.idx2id
    dl_src = DataLoader(ds_src, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Prepare Test Arrays for Viz & Eval
    X_te_full_raw = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full_raw, mins, maxs, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)
    
    # 對應回原始 RP ID (為了 compare_specific_rps 找人)
    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.zeros(len(df_te), dtype=int)
    
    # Mapped Index for Eval (0..C-1)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

    class TestDataset(Dataset):
        def __len__(self): return len(y_te_idx)
        def __getitem__(self, i):
            return torch.from_numpy(X_te_full[i]), torch.tensor(y_te_idx[i], dtype=torch.long), int(y_te_raw[i])

    ds_te = TestDataset()
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_ap = len(ap_cols)
    use_gating = not args.no_gating

    model = TransClassifier(
        input_dim=num_ap,
        vit_tokens=args.vit_tokens,
        n_classes=len(id2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        z_dim=args.z_dim,
        pred_hidden=args.pred_hidden,
        recon_hidden=args.recon_hidden,
        p_drop=args.dropout,
        use_mask=args.use_mask,
        mask_value=-1.0,
        use_gating=use_gating
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    os.makedirs(args.out_dir, exist_ok=True)

    print("---------------------------------------")
    print(f"ViT Model (Gating={use_gating}, With Specific RP Viz)")
    print("---------------------------------------")

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_total = 0.0
        tr_ce_total = 0.0
        tr_recon_s_total = 0.0
        tr_recon_t_total = 0.0
        tr_acc, n = 0.0, 0
        tgt_iter = cycle(dl_tgt)

        for xb_s, yb_s in dl_src:
            keep = (yb_s != -1)
            if not keep.any(): continue
            xb_s, yb_s = xb_s[keep], yb_s[keep]
            xb_s, yb_s = xb_s.to(device, non_blocking=True), yb_s.to(device, non_blocking=True)
            xb_t = next(tgt_iter).to(device, non_blocking=True)

            opt.zero_grad()
            logits_s, recon_s = model(xb_s)
            ce_loss = crit(logits_s, yb_s)
            recon_loss_s = reconstruction_loss(recon_s, xb_s.squeeze(1), mask_value=-1.0)
            _, recon_t = model(xb_t)
            recon_loss_t = reconstruction_loss(recon_t, xb_t.squeeze(1), mask_value=-1.0)
            
            total_loss = ce_loss + args.lambda_recon * (recon_loss_s + recon_loss_t)
            total_loss.backward()
            opt.step()

            bs = yb_s.size(0)
            tr_loss_total += total_loss.item() * bs
            tr_ce_total += ce_loss.item() * bs
            tr_recon_s_total += recon_loss_s.item() * bs
            tr_recon_t_total += recon_loss_t.item() * bs
            tr_acc += (logits_s.argmax(1) == yb_s).float().sum().item()
            n += bs

        if n > 0:
            tr_loss_total /= n
            tr_ce_total /= n
            tr_recon_s_total /= n
            tr_recon_t_total /= n
            tr_acc /= n
        print(f"Epoch {epoch:03d} | Total {tr_loss_total:.4f} | Acc {tr_acc:.4f}")

    # --- Final Viz: Specific RPs ---
    # 直接使用記憶體中的 model 進行繪圖，無需載入
    print(f"\n[Viz] Generating Head Comparisons for RPs: {args.target_rps}")
    compare_specific_rps(model, X_te_full, y_te_raw, args.target_rps, device, args.out_dir)

    # --- Final Eval ---
    model.eval()
    preds_idx, gts_idx, gts_rpid = [], [], []
    with torch.no_grad():
        for xb, yb_idx, yb_rpid in dl_te:
            xb = xb.to(device)
            logits, _ = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            preds_idx.append(pred)
            gts_idx.append(yb_idx.numpy())
            gts_rpid.append(yb_rpid.numpy())

    preds_idx = np.concatenate(preds_idx) if preds_idx else np.array([])
    gts_idx   = np.concatenate(gts_idx) if gts_idx else np.array([])
    gts_rpid  = np.concatenate(gts_rpid) if gts_rpid else np.array([])

    eval_mask = (gts_idx != -1)
    acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item() if eval_mask.any() else float("nan")

    rp_map = load_rp_map(args.rp_map_path)
    preds_rpid = np.array([idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)

    mde_distances = []
    for gt_id, pr_id, use in zip(gts_rpid, preds_rpid, eval_mask):
        if not use: continue
        gt_info = rp_map.get(int(gt_id), None)
        pr_info = rp_map.get(int(pr_id), None)
        if gt_info and pr_info:
            if gt_info[2] == pr_info[2]: # same floor
                d = sqrt((gt_info[0] - pr_info[0])**2 + (gt_info[1] - pr_info[1])**2)
                mde_distances.append(d)
    
    avg_mde = np.mean(mde_distances) if mde_distances else float("nan")

    print(f"Final Test Accuracy: {acc:.4f}")
    print(f"Final MDE (same floor): {avg_mde:.4f}")

if __name__ == "__main__":
    main()