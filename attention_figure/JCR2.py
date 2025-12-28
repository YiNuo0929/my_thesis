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
    """
    固定所有常見的隨機種子，讓實驗可重現。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 讓 cudnn 行為可重現
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
    """
    固定只抓 ap0 ~ ap(col-1) 共 col 維
    """
    all_cols = set(df.columns)
    ap_cols = [f"ap{i}" for i in range(col)]
    missing = [c for c in ap_cols if c not in all_cols]
    if missing:
        raise ValueError(f"資料集中缺少欄位（ap0~ap{col-1}）：{missing[:10]} ...")
    return ap_cols

def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    """
    Min-Max scaler
    """
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
    """
    Min-Max 正規化到 [0,1]，缺失值補 -1
    """
    x = x.copy().astype(np.float32)
    miss_mask = (x == missing_val)

    denom = (maxs - mins)
    denom[denom == 0.0] = 1.0
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)

    # 缺失的地方直接設成 -1
    x[miss_mask] = -1.0
    return x

class RSSISourceDataset(Dataset):
    """
    Source domain：有 label 的資料（rp_id）
    """
    def __init__(self, df: pd.DataFrame, ap_cols, label_col="rp_id",
                 mins=None, maxs=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.y_raw = df[label_col].values.astype(np.int64)  # 保留原始 rp_id（做 MDE 用）
        self.X = df[ap_cols].values.astype(np.float32)
        self.missing_val = missing_val
        self.mins = mins
        self.maxs = maxs
        if (mins is not None) and (maxs is not None):
            self.X = apply_scaler(self.X, self.mins, self.maxs, self.missing_val)
        # [N, 1, L]
        self.X = np.expand_dims(self.X, axis=1)

        # rp_id -> [0..C-1]
        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        # 不在映射的（例如 -1）設為 -1
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

class RSSITargetDataset(Dataset):
    """
    Target domain：unlabeled（不使用 rp_id，只做 reconstruction）
    """
    def __init__(self, df: pd.DataFrame, ap_cols,
                 mins=None, maxs=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.X = df[ap_cols].values.astype(np.float32)
        self.missing_val = missing_val
        self.mins = mins
        self.maxs = maxs
        if (mins is not None) and (maxs is not None):
            self.X = apply_scaler(self.X, self.mins, self.maxs, self.missing_val)
        # [N, 1, L]
        self.X = np.expand_dims(self.X, axis=1)

    def __len__(self):
        return len(self.X)

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
    def forward(self, x):
        return self.seq(x)

# -------- Positional Encoding --------
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
# Custom Standard Transformer Layer (With Weight Logging)
# 這是標準的 Transformer Layer (無 Gate)，但我們加上了儲存 Attention Weights 的功能
# 這樣才能畫出論文 Figure 2
# =========================================================================
class StandardTransformerEncoderLayerWithWeights(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        
        # [NEW] 用來暫存最後一次 forward 的 attention map
        self.last_attn_weights = None 

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Standard Transformer Encoder Layer Forward Pass
        """
        # --- Self Attention ---
        # [NEW] 設定 need_weights=True, average_attn_weights=False (保留所有 Heads)
        src2, attn_weights = self.self_attn(
            src, src, src, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask,
            need_weights=True, 
            average_attn_weights=False
        )
        
        # [NEW] 儲存權重 (detach cpu 避免佔用 gpu memory)
        self.last_attn_weights = attn_weights.detach().cpu()

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # --- Feed Forward ---
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

# -------- ViT-style Extractor (Updated for Viz) --------
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
        bottleneck_hidden: int = None,
        use_mask: bool = False,
    ):
        super().__init__()
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

        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.z_dim = z_dim
        self.use_mask = use_mask
        self.mask_value = mask_value

        self.patch_proj = nn.Linear(patch_dim, d_model)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            max_len = self.num_patches + 1
        else:
            self.cls_token = None
            max_len = self.num_patches

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # [MODIFIED] 使用自定義的 Layer 類別，方便存取 Attention Weights
        # 由於要存取內部狀態，我們不用 nn.TransformerEncoder，改用 ModuleList 手動串接
        layers = [
            StandardTransformerEncoderLayerWithWeights(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
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
        if x.dim() == 3:
            x = x.squeeze(1)

        B, L = x.shape
        img = x.view(B, 1, self.side, self.side)
        patches_raw = F.unfold(
            img, kernel_size=self.patch_size, stride=self.patch_size
        )
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

        # [MODIFIED] 手動迴圈執行 Encoder Layers
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
        logits = self.head(x)
        return logits

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
        out = self.head(x)
        return out

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
            mask_value=mask_value
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
        if x.dim() == 3:
            x_flat = x.squeeze(1)
        else:
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
# VIZ FUNCTION: Generate Plot Like Figure 2 in Paper
# =========================================================================
def visualize_attention_sink(model, test_loader, device, save_dir):
    """
    生成論文 Figure 2 風格的圖表：
    1. 左圖：每一層對 First Token (CLS) 的注意力佔比
    2. 右圖：最後一層的 Attention Matrix Heatmap
    """
    model.eval()
    
    # 1. 抓取一個 Batch 資料
    try:
        batch_data = next(iter(test_loader))
        if len(batch_data) == 3:
            xb, _, _ = batch_data
        else:
            xb = batch_data
    except Exception as e:
        print(f"Error grabbing batch: {e}")
        return

    xb = xb.to(device)
    
    # 2. 執行 Forward 以生成 Attention Weights
    print("Generating attention maps for visualization...")
    with torch.no_grad():
        _ = model(xb)

    # 3. 從模型中提取 Weights
    # 權重在 model.extractor.encoder (ModuleList) 裡的 layer.last_attn_weights
    all_layers_attn = []
    
    # 檢查是否為 ModuleList
    encoder_mod = model.extractor.encoder
    if isinstance(encoder_mod, nn.TransformerEncoder):
        # 如果是原生的 TransformerEncoder，很難抓權重，這段程式碼主要設計給 ModuleList 的版本
        print("警告：模型使用原生 TransformerEncoder，無法提取權重。Viz 跳過。")
        return
    
    for layer in encoder_mod:
        if hasattr(layer, 'last_attn_weights') and layer.last_attn_weights is not None:
            # Shape: [Batch, Heads, Seq_Len, Seq_Len]
            all_layers_attn.append(layer.last_attn_weights)
    
    if not all_layers_attn:
        print("No attention weights found.")
        return

    num_layers = len(all_layers_attn)
    
    # ==============================
    # Plot 1: First Token Attention Score (Line Chart)
    # ==============================
    first_token_scores = []
    
    # 我們取 Batch 中的第一個樣本來分析 (或者取平均)
    # 這裡我們取整個 Batch 的平均
    for layer_attn in all_layers_attn:
        # layer_attn: [Batch, Heads, Queries(L), Keys(L)]
        # 關注點：Query 是任意 Token，Key 是 First Token (Index 0)
        # 即 [:, :, :, 0]
        score = layer_attn[:, :, :, 0].mean().item()
        first_token_scores.append(score)
        
    plt.figure(figsize=(12, 5))
    
    # Subplot 1
    plt.subplot(1, 2, 1)
    plt.plot(range(num_layers), first_token_scores, marker='o', linewidth=2, label='Baseline (No Gate)')
    
    # 參考線：如果注意力是均勻分佈的 (1/L)
    seq_len = all_layers_attn[0].shape[-1]
    plt.axhline(y=1/seq_len, color='gray', linestyle='--', label='Uniform (1/L)')
    
    plt.title("Baseline: Attention Sink Analysis")
    plt.xlabel("Layer Index")
    plt.ylabel("First Token (CLS) Attention Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ==============================
    # Plot 2: Heatmap (Last Layer)
    # ==============================
    plt.subplot(1, 2, 2)
    target_layer_idx = num_layers - 3
    
    # 取第一個樣本 (Batch 0)，取所有 Heads 的平均
    # Shape: [Seq_Len, Seq_Len]
    attn_matrix = all_layers_attn[target_layer_idx][0].mean(dim=0).numpy()
    
    # 只畫前 50x50 以避免太密
    display_dim = min(50, seq_len)
    
    sns.heatmap(attn_matrix[:display_dim, :display_dim], cmap="viridis", vmin=0, vmax=0.2)
    plt.title(f"Baseline Heatmap (Layer {target_layer_idx})")
    plt.xlabel("Key Token Index")
    plt.ylabel("Query Token Index")
    
    # 標記 CLS
    plt.text(0.5, -0.5, "CLS", color='red', ha='center', fontweight='bold', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "baseline_attention_viz.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")
    # plt.show() # 如果是在 headless server 跑，這一行可註解掉

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vit_tokens", type=int, default=256)

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Load data ---
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

    X_te_full_raw = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full_raw, mins, maxs, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)
    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, dtype=int)
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
    
    # 這是 Baseline (無 Gate)
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
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    # Create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    print("---------------------------------------")
    print(f"Baseline Model (No Gate, With Viz Hook)")
    print(f"d_model: {args.d_model}, layers: {args.num_layers}")
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

    # --- Final Viz ---
    print("\n[Viz] Generating Figure 2 style plots...")
    visualize_attention_sink(model, dl_te, device, args.out_dir)
    
    # --- Eval ---
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
    
    eval_mask = (gts_idx != -1)
    acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item() if eval_mask.any() else float("nan")

    print(f"Final Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()