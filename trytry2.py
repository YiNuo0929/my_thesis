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
# NEW: Gated Transformer Encoder Layer for ViT
# 這是為了配合論文提出的 Gated Attention 機制 (SDPA Output Gating)
# 邏輯： Attention Output (Y) 乘上 Sigmoid(Gate Projection(X))
# =========================================================================
class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, batch_first=True):
        super().__init__()
        # 1. Standard Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # 2. Gate Projection (核心修改)
        # 輸入是 hidden state (X)，輸出是與 Attention Output 同維度的 Gate Score
        self.gate_linear = nn.Linear(d_model, d_model)

        # 3. Standard Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 4. Norms & Dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: [Batch, Num_Patches, Dim] (batch_first=True)
        """
        # --- Part 1: Attention ---
        # attn_output: [B, T, D]
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # --- Part 2: Gating (論文 G1 機制) ---
        # 計算 Gate Score: sigma(X * W)
        # src 是目前的 hidden state (Query 來源)
        gate_score = torch.sigmoid(self.gate_linear(src)) 
        
        # 應用 Gate: Y' = Y * Gate
        # 這裡會依賴輸入 src 來決定是否要保留 attn_output
        gated_output = attn_output * gate_score
        
        # --- Part 3: Residual & Norm ---
        # 原本是 src + dropout(attn_output)，現在變成 src + dropout(gated_output)
        src = src + self.dropout1(gated_output)
        src = self.norm1(src)

        # --- Part 4: Feed Forward (保持不變) ---
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

# -------- ViT-style Extractor (2D, side x side, patch-based) --------
class TransformerExtractor(nn.Module):
    """
    ViT-style：
    - input:  [B, 1, L]  (L = num_tokens，會 reshape 成 side x side)
    - 支援 Gated Attention (use_gating=True)
    """
    def __init__(
        self,
        num_tokens: int,          # 給 ViT 的 token 數，例如 1024
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool,
        mask_value: float,        # 哪個值當作 padding/missing (normalized -1)
        z_dim: int,               # 壓縮後 latent 維度
        use_mask: bool,
        bottleneck_hidden: int = None,
        use_gating: bool = True,  # 新增：控制是否使用 Gated Attention
    ):
        super().__init__()
        # ---- 2D reshape 設定 ----
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"num_tokens={num_tokens} 不能剛好 reshape 成正方形")
        self.side = side
        self.num_tokens = num_tokens
        self.patch_size = 2           # 固定用 4x4 (這裡代碼是 2，對應下面 stride 邏輯)
        if self.side % self.patch_size != 0:
            raise ValueError(f"side={self.side} 不能被 patch_size={self.patch_size} 整除")
        self.num_patches_per_side = self.side // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        patch_dim = 1 * self.patch_size * self.patch_size  # 單通道，所以 C=1

        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.z_dim = z_dim
        self.use_mask = use_mask
        self.mask_value = mask_value
        self.use_gating = use_gating

        # patch flatten -> d_model
        self.patch_proj = nn.Linear(patch_dim, d_model)

        # CLS token（可選）
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            max_len = self.num_patches + 1
        else:
            self.cls_token = None
            max_len = self.num_patches

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # ---- Encoder Layer 選擇 ----
        if self.use_gating:
            # 使用自定義 Gated Layer
            # 因為 nn.Sequential 對 forward 参数傳遞支援有限，我們用 ModuleList
            layers = [
                GatedTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ]
            self.encoder = nn.ModuleList(layers)
        else:
            # 標準 TransformerEncoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- bottleneck MLP ----
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
        if L != self.num_tokens:
            raise ValueError(f"輸入長度 L={L} 和 num_tokens={self.num_tokens} 不一致")

        # ---- 轉成 2D 影像 [B,1,H,W] ----
        img = x.view(B, 1, self.side, self.side)

        # ---- 用 unfold 切 patch ----
        patches_raw = F.unfold(
            img, kernel_size=self.patch_size, stride=self.patch_size
        )
        patches = patches_raw.transpose(1, 2)

        # ---- 針對 patch 做 mask（如果啟用）----
        key_padding_mask = None
        if self.use_mask:
            # 如果 patch 內所有值都是 missing_value，則 mask 掉
            eq_mask = (patches_raw == self.mask_value)
            key_padding_mask = eq_mask.all(dim=1)

        # ---- patch embedding ----
        h = self.patch_proj(patches)

        # ---- 加 CLS token ----
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            h = torch.cat([cls, h], dim=1)

        # ---- 位置編碼 ----
        h = self.pos_encoding(h)

        # ---- 準備給 encoder 的 key_padding_mask ----
        src_key_padding_mask = None
        if key_padding_mask is not None:
            if self.use_cls_token:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
                src_key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)
            else:
                src_key_padding_mask = key_padding_mask

        # ---- Encoder Forward (區分 Gated vs Standard) ----
        if self.use_gating:
            # ModuleList 需要手動迴圈
            for layer in self.encoder:
                h = layer(h, src_key_padding_mask=src_key_padding_mask)
        else:
            # Standard TransformerEncoder
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # ---- 取 CLS 或 mean pooling ----
        if self.use_cls_token:
            cls_feat = h[:, 0, :]
        else:
            cls_feat = h.mean(dim=1)

        z = self.bottleneck(cls_feat)
        return z

# -------- Predictor / Reconstruction (保持不變) --------
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

# -------- 整體模型 (Updated) --------
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
        use_gating: bool = True, # 新增
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
            use_gating=use_gating # 傳遞
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

# ---------- Metrics / Maps / Loss (保持不變) ----------
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
    parser.add_argument("--z_dim", type=int, default=16)

    parser.add_argument("--lambda_recon", type=float, default=0.1)

    parser.add_argument("--use_mask", action="store_true", help="啟用 key_padding_mask")
    # 預設使用 Gating，加 flag 則關閉
    parser.add_argument("--no_gating", action="store_true", help="關閉 Gated Attention")
    
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
        def __len__(self):
            return len(y_te_idx)
        def __getitem__(self, i):
            return torch.from_numpy(X_te_full[i]), torch.tensor(y_te_idx[i], dtype=torch.long), int(y_te_raw[i])

    ds_te = TestDataset()
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_ap = len(ap_cols)
    vit_tokens = args.vit_tokens
    
    use_gating = not args.no_gating

    model = TransClassifier(
        input_dim=num_ap,
        vit_tokens=vit_tokens,
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
        use_gating=use_gating # 傳入
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    print("---------------------------------------")
    print(f"ViT-style Model Configured")
    print(f"use mask: {model.extractor.use_mask}")
    print(f"use gating: {model.extractor.use_gating}")
    print(f"z_dim: {model.extractor.z_dim}")
    print(f"d_model: {model.extractor.d_model}")
    print(f"input_dim={model.input_dim}, vit_tokens={model.vit_tokens}")
    print(f"side={model.extractor.side}, patch_size={model.extractor.patch_size}, num_patches={model.extractor.num_patches}")
    print("---------------------------------------")

    # --- Train (保持不變) ---
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
            if not keep.any():
                continue
            xb_s, yb_s = xb_s[keep], yb_s[keep]

            xb_s = xb_s.to(device, non_blocking=True)
            yb_s = yb_s.to(device, non_blocking=True)
            xb_t = next(tgt_iter)
            xb_t = xb_t.to(device, non_blocking=True)

            opt.zero_grad()

            logits_s, recon_s = model(xb_s)
            ce_loss = crit(logits_s, yb_s)

            x_s_norm = xb_s.squeeze(1)
            x_t_norm = xb_t.squeeze(1)

            recon_loss_s = reconstruction_loss(recon_s, x_s_norm, mask_value=-1.0)
            _, recon_t = model(xb_t)
            recon_loss_t = reconstruction_loss(recon_t, x_t_norm, mask_value=-1.0)

            recon_loss_all = recon_loss_s + recon_loss_t
            total_loss = ce_loss + args.lambda_recon * recon_loss_all

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
        else:
            tr_loss_total = 0.0

        print(f"Epoch {epoch:03d} | "
              f"total {tr_loss_total:.4f} | CE {tr_ce_total:.4f} acc {tr_acc:.4f} | "
              f"recon_s {tr_recon_s_total:.4f} | recon_t {tr_recon_t_total:.4f}")

    # --- Eval (保持不變) ---
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
    if eval_mask.any():
        acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item()
    else:
        acc = float("nan")

    rp_map = load_rp_map(args.rp_map_path)
    preds_rpid = np.array([idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)

    mde_distances = []
    floor_mismatch = 0
    mde_skipped_notfound = 0

    for gt_id, pr_id, use in zip(gts_rpid, preds_rpid, eval_mask):
        if not use:
            continue
        gt_info = rp_map.get(int(gt_id), None)
        pr_info = rp_map.get(int(pr_id), None)
        if (gt_info is None) or (pr_info is None):
            mde_skipped_notfound += 1
            continue
        gx, gy, gf = gt_info
        px, py, pf = pr_info
        if gf != pf:
            floor_mismatch += 1
        else:
            d = sqrt((gx - px)**2 + (gy - py)**2)
            mde_distances.append(d)

    avg_mde = (float(np.mean(mde_distances)) if len(mde_distances) > 0 else float("nan"))

    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Test Accuracy               : {acc:.4f}")
    print(f"Mean Distance Error         : {avg_mde:.4f}")
    print(f"Floor mismatches             : {floor_mismatch}")
    print(f"Skipped                      : {mde_skipped_notfound}")

if __name__ == "__main__":
    main()