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
import random  # NEW

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

def get_feature_cols_fixed_168(df: pd.DataFrame):
    """
    只抓 ap0 ~ ap167 共 168 維
    """
    ap_cols_all = set(df.columns)
    ap_cols = [f"ap{i}" for i in range(168)]
    missing = [c for c in ap_cols if c not in ap_cols_all]
    if missing:
        raise ValueError(f"資料集中缺少欄位: {missing[:10]} ...")
    return ap_cols

def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    """
    Min-Max scaler：
    - 只用非 missing_val 的值算 per-AP min / max
    - 之後會把資料壓到 [0,1]
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
    Min-Max 正規化到 [0,1]：
    - 對非 missing 的位置做 (x - min) / (max - min)，clip 到 [0,1]
    - 原本是 missing_val 的位置最後設為 -1（給 mask 用）
    """
    x = x.copy().astype(np.float32)
    miss_mask = (x == missing_val)

    denom = (maxs - mins)
    denom[denom == 0.0] = 1.0
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)

    x[miss_mask] = -1.0
    return x

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask_val: float = -1.0):
    """
    只在 target != mask_val 的位置上算 MSE。
    這裡 mask_val=-1.0 對應「缺失 AP」。
    pred, target 形狀一樣，例如 [B,1,L]
    """
    mask = (target != mask_val)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - target) ** 2
    return diff[mask].mean()

# ---------- Dataset ----------
class RSSIDataset(Dataset):
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

        # [N, 1, L]，ViT Extractor 會再 reshape 成 4x42
        self.X = np.expand_dims(self.X, axis=1)  # [N, 1, L]

        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

class RSSITargetDataset(Dataset):
    """
    專門給 target domain 用的 Dataset：只有 X，沒有 label。
    """
    def __init__(self, df: pd.DataFrame, ap_cols, mins=None, maxs=None, missing_val=-110.0):
        X = df[ap_cols].values.astype(np.float32)
        if (mins is not None) and (maxs is not None):
            X = apply_scaler(X, mins, maxs, missing_val)
        self.X = np.expand_dims(X, axis=1)  # [N, 1, L]

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

# -------- ViT-style Extractor (4x42, patch PxP) --------
class ViTExtractor(nn.Module):
    """
    把一維 RSSI (L tokens) reshape 成 4x42「圖」，再用 ViT 風格處理：
    - 只取 ap0~ap167 → L=168 → 4x42
    - input:  x ∈ [B, 1, L] 或 [B, L]
    - 轉成影像 [B, 1, 4, 42]
    - patchify: patch_size = P → PxP patch
    - Linear 映射到 d_model、加 CLS + learnable pos embedding
    - TransformerEncoder
    - 取 CLS → bottleneck MLP → z_dim
    """
    def __init__(
        self,
        num_tokens: int,          # 期望 = 168
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        z_dim: int,
        patch_size: int,
        use_mask: bool = True,
        mask_value: float = -1.0,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.mask_value = mask_value
        self.d_model = d_model

        # 這裡固定設計成 4x42（4 * 42 = 168）
        self.image_h = 4
        self.image_w = 42
        if self.image_h * self.image_w != num_tokens:
            raise ValueError(f"num_tokens={num_tokens} 無法 reshape 成 4x42。")

        self.patch_size = patch_size
        self.in_chans = 1

        if (self.image_h % self.patch_size != 0) or (self.image_w % self.patch_size != 0):
            raise ValueError(f"image_h={self.image_h}, image_w={self.image_w} 無法被 patch_size={self.patch_size} 整除。")

        self.num_patches = (self.image_h // self.patch_size) * (self.image_w // self.patch_size)
        patch_dim = self.in_chans * (self.patch_size ** 2)

        # patch embedding：每個 patch flatten 後 Linear → d_model
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # CLS token + learnable pos embedding（ViT 標準作法）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # bottleneck: CLS(d_model) → z_dim
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, z_dim)
        )

        # 參數初始化（簡單一點：normal 初始化 pos & cls）
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 1, L] or [B, L], L 應該是 168
        return: z ∈ [B, z_dim]
        """
        if x.dim() == 3:
            # [B, 1, L] -> [B, L]
            x = x.squeeze(1)

        B, L = x.shape
        if L != self.image_h * self.image_w:
            raise ValueError(f"輸入長度 L={L} 與 image_h*image_w={self.image_h*self.image_w} 不符。")

        # reshape 成影像 [B, 1, 4, 42]
        img = x.view(B, 1, self.image_h, self.image_w)  # [B,1,H,W]

        # patchify：使用 F.unfold，kernel_size=(P,P), stride=(P,P)
        patches = F.unfold(
            img,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size)
        )  # [B, C*ps*ps, N_patches]
        patches = patches.transpose(1, 2)  # [B, N_patches, patch_dim]

        # patch-level mask（如果整個 patch 都是 mask_value，就當成 padding）
        src_key_padding_mask = None
        if self.use_mask:
            mask_map = (img == self.mask_value).float()  # [B,1,H,W]
            mask_patches = F.unfold(
                mask_map,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size)
            )
            mask_patches = (mask_patches.mean(dim=1) == 1.0)  # [B, N_patches] bool
            src_key_padding_mask = mask_patches  # 先不含 CLS

        # patch → embedding
        h = self.patch_embed(patches)  # [B, N_patches, d_model]

        # CLS token
        cls = self.cls_token.expand(B, 1, self.d_model)  # [B,1,d_model]
        h = torch.cat([cls, h], dim=1)                   # [B, 1+N_patches, d_model]

        # pos embedding
        h = h + self.pos_embed  # [B, 1+N_patches, d_model]

        # 準備 key_padding_mask（要補 CLS 的 False）
        if src_key_padding_mask is not None:
            pad_cls = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
            src_key_padding_mask = torch.cat([pad_cls, src_key_padding_mask], dim=1)  # [B, 1+N]

        # ViT backbone = TransformerEncoder
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # [B, 1+N_patches, d_model]

        # 取 CLS
        cls_feat = h[:, 0, :]  # [B, d_model]

        # bottleneck → z
        z = self.bottleneck(cls_feat)  # [B, z_dim]
        return z

# -------- Predictor (MLP head) --------
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

# -------- Reconstructor (z -> fingerprint) --------
class Reconstructor(nn.Module):
    """
    從 latent z 重建回原始 fingerprint：
    - input:  z [B, z_dim]
    - output: x_hat [B, 1, L]，這裡 L = num_ap
    """
    def __init__(self, z_dim: int, num_ap: int, hidden=[256]):
        super().__init__()
        dims = [z_dim] + hidden + [num_ap]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        x_hat = self.mlp(z)        # [B, num_ap]
        x_hat = x_hat.unsqueeze(1) # [B, 1, L]
        return x_hat

# -------- 整體模型：ViT Extractor + PredictorMLP + Reconstructor --------
class TransReconClassifier(nn.Module):
    """
    - extractor：ViT 版本 (4x42, patch PxP)
    - predictor：分類 head（做 RP 分類）
    - reconstructor：重建 fingerprint 的分支
    forward 回傳 (logits, x_hat)
    """
    def __init__(
        self,
        num_ap: int,
        n_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        z_dim: int,
        mlp_hidden,
        p_drop: float,
        use_mask: bool,
        mask_value: float,
        patch_size: int,
        recon_hidden,
    ):
        super().__init__()
        self.extractor = ViTExtractor(
            num_tokens=num_ap,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            z_dim=z_dim,
            patch_size=patch_size,
            use_mask=use_mask,
            mask_value=mask_value,
        )
        self.predictor = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=mlp_hidden,
            p_drop=p_drop,
        )
        self.reconstructor = Reconstructor(
            z_dim=z_dim,
            num_ap=num_ap,
            hidden=recon_hidden,
        )

    def forward(self, x):
        z = self.extractor(x)
        logits = self.predictor(z)
        x_hat = self.reconstructor(z)
        return logits, x_hat

# ---------- Metrics / Maps ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id", "x", "y", "floor"}
    if not need.issubset(df.columns):
        raise ValueError("rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp

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
    parser.add_argument("--out_dir", type=str, default="./rssi_vit_recon_ckpt_168")

    # predictor MLP 的設定
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--dropout", type=float, default=0.2)

    # Transformer / ViT 的超參數
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=32)

    # ViT patch 大小（P → PxP）
    parser.add_argument("--patch_size", type=int, default=2)

    # 是否啟用 key_padding_mask（整個 patch 都是 -1 才會被 mask 掉）
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="啟用 key_padding_mask（patch 內全部為 -1 視為 padding）"
    )

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")

    # Reconstructor hidden dims + λ
    parser.add_argument("--recon_hidden", type=int, nargs="+", default=[256],
                        help="reconstructor MLP hidden dims")
    parser.add_argument("--lambda_recon", type=float, default=1.0,
                        help="total loss = CE + lambda_recon * (recon_src + recon_tgt)")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data ---
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    # 只抓 ap0 ~ ap167
    ap_cols = get_feature_cols_fixed_168(df_src)
    # 確認 target/test 也都有
    assert set(ap_cols).issubset(df_te.columns),  "Test 缺少部分 ap0~ap167 欄位"
    assert set(ap_cols).issubset(df_tgt.columns), "Target 缺少部分 ap0~ap167 欄位"

    # 類別檢查（source）
    n_classes_tr = df_src["rp_id"].nunique()
    print(f"[INFO] 訓練集 rp 類別數={n_classes_tr}")

    # --- Fit scaler on source train (Min-Max) ---
    mins, maxs = fit_scaler(df_src[ap_cols].values.astype(np.float32),
                            missing_val=args.missing_val)

    # --- Build source / target dataset/loader ---
    ds_src = RSSIDataset(df_src, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_src.id2idx
    idx2id = ds_src.idx2id
    dl_src = DataLoader(ds_src, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, mins=mins, maxs=maxs,
                               missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    it_tgt = cycle(dl_tgt)

    # --- Prepare test arrays ---
    X_te_full_raw = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full_raw, mins, maxs, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)  # [N,1,L]

    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, dtype=int)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

    class TestDataset(Dataset):
        def __len__(self):
            return len(y_te_idx)
        def __getitem__(self, i):
            return torch.from_numpy(X_te_full[i]), torch.tensor(y_te_idx[i], dtype=torch.long), int(y_te_raw[i])

    ds_te = TestDataset()
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    # --- Model / Optim ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_ap = len(ap_cols)  # 應該是 168

    model = TransReconClassifier(
        num_ap=num_ap,
        n_classes=len(id2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        z_dim=args.z_dim,
        mlp_hidden=args.hidden,
        p_drop=args.dropout,
        use_mask=args.use_mask,
        mask_value=-1.0,
        patch_size=args.patch_size,
        recon_hidden=args.recon_hidden,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit_ce = nn.CrossEntropyLoss()

    img_h, img_w = 4, 42
    num_patches = (img_h // args.patch_size) * (img_w // args.patch_size)

    print("----------------config-----------------")
    print(f"use mask or not: {model.extractor.use_mask}")
    print(f"z_dim: {args.z_dim}")
    print(f"d_model: {args.d_model}")
    print(f"num_layers: {args.num_layers}")
    print(f"nhead: {args.nhead}")
    print(f"lambda_recon: {args.lambda_recon}")
    print(f"num_ap={num_ap}, image_shape={img_h}x{img_w}, patch={args.patch_size}x{args.patch_size}, num_patches={num_patches}")
    print("---------------------------------------")

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_ce, tr_acc = 0.0, 0.0, 0.0
        tr_rec_s, tr_rec_t = 0.0, 0.0
        n_ce = 0

        for xb_s, yb_s in dl_src:
            xb_t = next(it_tgt)

            xb_s, yb_s = xb_s.to(device, non_blocking=True), yb_s.to(device, non_blocking=True)
            xb_t = xb_t.to(device, non_blocking=True)

            keep = (yb_s != -1)

            opt.zero_grad()

            # ----- Source: 有 label，CE + reconstruction -----
            logits_s, xhat_s = model(xb_s)
            if keep.any():
                ce_loss = crit_ce(logits_s[keep], yb_s[keep])
                bs_ce = keep.sum().item()
                tr_ce += ce_loss.item() * bs_ce
                tr_acc += (logits_s[keep].argmax(1) == yb_s[keep]).float().sum().item()
                n_ce += bs_ce
            else:
                ce_loss = torch.tensor(0.0, device=device)

            rec_loss_s = masked_mse(xhat_s, xb_s, mask_val=-1.0)

            # ----- Target: 無 label，只 reconstruction -----
            _, xhat_t = model(xb_t)
            rec_loss_t = masked_mse(xhat_t, xb_t, mask_val=-1.0)

            rec_loss = rec_loss_s + rec_loss_t
            loss = ce_loss + args.lambda_recon * rec_loss

            loss.backward()
            opt.step()

            bs_s = xb_s.size(0)
            tr_loss += loss.item() * bs_s
            tr_rec_s += rec_loss_s.item() * bs_s
            tr_rec_t += rec_loss_t.item() * xb_t.size(0)

        # epoch 統計
        tr_loss = tr_loss / len(ds_src) if len(ds_src) > 0 else 0.0
        tr_ce   = tr_ce   / n_ce        if n_ce > 0      else 0.0
        tr_acc  = tr_acc  / n_ce        if n_ce > 0      else 0.0
        tr_rec_s = tr_rec_s / len(ds_src) if len(ds_src) > 0 else 0.0
        tr_rec_t = tr_rec_t / len(ds_tgt) if len(ds_tgt) > 0 else 0.0

        # 你要的 print 形式
        print(f"Epoch {epoch:03d} | total {tr_loss:.4f} | CE {tr_ce:.4f} acc {tr_acc:.4f} | "
              f"recon_s {tr_rec_s:.4f} | recon_t {tr_rec_t:.4f}")

        recon_weighted = args.lambda_recon * (tr_rec_s + tr_rec_t)
        if (tr_ce + recon_weighted) > 0:
            ratio = tr_ce / (tr_ce + recon_weighted)
        else:
            ratio = float('nan')
        print(f"... | CE {tr_ce:.4f} | λ*recon {recon_weighted:.4f} | CE/(CE+λrecon) {ratio:.2f}")

    # --- Final Evaluation on Test (只用分類分支) ---
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
            d = sqrt((gx - px) ** 2 + (gy - py) ** 2)
            mde_distances.append(d)

    avg_mde = (float(np.mean(mde_distances)) if len(mde_distances) > 0 else float("nan"))

    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")

if __name__ == "__main__":
    main()