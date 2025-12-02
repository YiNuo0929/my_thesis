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
    Min-Max scaler：
    - 只用非 missing_val 的值算 per-AP min / max
    - 之後會把資料壓到 [0,1]，缺失值之後會被設成 -1
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
    Min-Max 正規化到 [0,1]，缺失值補 -1：
    - 對非 missing 的位置做 (x - min) / (max - min)，clip 到 [0,1]
    - 原本是 missing_val 的位置，最後強制設為 -1
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

# ---------- Augmentation & Consistency Loss ----------
def augment_rssi(x: torch.Tensor, noise_std=0.05, drop_prob=0.1, missing_val=-1.0):
    """
    對 Target RSSI 進行增強：
    1. Jitter: 對非缺失值加上高斯雜訊。
    2. Masking: 隨機將部分非缺失值設為 missing_val (模擬 AP 消失)。
    x shape: [B, 1, L] or [B, L]
    """
    x_aug = x.clone()
    
    # 找出有訊號的地方 (mask 為 True 表示有值)
    valid_mask = (x_aug != missing_val)
    
    # 1. 訊號飄移 (Jitter)
    if noise_std > 0:
        noise = torch.randn_like(x_aug) * noise_std
        jittered = x_aug + noise
        # 數值截斷在 [0, 1] 之間 (因為原本資料是 Min-Max normalized)
        jittered = torch.clamp(jittered, 0.0, 1.0)
        # 將 noise 應用回去 (只改 valid 的部分，避免改到 -1)
        x_aug[valid_mask] = jittered[valid_mask]
    
    # 2. AP 消失 (Random Masking/Dropout)
    if drop_prob > 0:
        # 產生一個與 x 形狀相同的隨機遮罩
        drop_mask = torch.rand_like(x_aug) < drop_prob
        # 只有原本有效 且 被選中要 drop 的地方，才設為 missing_val
        final_drop_mask = valid_mask & drop_mask
        x_aug[final_drop_mask] = missing_val
    
    return x_aug

def consistency_loss(logits_aug, logits_clean):
    """
    計算兩個 Logits 之間的 KL Divergence。
    input (aug) 轉為 log_softmax，target (clean) 轉為 softmax
    """
    log_probs_aug = F.log_softmax(logits_aug, dim=1)
    probs_clean = F.softmax(logits_clean, dim=1)
    # KL(clean || aug) -> 我們希望 aug 的分佈趨近於 clean
    return F.kl_div(log_probs_aug, probs_clean, reduction='batchmean')

# --- NEW: Entropy Minimization Loss ---
def calc_entropy_loss(logits):
    """
    Conditional Entropy Minimization:
    H(p) = - sum( p(x) * log p(x) )
    讓模型對未標記資料的預測更有信心 (Low Entropy)
    """
    p = F.softmax(logits, dim=1)        # shape: [B, C]
    log_p = F.log_softmax(logits, dim=1) # shape: [B, C]
    # entropy per sample = - sum(p * log_p)
    entropy = -torch.sum(p * log_p, dim=1)
    return torch.mean(entropy)

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

# -------- DNN-based Extractor --------
class DNNExtractor(nn.Module):
    """
    DNN-based encoder：
    - input:  [B, 1, L] 或 [B, L]（normalized）
    - flatten → MLP → z_dim
    """
    def __init__(
        self,
        num_ap: int,
        z_dim: int,
        hidden=[512, 512],
        p_drop: float = 0.2,
    ):
        super().__init__()
        self.num_ap = num_ap
        self.z_dim = z_dim

        dims = [num_ap] + hidden
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks.append(MLPBlock(a, b, p_drop=p_drop))
        self.feat = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.proj = nn.Linear(dims[-1], z_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 1, L] or [B, L]
        return: z ∈ [B, z_dim]
        """
        if x.dim() == 3:
            # [B, 1, L] -> [B, L]
            x = x.squeeze(1)

        # x: [B, L]
        h = self.feat(x)           # [B, hidden_last]
        z = self.proj(h)           # [B, z_dim]
        return z

# -------- Predictor (MLP head) --------
class PredictorMLP(nn.Module):
    """
    接 encoder 抽出來的全局特徵 z [B, z_dim]，
    再接幾層 MLP + 最後分類器。
    """
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

# -------- Reconstruction head (MLP) --------
class ReconstructionMLP(nn.Module):
    """
    用同一個 z 走另一條 MLP，把它 reconstruct 回原始的 RSSI 向量（normalized 空間）。
    output: [B, num_ap]
    """
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
        out = self.head(x)   # [B, out_dim]
        return out

# -------- 整體模型 = DNNExtractor(產生 z) + PredictorMLP + ReconstructionMLP --------
class DNNClassifier(nn.Module):
    def __init__(
        self,
        num_ap: int,
        n_classes: int,
        z_dim: int,           # 壓縮後 latent 維度
        enc_hidden,
        pred_hidden,
        recon_hidden,
        p_drop: float,
    ):
        super().__init__()
        # encoder：學 AP 間關係並壓成 z（DNN-base）
        self.extractor = DNNExtractor(
            num_ap=num_ap,
            z_dim=z_dim,
            hidden=enc_hidden,
            p_drop=p_drop,
        )
        # predictor：分類 head，用 pred_hidden
        self.predictor = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=pred_hidden,
            p_drop=p_drop,
        )
        # recon head：重建 RSSI 的 head（輸出維度 = num_ap），用 recon_hidden
        self.reconstructor = ReconstructionMLP(
            in_dim=z_dim,
            out_dim=num_ap,
            hidden=recon_hidden,
            p_drop=p_drop,
        )

    def forward(self, x):
        # x: [B, 1, L]
        z = self.extractor(x)     # [B, z_dim]
        logits = self.predictor(z)
        recon = self.reconstructor(z)   # [B, num_ap]
        return logits, recon

# ---------- Metrics / Maps ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def load_rp_map(rp_map_path: str):
    """ 讀 rp_id 對應座標與樓層，回傳 dict: rid -> (x,y,floor) """
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
    if not need.issubset(df.columns):
        raise ValueError("rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp

# ---------- Reconstruction Loss ----------
def reconstruction_loss(pred, target, mask_value=-1.0):
    """
    pred, target: [B, L]
    只在 target != mask_value 的位置算 MSE，避免缺失值干擾。
    """
    mask = (target != mask_value)
    if mask.sum() == 0:
        return (pred * 0.0).sum()   # safe zero loss
    diff = pred - target
    diff = diff[mask]
    return torch.mean(diff * diff)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    # Source / Target 兩個 train path
    parser.add_argument("--source_train_path", type=str, required=True,
                        help="source domain train csv (有 label)")
    parser.add_argument("--target_train_path", type=str, required=True,
                        help="target domain train csv (unlabeled)")
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, default=r"rp_id.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./rssi_dnn_recon_ckpt")
    parser.add_argument("--column", type=int, default=256)

    # encoder / predictor / recon MLP 的設定
    parser.add_argument("--enc_hidden", type=int, nargs="+", default=[512, 512],
                        help="hidden sizes for encoder (DNN extractor)")
    parser.add_argument("--pred_hidden", type=int, nargs="+", default=[256, 256],
                        help="hidden sizes for predictor MLP")
    parser.add_argument("--recon_hidden", type=int, nargs="+", default=[256, 256],
                        help="hidden sizes for reconstruction MLP")
    parser.add_argument("--dropout", type=float, default=0.2)

    # 壓縮後 latent 維度 z_dim
    parser.add_argument("--z_dim", type=int, default=32)

    # reconstruction loss 的權重 λ
    parser.add_argument("--lambda_recon", type=float, default=0.1)

    # Consistency Loss 參數
    parser.add_argument("--lambda_consist", type=float, default=0.2, help="weight for consistency loss")
    parser.add_argument("--noise_std", type=float, default=0.05, help="std for gaussian noise jitter")
    parser.add_argument("--drop_prob", type=float, default=0.2, help="probability to drop AP signal")

    # NEW: Entropy Minimization 參數
    parser.add_argument("--lambda_entropy", type=float, default=0.2, help="weight for entropy minimization loss")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Load data ---
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src, args.column)   # 只會拿 ap0 ~ ap(column-1)
    # 確保 target / test 也有這些欄位
    assert set(ap_cols).issubset(df_tgt.columns), "Target train 缺少部分 AP 欄位"
    assert set(ap_cols).issubset(df_te.columns),  "Test 缺少部分 AP 欄位"

    # 類別檢查（只看 source）
    n_classes_src = df_src["rp_id"].nunique()
    if n_classes_src != 48:
        print(f"[WARN] Source 訓練集 rp 類別數={n_classes_src}（預期 48）")

    # --- Fit scaler on source train (Min-Max) ---
    mins, maxs = fit_scaler(df_src[ap_cols].values.astype(np.float32), missing_val=args.missing_val)

    # --- Build source / target dataset/loader ---
    ds_src = RSSISourceDataset(df_src, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_src.id2idx
    idx2id = ds_src.idx2id
    dl_src = DataLoader(ds_src, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # --- Prepare test arrays (只在最後 eval 用) ---
    X_te_full_raw = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full_raw, mins, maxs, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)

    # y 映射（未知類別或 -1 → -1）
    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, dtype=int)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

    class TestDataset(Dataset):
        def __len__(self):
            return len(y_te_idx)
        def __getitem__(self, i):
            return torch.from_numpy(X_te_full[i]), torch.tensor(y_te_idx[i], dtype=torch.long), int(y_te_raw[i])

    ds_te = TestDataset()
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model / Optim ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_ap = len(ap_cols)

    model = DNNClassifier(
        num_ap=num_ap,
        n_classes=len(id2idx),
        z_dim=args.z_dim,
        enc_hidden=args.enc_hidden,
        pred_hidden=args.pred_hidden,
        recon_hidden=args.recon_hidden,
        p_drop=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    print("---------------------------------------")
    print(f"z_dim: {model.extractor.z_dim}")
    print(f"num_ap={num_ap}")
    print(f"enc_hidden={args.enc_hidden}")
    print(f"pred_hidden={args.pred_hidden}, recon_hidden={args.recon_hidden}")
    print(f"Consistency: λ={args.lambda_consist}, noise={args.noise_std}, drop={args.drop_prob}")
    print(f"Entropy: λ={args.lambda_entropy}")
    print("---------------------------------------")

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_total = 0.0
        tr_ce_total = 0.0
        tr_recon_s_total = 0.0
        tr_recon_t_total = 0.0
        tr_consist_total = 0.0 
        tr_ent_total = 0.0 # NEW
        tr_acc, n = 0.0, 0

        tgt_iter = cycle(dl_tgt)  # target 比較少就循環使用

        for xb_s, yb_s in dl_src:
            keep = (yb_s != -1)
            if not keep.any():
                continue
            xb_s, yb_s = xb_s[keep], yb_s[keep]

            xb_s = xb_s.to(device, non_blocking=True)
            yb_s = yb_s.to(device, non_blocking=True)

            # 取一個 target batch（不需要 label）
            xb_t = next(tgt_iter)
            xb_t = xb_t.to(device, non_blocking=True)

            # Target Augmentation
            with torch.no_grad():
                xb_t_aug = augment_rssi(xb_t, noise_std=args.noise_std, drop_prob=args.drop_prob, missing_val=-1.0)
                xb_t_aug = xb_t_aug.to(device, non_blocking=True)

            opt.zero_grad()

            # 1. Source Flow: Predictor(CE) + Reconstructor
            logits_s, recon_s = model(xb_s)
            ce_loss = crit(logits_s, yb_s)

            x_s_norm = xb_s.squeeze(1)  # [B, L]
            recon_loss_s = reconstruction_loss(recon_s, x_s_norm, mask_value=-1.0)

            # 2. Target Flow (Original): Reconstructor + Predictor (for consistency & entropy)
            logits_t, recon_t = model(xb_t)
            x_t_norm = xb_t.squeeze(1)  # [B, L]
            recon_loss_t = reconstruction_loss(recon_t, x_t_norm, mask_value=-1.0)

            # NEW: Conditional Entropy Loss (只算在原始 Target 上)
            loss_ent = calc_entropy_loss(logits_t)

            # 3. Target Flow (Augmented): Predictor ONLY (for consistency)
            z_aug = model.extractor(xb_t_aug)
            logits_t_aug = model.predictor(z_aug)

            # 4. Consistency Loss
            loss_consist = consistency_loss(logits_t_aug, logits_t)

            recon_loss_all = recon_loss_s + recon_loss_t
            
            # 加總所有 Loss
            total_loss = ce_loss \
                         + args.lambda_recon * recon_loss_all \
                         + args.lambda_consist * loss_consist \
                         + args.lambda_entropy * loss_ent

            total_loss.backward()
            opt.step()

            bs = yb_s.size(0)
            tr_loss_total += total_loss.item() * bs
            tr_ce_total += ce_loss.item() * bs
            tr_recon_s_total += recon_loss_s.item() * bs
            tr_recon_t_total += recon_loss_t.item() * bs
            tr_consist_total += loss_consist.item() * bs
            tr_ent_total += loss_ent.item() * bs # NEW
            tr_acc += (logits_s.argmax(1) == yb_s).float().sum().item()
            n += bs

        if n > 0:
            tr_loss_total /= n
            tr_ce_total /= n
            tr_recon_s_total /= n
            tr_recon_t_total /= n
            tr_consist_total /= n
            tr_ent_total /= n
            tr_acc /= n
        else:
            tr_loss_total = tr_ce_total = tr_recon_s_total = tr_recon_t_total = tr_consist_total = tr_ent_total = tr_acc = 0.0

        print(f"Epoch {epoch:03d} | "
              f"Tot {tr_loss_total:.4f} | CE {tr_ce_total:.4f} | "
              f"RecS {tr_recon_s_total:.3f} RecT {tr_recon_t_total:.3f} | "
              f"Cst {tr_consist_total:.4f} Ent {tr_ent_total:.4f} | Acc {tr_acc:.4f}")

    # --- Final Evaluation on Test (一次，只看 predictor 分支) ---
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

    # acc：僅計算 gts_idx != -1 的樣本
    eval_mask = (gts_idx != -1)
    if eval_mask.any():
        acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item()
    else:
        acc = float("nan")

    # --- MDE 計算 ---
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
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")

if __name__ == "__main__":
    main()