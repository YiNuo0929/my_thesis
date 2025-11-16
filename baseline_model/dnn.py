# rssi_dnn_baseline_minmax.py
# Usage:
#   python rssi_dnn_baseline_minmax.py ^
#     --train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\01\train_all\all_trn_merged.csv" ^
#     --test_path  "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\01\test_all\all_tst_merged.csv" ^
#     --rp_map_path "C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv" ^
#     --epochs 50 --batch_size 256 --lr 1e-3
#
# 說明：
# - 每 AP 正規化 (Min–Max Normalization)
# - 缺值 (-110) 以該 AP 最小值補
# - 模型：多層全連接 MLP + BatchNorm + ReLU + Dropout
# - 評估輸出：Test Accuracy、MDE、樓層不一致數等

import argparse, os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt

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


def get_feature_cols(df: pd.DataFrame):
    """
    改成固定只抓 ap0 ~ ap255 共 256 維
    """
    all_cols = set(df.columns)
    ap_cols = [f"ap{i}" for i in range(168)]
    missing = [c for c in ap_cols if c not in all_cols]
    if missing:
        raise ValueError(f"資料集中缺少欄位（ap0~ap255）：{missing[:10]} ...")
    return ap_cols


# ---------- Min–Max Normalization ----------
def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    """
    為每個 AP 計算 min / max，用於 Min–Max Normalization。
    忽略 missing_val 的樣本。
    """
    mask = train_ap != missing_val
    mins = np.zeros(train_ap.shape[1], dtype=np.float32)
    maxs = np.zeros(train_ap.shape[1], dtype=np.float32)

    for j in range(train_ap.shape[1]):
        col = train_ap[:, j]
        m = mask[:, j]
        if m.any():
            valid = col[m]
            mn = valid.min()
            mx = valid.max()
            if abs(mx - mn) < 1e-6:
                # 避免除以 0 的情況
                mx = mn + 1.0
        else:
            mn, mx = -100.0, -30.0  # 預設範圍
        mins[j] = mn
        maxs[j] = mx
    return mins, maxs


def apply_scaler(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray, missing_val: float = -110.0):
    """
    對輸入 x 做 per-AP Min–Max Normalization：
      (x - min) / (max - min)
    缺值以該 AP 的最小值補。
    """
    x = x.copy()
    miss_mask = (x == missing_val)

    # 用各 AP 的最小值補缺失
    if miss_mask.any():
        x[miss_mask] = np.take(mins, np.where(miss_mask)[1])

    # per-AP min-max normalization
    x = (x - mins) / (maxs - mins)
    return x


# ---------- Dataset ----------
class RSSIDataset(Dataset):
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

        # 對齊 CNN：維度 [N, 1, L]
        self.X = np.expand_dims(self.X, axis=1)

        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)


# ---------- Model ----------
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


class DNNClassifier(nn.Module):
    """
    RSSI fingerprint DNN 模型：多層全連接 + BN + ReLU + Dropout。
    """
    def __init__(self, in_len: int, n_classes: int, hidden=[512, 256], p_drop=0.2):
        super().__init__()
        dims = [in_len] + hidden
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks.append(MLPBlock(a, b, p_drop=p_drop))
        self.feat = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(dims[-1], n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.feat(x)
        logits = self.head(x)
        return logits


# ---------- Evaluation ----------
def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id", "x", "y", "floor"}
    if not need.issubset(df.columns):
        raise ValueError(f"rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, default=r"C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./rssi_dnn_minmax_ckpt")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    # --- Load data ---
    df_tr = load_csvs(args.train_path)
    df_te = load_csvs(args.test_path)
    ap_cols = get_feature_cols(df_tr)
    assert set(ap_cols).issubset(df_te.columns), "Test 缺少部分 AP 欄位"

    n_classes_tr = df_tr["rp_id"].nunique()
    if n_classes_tr != 48:
        print(f"[WARN] 訓練集 rp 類別數={n_classes_tr}（預期 48）")

    # --- Fit scaler on train ---
    mins, maxs = fit_scaler(df_tr[ap_cols].values.astype(np.float32), missing_val=args.missing_val)

    # --- Build train dataset/loader ---
    ds_tr = RSSIDataset(df_tr, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_tr.id2idx
    idx2id = ds_tr.idx2id
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # --- Prepare test arrays ---
    X_te_full = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full, mins, maxs, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)

    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, dtype=int)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

    class TestDataset(Dataset):
        def __len__(self): return len(y_te_idx)
        def __getitem__(self, i):
            return torch.from_numpy(X_te_full[i]), torch.tensor(y_te_idx[i], dtype=torch.long), int(y_te_raw[i])

    ds_te = TestDataset()
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model / Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNClassifier(in_len=len(ap_cols), n_classes=len(id2idx),
                          hidden=args.hidden, p_drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        for xb, yb in dl_tr:
            keep = yb != -1
            if not keep.any():
                continue
            xb, yb = xb[keep], yb[keep]
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            bs = yb.size(0)
            tr_loss += loss.item() * bs
            tr_acc += (logits.argmax(1) == yb).float().sum().item()
            n += bs

        tr_loss /= n
        tr_acc /= n
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f}")

    # --- Test ---
    model.eval()
    preds_idx, gts_idx, gts_rpid = [], [], []
    with torch.no_grad():
        for xb, yb_idx, yb_rpid in dl_te:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            preds_idx.append(pred)
            gts_idx.append(yb_idx.numpy())
            gts_rpid.append(yb_rpid.numpy())

    preds_idx = np.concatenate(preds_idx)
    gts_idx = np.concatenate(gts_idx)
    gts_rpid = np.concatenate(gts_rpid)

    eval_mask = (gts_idx != -1)
    acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item() if eval_mask.any() else float("nan")

    # --- MDE ---
    rp_map = load_rp_map(args.rp_map_path)
    preds_rpid = np.array([idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)
    mde_distances, floor_mismatch, mde_skipped_notfound = [], 0, 0

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

    avg_mde = float(np.mean(mde_distances)) if mde_distances else float("nan")

    print("\n==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} m" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")


if __name__ == "__main__":
    main()
