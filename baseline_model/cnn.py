# rssi_cnn_baseline_minmax.py
# Usage:
#   python rssi_cnn_baseline_minmax.py ^
#     --train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\01\train_all\all_trn_merged.csv" ^
#     --test_path  "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\01\test_all\all_tst_merged.csv" ^
#     --rp_map_path "C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv" ^
#     --epochs 50 --batch_size 256 --lr 1e-3
#
# 說明：
# - 每 AP 做 Min–Max 正規化到約 [0,1]
# - 缺值 (-110) 以該 AP 的最小值補
# - CNN 1D 模型做 RP 分類，最後計算 Accuracy 與 MDE

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
    ap_cols = [f"ap{i}" for i in range(256)]
    missing = [c for c in ap_cols if c not in all_cols]
    if missing:
        raise ValueError(f"資料集中缺少欄位（ap0~ap255）：{missing[:10]} ...")
    return ap_cols

# ---------- Min–Max Scaler ----------
def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    """
    為每個 AP 計算 min / max，用於 Min–Max Normalization。
    忽略 missing_val。
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
                # 避免除以 0
                mx = mn + 1.0
        else:
            # 若整個 AP 都是缺值，給預設範圍
            mn, mx = -100.0, -30.0
        mins[j] = mn
        maxs[j] = mx
    return mins, maxs

def apply_scaler(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray, missing_val: float = -110.0):
    """
    對輸入 x 做 per-AP Min–Max Normalization：
      1. 缺值補成該 AP 的 min
      2. (x - min) / (max - min)
    """
    x = x.copy()
    miss_mask = (x == missing_val)

    # 用每個 AP 的 min 補缺失
    if miss_mask.any():
        x[miss_mask] = np.take(mins, np.where(miss_mask)[1])

    x = (x - mins) / (maxs - mins)
    return x

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

        # 轉成 [N, 1, L] 給 1D CNN 用
        self.X = np.expand_dims(self.X, axis=1)

        # rp_id -> [0..C-1]
        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        # 不在映射的（例如 -1）設為 -1
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

# ---------- Model ----------
class CNN1DClassifier(nn.Module):
    def __init__(self, in_len: int, n_classes: int):
        super().__init__()
        # in: [B, 1, L]
        self.feat = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                # L/2

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                # L/4

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1)         # -> [B, 256, 1]
        )
        self.cls = nn.Sequential(
            nn.Flatten(),                    # [B, 256]
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.feat(x)
        return self.cls(x)

def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def load_rp_map(rp_map_path: str):
    """ 讀 rp_id 對應座標與樓層，回傳 dict: rid -> (x,y,floor) """
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
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, default=r"C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./rssi_cnn_minmax_ckpt")
    args = parser.parse_args()

    # os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data ---
    df_tr = load_csvs(args.train_path)
    df_te = load_csvs(args.test_path)
    ap_cols = get_feature_cols(df_tr)
    assert set(ap_cols).issubset(df_te.columns), "Test 缺少部分 AP 欄位"

    # 類別檢查
    n_classes_tr = df_tr["rp_id"].nunique()
    if n_classes_tr != 48:
        print(f"[WARN] 訓練集 rp 類別數={n_classes_tr}（預期 48）")

    # --- Fit scaler on train ---
    mins, maxs = fit_scaler(df_tr[ap_cols].values.astype(np.float32), missing_val=args.missing_val)
    # np.savez(os.path.join(args.out_dir, "scaler_ap_mins_maxs.npz"), mins=mins, maxs=maxs)

    # --- Build train dataset/loader ---
    ds_tr = RSSIDataset(df_tr, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_tr.id2idx
    idx2id = ds_tr.idx2id

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # --- Prepare test arrays (只在最後 eval 用) ---
    X_te_full = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full, mins, maxs, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)

    # y 映射（未知類別或 -1 → -1）
    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, dtype=int)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

    # Test DataLoader（保留全部樣本，評估時再用 mask 過濾）
    class TestDataset(Dataset):
        def __len__(self): return len(y_te_idx)
        def __getitem__(self, i):
            # 傳回 idx label（可能為 -1）
            return torch.from_numpy(X_te_full[i]), torch.tensor(y_te_idx[i], dtype=torch.long), int(y_te_raw[i])

    ds_te = TestDataset()
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model / Optim ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DClassifier(in_len=len(ap_cols), n_classes=len(id2idx)).to(device)
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

            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            bs = yb.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += (logits.argmax(1) == yb).float().sum().item()
            n += bs

        tr_loss = tr_loss / n if n > 0 else 0.0
        tr_acc  = tr_acc / n if n > 0 else 0.0
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f}")

    # --- Final Evaluation on Test (一次) ---
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

    avg_mde = float(np.mean(mde_distances)) if len(mde_distances) > 0 else float("nan")

    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")

if __name__ == "__main__":
    main()
