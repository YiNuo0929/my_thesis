# rssi_reconstruct_semi.py
# ...（你的檔頭與註解原樣保留）...

import argparse, os, math, random
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import matplotlib.pyplot as plt  

# -------------------- Utils: I/O --------------------
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
    ap_cols = [c for c in df.columns if c.startswith("ap")]
    if not ap_cols:
        raise ValueError("No AP columns (prefix 'ap') found.")
    return ap_cols

def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    # 與 DANN 相同：僅以 source 訓練資料估計
    mask = train_ap != missing_val
    means = np.zeros(train_ap.shape[1], dtype=np.float32)
    stds  = np.ones(train_ap.shape[1], dtype=np.float32)
    for j in range(train_ap.shape[1]):
        col = train_ap[:, j]; m = mask[:, j]
        if m.any():
            mu = col[m].mean(); sigma = col[m].std()
            if sigma < 1e-6: sigma = 1.0
        else:
            mu, sigma = -100.0, 10.0
        means[j] = mu; stds[j] = sigma
    return means, stds

def apply_scaler(x: np.ndarray, means: np.ndarray, stds: np.ndarray, missing_val: float = -110.0):
    x = x.copy()
    miss_mask = (x == missing_val)
    if miss_mask.any():
        ap_idx = np.where(miss_mask)[1]
        x[miss_mask] = means[ap_idx]
    x = (x - means) / stds
    return x

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------- Dataset --------------------
class RSSISourceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, label_col="rp_id",
                 means=None, stds=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.X_raw = df[ap_cols].values.astype(np.float32)
        self.X = apply_scaler(self.X_raw, means, stds, missing_val) if (means is not None) else self.X_raw
        self.miss_mask = (self.X_raw == missing_val).astype(np.float32)  # 1 表缺值

        self.y_raw = df[label_col].values.astype(np.int64)
        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),                # x (standardized)
            torch.tensor(self.y[i], dtype=torch.long),  # y (index)
            torch.from_numpy(self.miss_mask[i])         # mask_missing (1=missing)
        )

class RSSITargetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, means, stds, missing_val=-110.0):
        X_raw = df[ap_cols].values.astype(np.float32)
        self.X = apply_scaler(X_raw, means, stds, missing_val)
        self.miss_mask = (X_raw == missing_val).astype(np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(-1, dtype=torch.long),
            torch.from_numpy(self.miss_mask[i])
        )

# -------------------- Model --------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=None, dropout=0.4, last_activation=False):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            last = h
        if out_dim is not None:
            layers += [nn.Linear(last, out_dim)]
            if last_activation:
                layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x): return self.net(x)

class FullModel(nn.Module):
    def __init__(self, in_dim, n_classes, feat_dim=256,
                 ext_hidden=[512,512], cls_hidden=[256], rec_hidden=[256,256], dropout=0.4):
        super().__init__()
        self.extractor = MLP(in_dim, ext_hidden + [feat_dim], out_dim=None, dropout=dropout)
        self.cls_head  = MLP(feat_dim, cls_hidden, out_dim=n_classes, dropout=dropout)
        self.rec_head  = MLP(feat_dim, rec_hidden, out_dim=in_dim, dropout=dropout)

    def forward(self, x):
        f = self.extractor(x)
        logits = self.cls_head(f)
        recon  = self.rec_head(f)
        return logits, recon, f

# -------------------- Loss / Metrics --------------------
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask_missing: torch.Tensor, ignore_missing: bool = True):
    if ignore_missing:
        valid = (mask_missing == 0.0).float()
        denom = valid.sum().clamp(min=1.0)
        return ((pred - target) ** 2 * valid).sum() / denom
    else:
        return nn.functional.mse_loss(pred, target)

def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
    if not need.issubset(df.columns):
        raise ValueError(f"rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp

# -------------------- Train / Eval --------------------
@torch.no_grad()
def evaluate_classification(model, dl_te, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    for xb, yb_idx, _ in dl_te:
        xb = xb.to(device, non_blocking=True)
        yb_idx = yb_idx.to(device, non_blocking=True)
        logits, _, _ = model(xb)
        loss = ce(logits[yb_idx != -1], yb_idx[yb_idx != -1]) if (yb_idx != -1).any() else torch.tensor(0.0, device=device)
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += ((pred == yb_idx) & (yb_idx != -1)).sum().item()
        total += (yb_idx != -1).sum().item()
    acc = (correct / max(1, total)) if total > 0 else float("nan")
    return (total_loss / max(1, total)) if total > 0 else float("nan"), acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_path", type=str, required=True)
    parser.add_argument("--target_train_path", type=str, required=True)
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size_src", type=int, default=256)
    parser.add_argument("--batch_size_tgt", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--ext_hidden", type=int, nargs="+", default=[512,512])
    parser.add_argument("--cls_hidden", type=int, nargs="+", default=[256])
    parser.add_argument("--rec_hidden", type=int, nargs="+", default=[256,256])
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--ignore_missing_in_recon", action="store_true")

    parser.add_argument("--out_dir", type=str, default="./rssi_recon_ckpt")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # --------- Load DataFrames ---------
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src)
    assert set(ap_cols).issubset(df_tgt.columns), "Target 訓練資料缺少部分 AP 欄位"
    assert set(ap_cols).issubset(df_te.columns),  "Test 資料缺少部分 AP 欄位"

    # --------- Scaler (fit on source only) ---------
    means, stds = fit_scaler(df_src[ap_cols].values.astype(np.float32), missing_val=args.missing_val)
    np.savez(os.path.join(args.out_dir, "scaler_ap_means_stds.npz"), means=means, stds=stds)

    # --------- Datasets / Loaders ---------
    ds_src = RSSISourceDataset(df_src, ap_cols, label_col="rp_id", means=means, stds=stds, missing_val=args.missing_val)
    id2idx, idx2id = ds_src.id2idx, ds_src.idx2id
    n_classes = len(id2idx)

    dl_src = DataLoader(ds_src, batch_size=args.batch_size_src, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, means=means, stds=stds, missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size_tgt, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

    # --------- Test set 轉 index（未知=-1） ---------
    X_te_raw = df_te[ap_cols].values.astype(np.float32)
    X_te = apply_scaler(X_te_raw, means, stds, args.missing_val)
    miss_te = (X_te_raw == args.missing_val).astype(np.float32)

    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, dtype=int)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

    class TestDataset(Dataset):
        def __len__(self): return len(y_te_idx)
        def __getitem__(self, i):
            return torch.from_numpy(X_te[i]), torch.tensor(y_te_idx[i], dtype=torch.long), torch.from_numpy(miss_te[i])

    dl_te = DataLoader(TestDataset(), batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    # --------- Model / Optim ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(
        in_dim=len(ap_cols),
        n_classes=n_classes,
        feat_dim=args.feat_dim,
        ext_hidden=args.ext_hidden,
        cls_hidden=args.cls_hidden,
        rec_hidden=args.rec_hidden,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    # --------- Train (source CE+Recon, target Recon) ---------
    steps_per_epoch = len(dl_src)
    it_tgt = cycle(dl_tgt)
    best_acc = -1.0

    # ### NEW: histories for plotting
    hist_epoch = []
    hist_train_ce = []
    hist_train_recon_s = []
    hist_train_recon_t = []
    hist_val_ce = []
    hist_val_acc = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_ce = run_rec_s = run_rec_t = 0.0

        for xs, ys, ms in dl_src:
            xt, _, mt = next(it_tgt)

            xs, ys, ms = xs.to(device), ys.to(device), ms.to(device)
            xt, mt = xt.to(device), mt.to(device)

            # Source: CE + Recon
            logits_s, recon_s, _ = model(xs)
            ce_loss = ce(logits_s, ys)
            rec_loss_s = masked_mse(recon_s, xs, ms, ignore_missing=args.ignore_missing_in_recon)

            # Target: only Recon
            _, recon_t, _ = model(xt)
            rec_loss_t = masked_mse(recon_t, xt, mt, ignore_missing=args.ignore_missing_in_recon)

            loss = ce_loss + args.lambda_recon * (rec_loss_s + rec_loss_t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            run_ce += ce_loss.item()
            run_rec_s += rec_loss_s.item()
            run_rec_t += rec_loss_t.item()

        # ---- Eval ----
        val_ce, val_acc = evaluate_classification(model, dl_te, device)

        print(f"Epoch {epoch:03d} | train_CE {run_ce/steps_per_epoch:.4f} | "
              f"train_Recon_s {run_rec_s/steps_per_epoch:.4f} | train_Recon_t {run_rec_t/steps_per_epoch:.4f} | "
              f"val_CE {val_ce:.4f} | val_acc {val_acc:.4f}")

        if (val_acc == val_acc) and (val_acc > best_acc):  # check not NaN
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "ap_cols": ap_cols,
                "id2idx": id2idx,
                "idx2id": idx2id,
                "means": means,
                "stds": stds,
                "args": vars(args),
            }, os.path.join(args.out_dir, "best_recon_model.pth"))
            print(f"[Info] Saved new best (acc={best_acc:.4f})")

        # ### NEW: record histories
        hist_epoch.append(epoch)
        hist_train_ce.append(run_ce / steps_per_epoch)
        hist_train_recon_s.append(run_rec_s / steps_per_epoch)
        hist_train_recon_t.append(run_rec_t / steps_per_epoch)
        hist_val_ce.append(val_ce)
        hist_val_acc.append(val_acc)

    # ===== NEW: Save training log (CSV) and plots =====
    try:
        log_df = pd.DataFrame({
            "epoch": hist_epoch,
            "train_CE": hist_train_ce,
            "train_Recon_s": hist_train_recon_s,
            "train_Recon_t": hist_train_recon_t,
            "val_CE": hist_val_ce,
            "val_acc": hist_val_acc,
        })
        log_path = os.path.join(args.out_dir, "training_log.csv")
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"[Info] Saved training log -> {log_path}")
    except Exception as e:
        print(f"[Warn] Failed to save training_log.csv: {e}")

    # Plot 1: train_CE vs val_CE
    try:
        plt.figure()
        plt.plot(hist_epoch, hist_train_ce, label="train_CE")
        plt.plot(hist_epoch, hist_val_ce, label="val_CE")
        plt.xlabel("epoch"); plt.ylabel("cross entropy loss")
        plt.title("Classification Loss (train vs val)")
        plt.grid(True); plt.legend()
        fig1 = os.path.join(args.out_dir, "loss_curves.png")
        plt.savefig(fig1, bbox_inches="tight", dpi=160)
        plt.close()
        print(f"[Info] Saved plot -> {fig1}")
    except Exception as e:
        print(f"[Warn] Failed to save loss_curves.png: {e}")

    # Plot 2: train_Recon_s vs train_Recon_t
    try:
        plt.figure()
        plt.plot(hist_epoch, hist_train_recon_s, label="train_Recon_s")
        plt.plot(hist_epoch, hist_train_recon_t, label="train_Recon_t")
        plt.xlabel("epoch"); plt.ylabel("reconstruction MSE")
        plt.title("Reconstruction Loss (source vs target)")
        plt.grid(True); plt.legend()
        fig2 = os.path.join(args.out_dir, "recon_curves.png")
        plt.savefig(fig2, bbox_inches="tight", dpi=160)
        plt.close()
        print(f"[Info] Saved plot -> {fig2}")
    except Exception as e:
        print(f"[Warn] Failed to save recon_curves.png: {e}")

    # Plot 3: val_acc
    try:
        plt.figure()
        plt.plot(hist_epoch, hist_val_acc, label="val_acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy")
        plt.title("Validation Accuracy")
        plt.grid(True); plt.legend()
        fig3 = os.path.join(args.out_dir, "val_acc_curve.png")
        plt.savefig(fig3, bbox_inches="tight", dpi=160)
        plt.close()
        print(f"[Info] Saved plot -> {fig3}")
    except Exception as e:
        print(f"[Warn] Failed to save val_acc_curve.png: {e}")

    # --------- Final Evaluate on Test with MDE (同樓層) ---------
    model.eval()
    preds_idx = []
    with torch.no_grad():
        for xb, yb_idx, _ in dl_te:
            xb = xb.to(device)
            logits, _, _ = model(xb)
            preds_idx.append(logits.argmax(1).cpu().numpy())
    preds_idx = np.concatenate(preds_idx) if preds_idx else np.array([])

    preds_rpid = np.array([idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)
    gts_rpid   = y_te_raw
    eval_mask  = (y_te_idx != -1)
    acc = (preds_idx[eval_mask] == y_te_idx[eval_mask]).mean().item() if eval_mask.any() else float("nan")

    rp_map = load_rp_map(args.rp_map_path)
    mde_distances, floor_mismatch, mde_skipped_notfound = [], 0, 0
    for gt_id, pr_id, use in zip(gts_rpid, preds_rpid, eval_mask):
        if not use: continue
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
    print(f"Test samples total          : {len(y_te_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")

if __name__ == "__main__":
    main()
