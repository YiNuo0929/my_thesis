# rssi_dann_minmax.py  (target 批次補齊 + domain acc 版)
# Usage:
#   python rssi_dann_minmax.py ^
#     --source_train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\01\train_all\all_trn_merged.csv" ^
#     --target_train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\04\train_all\all_trn_merged.csv" ^
#     --test_path         "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\04\test_all\all_tst_merged.csv" ^
#     --rp_map_path       "C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv" ^
#     --epochs 50 --batch_size_src 256 --batch_size_tgt 256 --lr 1e-3
#
# 說明（前處理部分已改為 min–max）：
# - Scaler：source domain 每 AP min–max 正規化
# - 缺值 (-110) 以該 AP 的最小值補
# - DANN 架構：feature extractor + classifier + domain discriminator

import argparse, os, math, random
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import cycle

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

def get_feature_cols(df: pd.DataFrame, col):
    """
    改成固定只抓 ap0 ~ ap255 共 256 維
    """
    all_cols = set(df.columns)
    ap_cols = [f"ap{i}" for i in range(col)]
    missing = [c for c in ap_cols if c not in all_cols]
    if missing:
        raise ValueError(f"資料集中缺少欄位（ap0~ap255）：{missing[:10]} ...")
    return ap_cols

# -------------------- Min–Max Scaler --------------------
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
            # 若整個 AP 都是缺值，給一個預設範圍
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
    if miss_mask.any():
        ap_idx = np.where(miss_mask)[1]
        x[miss_mask] = mins[ap_idx]
    x = (x - mins) / (maxs - mins)
    return x

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------- Datasets --------------------
class RSSISourceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, label_col="rp_id",
                 mins=None, maxs=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.X = df[ap_cols].values.astype(np.float32)
        self.y_raw = df[label_col].values.astype(np.int64)

        self.missing_val = missing_val
        self.mins = mins
        self.maxs = maxs
        if (mins is not None) and (maxs is not None):
            self.X = apply_scaler(self.X, self.mins, self.maxs, self.missing_val)

        self.X = np.expand_dims(self.X, axis=1)  # [N,1,L]

        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

class RSSITargetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, mins, maxs, missing_val=-110.0):
        X = df[ap_cols].values.astype(np.float32)
        X = apply_scaler(X, mins, maxs, missing_val)
        self.X = np.expand_dims(X, axis=1)  # [N,1,L]

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(-1, dtype=torch.long)

# -------------------- Model (DANN) --------------------
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

class FeatureExtractor(nn.Module):
    def __init__(self, in_len: int, hidden=[512, 256], p_drop=0.2):
        super().__init__()
        dims = [in_len] + hidden
        blocks = [MLPBlock(a, b, p_drop=p_drop) for a, b in zip(dims[:-1], dims[1:])]
        self.net = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.out_dim = dims[-1]

    def forward(self, x):
        if x.dim() == 3:  # [B,1,L] -> [B,L]
            x = x.squeeze(1)
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, f): return self.fc(f)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grl_lambda):
        ctx.lambda_ = grl_lambda
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden=32, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 2)
        )
    def forward(self, f): return self.net(f)

class DANN(nn.Module):
    def __init__(self, in_len: int, n_classes: int, feat_hidden=[512,256,128], p_drop=0.2, disc_hidden=128):
        super().__init__()
        self.feature = FeatureExtractor(in_len, hidden=feat_hidden, p_drop=p_drop)
        self.classifier = ClassifierHead(self.feature.out_dim, n_classes)
        self.discriminator = DomainDiscriminator(self.feature.out_dim, hidden=disc_hidden, p_drop=p_drop)

    def forward(self, x, grl_lambda=0.0):
        f = self.feature(x)
        cls_logits = self.classifier(f)
        f_rev = GradReverse.apply(f, grl_lambda)
        dom_logits = self.discriminator(f_rev)
        return cls_logits, dom_logits

# -------------------- Metrics / Maps --------------------
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
def dann_lambda_schedule(global_step, total_steps, gamma=10.0):
    p = min(1.0, max(0.0, global_step / max(1, total_steps)))
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

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

    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--disc_hidden", type=int, default=128)

    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./rssi_dann_minmax_ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--column", type=int, default=256)

    parser.add_argument("--grl_gamma", type=float, default=10.0)
    parser.add_argument("--domain_loss_w", type=float, default=0.7)

    parser.add_argument("--model_dir", type=str, default="./models", help="儲存訓練完成模型的資料夾")
    parser.add_argument("--model_name", type=str, default="dann.pth", help="模型檔名")

    args = parser.parse_args()
    # os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    # --------- Load DataFrames ---------
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)


    ap_cols = get_feature_cols(df_src, args.column)
    assert set(ap_cols).issubset(df_tgt.columns), "Target 訓練資料缺少部分 AP 欄位"
    assert set(ap_cols).issubset(df_te.columns),  "Test 資料缺少部分 AP 欄位"

    # --------- Scaler (fit on source only, min-max) ---------
    mins, maxs = fit_scaler(df_src[ap_cols].values.astype(np.float32), missing_val=args.missing_val)
    # np.savez(os.path.join(args.out_dir, "scaler_ap_mins_maxs.npz"), mins=mins, maxs=maxs)

    # --------- Datasets / Loaders ---------
    ds_src = RSSISourceDataset(df_src, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx, idx2id = ds_src.id2idx, ds_src.idx2id
    n_classes = len(id2idx)

    dl_src = DataLoader(ds_src, batch_size=args.batch_size_src, shuffle=True,
                        drop_last=True, num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size_tgt, shuffle=True,
                        drop_last=True, num_workers=0, pin_memory=True)

    # --------- Test set (one-shot eval) ---------
    X_te = df_te[ap_cols].values.astype(np.float32)
    X_te = apply_scaler(X_te, mins, maxs, args.missing_val)
    X_te = np.expand_dims(X_te, axis=1)

    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, dtype=int)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

    class TestDataset(Dataset):
        def __len__(self): return len(y_te_idx)
        def __getitem__(self, i):
            return torch.from_numpy(X_te[i]), torch.tensor(y_te_idx[i], dtype=torch.long), int(y_te_raw[i])

    ds_te = TestDataset()
    dl_te = DataLoader(ds_te, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    # --------- Model / Optim ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DANN(
        in_len=len(ap_cols),
        n_classes=n_classes,
        feat_hidden=args.hidden,
        p_drop=args.dropout,
        disc_hidden=args.disc_hidden
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_cls = nn.CrossEntropyLoss()
    ce_dom = nn.CrossEntropyLoss()

    # --------- Train (target 批次補齊 + domain acc) ---------
    steps_per_epoch = len(dl_src)                  # 以 source 為基準
    total_steps = args.epochs * steps_per_epoch
    print(f"[Info] len(dl_src)={len(dl_src)}, len(dl_tgt)={len(dl_tgt)} | steps/epoch={steps_per_epoch}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        it_src = iter(dl_src)
        it_tgt = cycle(dl_tgt)                     # target 循環補齊

        running_cls = running_dom = running_src_acc = 0.0
        running_dom_acc_s = running_dom_acc_t = 0.0

        for step in range(steps_per_epoch):
            try:
                xs, ys = next(it_src)
            except StopIteration:
                it_src = iter(dl_src)
                xs, ys = next(it_src)

            xt, _ = next(it_tgt)                   # 無限循環

            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)

            global_step = (epoch - 1) * steps_per_epoch + step
            grl_lambda = dann_lambda_schedule(global_step, total_steps, gamma=args.grl_gamma)

            opt.zero_grad()

            # Source：分類 + 領域（domain=0）
            cls_logits_s, dom_logits_s = model(xs, grl_lambda=grl_lambda)
            cls_loss = ce_cls(cls_logits_s, ys)
            dom_label_s = torch.zeros(xs.size(0), dtype=torch.long, device=device)
            dom_loss_s = ce_dom(dom_logits_s, dom_label_s)

            # Target：只有領域（domain=1）
            _, dom_logits_t = model(xt, grl_lambda=grl_lambda)
            dom_label_t = torch.ones(xt.size(0), dtype=torch.long, device=device)
            dom_loss_t = ce_dom(dom_logits_t, dom_label_t)

            dom_loss = 0.5 * (dom_loss_s + dom_loss_t)
            loss = cls_loss + args.domain_loss_w * dom_loss
            loss.backward()
            opt.step()

            with torch.no_grad():
                src_acc = (cls_logits_s.argmax(1) == ys).float().mean().item()
                dom_acc_s = (dom_logits_s.argmax(1) == dom_label_s).float().mean().item()
                dom_acc_t = (dom_logits_t.argmax(1) == dom_label_t).float().mean().item()

            running_cls       += cls_loss.item()
            running_dom       += dom_loss.item()
            running_src_acc   += src_acc
            running_dom_acc_s += dom_acc_s
            running_dom_acc_t += dom_acc_t

        avg_src_cls   = running_cls / steps_per_epoch
        avg_dom       = running_dom / steps_per_epoch
        avg_src_acc   = running_src_acc / steps_per_epoch
        avg_dom_acc_s = running_dom_acc_s / steps_per_epoch
        avg_dom_acc_t = running_dom_acc_t / steps_per_epoch
        avg_dom_acc   = 0.5 * (avg_dom_acc_s + avg_dom_acc_t)

        print(f"Epoch {epoch:03d} | src_cls_loss {avg_src_cls:.4f} "
              f"| dom_loss {avg_dom:.4f} "
              f"| dom_acc_s {avg_dom_acc_s:.3f} | dom_acc_t {avg_dom_acc_t:.3f} | dom_acc {avg_dom_acc:.3f} "
              f"| src_acc {avg_src_acc:.4f} "
              f"| lambda {grl_lambda:.3f}")

    # --------- Save trained model ---------
    model_save_path = os.path.join(args.model_dir, args.model_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "ap_cols": ap_cols,
        "id2idx": id2idx,
        "idx2id": idx2id,
        "mins": mins,
        "maxs": maxs,
        "n_classes": n_classes,
        "args": vars(args),
    }, model_save_path)

    print(f"[*] Model saved to: {model_save_path}")
    
    # --------- Evaluate on Test ---------
    model.eval()
    preds_idx, gts_idx, gts_rpid = [], [], []
    with torch.no_grad():
        for xb, yb_idx, yb_rpid in dl_te:
            xb = xb.to(device, non_blocking=True)
            cls_logits, _ = model(xb, grl_lambda=0.0)
            pred = cls_logits.argmax(1).cpu().numpy()
            preds_idx.append(pred)
            gts_idx.append(yb_idx.numpy())
            gts_rpid.append(yb_rpid.numpy())

    preds_idx = np.concatenate(preds_idx) if preds_idx else np.array([])
    gts_idx   = np.concatenate(gts_idx) if gts_idx else np.array([])
    gts_rpid  = np.concatenate(gts_rpid) if gts_rpid else np.array([])

    eval_mask = (gts_idx != -1)
    acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item() if eval_mask.any() else float("nan")

    rp_map = load_rp_map(args.rp_map_path)
    preds_rpid = np.array([ds_src.idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)

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
