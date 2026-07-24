import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id", "x", "y", "floor"}
    if not need.issubset(df.columns):
        raise ValueError(f"rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- Min-Max Scaler --------------------
def apply_scaler(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray, missing_val: float = -110.0):
    x = x.copy()
    miss_mask = (x == missing_val)
    if miss_mask.any():
        ap_idx = np.where(miss_mask)[1]
        x[miss_mask] = mins[ap_idx]
    x = (x - mins) / (maxs - mins)
    return x


# -------------------- Dataset --------------------
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, id2idx, mins, maxs, missing_val=-110.0):
        X = df[ap_cols].values.astype(np.float32)
        X = apply_scaler(X, mins, maxs, missing_val)
        self.X = np.expand_dims(X, axis=1)  # [N,1,L]

        if "rp_id" in df.columns:
            self.y_raw = df["rp_id"].astype(int).values
        else:
            self.y_raw = np.full(len(df), -1, dtype=int)

        self.y_idx = np.array([id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self):
        return len(self.y_idx)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y_idx[i], dtype=torch.long),
            int(self.y_raw[i])
        )


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

    def forward(self, x):
        return self.seq(x)


class FeatureExtractor(nn.Module):
    def __init__(self, in_len: int, hidden=[512, 256], p_drop=0.2):
        super().__init__()
        dims = [in_len] + hidden
        blocks = [MLPBlock(a, b, p_drop=p_drop) for a, b in zip(dims[:-1], dims[1:])]
        self.net = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.out_dim = dims[-1]

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, f):
        return self.fc(f)


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

    def forward(self, f):
        return self.net(f)


class DANN(nn.Module):
    def __init__(self, in_len: int, n_classes: int, feat_hidden=[512, 256, 128], p_drop=0.2, disc_hidden=128):
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


# -------------------- Main --------------------
def main():
    set_seed(42)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    model_path = os.path.join(BASE_DIR, "dann.pth")
    test_path = os.path.join(
        PROJECT_ROOT,
        "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-06-11/test.csv"
    )
    rp_map_path = os.path.join(PROJECT_ROOT, "rp_id_um.csv")

    batch_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 載入 checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    ap_cols = ckpt["ap_cols"]
    id2idx = ckpt["id2idx"]
    idx2id = ckpt["idx2id"]
    mins = np.array(ckpt["mins"], dtype=np.float32)
    maxs = np.array(ckpt["maxs"], dtype=np.float32)
    n_classes = ckpt["n_classes"]
    saved_args = ckpt.get("args", {})

    # 2) 建立模型
    model = DANN(
        in_len=len(ap_cols),
        n_classes=n_classes,
        feat_hidden=saved_args.get("hidden", [256, 256]),
        p_drop=saved_args.get("dropout", 0.4),
        disc_hidden=saved_args.get("disc_hidden", 128)
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3) 讀 test.csv
    df_te = load_csvs(test_path)

    missing_cols = [c for c in ap_cols if c not in df_te.columns]
    if missing_cols:
        raise ValueError(f"Test 缺少以下 AP 欄位：{missing_cols[:10]}")

    ds_te = TestDataset(
        df=df_te,
        ap_cols=ap_cols,
        id2idx=id2idx,
        mins=mins,
        maxs=maxs,
        missing_val=saved_args.get("missing_val", -110.0)
    )
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 4) 載入 rp map
    rp_map = load_rp_map(rp_map_path)

    # 5) 測試
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

    preds_rpid = np.array([idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)

    mde_distances = []
    floor_mismatch = 0
    skipped_notfound = 0

    for gt_id, pr_id, use in zip(gts_rpid, preds_rpid, eval_mask):
        if not use:
            continue

        gt_info = rp_map.get(int(gt_id), None)
        pr_info = rp_map.get(int(pr_id), None)

        if (gt_info is None) or (pr_info is None):
            skipped_notfound += 1
            continue

        gx, gy, gf = gt_info
        px, py, pf = pr_info

        if gf != pf:
            floor_mismatch += 1
        else:
            d = sqrt((gx - px) ** 2 + (gy - py) ** 2)
            mde_distances.append(d)

    avg_mde = float(np.mean(mde_distances)) if len(mde_distances) > 0 else float("nan")

    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(
        f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)"
        if not np.isnan(avg_mde)
        else "Mean Distance Error (same floor only): N/A"
    )
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {skipped_notfound}")


if __name__ == "__main__":
    main()