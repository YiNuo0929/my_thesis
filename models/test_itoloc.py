import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------- Utils --------------------
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


# -------------------- Dataset --------------------
class RSSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, label_col="rp_id", id2idx=None, missing_val=-110.0, is_target=False):
        self.X = df[ap_cols].values.astype(np.float32)

        miss_mask = (self.X == missing_val)
        self.X[miss_mask] = missing_val

        self.is_target = is_target
        if not is_target:
            self.y_raw = df[label_col].values.astype(np.int64)
            if id2idx is None:
                uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
                self.id2idx = {rid: i for i, rid in enumerate(uniq)}
            else:
                self.id2idx = id2idx

            self.idx2id = {i: rid for rid, i in self.id2idx.items()}
            self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)
            self.y_raw_list = self.y_raw
        else:
            self.y = np.full(len(self.X), -1, dtype=np.int64)
            self.y_raw_list = np.full(len(self.X), -1, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i], dtype=torch.long),
            int(self.y_raw_list[i])
        )


# -------------------- Model --------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grl_lambda):
        ctx.lambda_ = grl_lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)


class iToLocModel(nn.Module):
    def __init__(self, n_classes: int, p_drop=0.5):
        super().__init__()
        self.R = -40.0
        self.eta = 3.0

        self.M_E = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.M1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes)
        )

        self.M2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes)
        )

        self.M3 = nn.Sequential(
            SimpleResBlock(32),
            nn.MaxPool2d(2, 2),
            SimpleResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes)
        )

        self.M_D = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, 2)
        )

    def fingerprint_image_transform(self, rssi):
        d = torch.pow(10.0, (self.R - rssi) / (10.0 * self.eta))
        d_j = d.unsqueeze(2)
        d_k = d.unsqueeze(1)
        X = (d_j - d_k) / (d_k + 1e-6)
        return X.unsqueeze(1)

    def forward(self, x, grl_lambda=0.0):
        x_img = self.fingerprint_image_transform(x)
        z = self.M_E(x_img)
        l1 = self.M1(z)
        l2 = self.M2(z)
        l3 = self.M3(z)
        z_rev = GradReverse.apply(z, grl_lambda)
        d_logits = self.M_D(z_rev)
        return l1, l2, l3, d_logits


# -------------------- Main --------------------
def main():
    set_seed(42)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    model_path = os.path.join(BASE_DIR, "itoloc.pth")
    test_path = os.path.join(
        PROJECT_ROOT,
        "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-06-11/test.csv"
    )
    rp_map_path = os.path.join(PROJECT_ROOT, "rp_id_um.csv")

    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 載入 checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    ap_cols = ckpt["ap_cols"]
    id2idx = ckpt["id2idx"]
    idx2id = ckpt["idx2id"]
    n_classes = ckpt["n_classes"]
    saved_args = ckpt.get("args", {})

    # 2) 建立模型
    model = iToLocModel(
        n_classes=n_classes,
        p_drop=0.5
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3) 讀 test.csv
    df_te = load_csvs(test_path)

    missing_cols = [c for c in ap_cols if c not in df_te.columns]
    if missing_cols:
        raise ValueError(f"Test 缺少以下 AP 欄位：{missing_cols[:10]}")

    ds_te = RSSIDataset(
        df=df_te,
        ap_cols=ap_cols,
        id2idx=id2idx,
        missing_val=saved_args.get("missing_val", -110.0),
        is_target=False
    )
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4) 載入 rp map
    rp_map = load_rp_map(rp_map_path)

    # 5) 測試
    preds_idx, gts_idx, gts_rpid = [], [], []

    with torch.no_grad():
        for xb, yb_idx, yb_rpid in dl_te:
            xb = xb.to(device)
            l1, l2, l3, _ = model(xb, grl_lambda=0.0)
            pred = (l1 + l2 + l3).argmax(1).cpu().numpy()

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
            mde_distances.append(sqrt((gx - px) ** 2 + (gy - py) ** 2))

    avg_mde = float(np.mean(mde_distances)) if len(mde_distances) > 0 else float("nan")

    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy (ensemble)    : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy (ensemble)    : N/A")
    print(
        f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)"
        if not np.isnan(avg_mde)
        else "Mean Distance Error (same floor only): N/A"
    )
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {skipped_notfound}")


if __name__ == "__main__":
    main()