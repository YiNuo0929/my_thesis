import os
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
        raise ValueError(f"rp_id.csv 缺少欄位，需要：{need}")
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

        # 與 fidora_dynamic.py 相同的 normalization 邏輯
        miss_mask = (self.X == missing_val)
        n_feats = self.X.shape[1]

        for j in range(n_feats):
            col = self.X[:, j]
            m = (col != missing_val)
            if m.any():
                vmin = col[m].min()
                vmax = col[m].max()
                if abs(vmax - vmin) < 1e-6:
                    vmax = vmin + 1.0
                self.X[:, j] = np.clip((col - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                self.X[:, j] = 0.0

        self.X[miss_mask] = -1.0

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
class FidoraJCRModel(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, p_drop: float = 0.3):
        super().__init__()

        # Feature Extraction Layers
        self.feat_fc1 = nn.Linear(in_dim, 360)
        self.feat_bn1 = nn.BatchNorm1d(360)
        self.feat_fc2 = nn.Linear(360, 480)
        self.feat_bn2 = nn.BatchNorm1d(480)
        self.feat_fc3 = nn.Linear(480, 600)
        self.drop = nn.Dropout(p_drop)

        # Classification Layers
        self.cls_fc1 = nn.Linear(600, 300)
        self.cls_bn1 = nn.BatchNorm1d(300)
        self.cls_fc2 = nn.Linear(300, 100)
        self.cls_bn2 = nn.BatchNorm1d(100)
        self.cls_out = nn.Linear(100, n_classes)

        # Reconstruction Layers
        self.rec_fc1 = nn.Linear(600, 480)
        self.rec_bn1 = nn.BatchNorm1d(480)
        self.rec_fc2 = nn.Linear(480, 360)
        self.rec_bn2 = nn.BatchNorm1d(360)
        self.rec_out = nn.Linear(360, in_dim)

    def extract_features(self, x):
        h = self.drop(self.feat_bn1(torch.sigmoid(self.feat_fc1(x))))
        h = self.drop(self.feat_bn2(torch.sigmoid(self.feat_fc2(h))))
        z = self.feat_fc3(h)
        return z

    def forward(self, x):
        z = self.extract_features(x)

        c = self.drop(self.cls_bn1(torch.sigmoid(self.cls_fc1(z))))
        c = self.cls_bn2(torch.sigmoid(self.cls_fc2(c)))
        logits = self.cls_out(c)

        r = self.drop(self.rec_bn1(torch.sigmoid(self.rec_fc1(z))))
        r = self.drop(self.rec_bn2(torch.sigmoid(self.rec_fc2(r))))
        x_recon = self.rec_out(r)

        return logits, x_recon


# -------------------- Main --------------------
def main():
    set_seed(42)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    model_path = os.path.join(BASE_DIR, "fidora.pth")
    test_path = os.path.join(PROJECT_ROOT, "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-06-11/test.csv")
    rp_map_path = os.path.join(PROJECT_ROOT, "rp_id_um.csv")

    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 載入 checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    ap_cols = ckpt["ap_cols"]
    id2idx = ckpt["id2idx"]
    idx2id = ckpt["idx2id"]
    in_dim = ckpt["in_dim"]
    n_classes = ckpt["n_classes"]
    saved_args = ckpt.get("args", {})

    # 2) 建立模型
    model = FidoraJCRModel(
        in_dim=in_dim,
        n_classes=n_classes,
        p_drop=0.3
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
            logits, _ = model(xb)
            pred = logits.argmax(1).cpu().numpy()

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

    # 6) 輸出格式
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