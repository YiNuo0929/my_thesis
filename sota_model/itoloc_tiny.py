# itoloc_dynamic_dann.py
# Usage:
#   python itoloc_dynamic_dann.py ^
#     --source_train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\01\train_all\all_trn_merged.csv" ^
#     --target_train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\04\train_all\all_trn_merged.csv" ^
#     --test_path         "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\04\test_all\all_tst_merged.csv" ^
#     --rp_map_path       "C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv" ^
#     --epochs 50 --batch_size_src 128 --batch_size_tgt 128 --lr 1e-3 --img_size 128
#
# 特色：
# - 完全動態適應 AP 數量 (自動偵測 CSV 內的 ap 欄位)
# - 加入 img_size 強制降維機制，解決 VRAM (顯存) OOM 溢出問題

import argparse, os, math, random
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import cycle

# -------------------- Utils: I/O & Mapping --------------------
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
    ap_cols = [c for c in df.columns if str(c).startswith('ap')]
    ap_cols = sorted(ap_cols, key=lambda x: int(x.replace('ap', '')))
    if not ap_cols:
        raise ValueError("資料集中找不到任何以 'ap' 開頭的欄位！")
    return ap_cols

def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
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

# -------------------- Datasets --------------------
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

    def __len__(self): return len(self.X)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long), int(self.y_raw_list[i])

# -------------------- Network Modules (iToLoc) --------------------
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
    def __init__(self, n_classes: int, p_drop=0.5, img_size=128):
        super().__init__()
        self.R = -40.0
        self.eta = 3.0
        
        # 接收外部指定的影像大小，預設將 N x N 壓縮為 img_size x img_size
        self.img_size = img_size
        
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
        # 1. 計算實體距離矩陣
        d = torch.pow(10.0, (self.R - rssi) / (10.0 * self.eta)) 
        d_j = d.unsqueeze(2) 
        d_k = d.unsqueeze(1) 
        X = (d_j - d_k) / (d_k + 1e-6)
        X = X.unsqueeze(1) # shape: (B, 1, N, N)
        
        # 2. 關鍵降維：如果 N 太大，強制 Pooling 壓縮為 (B, 1, img_size, img_size)
        if X.shape[-1] != self.img_size:
            X = F.adaptive_avg_pool2d(X, (self.img_size, self.img_size))
            
        return X

    def forward(self, x, grl_lambda=0.0):
        x_img = self.fingerprint_image_transform(x)
        z = self.M_E(x_img)
        l1 = self.M1(z)
        l2 = self.M2(z)
        l3 = self.M3(z)
        z_rev = GradReverse.apply(z, grl_lambda)
        d_logits = self.M_D(z_rev)
        return l1, l2, l3, d_logits

# -------------------- Spatial Constraint Utils --------------------
def build_spatial_weight_matrix(id2idx, rp_map):
    n_classes = len(id2idx)
    idx2id = {v: k for k, v in id2idx.items()}
    W = np.zeros((n_classes, n_classes), dtype=np.float32)
    max_dist = 1.0
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j: continue
            rpid_i, rpid_j = idx2id[i], idx2id[j]
            xi, yi, fi = rp_map[rpid_i]
            xj, yj, fj = rp_map[rpid_j]
            dist = sqrt((xi - xj)**2 + (yi - yj)**2) + abs(fi - fj) * 15.0
            W[i, j] = dist
            if dist > max_dist:
                max_dist = dist
    W = W / max_dist
    return torch.tensor(W, dtype=torch.float32)

def spatial_constraint_loss(logits, targets, W_tensor):
    probs = F.softmax(logits, dim=1)
    W_batch = W_tensor[targets] 
    loss = torch.sum(W_batch * probs, dim=1).mean()
    return loss

# -------------------- Main --------------------
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
    parser.add_argument("--batch_size_src", type=int, default=128)
    parser.add_argument("--batch_size_tgt", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--seed", type=int, default=42)

    # 允許外部指定壓縮後的圖片大小 (預設 128)
    parser.add_argument("--img_size", type=int, default=128, help="Resize 2D input to img_size x img_size to save VRAM")
    
    parser.add_argument("--grl_gamma", type=float, default=10.0)
    parser.add_argument("--lambda_d", type=float, default=0.4)
    parser.add_argument("--gamma_s", type=float, default=0.2)

    args = parser.parse_args()
    set_seed(args.seed)

    # --------- Load DataFrames ---------
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src)
    num_aps = len(ap_cols)
    print(f"[*] 成功動態偵測到 {num_aps} 個 AP 欄位。")
    print(f"[*] 設定將 {num_aps}x{num_aps} 的 2D 關聯圖壓縮至 {args.img_size}x{args.img_size} 以節省顯存。")
    
    assert set(ap_cols).issubset(df_tgt.columns), "Target 訓練資料缺少部分 AP 欄位"
    assert set(ap_cols).issubset(df_te.columns),  "Test 資料缺少部分 AP 欄位"

    # --------- Datasets / Loaders ---------
    ds_src = RSSIDataset(df_src, ap_cols, missing_val=args.missing_val, is_target=False)
    id2idx, idx2id = ds_src.id2idx, ds_src.idx2id
    n_classes = len(id2idx)

    dl_src = DataLoader(ds_src, batch_size=args.batch_size_src, shuffle=True, drop_last=True)
    ds_tgt = RSSIDataset(df_tgt, ap_cols, missing_val=args.missing_val, is_target=True)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size_tgt, shuffle=True, drop_last=True)
    ds_te = RSSIDataset(df_te, ap_cols, id2idx=id2idx, missing_val=args.missing_val, is_target=False)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)

    # --------- Build Spatial Constraint Matrix ---------
    rp_map = load_rp_map(args.rp_map_path)
    W_tensor = build_spatial_weight_matrix(id2idx, rp_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_tensor = W_tensor.to(device)

    # --------- Model / Optim ---------
    model = iToLocModel(n_classes=n_classes, p_drop=0.5, img_size=args.img_size).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    ce_cls = nn.CrossEntropyLoss()
    ce_dom = nn.CrossEntropyLoss()

    # --------- Train Loop ---------
    steps_per_epoch = len(dl_src)
    total_steps = args.epochs * steps_per_epoch
    print(f"[*] 開始訓練： src batches={len(dl_src)}, tgt batches={len(dl_tgt)} | Classes={n_classes}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        it_src = iter(dl_src)
        it_tgt = cycle(dl_tgt)

        running_cls = running_spa = running_dom = running_acc = 0.0

        for step in range(steps_per_epoch):
            try:
                xs, ys, _ = next(it_src)
            except StopIteration:
                it_src = iter(dl_src)
                xs, ys, _ = next(it_src)

            xt, _, _ = next(it_tgt)

            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            global_step = (epoch - 1) * steps_per_epoch + step
            grl_lambda = dann_lambda_schedule(global_step, total_steps, gamma=args.grl_gamma)

            opt.zero_grad()

            l1_s, l2_s, l3_s, dom_s = model(xs, grl_lambda)
            loss_a = (ce_cls(l1_s, ys) + ce_cls(l2_s, ys) + ce_cls(l3_s, ys)) / 3.0
            loss_s = (spatial_constraint_loss(l1_s, ys, W_tensor) + 
                      spatial_constraint_loss(l2_s, ys, W_tensor) + 
                      spatial_constraint_loss(l3_s, ys, W_tensor)) / 3.0
            loss_d_s = ce_dom(dom_s, torch.zeros(xs.size(0), dtype=torch.long, device=device))

            _, _, _, dom_t = model(xt, grl_lambda)
            loss_d_t = ce_dom(dom_t, torch.ones(xt.size(0), dtype=torch.long, device=device))
            
            loss_d = 0.5 * (loss_d_s + loss_d_t)
            loss = loss_a + args.gamma_s * loss_s + args.lambda_d * loss_d
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred_ens = (l1_s + l2_s + l3_s).argmax(1)
                acc = (pred_ens == ys).float().mean().item()

            running_cls += loss_a.item()
            running_spa += loss_s.item()
            running_dom += loss_d.item()
            running_acc += acc

        print(f"Epoch {epoch:03d} | cls_loss: {running_cls/steps_per_epoch:.3f} | "
              f"spa_loss: {running_spa/steps_per_epoch:.3f} | dom_loss: {running_dom/steps_per_epoch:.3f} | "
              f"src_acc: {running_acc/steps_per_epoch:.4f}")

    # --------- Evaluation ---------
    model.eval()
    preds_idx, gts_idx, gts_rpid = [], [], []
    with torch.no_grad():
        for xb, yb_idx, yb_rpid in dl_te:
            xb = xb.to(device)
            l1, l2, l3, _ = model(xb, grl_lambda=0.0)
            pred = (l1 + l2 + l3).argmax(1).cpu().numpy()
            
            preds_idx.append(pred)
            gts_idx.append(yb_idx.numpy())
            gts_rpid.append(yb_rpid.numpy())

    preds_idx = np.concatenate(preds_idx)
    gts_idx   = np.concatenate(gts_idx)
    gts_rpid  = np.concatenate(gts_rpid)

    eval_mask = (gts_idx != -1)
    acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item() if eval_mask.any() else float("nan")
    preds_rpid = np.array([idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)

    mde_distances, floor_mismatch = [], 0
    for gt_id, pr_id, use in zip(gts_rpid, preds_rpid, eval_mask):
        if not use: continue
        gt_info, pr_info = rp_map.get(int(gt_id)), rp_map.get(int(pr_id))
        if not gt_info or not pr_info: continue
            
        gx, gy, gf = gt_info
        px, py, pf = pr_info
        if gf != pf:
            floor_mismatch += 1
        else:
            mde_distances.append(sqrt((gx - px)**2 + (gy - py)**2))

    avg_mde = float(np.mean(mde_distances)) if mde_distances else float("nan")

    print("\n==== Final Test Metrics (Dynamic AP with Size Compression) ====")
    print(f"Evaluated Samples                    : {int(eval_mask.sum())}")
    print(f"Test Accuracy (Ensemble)             : {acc:.4f}")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)")
    print(f"Floor Mismatches (excluded from MDE) : {floor_mismatch}")

if __name__ == "__main__":
    main()