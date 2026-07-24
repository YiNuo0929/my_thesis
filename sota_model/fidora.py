# fidora_dynamic.py
# Usage:
#   python fidora_dynamic.py ^
#     --source_train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\01\train_all\all_trn_merged.csv" ^
#     --target_train_path "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\04\train_all\all_trn_merged.csv" ^
#     --test_path         "C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\04\test_all\all_tst_merged.csv" ^
#     --rp_map_path       "C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv" ^
#     --epochs 50 --batch_size 128 --lr 1e-3

import argparse, os, math, random
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
        
        # --- 新的正規化邏輯：缺失值補 -1，有效值 0~1 ---
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
                # Min-Max 到 [0, 1]
                self.X[:, j] = np.clip((col - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                self.X[:, j] = 0.0
        
        # 缺失值設成 -1.0
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

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long), int(self.y_raw_list[i])

# -------------------- Fidora: Data Augmenter (VAE) --------------------
class FidoraVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 10):
        super().__init__()
        self.enc_fc1 = nn.Linear(in_dim, 360)
        self.enc_bn1 = nn.BatchNorm1d(360)
        self.enc_fc2 = nn.Linear(360, 50)
        self.enc_bn2 = nn.BatchNorm1d(50)
        
        self.fc_mu = nn.Linear(50, latent_dim)
        self.fc_logvar = nn.Linear(50, latent_dim)
        
        self.dec_fc1 = nn.Linear(latent_dim, 50)
        self.dec_bn1 = nn.BatchNorm1d(50)
        self.dec_fc2 = nn.Linear(50, 360)
        self.dec_bn2 = nn.BatchNorm1d(360)
        self.dec_fc3 = nn.Linear(360, in_dim)

    def encode(self, x):
        h = F.relu(self.enc_bn1(self.enc_fc1(x)))
        h = F.relu(self.enc_bn2(self.enc_fc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_bn1(self.dec_fc1(z)))
        h = F.relu(self.dec_bn2(self.dec_fc2(h)))
        # 為了重建 -1.0 到 1.0 的範圍，拿掉 Sigmoid 改用線性輸出
        return self.dec_fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def train_vae_and_augment(ds_src, device, aug_ratio=5, epochs=10):
    print(f"[*] 啟動 Fidora VAE Data Augmenter (每類別擴增 {aug_ratio} 倍)...")
    X_all, Y_all = ds_src.X, ds_src.y
    in_dim = X_all.shape[1]
    n_classes = len(ds_src.id2idx)
    
    aug_X_list, aug_Y_list = [X_all], [Y_all]
    
    for c in range(n_classes):
        X_c = X_all[Y_all == c]
        if len(X_c) < 5: continue
        
        vae = FidoraVAE(in_dim=in_dim).to(device)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        X_c_tensor = torch.tensor(X_c, dtype=torch.float32).to(device)
        dataset_c = TensorDataset(X_c_tensor)
        dl_c = DataLoader(dataset_c, batch_size=32, shuffle=True)
        
        vae.train()
        for ep in range(epochs):
            for (bx,) in dl_c:
                opt.zero_grad()
                recon_x, mu, logvar = vae(bx)
                recon_loss = F.mse_loss(recon_x, bx, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                loss.backward()
                opt.step()
                
        vae.eval()
        n_samples = len(X_c) * aug_ratio
        with torch.no_grad():
            z = torch.randn(n_samples, 10).to(device)
            synthetic_X = vae.decode(z).cpu().numpy()
            synthetic_Y = np.full(n_samples, c, dtype=np.int64)
            
        aug_X_list.append(synthetic_X)
        aug_Y_list.append(synthetic_Y)
        
    final_X = np.concatenate(aug_X_list, axis=0)
    final_Y = np.concatenate(aug_Y_list, axis=0)
    print(f"[*] VAE 擴增完成！資料量從 {len(X_all)} 提升至 {len(final_X)}")
    
    ds_aug = torch.utils.data.TensorDataset(
        torch.tensor(final_X, dtype=torch.float32), 
        torch.tensor(final_Y, dtype=torch.long)
    )
    return ds_aug

# -------------------- Fidora: Domain-Adaptive Classifier (JCR) --------------------
class FidoraJCRModel(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, p_drop: float = 0.3):
        super().__init__()
        
        # 1. Feature Extraction Layers (Theta_F)
        self.feat_fc1 = nn.Linear(in_dim, 360)
        self.feat_bn1 = nn.BatchNorm1d(360)
        self.feat_fc2 = nn.Linear(360, 480)
        self.feat_bn2 = nn.BatchNorm1d(480)
        self.feat_fc3 = nn.Linear(480, 600)
        self.drop = nn.Dropout(p_drop)
        
        # 2. Classification Layers (Theta_C)
        self.cls_fc1 = nn.Linear(600, 300)
        self.cls_bn1 = nn.BatchNorm1d(300)
        self.cls_fc2 = nn.Linear(300, 100)
        self.cls_bn2 = nn.BatchNorm1d(100)
        self.cls_out = nn.Linear(100, n_classes)
        
        # 3. Reconstruction Layers (Theta_R)
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
        
        # Classification Path
        c = self.drop(self.cls_bn1(torch.sigmoid(self.cls_fc1(z))))
        c = self.cls_bn2(torch.sigmoid(self.cls_fc2(c)))
        logits = self.cls_out(c)
        
        # Reconstruction Path
        r = self.drop(self.rec_bn1(torch.sigmoid(self.rec_fc1(z))))
        r = self.drop(self.rec_bn2(torch.sigmoid(self.rec_fc2(r))))
        # 拿掉最後的 Sigmoid，改用線性輸出以重建 -1.0 到 1.0 的資料
        x_recon = self.rec_out(r)
        
        return logits, x_recon

# -------------------- Cluster Assumption Loss --------------------
def conditional_entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
    return entropy.mean()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_path", type=str, required=True)
    parser.add_argument("--target_train_path", type=str, required=True)
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--aug_ratio", type=int, default=10)
    parser.add_argument("--lambda_c", type=float, default=13.0)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_r", type=float, default=4.0)

    parser.add_argument("--model_dir", type=str, default="./models", help="儲存訓練完成模型的資料夾")
    parser.add_argument("--model_name", type=str, default="fidora.pth", help="模型檔名")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src)
    in_dim = len(ap_cols)
    print(f"[*] 成功動態偵測到 {in_dim} 個 AP 欄位。")

    ds_src_raw = RSSIDataset(df_src, ap_cols, missing_val=args.missing_val, is_target=False)
    id2idx, idx2id = ds_src_raw.id2idx, ds_src_raw.idx2id
    n_classes = len(id2idx)

    ds_src_aug = train_vae_and_augment(ds_src_raw, device, aug_ratio=args.aug_ratio, epochs=10)
    dl_src = DataLoader(ds_src_aug, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ds_tgt = RSSIDataset(df_tgt, ap_cols, missing_val=args.missing_val, is_target=True)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    ds_te = RSSIDataset(df_te, ap_cols, id2idx=id2idx, missing_val=args.missing_val, is_target=False)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)

    rp_map = load_rp_map(args.rp_map_path)

    model = FidoraJCRModel(in_dim=in_dim, n_classes=n_classes, p_drop=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce_loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = len(dl_src)
    print(f"[*] 開始 JCR 聯合訓練： src batches={len(dl_src)}, tgt batches={len(dl_tgt)} | Classes={n_classes}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        it_src = iter(dl_src)
        it_tgt = cycle(dl_tgt)

        running_cls = running_clu = running_rec = running_acc = 0.0

        for step in range(steps_per_epoch):
            xs, ys = next(it_src)
            xt, _, _ = next(it_tgt)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            opt.zero_grad()
            logits_s, x_recon_s = model(xs)
            loss_a = ce_loss_fn(logits_s, ys) 
            
            logits_t, x_recon_t = model(xt)
            loss_u = conditional_entropy_loss(logits_t) 
            
            loss_r = F.mse_loss(x_recon_s, xs) + F.mse_loss(x_recon_t, xt)

            loss = args.lambda_c * loss_a + args.lambda_u * loss_u + args.lambda_r * loss_r
            loss.backward()
            opt.step()

            with torch.no_grad():
                acc = (logits_s.argmax(1) == ys).float().mean().item()

            running_cls += loss_a.item()
            running_clu += loss_u.item()
            running_rec += loss_r.item()
            running_acc += acc

        print(f"Epoch {epoch:03d} | cls_loss (LA): {running_cls/steps_per_epoch:.3f} | "
              f"clu_loss (LU): {running_clu/steps_per_epoch:.3f} | rec_loss (LR): {running_rec/steps_per_epoch:.3f} | "
              f"src_acc: {running_acc/steps_per_epoch:.4f}")

    # ---------- Save trained model ----------
    model_save_path = os.path.join(args.model_dir, args.model_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "ap_cols": ap_cols,
        "id2idx": id2idx,
        "idx2id": idx2id,
        "in_dim": in_dim,
        "n_classes": n_classes,
        "args": vars(args),
    }, model_save_path)

    print(f"[*] Model saved to: {model_save_path}")

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

    print("\n==== Final Test Metrics (Fidora Architecture) ====")
    print(f"Evaluated Samples                    : {int(eval_mask.sum())}")
    print(f"Test Accuracy                        : {acc:.4f}")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)")
    print(f"Floor Mismatches (excluded from MDE) : {floor_mismatch}")

if __name__ == "__main__":
    main()