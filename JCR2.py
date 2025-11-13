import argparse, os
import math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from math import sqrt
from itertools import cycle
import matplotlib.pyplot as plt   # 畫圖用

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

def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    """
    Min-Max scaler：
    - 只用非 missing_val 的值算 per-AP min / max
    - 之後會把資料壓到 [0,1]
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
    Min-Max 正規化到 [0,1]，缺失值補 0：
    - 對非 missing 的位置做 (x - min) / (max - min)，clip 到 [0,1]
    - 原本是 missing_val 的位置，最後強制設為 0
    """
    x = x.copy().astype(np.float32)
    miss_mask = (x == missing_val)

    denom = (maxs - mins)
    denom[denom == 0.0] = 1.0
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)

    x[miss_mask] = -1.0
    return x

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask_val: float = 0.0):
    """
    只在 target != mask_val 的位置上算 MSE。
    這裡 mask_val=0.0 對應「缺失 AP / padding」。
    """
    mask = (target != mask_val)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - target) ** 2
    return diff[mask].mean()

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
        self.X = np.expand_dims(self.X, axis=1)  # [N, 1, L]

        uniq = np.sort(np.unique(self.y_raw[self.y_raw!=-1]))
        self.id2idx = {rid:i for i, rid in enumerate(uniq)}
        self.idx2id = {i:rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

class RSSITargetDataset(Dataset):
    """
    專門給 target domain 用的 Dataset：只有 X，沒有 label。
    """
    def __init__(self, df: pd.DataFrame, ap_cols, mins=None, maxs=None, missing_val=-110.0):
        X = df[ap_cols].values.astype(np.float32)
        if (mins is not None) and (maxs is not None):
            X = apply_scaler(X, mins, maxs, missing_val)
        self.X = np.expand_dims(X, axis=1)  # [N, 1, L]

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

class PositionalEncoding(nn.Module):
    """
    標準 sin/cos 位置編碼，batch_first = True 對應 [B, L, d_model]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return x

class TransformerExtractor(nn.Module):
    """
    把每個 AP 當成一個 token：
    - input:  [B, 1, L]
    - output: z ∈ [B, z_dim]
    """
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool,
        z_dim: int,
        mask_value: float,
        bottleneck_hidden: int = None,
        use_mask: bool = False,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.z_dim = z_dim
        self.use_mask = use_mask
        self.mask_value = mask_value

        self.input_proj = nn.Linear(1, d_model)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            max_len = num_tokens + 1
        else:
            self.cls_token = None
            max_len = num_tokens

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if bottleneck_hidden is None:
            bottleneck_hidden = d_model

        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_hidden, z_dim)
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, 1, L] -> [B, L]

        B, L = x.shape

        key_padding_mask = None
        if self.use_mask:
            key_padding_mask = (x == self.mask_value)

        x = x.unsqueeze(-1)          # [B, L, 1]
        h = self.input_proj(x)       # [B, L, d_model]

        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            h = torch.cat([cls, h], dim=1)  # [B, 1+L, d_model]

        h = self.pos_encoding(h)

        src_key_padding_mask = None
        if key_padding_mask is not None:
            if self.use_cls_token:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
                src_key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)
            else:
                src_key_padding_mask = key_padding_mask

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        if self.use_cls_token:
            cls_feat = h[:, 0, :]
        else:
            cls_feat = h.mean(dim=1)

        z = self.bottleneck(cls_feat)
        return z

class PredictorMLP(nn.Module):
    """
    接 transformer 抽出來的全局特徵 z [B, z_dim]，
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

class Reconstructor(nn.Module):
    """
    從 latent z 重建回原始 fingerprint：
    - input:  z [B, z_dim]
    - output: x_hat [B, 1, L]
    """
    def __init__(self, z_dim: int, num_ap: int, hidden=[128, 128]):
        super().__init__()
        dims = [z_dim] + hidden + [num_ap]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        x_hat = self.mlp(z)       # [B, num_ap]
        x_hat = x_hat.unsqueeze(1)  # [B, 1, L]
        return x_hat

class TransReconModel(nn.Module):
    def __init__(
        self,
        num_ap: int,
        n_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        z_dim: int,
        mlp_hidden,
        p_drop: float,
        use_mask: bool,
        recon_hidden,
        mask_value: float
    ):
        super().__init__()
        self.extractor = TransformerExtractor(
            num_tokens=num_ap,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_cls_token=True,
            z_dim=z_dim,
            bottleneck_hidden=None,
            use_mask=use_mask,
            mask_value=mask_value
        )
        self.predictor = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=mlp_hidden,
            p_drop=p_drop,
        )
        self.reconstructor = Reconstructor(
            z_dim=z_dim,
            num_ap=num_ap,
            hidden=recon_hidden
        )

    def forward(self, x):
        z = self.extractor(x)
        logits = self.predictor(z)
        x_hat = self.reconstructor(z)
        return logits, x_hat

# ---------- Metrics / Maps ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
    if not need.issubset(df.columns):
        raise ValueError("rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp

# ---------- Validation on test (當作 val set 用，不影響 train) ----------
def eval_on_val(model, dl_te, device):
    model.eval()
    preds_idx, gts_idx = [], []
    with torch.no_grad():
        for xb, yb_idx, _ in dl_te:
            xb = xb.to(device)
            logits, _ = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            preds_idx.append(pred)
            gts_idx.append(yb_idx.numpy())

    if not preds_idx:
        return float("nan")

    preds_idx = np.concatenate(preds_idx)
    gts_idx   = np.concatenate(gts_idx)

    eval_mask = (gts_idx != -1)
    if eval_mask.any():
        acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item()
    else:
        acc = float("nan")
    return acc

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_path", type=str, required=True)
    parser.add_argument("--target_train_path", type=str, required=True)
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, default=r"rp_id.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./result")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)

    # Transformer 超參
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--recon_hidden",type=int, nargs="+", default=[128])

    parser.add_argument("--z_dim", type=int, default=128)

    parser.add_argument("--use_mask", type=bool, default=True,
                        help="啟用 key_padding_mask，將值==0 的 AP 當作 padding/missing 忽略掉")

    parser.add_argument("--lambda_recon", type=float, default=1.0,
                        help="total loss = CE + lambda_recon * (recon_src + recon_tgt)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data ---
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src)
    assert set(ap_cols).issubset(df_te.columns), "Test 缺少部分 AP 欄位"
    assert set(ap_cols).issubset(df_tgt.columns), "Target 缺少部分 AP 欄位"

    n_classes_tr = df_src["rp_id"].nunique()
    print(f"[WARN] 訓練集 rp 類別數={n_classes_tr}")

    # --- Fit scaler on source train (Min-Max) ---
    mins, maxs = fit_scaler(df_src[ap_cols].values.astype(np.float32), missing_val=args.missing_val)

    # --- Build datasets/loaders ---
    ds_src = RSSIDataset(df_src, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_src.id2idx
    idx2id = ds_src.idx2id
    dl_src = DataLoader(ds_src, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    it_tgt = cycle(dl_tgt)

    X_te_full_raw = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full_raw, mins, maxs, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)

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

    model = TransReconModel(
        num_ap=num_ap,
        n_classes=len(id2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        z_dim=args.z_dim,
        mlp_hidden=args.hidden,
        p_drop=args.dropout,
        use_mask=args.use_mask,
        mask_value= -1.0,
        recon_hidden=args.recon_hidden
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit_ce = nn.CrossEntropyLoss()

    print(f"use mask or not: {model.extractor.use_mask}")
    print(f"z_dim : {model.extractor.z_dim}")

    # ====== 歷史記錄（畫圖用）======
    hist_epoch = []
    hist_tr_loss = []
    hist_tr_ce = []
    hist_tr_acc = []
    hist_rec_s = []
    hist_rec_t = []
    hist_val_acc = []

    val_epochs = []   # << 新增：只記有做 val 的 epoch
    val_accs   = []   # << 新增：對應的 val accuracy

    # ====== 追蹤最佳 val model ======  ### NEW
    best_val_acc = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.out_dir, "best_val_model.pth")

    # --- Train ---
    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss, tr_ce, tr_acc, tr_rec_s, tr_rec_t, n_ce = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        for xb_s, yb_s in dl_src:
            xb_t = next(it_tgt)

            xb_s, yb_s = xb_s.to(device, non_blocking=True), yb_s.to(device, non_blocking=True)
            xb_t = xb_t.to(device, non_blocking=True)

            keep = (yb_s != -1)
            opt.zero_grad()

            logits_s, xhat_s = model(xb_s)

            if keep.any():
                ce_loss = crit_ce(logits_s[keep], yb_s[keep])
                bs_ce = keep.sum().item()
                tr_ce += ce_loss.item() * bs_ce
                tr_acc += (logits_s[keep].argmax(1) == yb_s[keep]).float().sum().item()
                n_ce += bs_ce
            else:
                ce_loss = torch.tensor(0.0, device=device)

            rec_loss_s = masked_mse(xhat_s, xb_s, mask_val=0.0)

            _, xhat_t = model(xb_t)
            rec_loss_t = masked_mse(xhat_t, xb_t, mask_val=0.0)

            rec_loss = rec_loss_s + rec_loss_t

            loss = ce_loss + args.lambda_recon * rec_loss
            loss.backward()
            opt.step()

            bs_s = xb_s.size(0)
            tr_loss += loss.item() * bs_s
            tr_rec_s += rec_loss_s.item() * bs_s
            tr_rec_t += rec_loss_t.item() * xb_t.size(0)

        tr_loss = tr_loss / len(ds_src) if len(ds_src) > 0 else 0.0
        tr_ce   = tr_ce / n_ce if n_ce > 0 else 0.0
        tr_acc  = tr_acc / n_ce if n_ce > 0 else 0.0
        tr_rec_s = tr_rec_s / len(ds_src) if len(ds_src) > 0 else 0.0
        tr_rec_t = tr_rec_t / len(ds_tgt) if len(ds_tgt) > 0 else 0.0

        print(f"Epoch {epoch:03d} | total {tr_loss:.4f} | CE {tr_ce:.4f} acc {tr_acc:.4f} | "
              f"recon_s {tr_rec_s:.4f} | recon_t {tr_rec_t:.4f}")
        recon_weighted = args.lambda_recon * (tr_rec_s + tr_rec_t)
        print(f"... | CE {tr_ce:.4f} | λ*recon {recon_weighted:.4f} | "
              f"CE/(CE+λrecon) {tr_ce/(tr_ce+recon_weighted):.2f}")
        if epoch % 5 == 0:
            # --- 每 5 個 epoch 做一次 Validation ---
            val_acc = eval_on_val(model, dl_te, device)
            if not np.isnan(val_acc):
                print("------------------------------------------------")
                print(f"[Val]  Epoch {epoch:03d} | val_acc {val_acc:.4f}")
                print("------------------------------------------------")

                # 先記錄到 val 的歷史（畫圖用）
                val_epochs.append(epoch)
                val_accs.append(val_acc)

                # 再看是否更新 best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "val_acc": val_acc,
                            "args": vars(args),
                        },
                        best_model_path,
                    )
                    print(f"[Val]  New best model at epoch {epoch:03d} | val_acc {val_acc:.4f} -> saved to {best_model_path}")
            else:
                print(f"[Val]  Epoch {epoch:03d} | val_acc N/A (no valid labels)")

        # 存到歷史記錄
        hist_epoch.append(epoch)
        hist_tr_loss.append(tr_loss)
        hist_tr_ce.append(tr_ce)
        hist_tr_acc.append(tr_acc)
        hist_rec_s.append(tr_rec_s)
        hist_rec_t.append(tr_rec_t)

    # ====== 訓練完後：畫圖並存檔 ======
    recon_weighted_list = [args.lambda_recon * (rs + rt) for rs, rt in zip(hist_rec_s, hist_rec_t)]

    plt.figure()
    plt.plot(hist_epoch, hist_tr_loss, label="total_loss")
    plt.plot(hist_epoch, hist_tr_ce, label="CE_loss")
    plt.plot(hist_epoch, recon_weighted_list, label="lambda*recon_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curves.png"))
    plt.close()

    plt.figure()
    plt.plot(hist_epoch, hist_tr_acc, label="train_acc")

    # 只在有做 val 的 epoch 畫 val_acc
    if len(val_epochs) > 0:
        plt.plot(val_epochs, val_accs, label="val_acc")  # 也可以加 marker='o'

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "acc_curves.png"))
    plt.close()

    plt.figure()
    plt.plot(hist_epoch, hist_rec_s, label="recon_src")
    plt.plot(hist_epoch, hist_rec_t, label="recon_tgt")
    plt.xlabel("Epoch")
    plt.ylabel("Recon Loss (MSE)")
    plt.title("Reconstruction Loss (src vs tgt)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "recon_curves.png"))
    plt.close()

    print(f"[Plot] Saved curves to {args.out_dir}")

    # ========= Helper：共用一個 test 評估函式 =========
    def eval_on_test(model_for_eval, dl, idx2id, rp_map_path):
        model_for_eval.eval()
        preds_idx, gts_idx, gts_rpid = [], [], []
        with torch.no_grad():
            for xb, yb_idx, yb_rpid in dl:
                xb = xb.to(device)
                logits, _ = model_for_eval(xb)
                pred = logits.argmax(1).cpu().numpy()
                preds_idx.append(pred)
                gts_idx.append(yb_idx.numpy())
                gts_rpid.append(yb_rpid.numpy())

        preds_idx = np.concatenate(preds_idx) if preds_idx else np.array([])
        gts_idx   = np.concatenate(gts_idx) if gts_idx else np.array([])
        gts_rpid  = np.concatenate(gts_rpid) if gts_rpid else np.array([])

        eval_mask = (gts_idx != -1)
        if eval_mask.any():
            acc = (preds_idx[eval_mask] == gts_idx[eval_mask]).mean().item()
        else:
            acc = float("nan")

        rp_map = load_rp_map(rp_map_path)
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

        avg_mde = (float(np.mean(mde_distances)) if len(mde_distances)>0 else float("nan"))
        return acc, avg_mde, int(eval_mask.sum()), len(gts_idx), floor_mismatch, mde_skipped_notfound

    # --- Test 1：最後一個 epoch 的 model ---
    acc_last, mde_last, eval_cnt_last, total_cnt, floor_m_last, skipped_last = \
        eval_on_test(model, dl_te, idx2id, args.rp_map_path)

    print("==== Final Test Metrics (Last Epoch Model) ====")
    print(f"Test samples total          : {total_cnt}")
    print(f"Evaluated (label available) : {eval_cnt_last}")
    print(f"Test Accuracy               : {acc_last:.4f}" if not np.isnan(acc_last) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {mde_last:.4f} (meters)" if not np.isnan(mde_last) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_m_last}")
    print(f"Skipped (rp_id not in rp_map)        : {skipped_last}")

    # --- Test 2：val_acc 最好的那個 model ---
    if best_epoch > 0 and os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        acc_best, mde_best, eval_cnt_best, total_cnt2, floor_m_best, skipped_best = \
            eval_on_test(model, dl_te, idx2id, args.rp_map_path)

        print("==== Test Metrics (Best Val Model) ====")
        print(f"Best val epoch               : {best_epoch}")
        print(f"Best val acc                 : {best_val_acc:.4f}")
        print(f"Test samples total           : {total_cnt2}")
        print(f"Evaluated (label available)  : {eval_cnt_best}")
        print(f"Test Accuracy (best val mdl) : {acc_best:.4f}" if not np.isnan(acc_best) else "Test Accuracy (best val mdl) : N/A")
        print(f"Mean Distance Error (same floor only): {mde_best:.4f} (meters)" if not np.isnan(mde_best) else "Mean Distance Error (same floor only, best val mdl): N/A")
        print(f"Floor mismatches (excluded from MDE) : {floor_m_best}")
        print(f"Skipped (rp_id not in rp_map, best val mdl)        : {skipped_best}")
    else:
        print("==== Best Val Model ====")
        print("No best model saved (maybe all val_acc are NaN).")

if __name__ == "__main__":
    main()