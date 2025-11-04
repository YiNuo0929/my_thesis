import argparse, os
import math
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
    ap_cols = [c for c in df.columns if c.startswith("ap")]
    if not ap_cols:
        raise ValueError("No AP columns (prefix 'ap') found.")
    return ap_cols

def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
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
    # 用每個 AP 的平均值補缺失
    x[miss_mask] = np.take(means, np.where(miss_mask)[1])
    # per-AP z-score
    x = (x - means) / stds
    return x

class RSSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols, label_col="rp_id",
                 means=None, stds=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.y_raw = df[label_col].values.astype(np.int64)  # 保留原始 rp_id（做 MDE 用）
        self.X = df[ap_cols].values.astype(np.float32)
        self.missing_val = missing_val
        self.means = means
        self.stds  = stds
        if (means is not None) and (stds is not None):
            self.X = apply_scaler(self.X, self.means, self.stds, self.missing_val)
        # 與 CNN 版對齊：維度 [N, 1, L]；之後在 model 內再 squeeze
        self.X = np.expand_dims(self.X, axis=1)

        # rp_id -> [0..C-1]
        uniq = np.sort(np.unique(self.y_raw[self.y_raw!=-1]))
        self.id2idx = {rid:i for i, rid in enumerate(uniq)}
        self.idx2id = {i:rid for rid, i in self.id2idx.items()}
        # 不在映射的（例如 -1）設為 -1
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

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


# -------- 新增：Positional Encoding --------
class PositionalEncoding(nn.Module):
    """
    標準 sin/cos 位置編碼，batch_first = True 對應 [B, L, d_model]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)            # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, d_model]
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return x

# -------- 新增：Transformer-based Extractor --------
class TransformerExtractor(nn.Module):
    """
    把每個 AP 當成一個 token：
    - input:  [B, 1, L]  (L 個 AP)
    - 先轉成 [B, L, 1] 再線性投影到 d_model 維度
    - 加上位置編碼
    - 通過 TransformerEncoder
    - 使用 CLS token 或 mean pooling 取得整體 fingerprint 向量
    """
    def __init__(
        self,
        num_tokens: int,          # AP 數量 = 序列長度 L
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.d_model = d_model

        # 每個 RSSI scalar -> d_model 維
        self.input_proj = nn.Linear(1, d_model)

        # CLS token（可選）
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

    def forward(self, x: torch.Tensor):
        """
        x: [B, 1, L] 或 [B, L]
        return: [B, d_model]  (整個 fingerprint 的向量)
        """
        if x.dim() == 3:
            # [B, 1, L] -> [B, L]
            x = x.squeeze(1)
        # [B, L] -> [B, L, 1]
        x = x.unsqueeze(-1)
        # [B, L, 1] -> [B, L, d_model]
        h = self.input_proj(x)

        # 加 CLS token
        if self.use_cls_token:
            B = h.size(0)
            cls = self.cls_token.expand(B, -1, -1)   # [B, 1, d_model]
            h = torch.cat([cls, h], dim=1)           # [B, 1+L, d_model]

        # 位置編碼 + encoder
        h = self.pos_encoding(h)
        h = self.encoder(h)                          # [B, seq_len, d_model]

        if self.use_cls_token:
            # 回傳 CLS 位置的向量
            return h[:, 0, :]                        # [B, d_model]
        else:
            # 或者用 mean pooling
            return h.mean(dim=1)                     # [B, d_model]

# -------- 新增：Predictor (MLP head) --------
class PredictorMLP(nn.Module):
    """
    接 transformer 抽出來的全局特徵 [B, d_model]，
    再接幾層 MLP + 最後分類器。
    """
    def __init__(self, in_dim: int, n_classes: int, hidden=[256, 128], p_drop=0.2):
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

# -------- 新增：整體模型 = TransformerExtractor + PredictorMLP --------
class TransClassifier(nn.Module):
    def __init__(
        self,
        num_ap: int,
        n_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        mlp_hidden=[256, 128],
        p_drop: float = 0.2,
    ):
        super().__init__()
        # extractor：學 AP 間關係
        self.extractor = TransformerExtractor(
            num_tokens=num_ap,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_cls_token=True,
        )
        # predictor：接上 MLP head 做分類
        self.predictor = PredictorMLP(
            in_dim=d_model,
            n_classes=n_classes,
            hidden=mlp_hidden,
            p_drop=p_drop,
        )

    def forward(self, x):
        # x: [B, 1, L]
        feat = self.extractor(x)     # [B, d_model]
        logits = self.predictor(feat)
        return logits

# ---------- Metrics / Maps ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def load_rp_map(rp_map_path: str):
    """ 讀 rp_id 對應座標與樓層，回傳 dict: rid -> (x,y,floor) """
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
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
    parser.add_argument("--out_dir", type=str, default="./rssi_dnn_baseline_ckpt")
    # 原本 DNN 用的 hidden & dropout，現在當作 predictor MLP 的設定
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)

    # 新增 Transformer 的超參數
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
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
    means, stds = fit_scaler(df_tr[ap_cols].values.astype(np.float32), missing_val=args.missing_val)
    # np.savez(os.path.join(args.out_dir, "scaler_ap_means_stds.npz"), means=means, stds=stds)

    # --- Build train dataset/loader ---
    ds_tr = RSSIDataset(df_tr, ap_cols, means=means, stds=stds, missing_val=args.missing_val)
    id2idx = ds_tr.id2idx
    idx2id = ds_tr.idx2id
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # --- Prepare test arrays (只在最後 eval 用) ---
    X_te_full = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full, means, stds, args.missing_val)
    X_te_full = np.expand_dims(X_te_full, axis=1)

    # y 映射（未知類別或 -1 → -1）
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

    # 使用 Transformer-based 模型
    model = TransClassifier(
        num_ap=num_ap,
        n_classes=len(id2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        mlp_hidden=args.hidden,
        p_drop=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    # --- Train ---
    for epoch in range(1, args.epochs+1):
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

        tr_loss, tr_acc = (tr_loss/n if n>0 else 0.0), (tr_acc/n if n>0 else 0.0)
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
    preds_rpid = np.array([idx2id[int(i)] if int(i) in idx2id else -999999 for i in preds_idx], dtype=int)

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

    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")

if __name__ == "__main__":
    main()
