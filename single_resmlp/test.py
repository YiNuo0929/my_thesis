import argparse, os
import math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from math import sqrt
from itertools import cycle
import random
import matplotlib.pyplot as plt

# ---------- Reproducibility ----------
def set_seed(seed: int = 42):
    """
    固定所有常見的隨機種子，讓實驗可重現。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 讓 cudnn 行為可重現
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def get_feature_cols(df: pd.DataFrame, col):
    """
    固定只抓 ap0 ~ ap(col-1) 共 col 維
    """
    all_cols = set(df.columns)
    ap_cols = [f"ap{i}" for i in range(col)]
    missing = [c for c in ap_cols if c not in all_cols]
    if missing:
        raise ValueError(f"資料集中缺少欄位（ap0~ap{col-1}）：{missing[:10]} ...")
    return ap_cols

def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    """
    Min-Max scaler：
    - 只用非 missing_val 的值算 per-AP min / max
    - 之後會把資料壓到 [0,1]，缺失值之後會被設成 -1
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
    Min-Max 正規化到 [0,1]，缺失值補 -1：
    - 對非 missing 的位置做 (x - min) / (max - min)，clip 到 [0,1]
    - 原本是 missing_val 的位置，最後強制設為 -1
    """
    x = x.copy().astype(np.float32)
    miss_mask = (x == missing_val)

    denom = (maxs - mins)
    denom[denom == 0.0] = 1.0
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)

    # 缺失的地方直接設成 -1
    x[miss_mask] = -1.0
    return x

# ---------- Augmentation & Loss Functions (NEW) ----------
def augment_rssi(x: torch.Tensor, noise_std=0.05, drop_prob=0.1, missing_val=-1.0):
    """
    對 Target RSSI 進行增強：
    1. Jitter: 對非缺失值加上高斯雜訊。
    2. Masking: 隨機將部分非缺失值設為 missing_val (模擬 AP 消失)。
    """
    x_aug = x.clone()
    valid_mask = (x_aug != missing_val)
    
    # 1. Jitter
    if noise_std > 0:
        noise = torch.randn_like(x_aug) * noise_std
        jittered = x_aug + noise
        jittered = torch.clamp(jittered, 0.0, 1.0)
        x_aug[valid_mask] = jittered[valid_mask]
    
    # 2. Dropout
    if drop_prob > 0:
        drop_mask = torch.rand_like(x_aug) < drop_prob
        final_drop_mask = valid_mask & drop_mask
        x_aug[final_drop_mask] = missing_val
    
    return x_aug

def consistency_loss(logits_aug, logits_clean):
    """
    計算兩個 Logits 之間的 KL Divergence。
    input (aug) 轉為 log_softmax，target (clean) 轉為 softmax
    """
    log_probs_aug = F.log_softmax(logits_aug, dim=1)
    probs_clean = F.softmax(logits_clean, dim=1)
    return F.kl_div(log_probs_aug, probs_clean, reduction='batchmean')

def calc_entropy_loss(logits):
    """
    Conditional Entropy Minimization:
    H(p) = - sum( p(x) * log p(x) )
    """
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(p * log_p, dim=1)
    return torch.mean(entropy)

# ---------- Datasets ----------
class RSSISourceDataset(Dataset):
    """
    Source domain：有 label 的資料（rp_id）
    """
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
        # [N, 1, L]
        self.X = np.expand_dims(self.X, axis=1)

        # rp_id -> [0..C-1]
        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        # 不在映射的（例如 -1）設為 -1
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

class RSSITargetDataset(Dataset):
    """
    Target domain：unlabeled（不使用 rp_id，只做 reconstruction）
    """
    def __init__(self, df: pd.DataFrame, ap_cols,
                 mins=None, maxs=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.X = df[ap_cols].values.astype(np.float32)
        self.missing_val = missing_val
        self.mins = mins
        self.maxs = maxs
        if (mins is not None) and (maxs is not None):
            self.X = apply_scaler(self.X, self.mins, self.maxs, self.missing_val)
        # [N, 1, L]
        self.X = np.expand_dims(self.X, axis=1)

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

# -------- Positional Encoding --------
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

# -------- ViT-style Extractor --------
class TransformerExtractor(nn.Module):
    def __init__(
        self,
        num_tokens: int,          # 給 ViT 的 token 數，例如 1024
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool,
        mask_value: float,        # 哪個值當作 padding/missing (normalized -1)
        z_dim: int,               # 壓縮後 latent 維度
        bottleneck_hidden: int = None,  # 中間 hidden 維度（如果 None 就用 d_model）
        use_mask: bool = False,
    ):
        super().__init__()
        # ---- 2D reshape 設定 ----
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"num_tokens={num_tokens} 不能剛好 reshape 成正方形")
        self.side = side              # e.g. 32 for 1024 tokens
        self.num_tokens = num_tokens
        self.patch_size = 8           # 固定用 4x4 patch
        if self.side % self.patch_size != 0:
            raise ValueError(f"side={self.side} 不能被 patch_size={self.patch_size} 整除")
        self.num_patches_per_side = self.side // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        patch_dim = 1 * self.patch_size * self.patch_size  # 單通道，所以 C=1

        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.z_dim = z_dim
        self.use_mask = use_mask
        self.mask_value = mask_value

        # patch flatten -> d_model
        self.patch_proj = nn.Linear(patch_dim, d_model)

        # CLS token（可選）
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            max_len = self.num_patches + 1
        else:
            self.cls_token = None
            max_len = self.num_patches

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- bottleneck MLP: CLS(d_model) → hidden → z_dim ----
        if bottleneck_hidden is None:
            bottleneck_hidden = d_model  # 預設 hidden = d_model

        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_hidden, z_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, 1, L] 或 [B, L]（normalized, L=num_tokens ）
        return: z ∈ [B, z_dim]
        """
        if x.dim() == 3:
            # [B, 1, L] -> [B, L]
            x = x.squeeze(1)

        B, L = x.shape
        if L != self.num_tokens:
            raise ValueError(f"輸入長度 L={L} 和 num_tokens={self.num_tokens} 不一致")

        # ---- 轉成 2D 影像 [B,1,H,W] ----
        img = x.view(B, 1, self.side, self.side)   # [B,1,side,side]

        # ---- 用 unfold 切 4x4 patch ----
        # patches_raw: [B, patch_dim, num_patches]
        patches_raw = F.unfold(
            img, kernel_size=self.patch_size, stride=self.patch_size
        )
        # [B, num_patches, patch_dim]
        patches = patches_raw.transpose(1, 2)

        # ---- 針對 patch 做 mask（如果啟用）----
        key_padding_mask = None  # [B, num_patches]
        if self.use_mask:
            # 一個 patch 裡所有元素都等於 mask_value（-1）就當作 missing
            # patches_raw: [B, patch_dim, num_patches]
            eq_mask = (patches_raw == self.mask_value)      # [B, patch_dim, num_patches]
            key_padding_mask = eq_mask.all(dim=1)           # [B, num_patches] bool

        # ---- patch embedding ----
        # patches: [B, num_patches, patch_dim] -> [B, num_patches, d_model]
        h = self.patch_proj(patches)                        # [B, T, d_model], T=num_patches

        # ---- 加 CLS token ----
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)   # [B,1,d_model]
            h = torch.cat([cls, h], dim=1)                    # [B,1+T,d_model]

        # ---- 位置編碼 ----
        h = self.pos_encoding(h)                              # [B,T',d_model]

        # ---- 準備給 encoder 的 key_padding_mask ----
        src_key_padding_mask = None
        if key_padding_mask is not None:
            if self.use_cls_token:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)  # CLS 不 mask
                src_key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # [B,1+num_patches]
            else:
                src_key_padding_mask = key_padding_mask                       # [B,num_patches]

        # ---- Encoder ----
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)        # [B,T',d_model]

        # ---- 取 CLS 或 mean pooling ----
        if self.use_cls_token:
            cls_feat = h[:, 0, :]                    # [B, d_model]
        else:
            cls_feat = h.mean(dim=1)                 # [B, d_model]

        # ---- 經 bottleneck MLP 壓縮成 z ----
        z = self.bottleneck(cls_feat)                # [B, z_dim]
        return z

# -------- Predictor (MLP head) --------
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
# -------- Residual MLP head (single-head) --------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return x + self.net(x)


class PredictorResidualMLP(nn.Module):
    """
    z -> (Linear to hidden_dim) -> N residual blocks -> classifier
    """
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 256, num_blocks: int = 4, p_drop=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, p_drop=p_drop) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.head(h)
# -------- Reconstruction head (MLP) --------
class ReconstructionMLP(nn.Module):
    """
    用同一個 z 走另一條 MLP，把它 reconstruct 回原始的 RSSI 向量（normalized 空間）。
    output: [B, num_ap]
    """
    def __init__(self, in_dim: int, out_dim: int, hidden=[256, 256], p_drop=0.2):
        super().__init__()
        dims = [in_dim] + hidden
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks.append(MLPBlock(a, b, p_drop=p_drop))
        self.feat = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        x = self.feat(x)
        out = self.head(x)   # [B, out_dim]
        return out

# -------- 整體模型 = (Linear 1033→1024) + TransformerExtractor + PredictorMLP + ReconstructionMLP --------
class TransClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,       # 原始 AP 維度（例如 1033）
        vit_tokens: int,      # 給 ViT 的 token 數（例如 1024）
        n_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        z_dim: int,           # 壓縮後 latent 維度
        pred_hidden,
        recon_hidden,
        p_drop: float,
        use_mask: bool,
        mask_value: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.vit_tokens = vit_tokens

        # 如果 input_dim != vit_tokens，前面加一層 Linear 做維度轉換
        if input_dim != vit_tokens:
            self.pre_linear = nn.Linear(input_dim, vit_tokens)
        else:
            self.pre_linear = None

        # extractor：2D ViT-style，學 AP 間關係並壓成 z
        self.extractor = TransformerExtractor(
            num_tokens=vit_tokens,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_cls_token=True,
            z_dim=z_dim,
            bottleneck_hidden=None,  # 預設 hidden = d_model
            use_mask=use_mask,
            mask_value=mask_value
        )
        # predictor：分類 head，用 pred_hidden
        # predictor：單頭 residual MLP（固定 head2 設計）
        self.predictor = PredictorResidualMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden_dim=256,    # residual block hidden 寬度
            num_blocks=4,      # residual block 數量（你的「層數」概念）
            p_drop=p_drop,
        )
        # recon head：重建「原始維度」的 RSSI（輸出維度 = input_dim），用 recon_hidden
        self.reconstructor = ReconstructionMLP(
            in_dim=z_dim,
            out_dim=input_dim,
            hidden=recon_hidden,
            p_drop=p_drop,
        )

    def forward(self, x):
        # x: [B, 1, L_in]，L_in = input_dim (例如 1033)
        if x.dim() == 3:
            x_flat = x.squeeze(1)   # [B, L_in]
        else:
            x_flat = x              # [B, L_in]

        # 若需要，先用 Linear 把 1033 壓成 1024
        if self.pre_linear is not None:
            x_vit = self.pre_linear(x_flat)  # [B, vit_tokens=1024]
        else:
            x_vit = x_flat                   # [B, L_in]

        # 給 ViT 的是 [B, 1, vit_tokens]
        x_vit = x_vit.unsqueeze(1)           # [B, 1, vit_tokens]

        # 取 z
        z = self.extractor(x_vit)            # [B, z_dim]

        # classifier
        logits = self.predictor(z)

        # recon 回原始維度（1033）
        recon = self.reconstructor(z)        # [B, input_dim]

        return logits, recon

# ---------- Metrics / Maps ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def load_rp_map(rp_map_path: str):
    """ 讀 rp_id 對應座標與樓層，回傳 dict: rid -> (x,y,floor) """
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
    if not need.issubset(df.columns):
        raise ValueError("rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp
# ---------- Viz utils (single-head: clean / aug / mc-dropout-ens) ----------
def _entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)

@torch.no_grad()
def _predict_probs(model, xb: torch.Tensor) -> torch.Tensor:
    logits, _ = model(xb)
    return F.softmax(logits, dim=1)

def _set_bn_eval(m: nn.Module):
    if isinstance(m, nn.BatchNorm1d):
        m.eval()

@torch.no_grad()
def collect_probs_singlehead(model, dl, device, mode: str,
                             mc_passes: int = 20,
                             noise_std: float = 0.1,
                             drop_prob: float = 0.3):
    """
    mode:
      - 'clean' : normal inference
      - 'aug'   : apply augment_rssi then inference
      - 'mc'    : MC dropout ensemble (avg softmax over mc_passes)
    return: probs [N, C] on CPU (float32)
    """
    probs_all = []

    if mode == "mc":
        # enable dropout but keep BN frozen
        model.train()
        model.apply(_set_bn_eval)
    else:
        model.eval()

    for batch in dl:
        # dl_te: (xb, yb_idx, yb_rpid) ; dl_tgt: xb only
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device)

        if mode == "clean":
            p = _predict_probs(model, xb)

        elif mode == "aug":
            xb_aug = augment_rssi(xb, noise_std=noise_std, drop_prob=drop_prob, missing_val=-1.0)
            p = _predict_probs(model, xb_aug)

        elif mode == "mc":
            ps = []
            for _ in range(mc_passes):
                ps.append(_predict_probs(model, xb))
            p = torch.stack(ps, dim=0).mean(dim=0)  # [B,C]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        probs_all.append(p.detach().cpu())

    return torch.cat(probs_all, dim=0).numpy().astype(np.float32)

def make_viz_three_plots_singlehead(
    probs_clean: np.ndarray,
    probs_aug: np.ndarray,
    probs_mc: np.ndarray,
    out_dir: str,
    prefix: str = "singlehead",
):
    os.makedirs(out_dir, exist_ok=True)

    # confidence & entropy
    conf_clean = probs_clean.max(axis=1)
    conf_aug   = probs_aug.max(axis=1)
    conf_mc    = probs_mc.max(axis=1)

    ent_clean = _entropy_from_probs(probs_clean)
    ent_aug   = _entropy_from_probs(probs_aug)
    ent_mc    = _entropy_from_probs(probs_mc)

    # 1) Confidence histogram
    plt.figure()
    plt.hist(conf_clean, bins=40, alpha=0.5, label="clean")
    plt.hist(conf_aug,   bins=40, alpha=0.5, label="aug")
    plt.hist(conf_mc,    bins=40, alpha=0.5, label="mc_ens")
    plt.xlabel("max probability (confidence)")
    plt.ylabel("count")
    plt.title("Confidence distribution: clean vs aug vs mc-ensemble")
    plt.legend()
    p1 = os.path.join(out_dir, f"{prefix}_viz_confidence_hist.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Disagreement heatmap (argmax mismatch rate)
    pred_clean = probs_clean.argmax(axis=1)
    pred_aug   = probs_aug.argmax(axis=1)
    pred_mc    = probs_mc.argmax(axis=1)

    names = ["clean", "aug", "mc_ens"]
    preds = [pred_clean, pred_aug, pred_mc]
    M = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            M[i, j] = np.mean(preds[i] != preds[j])

    plt.figure()
    im = plt.imshow(M)
    plt.xticks(range(3), names, rotation=45, ha="right")
    plt.yticks(range(3), names)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="disagreement rate")
    plt.title("Argmax disagreement rate (lower = more similar)")
    p2 = os.path.join(out_dir, f"{prefix}_viz_disagreement_heatmap.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Entropy boxplot
    plt.figure()
    plt.boxplot([ent_clean, ent_aug, ent_mc], labels=names, showfliers=False)
    plt.ylabel("entropy")
    plt.title("Entropy distribution: clean vs aug vs mc-ensemble")
    p3 = os.path.join(out_dir, f"{prefix}_viz_entropy_boxplot.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    print("[VIZ saved]")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)
# ---------- Reconstruction Loss ----------
def reconstruction_loss(pred, target, mask_value=-1.0):
    """
    pred, target: [B, L]
    只在 target != mask_value 的位置算 MSE，避免缺失值干擾。
    """
    mask = (target != mask_value)
    if mask.sum() == 0:
        return (pred * 0.0).sum()   # safe zero loss
    diff = pred - target
    diff = diff[mask]
    return torch.mean(diff * diff)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    # 這裡改成 Source / Target 兩個 train path
    parser.add_argument("--source_train_path", type=str, required=True,
                        help="source domain train csv (有 label)")
    parser.add_argument("--target_train_path", type=str, required=True,
                        help="target domain train csv (unlabeled)")
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, default=r"rp_id.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./single_resmlp")
    parser.add_argument("--column", type=int, default=256)   # 新的 case 記得給 1033

    # predictor / recon MLP 的設定（已拆開）
    parser.add_argument("--pred_hidden", type=int, nargs="+", default=[256, 256],
                        help="hidden sizes for predictor MLP")
    parser.add_argument("--recon_hidden", type=int, nargs="+", default=[256],
                        help="hidden sizes for reconstruction MLP")
    parser.add_argument("--dropout", type=float, default=0.2)

    # Transformer 的超參數
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)

    # 壓縮後 latent 維度 z_dim
    parser.add_argument("--z_dim", type=int, default=64)

    # reconstruction loss 的權重 λ
    parser.add_argument("--lambda_recon", type=float, default=0.1)

    # Consistency Loss & Augmentation 參數
    parser.add_argument("--lambda_consist", type=float, default=0.07, help="weight for consistency loss")
    parser.add_argument("--noise_std", type=float, default=0.1, help="std for gaussian noise jitter")
    parser.add_argument("--drop_prob", type=float, default=0.3, help="probability to drop AP signal")
    
    # Entropy Minimization 參數
    parser.add_argument("--lambda_entropy", type=float, default=0.05, help="weight for entropy minimization loss")

    # 是否啟用 key_padding_mask
    parser.add_argument("--use_mask", action="store_true", help="啟用 key_padding_mask")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")

    # 給 ViT 用的 token 數，這裡固定用 1024
    parser.add_argument("--vit_tokens", type=int, default=256,
                        help="token 數（必須能 sqrt 成整數，且 side 能被 patch_size=4 整除）")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Load data ---
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src, args.column)   # 只會拿 ap0 ~ ap(column-1)
    # 確保 target / test 也有這些欄位
    assert set(ap_cols).issubset(df_tgt.columns), "Target train 缺少部分 AP 欄位"
    assert set(ap_cols).issubset(df_te.columns),  "Test 缺少部分 AP 欄位"

    # 類別檢查（只看 source）
    n_classes_src = df_src["rp_id"].nunique()
    if n_classes_src != 48:
        print(f"[WARN] Source 訓練集 rp 類別數={n_classes_src}（預期 48）")

    # --- Fit scaler on source train (Min-Max) ---
    mins, maxs = fit_scaler(df_src[ap_cols].values.astype(np.float32), missing_val=args.missing_val)

    # --- Build source / target dataset/loader ---
    ds_src = RSSISourceDataset(df_src, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_src.id2idx
    idx2id = ds_src.idx2id
    dl_src = DataLoader(ds_src, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # --- Prepare test arrays (只在最後 eval 用) ---
    X_te_full_raw = df_te[ap_cols].values.astype(np.float32)
    X_te_full = apply_scaler(X_te_full_raw, mins, maxs, args.missing_val)
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
    num_ap = len(ap_cols)           # 例如 1033
    vit_tokens = args.vit_tokens    # 例如 1024

    model = TransClassifier(
        input_dim=num_ap,
        vit_tokens=vit_tokens,
        n_classes=len(id2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        z_dim=args.z_dim,
        pred_hidden=args.pred_hidden,
        recon_hidden=args.recon_hidden,
        p_drop=args.dropout,
        use_mask=args.use_mask,
        mask_value=-1.0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    print("---------------------------------------")
    print(f"use mask or not: {model.extractor.use_mask}")
    print(f"z_dim: {model.extractor.z_dim}")
    print(f"d_model: {model.extractor.d_model}")
    print(f"input_dim={model.input_dim}, vit_tokens={model.vit_tokens}")
    print(f"pred_hidden={args.pred_hidden}, recon_hidden={args.recon_hidden}")
    print(f"Consistency: λ={args.lambda_consist}, noise={args.noise_std}, drop={args.drop_prob}")
    print(f"Entropy: λ={args.lambda_entropy}")
    print("---------------------------------------")

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_total = 0.0
        tr_ce_total = 0.0
        tr_recon_s_total = 0.0
        tr_recon_t_total = 0.0
        tr_consist_total = 0.0
        tr_ent_total = 0.0
        tr_acc, n = 0.0, 0

        tgt_iter = cycle(dl_tgt)  # target 比較少就循環使用

        for xb_s, yb_s in dl_src:
            keep = (yb_s != -1)
            if not keep.any():
                continue
            xb_s, yb_s = xb_s[keep], yb_s[keep]

            xb_s = xb_s.to(device, non_blocking=True)
            yb_s = yb_s.to(device, non_blocking=True)

            # 取一個 target batch（不需要 label）
            xb_t = next(tgt_iter)
            xb_t = xb_t.to(device, non_blocking=True)
            
            # --- Augmentation (Target) ---
            with torch.no_grad():
                xb_t_aug = augment_rssi(xb_t, noise_std=args.noise_std, drop_prob=args.drop_prob, missing_val=-1.0)
                xb_t_aug = xb_t_aug.to(device, non_blocking=True)

            opt.zero_grad()

            # 1. Source Flow: CE + Recon
            logits_s, recon_s = model(xb_s)
            ce_loss = crit(logits_s, yb_s)

            x_s_norm = xb_s.squeeze(1)  # [B, L_in]
            recon_loss_s = reconstruction_loss(recon_s, x_s_norm, mask_value=-1.0)

            # 2. Target Flow (Original): Recon + Entropy + (Target for Consistency)
            logits_t, recon_t = model(xb_t)
            x_t_norm = xb_t.squeeze(1)  # [B, L_in]
            recon_loss_t = reconstruction_loss(recon_t, x_t_norm, mask_value=-1.0)
            
            # NEW: Entropy Loss (只針對原始 Target)
            loss_ent = calc_entropy_loss(logits_t)

            # 3. Target Flow (Augmented): Predictor ONLY (for Consistency)
            # 為了效率，這裡手動執行 TransClassifier 的前半段邏輯，只跑 predictor 分支
            x_aug_flat = xb_t_aug.squeeze(1)
            if model.pre_linear is not None:
                x_vit = model.pre_linear(x_aug_flat)
            else:
                x_vit = x_aug_flat
            x_vit = x_vit.unsqueeze(1)
            z_aug = model.extractor(x_vit)
            logits_t_aug = model.predictor(z_aug)

            # 4. Consistency Loss
            loss_consist = consistency_loss(logits_t_aug, logits_t)

            recon_loss_all = recon_loss_s + recon_loss_t
            total_loss = ce_loss \
                         + args.lambda_recon * recon_loss_all \
                         + args.lambda_consist * loss_consist \
                         + args.lambda_entropy * loss_ent

            total_loss.backward()
            opt.step()

            bs = yb_s.size(0)
            tr_loss_total += total_loss.item() * bs
            tr_ce_total += ce_loss.item() * bs
            tr_recon_s_total += recon_loss_s.item() * bs
            tr_recon_t_total += recon_loss_t.item() * bs
            tr_consist_total += loss_consist.item() * bs
            tr_ent_total += loss_ent.item() * bs
            tr_acc += (logits_s.argmax(1) == yb_s).float().sum().item()
            n += bs

        if n > 0:
            tr_loss_total /= n
            tr_ce_total /= n
            tr_recon_s_total /= n
            tr_recon_t_total /= n
            tr_consist_total /= n
            tr_ent_total /= n
            tr_acc /= n
        else:
            tr_loss_total = tr_ce_total = tr_recon_s_total = tr_recon_t_total = tr_acc = 0.0

        print(f"Epoch {epoch:03d} | "
              f"Tot {tr_loss_total:.3f} | CE {tr_ce_total:.3f} | "
              f"RecS {tr_recon_s_total:.3f} RecT {tr_recon_t_total:.3f} | "
              f"Cst {tr_consist_total:.3f} Ent {tr_ent_total:.3f} | Acc {tr_acc:.3f}")

    # --- Final Evaluation on Test (一次，只看 predictor 分支) ---
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

    avg_mde = (float(np.mean(mde_distances)) if len(mde_distances) > 0 else float("nan"))

    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(gts_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")
        # --- VIZ (single-head) ---
    # 用 test set 畫：clean / aug / mc-dropout-ens
    probs_clean = collect_probs_singlehead(
        model, dl_te, device, mode="clean"
    )
    probs_aug = collect_probs_singlehead(
        model, dl_te, device, mode="aug",
        noise_std=args.noise_std, drop_prob=args.drop_prob
    )
    probs_mc = collect_probs_singlehead(
        model, dl_te, device, mode="mc",
        mc_passes=20
    )

    make_viz_three_plots_singlehead(
        probs_clean=probs_clean,
        probs_aug=probs_aug,
        probs_mc=probs_mc,
        out_dir=args.out_dir,
        prefix="singlehead_resmlp"
    )
if __name__ == "__main__":
    main()