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

# NEW: visualization
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
    """
    x = x.copy().astype(np.float32)
    miss_mask = (x == missing_val)

    denom = (maxs - mins)
    denom[denom == 0.0] = 1.0
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)
    x[miss_mask] = -1.0
    return x

# ---------- Augmentation & Loss Functions ----------
def augment_rssi(x: torch.Tensor, noise_std=0.05, drop_prob=0.1, missing_val=-1.0):
    """
    對 Target RSSI 進行增強：
    1) Jitter  2) Masking
    """
    x_aug = x.clone()
    valid_mask = (x_aug != missing_val)

    if noise_std > 0:
        noise = torch.randn_like(x_aug) * noise_std
        jittered = x_aug + noise
        jittered = torch.clamp(jittered, 0.0, 1.0)
        x_aug[valid_mask] = jittered[valid_mask]

    if drop_prob > 0:
        drop_mask = torch.rand_like(x_aug) < drop_prob
        final_drop_mask = valid_mask & drop_mask
        x_aug[final_drop_mask] = missing_val

    return x_aug

def consistency_loss(logits_aug, logits_clean):
    """
    KL(aug || clean)
    """
    log_probs_aug = F.log_softmax(logits_aug, dim=1)
    probs_clean = F.softmax(logits_clean, dim=1)
    return F.kl_div(log_probs_aug, probs_clean, reduction='batchmean')

def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12):
    """
    H(p) = -sum p log p
    p: [B,C]
    """
    p = torch.clamp(p, eps, 1.0)
    return torch.mean(-torch.sum(p * torch.log(p), dim=1))

def ensemble_probs_from_logits_list(logits_list):
    """
    logits_list: list of [B,C]
    return p_ens: [B,C]
    """
    ps = [F.softmax(lg, dim=1) for lg in logits_list]
    return torch.stack(ps, dim=0).mean(dim=0)

# ---------- Datasets ----------
class RSSISourceDataset(Dataset):
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
        self.X = np.expand_dims(self.X, axis=1)

        uniq = np.sort(np.unique(self.y_raw[self.y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in self.y_raw], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

class RSSITargetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ap_cols,
                 mins=None, maxs=None, missing_val=-110.0):
        self.ap_cols = ap_cols
        self.X = df[ap_cols].values.astype(np.float32)
        self.missing_val = missing_val
        self.mins = mins
        self.maxs = maxs
        if (mins is not None) and (maxs is not None):
            self.X = apply_scaler(self.X, self.mins, self.maxs, self.missing_val)
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

class ResidualMLPBlock(nn.Module):
    """
    NEW: residual MLP block for 3rd head
    """
    def __init__(self, dim, p_drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.bn2(h)
        return F.relu(x + h, inplace=True)

# -------- Positional Encoding --------
class PositionalEncoding(nn.Module):
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

# -------- ViT-style Extractor --------
class TransformerExtractor(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool,
        mask_value: float,
        z_dim: int,
        bottleneck_hidden: int = None,
        use_mask: bool = False,
    ):
        super().__init__()
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"num_tokens={num_tokens} 不能剛好 reshape 成正方形")
        self.side = side
        self.num_tokens = num_tokens
        self.patch_size = 8
        if self.side % self.patch_size != 0:
            raise ValueError(f"side={self.side} 不能被 patch_size={self.patch_size} 整除")
        self.num_patches_per_side = self.side // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        patch_dim = 1 * self.patch_size * self.patch_size

        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.z_dim = z_dim
        self.use_mask = use_mask
        self.mask_value = mask_value

        self.patch_proj = nn.Linear(patch_dim, d_model)

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
            x = x.squeeze(1)

        B, L = x.shape
        if L != self.num_tokens:
            raise ValueError(f"輸入長度 L={L} 和 num_tokens={self.num_tokens} 不一致")

        img = x.view(B, 1, self.side, self.side)

        patches_raw = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches_raw.transpose(1, 2)

        key_padding_mask = None
        if self.use_mask:
            eq_mask = (patches_raw == self.mask_value)
            key_padding_mask = eq_mask.all(dim=1)

        h = self.patch_proj(patches)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            h = torch.cat([cls, h], dim=1)

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

# -------- Predictor heads --------
class PredictorMLP(nn.Module):
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
        return self.head(x)

class PredictorResidualMLP(nn.Module):
    """
    NEW: residual head
    """
    def __init__(self, in_dim: int, n_classes: int, hidden_dim=256, num_blocks=2, p_drop=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, p_drop=p_drop) for _ in range(num_blocks)])
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out(x)

# -------- Reconstruction head (MLP) --------
class ReconstructionMLP(nn.Module):
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
        return self.head(x)

# -------- TransClassifier (3-head predictors) --------
class TransClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vit_tokens: int,
        n_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        z_dim: int,
        #pred_hidden,
        recon_hidden,
        p_drop: float,
        use_mask: bool,
        mask_value: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.vit_tokens = vit_tokens

        if input_dim != vit_tokens:
            self.pre_linear = nn.Linear(input_dim, vit_tokens)
        else:
            self.pre_linear = None

        self.extractor = TransformerExtractor(
            num_tokens=vit_tokens,
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

        # ===== NEW: 3 heads =====
        '''
        # head0: shallow MLP
        head0 = PredictorMLP(in_dim=z_dim, n_classes=n_classes, hidden=[pred_hidden[0]] if len(pred_hidden) > 0 else [], p_drop=p_drop)
        # head1: deep MLP
        head1 = PredictorMLP(in_dim=z_dim, n_classes=n_classes, hidden=pred_hidden, p_drop=p_drop)
        # head2: residual MLP
        head2 = PredictorResidualMLP(in_dim=z_dim, n_classes=n_classes, hidden_dim=pred_hidden[0] if len(pred_hidden) > 0 else 256, num_blocks=2, p_drop=p_drop)
        '''
        head0 = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=[256, 256],          # 你要幾層就放幾個數字
            p_drop=p_drop
        )

        # head1: deep MLP (3 layers)
        head1 = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=[512, 256, 256],  # 這裡就是層數與每層寬度
            p_drop=p_drop
        )

        # head2: residual MLP (4 residual blocks)
        
        head2 = PredictorResidualMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden_dim=256,        # residual block 的 hidden 寬度
            num_blocks=4,          # residual block 數量 = 你要的「層數」概念
            p_drop=p_drop
        )
        '''
        head2 = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=[512, 256, 256],  # 這裡就是層數與每層寬度
            p_drop=p_drop
        )
        '''
        self.predictors = nn.ModuleList([head0, head1, head2])

        self.reconstructor = ReconstructionMLP(
            in_dim=z_dim,
            out_dim=input_dim,
            hidden=recon_hidden,
            p_drop=p_drop,
        )

    def forward(self, x):
        if x.dim() == 3:
            x_flat = x.squeeze(1)
        else:
            x_flat = x

        if self.pre_linear is not None:
            x_vit = self.pre_linear(x_flat)
        else:
            x_vit = x_flat

        x_vit = x_vit.unsqueeze(1)
        z = self.extractor(x_vit)

        # NEW: logits_list
        logits_list = [head(z) for head in self.predictors]

        recon = self.reconstructor(z)
        return logits_list, recon

    # NEW: predictor-only helper (for consistency path)
    def forward_predictors_only(self, x):
        """
        x: [B,1,input_dim] normalized
        return logits_list
        """
        if x.dim() == 3:
            x_flat = x.squeeze(1)
        else:
            x_flat = x
        if self.pre_linear is not None:
            x_vit = self.pre_linear(x_flat)
        else:
            x_vit = x_flat
        x_vit = x_vit.unsqueeze(1)
        z = self.extractor(x_vit)
        return [head(z) for head in self.predictors]
    # NEW: extract z helper
    def forward_z(self, x):
        """
        x: [B,1,input_dim] normalized
        return z: [B,z_dim]
        """
        if x.dim() == 3:
            x_flat = x.squeeze(1)
        else:
            x_flat = x

        if self.pre_linear is not None:
            x_vit = self.pre_linear(x_flat)
        else:
            x_vit = x_flat

        x_vit = x_vit.unsqueeze(1)
        z = self.extractor(x_vit)
        return z

# ---------- Metrics / Maps ----------
def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id","x","y","floor"}
    if not need.issubset(df.columns):
        raise ValueError("rp_id.csv 缺少欄位（需要 {need}）")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp

# ---------- Reconstruction Loss ----------
def reconstruction_loss(pred, target, mask_value=-1.0):
    mask = (target != mask_value)
    if mask.sum() == 0:
        return (pred * 0.0).sum()
    diff = pred - target
    diff = diff[mask]
    return torch.mean(diff * diff)

# ---------- Visualization (NEW) ----------
def save_three_head_plots(out_dir, head_probs_np, ens_probs_np, gt_np=None):
    """
    head_probs_np: list of [N,C] numpy
    ens_probs_np:  [N,C] numpy
    gt_np: [N] numpy or None
    會輸出 3 張圖：
      1) 每個 head/ensemble 的 confidence(max prob) 分佈
      2) head 之間 argmax disagreement heatmap
      3) 每個 head/ensemble 的 entropy boxplot
    """
    os.makedirs(out_dir, exist_ok=True)
    H = len(head_probs_np)
    labels = [f"head{i}" for i in range(H)] + ["ens"]

    # ---- 1) confidence histogram ----
    plt.figure()
    for i in range(H):
        conf = head_probs_np[i].max(axis=1)
        plt.hist(conf, bins=40, alpha=0.5, label=f"head{i}")
    plt.hist(ens_probs_np.max(axis=1), bins=40, alpha=0.5, label="ens")
    plt.xlabel("max probability (confidence)")
    plt.ylabel("count")
    plt.title("Confidence distribution: heads vs ensemble")
    plt.legend()
    fp1 = os.path.join(out_dir, "viz_confidence_hist.png")
    plt.savefig(fp1, dpi=200, bbox_inches="tight")
    plt.close()

    # ---- 2) disagreement heatmap ----
    def argmax_np(p): return np.argmax(p, axis=1)
    preds = [argmax_np(p) for p in head_probs_np] + [argmax_np(ens_probs_np)]
    M = len(preds)
    mat = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            mat[i, j] = np.mean(preds[i] != preds[j])

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.xticks(range(M), labels, rotation=45)
    plt.yticks(range(M), labels)
    plt.colorbar(label="disagreement rate")
    plt.title("Argmax disagreement rate (lower = more similar)")
    fp2 = os.path.join(out_dir, "viz_disagreement_heatmap.png")
    plt.savefig(fp2, dpi=200, bbox_inches="tight")
    plt.close()

    # ---- 3) entropy boxplot ----
    def entropy_np(p, eps=1e-12):
        p = np.clip(p, eps, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    ent_list = [entropy_np(p) for p in head_probs_np] + [entropy_np(ens_probs_np)]
    plt.figure()
    plt.boxplot(ent_list, labels=labels, showfliers=False)
    plt.ylabel("entropy")
    plt.title("Entropy distribution: heads vs ensemble")
    fp3 = os.path.join(out_dir, "viz_entropy_boxplot.png")
    plt.savefig(fp3, dpi=200, bbox_inches="tight")
    plt.close()

    print("==== Saved visualizations ====")
    print(fp1)
    print(fp2)
    print(fp3)

# ---------- Attribution on z (NEW) ----------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def select_target_class(logits_np, mode="pred"):
    """
    logits_np: [N,C]
    mode:
      - "pred": 使用 model argmax 當 target class
    return y_sel: [N]
    """
    if mode == "pred":
        return np.argmax(logits_np, axis=1).astype(np.int64)
    else:
        raise ValueError("select_target_class: unsupported mode")

def grad_x_input_attrib(head_module, z_batch, target_y):
    """
    head_module: 某個 head (nn.Module), input z -> logits
    z_batch: [B, z_dim] torch, requires_grad will be set
    target_y: [B] torch long, target class for each sample
    return: attr [B, z_dim] torch
    """
    z = z_batch.detach().clone().requires_grad_(True)
    logits = head_module(z)  # [B,C]
    # gather each sample's target logit
    tlogit = logits.gather(1, target_y.view(-1,1)).squeeze(1)  # [B]
    loss = tlogit.sum()
    loss.backward()
    grad = z.grad  # [B, z_dim]
    attr = grad * z
    return attr.detach()

def integrated_gradients_attrib(head_module, z_batch, target_y, steps=50, baseline="zero"):
    """
    IG on z:
      baseline: "zero" 或 "mean"（mean 需要你外面先算好再傳進來，這裡先給 zero）
    return: attr [B, z_dim]
    """
    z = z_batch.detach()
    if baseline == "zero":
        z0 = torch.zeros_like(z)
    else:
        raise ValueError("integrated_gradients_attrib: baseline only supports 'zero' here")

    total_grad = torch.zeros_like(z)
    # Riemann sum
    for k in range(1, steps+1):
        alpha = float(k) / steps
        zk = (z0 + alpha * (z - z0)).detach().clone().requires_grad_(True)
        logits = head_module(zk)
        tlogit = logits.gather(1, target_y.view(-1,1)).squeeze(1)
        loss = tlogit.sum()
        loss.backward()
        total_grad += zk.grad.detach()
    avg_grad = total_grad / steps
    attr = (z - z0) * avg_grad
    return attr.detach()

def occlusion_attrib(head_module, z_batch, target_y, baseline="zero"):
    """
    Occlusion on z:
      對每個維度 j，把 z[:,j] 換成 baseline，再看 target logit 掉多少（Δlogit）。
    return attr [B,z_dim] where larger = more important
    """
    z = z_batch.detach()
    B, D = z.shape
    if baseline == "zero":
        z0 = torch.zeros_like(z)
    else:
        raise ValueError("occlusion_attrib: baseline only supports 'zero'")

    with torch.no_grad():
        logits = head_module(z)  # [B,C]
        base = logits.gather(1, target_y.view(-1,1)).squeeze(1)  # [B]

    attrs = []
    for j in range(D):
        z_occ = z.clone()
        z_occ[:, j] = z0[:, j]
        with torch.no_grad():
            logits_occ = head_module(z_occ)
            occ = logits_occ.gather(1, target_y.view(-1,1)).squeeze(1)  # [B]
        # drop in target logit
        attrs.append((base - occ).unsqueeze(1))  # [B,1]
    attr = torch.cat(attrs, dim=1)  # [B,D]
    return attr

def plot_attribution_bundle(out_dir, tag, attr_np, topk=20):
    """
    attr_np: [N, D] numpy
    產 3 張圖：
      1) mean abs attribution per dim
      2) topk bar
      3) heatmap (N x topk dims) using signed attribution
    """
    _ensure_dir(out_dir)
    N, D = attr_np.shape
    mean_abs = np.mean(np.abs(attr_np), axis=0)  # [D]
    top_idx = np.argsort(-mean_abs)[:topk]

    # 1) mean abs line plot
    plt.figure()
    plt.plot(mean_abs)
    plt.xlabel("z dimension")
    plt.ylabel("mean |attribution|")
    plt.title(f"{tag} - mean |attr| over z dims")
    fp1 = os.path.join(out_dir, f"{tag}_mean_abs.png")
    plt.savefig(fp1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) topk bar
    plt.figure()
    plt.bar(np.arange(topk), mean_abs[top_idx])
    plt.xticks(np.arange(topk), [str(i) for i in top_idx], rotation=45)
    plt.xlabel("top z dims")
    plt.ylabel("mean |attribution|")
    plt.title(f"{tag} - top{topk} dims")
    fp2 = os.path.join(out_dir, f"{tag}_top{topk}_bar.png")
    plt.savefig(fp2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) heatmap for topk dims (signed)
    heat = attr_np[:, top_idx]  # [N, topk]
    plt.figure()
    plt.imshow(heat, aspect="auto")
    plt.colorbar(label="attribution (signed)")
    plt.xlabel("top z dims (sorted by mean|attr|)")
    plt.ylabel("samples")
    plt.title(f"{tag} - heatmap (samples x top{topk} dims)")
    fp3 = os.path.join(out_dir, f"{tag}_top{topk}_heatmap.png")
    plt.savefig(fp3, dpi=200, bbox_inches="tight")
    plt.close()

    print("[ATTR saved]", fp1)
    print("[ATTR saved]", fp2)
    print("[ATTR saved]", fp3)

def run_z_attribution(
    model,
    dl,
    device,
    out_dir,
    max_batches=2,          # 控制用多少 batch 做 attribution（避免太慢）
    topk=20,
    ig_steps=50,
):
    """
    dl: 通常用 test loader 或你想看的 target loader（乾淨資料）
    這裡用 'pred class' 當 target class（每個 head 自己的 pred class）
    另外做 ensemble 的 pred class（用平均機率）。
    """
    _ensure_dir(out_dir)
    model.eval()

    # collect a subset
    zs_all = []
    logits_heads_all = None  # list of list
    with torch.no_grad():
        for bidx, batch in enumerate(dl):
            if bidx >= max_batches:
                break
            # batch could be (xb, y, ...) or just xb
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device)
            z = model.forward_z(xb)  # [B,D]
            logits_list, _ = model(xb)  # list of [B,C]
            zs_all.append(z.cpu())

            if logits_heads_all is None:
                logits_heads_all = [[] for _ in range(len(logits_list))]
            for i, lg in enumerate(logits_list):
                logits_heads_all[i].append(lg.cpu())

    if not zs_all:
        print("[ATTR] No batches collected.")
        return

    z_all = torch.cat(zs_all, dim=0).to(device)  # [N,D]
    logits_heads = [torch.cat(chunks, dim=0).to(device) for chunks in logits_heads_all]  # each [N,C]
    H = len(logits_heads)
    N, D = z_all.shape

    # ensemble logits -> probs -> ensemble pred class
    with torch.no_grad():
        probs_heads = [F.softmax(lg, dim=1) for lg in logits_heads]
        probs_ens = torch.stack(probs_heads, dim=0).mean(dim=0)  # [N,C]
        y_ens = torch.argmax(probs_ens, dim=1)  # [N]

    # ---- Per-head target class = that head's own pred ----
    y_heads = [torch.argmax(lg.detach(), dim=1) for lg in logits_heads]  # list of [N]

    # ---- Attribution for each head (3 methods) ----
    head_ig_mean_abs = {}
    for hi in range(H):
        head = model.predictors[hi]

        # Grad×Input
        attr_gxi = grad_x_input_attrib(head, z_all, y_heads[hi]).cpu().numpy()
        plot_attribution_bundle(out_dir, f"head{hi}_gradxinput", attr_gxi, topk=topk)

        # IG
        attr_ig = integrated_gradients_attrib(head, z_all, y_heads[hi], steps=ig_steps).cpu().numpy()
        plot_attribution_bundle(out_dir, f"head{hi}_intgrad", attr_ig, topk=topk)
        head_ig_mean_abs[hi] = np.mean(np.abs(attr_ig), axis=0)

        # Occlusion
        attr_occ = occlusion_attrib(head, z_all, y_heads[hi]).cpu().numpy()
        plot_attribution_bundle(out_dir, f"head{hi}_occlusion", attr_occ, topk=topk)

    # ---- Ensemble attribution（把每個 head attribution 取平均，對應 ensemble 行為）----
    # 這裡的 target class 用 ensemble pred class y_ens
    # 做法：各 head 用同一個 y_ens 算 attribution，再平均
    attr_ens_gxi = []
    attr_ens_ig = []
    attr_ens_occ = []
    for hi in range(H):
        head = model.predictors[hi]
        attr_ens_gxi.append(grad_x_input_attrib(head, z_all, y_ens).cpu().numpy())
        attr_ens_ig.append(integrated_gradients_attrib(head, z_all, y_ens, steps=ig_steps).cpu().numpy())
        attr_ens_occ.append(occlusion_attrib(head, z_all, y_ens).cpu().numpy())

    attr_ens_gxi = np.mean(np.stack(attr_ens_gxi, axis=0), axis=0)
    attr_ens_ig  = np.mean(np.stack(attr_ens_ig, axis=0), axis=0)
    attr_ens_occ = np.mean(np.stack(attr_ens_occ, axis=0), axis=0)

    plot_attribution_bundle(out_dir, f"ens_gradxinput", attr_ens_gxi, topk=topk)
    plot_attribution_bundle(out_dir, f"ens_intgrad", attr_ens_ig, topk=topk)
    plot_attribution_bundle(out_dir, f"ens_occlusion", attr_ens_occ, topk=topk)

    ens_ig_mean_abs = np.mean(np.abs(attr_ens_ig), axis=0)

    curves = {"ens": ens_ig_mean_abs}
    for hi in range(H):
        curves[f"head{hi}"] = head_ig_mean_abs[hi]
    # 原本 overlay 圖：可以保留，但建議只放 appendix

    plot_ig_mean_abs_overlay(out_dir, curves)
    # Thesis 主文建議放這張：三個 heads 對 bottleneck dimensions 的 attribution heatmap

    head_only_curves = {
        f"head{hi}": head_ig_mean_abs[hi]
        for hi in range(H)
    }

    plot_head_attribution_heatmap(
        out_dir=out_dir,
        head_curves_dict=head_only_curves,
        topk=topk
    )

    print("[ATTR] Done. Output dir =", out_dir)

def plot_ig_mean_abs_overlay(out_dir, curves_dict):
    """
    curves_dict: {"ens": [D], "head0": [D], "head1": [D], "head2": [D]}
    全部疊在同一張 x/y 軸圖上
    """
    _ensure_dir(out_dir)
    plt.figure()
    for name, y in curves_dict.items():
        plt.plot(y, label=name)
    plt.xlabel("z dimension")
    plt.ylabel("mean |IG attribution|")
    plt.title("Integrated Gradients: mean |attr| overlay")
    plt.legend()
    fp = os.path.join(out_dir, "ig_mean_abs_overlay.png")
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print("[ATTR saved]", fp)

def plot_head_attribution_heatmap(out_dir, head_curves_dict, topk=20):
    """
    Thesis-ready multi-head attribution heatmap

    head_curves_dict:
      {
        "head0": mean_abs_attr [D],
        "head1": mean_abs_attr [D],
        "head2": mean_abs_attr [D],
      }

    圖的概念：
    - x-axis: union of top-k latent dims from all heads
    - y-axis: predictor heads
    - color: normalized attribution strength
    """

    _ensure_dir(out_dir)

    head_names = list(head_curves_dict.keys())
    curves = [head_curves_dict[h] for h in head_names]

    # =========================
    # Step 1: 取各 head top-k union
    # =========================
    union_dims = set()
    for c in curves:
        top_idx = np.argsort(-c)[:topk]
        union_dims.update(top_idx.tolist())

    union_dims = list(union_dims)

    # =========================
    # Step 2: 用 mean importance 排序（回到原本版本）
    # shared + complementary 比較自然
    # =========================
    mean_importance = np.mean(
        np.stack([c[union_dims] for c in curves], axis=0),
        axis=0
    )

    sorted_order = np.argsort(-mean_importance)
    union_dims = [union_dims[i] for i in sorted_order]

    # =========================
    # Step 3: 建立 heatmap matrix
    # [num_heads, num_union_dims]
    # =========================
    mat = np.stack(
        [c[union_dims] for c in curves],
        axis=0
    )

    # =========================
    # Step 4: row-wise normalization
    # 每個 head 比自己內部相對重要性
    # =========================
    mat_norm = mat / (mat.max(axis=1, keepdims=True) + 1e-12)

    # =========================
    # Step 5: plot
    # =========================
    plt.figure(
        figsize=(max(8, len(union_dims) * 0.35), 2.8)
    )

    im = plt.imshow(
        mat_norm,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.yticks(
        np.arange(len(head_names)),
        head_names
    )

    plt.xticks(
        np.arange(len(union_dims)),
        [str(d) for d in union_dims],
        rotation=45
    )

    plt.xlabel("Bottleneck latent dimensions")
    plt.ylabel("Predictor heads")

    cbar = plt.colorbar(im)
    cbar.set_label("normalized attribution strength")

    fp = os.path.join(
        out_dir,
        f"multihead_top{topk}_attribution_heatmap.png"
    )

    plt.savefig(
        fp,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("[ATTR saved]", fp)

    # =========================
    # Extra: overlap analysis
    # =========================
    print("==== Top-k overlap analysis ====")

    top_sets = {}
    for h, c in head_curves_dict.items():
        top_sets[h] = set(
            np.argsort(-c)[:topk].tolist()
        )

    for i in range(len(head_names)):
        for j in range(i + 1, len(head_names)):
            hi = head_names[i]
            hj = head_names[j]

            inter = len(
                top_sets[hi] & top_sets[hj]
            )

            union = len(
                top_sets[hi] | top_sets[hj]
            )

            jaccard = (
                inter / union
                if union > 0 else 0.0
            )

            print(
                f"{hi} vs {hj}: "
                f"overlap={inter}/{topk}, "
                f"Jaccard={jaccard:.3f}"
            )
# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_path", type=str, required=True, help="source domain train csv (有 label)")
    parser.add_argument("--target_train_path", type=str, required=True, help="target domain train csv (unlabeled)")
    parser.add_argument("--test_path",  type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, default=r"rp_id.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--out_dir", type=str, default="./3head_result")
    parser.add_argument("--column", type=int, default=256)

    #parser.add_argument("--pred_hidden", type=int, nargs="+", default=[256, 256], help="hidden sizes for predictor MLP")
    parser.add_argument("--recon_hidden", type=int, nargs="+", default=[256], help="hidden sizes for reconstruction MLP")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)

    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--lambda_recon", type=float, default=0.1)

    parser.add_argument("--lambda_consist", type=float, default=0.07, help="weight for consistency loss")
    parser.add_argument("--noise_std", type=float, default=0.1, help="std for gaussian noise jitter")
    parser.add_argument("--drop_prob", type=float, default=0.3, help="probability to drop AP signal")

    parser.add_argument("--lambda_entropy", type=float, default=0.05, help="weight for entropy minimization loss")
    parser.add_argument("--use_mask", action="store_true", help="啟用 key_padding_mask")

    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    parser.add_argument("--vit_tokens", type=int, default=256,
                        help="token 數（必須能 sqrt 成整數，且 side 能被 patch_size=8 整除）")

        # ---- Attribution args (NEW) ----
    parser.add_argument("--attr_out_dir", type=str, default=None,
                        help="輸出 attribution 圖的資料夾；若 None 則用 out_dir/attribution")
    parser.add_argument("--attr_max_batches", type=int, default=2,
                        help="用多少個 batch 做 attribution（越大越慢）")
    parser.add_argument("--attr_topk", type=int, default=20,
                        help="熱點圖/柱狀圖顯示 top-k 維度")
    parser.add_argument("--attr_ig_steps", type=int, default=50,
                        help="Integrated Gradients steps（越大越穩但越慢）")
    
    parser.add_argument("--model_dir", type=str, default="./models", help="儲存訓練完成模型的資料夾")
    parser.add_argument("--model_name", type=str, default="TransJCR.pth", help="模型檔名")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # --- Load data ---
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te  = load_csvs(args.test_path)

    ap_cols = get_feature_cols(df_src, args.column)
    assert set(ap_cols).issubset(df_tgt.columns), "Target train 缺少部分 AP 欄位"
    assert set(ap_cols).issubset(df_te.columns),  "Test 缺少部分 AP 欄位"

    n_classes_src = df_src["rp_id"].nunique()
    if n_classes_src != 48:
        print(f"[WARN] Source 訓練集 rp 類別數={n_classes_src}（預期 48）")

    mins, maxs = fit_scaler(df_src[ap_cols].values.astype(np.float32), missing_val=args.missing_val)

    ds_src = RSSISourceDataset(df_src, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    id2idx = ds_src.id2idx
    idx2id = ds_src.idx2id
    dl_src = DataLoader(ds_src, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    ds_tgt = RSSITargetDataset(df_tgt, ap_cols, mins=mins, maxs=maxs, missing_val=args.missing_val)
    dl_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # --- Prepare test arrays ---
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
    vit_tokens = args.vit_tokens

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
        #pred_hidden=args.pred_hidden,
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
    print(f"recon_hidden={args.recon_hidden}")
    print("Predictors: 3 heads (shallow / deep / residual)")
    print(f"Consistency: λ={args.lambda_consist}, noise={args.noise_std}, drop={args.drop_prob}")
    print(f"Entropy(ensemble): λ={args.lambda_entropy}")
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

        tgt_iter = cycle(dl_tgt)

        for xb_s, yb_s in dl_src:
            keep = (yb_s != -1)
            if not keep.any():
                continue
            xb_s, yb_s = xb_s[keep], yb_s[keep]

            xb_s = xb_s.to(device, non_blocking=True)
            yb_s = yb_s.to(device, non_blocking=True)

            xb_t = next(tgt_iter).to(device, non_blocking=True)

            # --- Augmentation (Target) ---
            with torch.no_grad():
                xb_t_aug = augment_rssi(xb_t, noise_std=args.noise_std, drop_prob=args.drop_prob, missing_val=-1.0)

            opt.zero_grad()

            # 1) Source Flow: CE (3-head avg) + Recon
            logits_s_list, recon_s = model(xb_s)
            ce_each = [crit(lg, yb_s) for lg in logits_s_list]
            ce_loss = torch.stack(ce_each).mean()

            x_s_norm = xb_s.squeeze(1)
            recon_loss_s = reconstruction_loss(recon_s, x_s_norm, mask_value=-1.0)

            # 2) Target Flow (Original): Recon + ensemble entropy
            logits_t_list, recon_t = model(xb_t)
            x_t_norm = xb_t.squeeze(1)
            recon_loss_t = reconstruction_loss(recon_t, x_t_norm, mask_value=-1.0)

            p_ens_t = ensemble_probs_from_logits_list(logits_t_list)
            loss_ent = entropy_from_probs(p_ens_t)

            # 3) Target Flow (Augmented): predictor-only (3 heads)
            logits_t_aug_list = model.forward_predictors_only(xb_t_aug)

            # 4) Consistency (head-wise KL avg)
            kl_each = [consistency_loss(logits_t_aug_list[i], logits_t_list[i]) for i in range(len(logits_t_list))]
            loss_consist = torch.stack(kl_each).mean()

            recon_loss_all = recon_loss_s + recon_loss_t
            total_loss = ce_loss \
                         + args.lambda_recon * recon_loss_all \
                         + args.lambda_consist * loss_consist \
                         + args.lambda_entropy * loss_ent

            total_loss.backward()
            opt.step()

            # ---- train acc: ensemble argmax ----
            with torch.no_grad():
                p_ens_s = ensemble_probs_from_logits_list(logits_s_list)
                pred_s = torch.argmax(p_ens_s, dim=1)

            bs = yb_s.size(0)
            tr_loss_total += total_loss.item() * bs
            tr_ce_total += ce_loss.item() * bs
            tr_recon_s_total += recon_loss_s.item() * bs
            tr_recon_t_total += recon_loss_t.item() * bs
            tr_consist_total += loss_consist.item() * bs
            tr_ent_total += loss_ent.item() * bs
            tr_acc += (pred_s == yb_s).float().sum().item()
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
              f"Cst {tr_consist_total:.3f} Ent {tr_ent_total:.3f} | Acc(ens) {tr_acc:.3f}")

    # --- Save trained model ---
    model_save_path = os.path.join(args.model_dir, args.model_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "id2idx": id2idx,
        "idx2id": idx2id,
        "ap_cols": ap_cols,
        "mins": mins,
        "maxs": maxs,
        "args": vars(args),
    }, model_save_path)

    print(f"==== Model saved to: {model_save_path} ====")
    # --- Final Evaluation on Test (ensemble) + collect probs for visualization ---
    model.eval()
    preds_idx, gts_idx, gts_rpid = [], [], []
    head_probs_all = None
    ens_probs_all = []

    with torch.no_grad():
        for xb, yb_idx, yb_rpid in dl_te:
            xb = xb.to(device)
            logits_list, _ = model(xb)

            # head probs
            ps = [F.softmax(lg, dim=1).cpu().numpy() for lg in logits_list]
            p_ens = ensemble_probs_from_logits_list(logits_list).cpu().numpy()

            if head_probs_all is None:
                head_probs_all = [[] for _ in range(len(ps))]
            for i in range(len(ps)):
                head_probs_all[i].append(ps[i])
            ens_probs_all.append(p_ens)

            pred = np.argmax(p_ens, axis=1)
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

    # --- MDE ---
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
    print(f"Test Accuracy (ensemble)    : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)" if not np.isnan(avg_mde) else "Mean Distance Error (same floor only): N/A")
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")

    # --- Save 3 plots ---
    if head_probs_all is not None:
        head_probs_np = [np.concatenate(head_probs_all[i], axis=0) for i in range(len(head_probs_all))]
        ens_probs_np = np.concatenate(ens_probs_all, axis=0)
        save_three_head_plots(args.out_dir, head_probs_np, ens_probs_np, gt_np=gts_idx)
        # --- Run attribution on z (NEW) ---
        attr_dir = args.attr_out_dir if args.attr_out_dir is not None else os.path.join(args.out_dir, "attribution")
        # 你要看「test」就用 dl_te；如果你要看「target clean」就改成 dl_tgt
        run_z_attribution(
            model=model,
            dl=dl_te,  # 或 dl_tgt（注意 dl_tgt 只有 xb）
            device=device,
            out_dir=attr_dir,
            max_batches=args.attr_max_batches,
            topk=args.attr_topk,
            ig_steps=args.attr_ig_steps,
        )

if __name__ == "__main__":
    main()