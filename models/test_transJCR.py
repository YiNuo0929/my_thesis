import os
import math
import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


def apply_scaler(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray, missing_val: float = -110.0):
    x = x.copy().astype(np.float32)
    miss_mask = (x == missing_val)

    denom = (maxs - mins)
    denom[denom == 0.0] = 1.0
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)
    x[miss_mask] = -1.0
    return x


def load_rp_map(rp_map_path: str):
    df = pd.read_csv(rp_map_path, encoding="utf-8-sig")
    need = {"rp_id", "x", "y", "floor"}
    if not need.issubset(df.columns):
        raise ValueError(f"rp_id.csv 缺少欄位，需要：{need}")
    mp = {}
    for _, r in df.iterrows():
        mp[int(r["rp_id"])] = (float(r["x"]), float(r["y"]), int(r["floor"]))
    return mp


def ensemble_probs_from_logits_list(logits_list):
    ps = [F.softmax(lg, dim=1) for lg in logits_list]
    return torch.stack(ps, dim=0).mean(dim=0)


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
    def __init__(self, in_dim: int, n_classes: int, hidden_dim=256, num_blocks=4, p_drop=0.2):
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


class ReconstructionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=[256], p_drop=0.2):
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

        head0 = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=[256, 256],
            p_drop=p_drop
        )

        head1 = PredictorMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden=[512, 256, 256],
            p_drop=p_drop
        )

        head2 = PredictorResidualMLP(
            in_dim=z_dim,
            n_classes=n_classes,
            hidden_dim=256,
            num_blocks=4,
            p_drop=p_drop
        )

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

        logits_list = [head(z) for head in self.predictors]
        recon = self.reconstructor(z)
        return logits_list, recon


# ---------- Test Dataset ----------
class TestDataset(Dataset):
    def __init__(self, df_te, ap_cols, mins, maxs, id2idx, missing_val=-110.0):
        X_te_full_raw = df_te[ap_cols].values.astype(np.float32)
        X_te_full = apply_scaler(X_te_full_raw, mins, maxs, missing_val)
        X_te_full = np.expand_dims(X_te_full, axis=1)

        if "rp_id" in df_te.columns:
            y_te_raw = df_te["rp_id"].astype(int).values
        else:
            y_te_raw = np.full(len(df_te), -1, dtype=int)

        y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], dtype=np.int64)

        self.X = X_te_full
        self.y_idx = y_te_idx
        self.y_raw = y_te_raw

    def __len__(self):
        return len(self.y_idx)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y_idx[i], dtype=torch.long),
            int(self.y_raw[i])
        )


# ---------- Main ----------
def main():
    model_path = "./models/TransJCR.pth"
    test_path = "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-06-11/test.csv"
    rp_map_path = "./rp_id_um.csv"
    batch_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 載入 checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    id2idx = ckpt["id2idx"]
    idx2id = ckpt["idx2id"]
    ap_cols = ckpt["ap_cols"]
    mins = np.array(ckpt["mins"], dtype=np.float32)
    maxs = np.array(ckpt["maxs"], dtype=np.float32)
    saved_args = ckpt["args"]

    # 2) 建立 model
    model = TransClassifier(
        input_dim=len(ap_cols),
        vit_tokens=saved_args["vit_tokens"],
        n_classes=len(id2idx),
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_layers=saved_args["num_layers"],
        dim_feedforward=saved_args["dim_feedforward"],
        dropout=saved_args["dropout"],
        z_dim=saved_args["z_dim"],
        recon_hidden=saved_args["recon_hidden"],
        p_drop=saved_args["dropout"],
        use_mask=saved_args["use_mask"],
        mask_value=-1.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3) 載入 test
    df_te = load_csvs(test_path)

    # 檢查 test 是否有需要的 AP 欄位
    missing_cols = [c for c in ap_cols if c not in df_te.columns]
    if missing_cols:
        raise ValueError(f"Test 缺少以下 AP 欄位：{missing_cols[:10]}")

    ds_te = TestDataset(
        df_te=df_te,
        ap_cols=ap_cols,
        mins=mins,
        maxs=maxs,
        id2idx=id2idx,
        missing_val=saved_args.get("missing_val", -110.0)
    )
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 4) 測試
    preds_idx, gts_idx, gts_rpid = [], [], []

    with torch.no_grad():
        for xb, yb_idx, yb_rpid in dl_te:
            xb = xb.to(device)
            logits_list, _ = model(xb)
            p_ens = ensemble_probs_from_logits_list(logits_list).cpu().numpy()

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

    # 5) 計算 MDE
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
            d = sqrt((gx - px) ** 2 + (gy - py) ** 2)
            mde_distances.append(d)

    avg_mde = float(np.mean(mde_distances)) if len(mde_distances) > 0 else float("nan")

    # 6) 輸出格式和原本一樣
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
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")


if __name__ == "__main__":
    main()