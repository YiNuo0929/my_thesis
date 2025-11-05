# models.py
import math
import torch
import torch.nn as nn
from torch import Tensor


# ============================================================
# 基本小積木
# ============================================================

class MLPBlock(nn.Module):
    """Linear + BN + ReLU + Dropout"""
    def __init__(self, in_dim: int, out_dim: int, p_drop: float):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class PositionalEncoding(nn.Module):
    """標準 sin/cos 位置編碼，batch_first=True 對應 [B, L, d_model]"""
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


# ============================================================
# Transformer-based Extractor（不帶預設超參數）
# ============================================================

class TransformerExtractor(nn.Module):
    """
    將每個 AP 當成 token，經 TransformerEncoder 抽 bottleneck 特徵：
    input:  [B, 1, L] 或 [B, L]，L = AP 數量
    output: z ∈ [B, z_dim]
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
        bottleneck_hidden: int,
        use_mask: bool,
        mask_value: float,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.use_mask = use_mask
        self.mask_value = mask_value
        self.d_model = d_model
        self.z_dim = z_dim

        # RSSI scalar → d_model
        self.input_proj = nn.Linear(1, d_model)

        # CLS token
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

        # bottleneck MLP: d_model -> bottleneck_hidden -> z_dim
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_hidden, z_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # [B, 1, L] -> [B, L]
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        B, L = x.shape

        key_padding_mask = None
        if self.use_mask:
            key_padding_mask = (x == self.mask_value)  # [B, L]

        # [B, L] -> [B, L, 1] -> [B, L, d_model]
        x = x.unsqueeze(-1)
        h = self.input_proj(x)

        # CLS token
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            h = torch.cat([cls, h], dim=1)  # [B, 1+L, d_model]

        # 位置編碼
        h = self.pos_encoding(h)

        # mask 對齊
        src_key_padding_mask = None
        if key_padding_mask is not None:
            if self.use_cls_token:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
                src_key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # [B, 1+L]
            else:
                src_key_padding_mask = key_padding_mask                          # [B, L]

        # Transformer encoder
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # 取 CLS 或 mean
        if self.use_cls_token:
            cls_feat = h[:, 0, :]
        else:
            cls_feat = h.mean(dim=1)

        z = self.bottleneck(cls_feat)
        return z


# ============================================================
# Classifier / Reconstruction Head（不帶預設超參數）
# ============================================================

class ClassifierHead(nn.Module):
    """f (feat_dim) → logits (n_classes)"""
    def __init__(self, in_dim: int, n_classes: int, hidden: list, dropout: float):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(MLPBlock(last, h, p_drop=dropout))
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ReconstructionHead(nn.Module):
    """f (feat_dim) → recon (in_dim)"""
    def __init__(self, in_dim: int, out_dim: int, hidden: list, dropout: float):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(MLPBlock(last, h, p_drop=dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================
# trans_FullModel: 統一設定所有超參數的地方
# ============================================================

class trans_FullModel(nn.Module):
    """
    Transformer-based RSSI 室內定位模型：
    - 統一在這裡設定所有超參數
    - 內部再把參數丟給 TransformerExtractor / ClassifierHead / ReconstructionHead
    forward 回傳：
        logits: [B, n_classes]
        recon : [B, in_dim]
        f     : [B, feat_dim]
    """
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        # Transformer 超參數
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        # bottleneck feature 維度
        feat_dim: int = 32,
        # head 結構
        cls_hidden: list = None,
        rec_hidden: list = None,
        # mask 設定
        use_mask: bool = True,
        mask_value: float = 0.0,
    ):
        super().__init__()

        # -------- 在這裡統一處理預設值 --------
        if cls_hidden is None:
            cls_hidden = [256]
        if rec_hidden is None:
            rec_hidden = [256]

        bottleneck_hidden = d_model  # 你要改成別的也可以在這裡改

        # ----- Transformer-based extractor -----
        self.extractor = TransformerExtractor(
            num_tokens=in_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_cls_token=True,
            z_dim=feat_dim,
            bottleneck_hidden=bottleneck_hidden,
            use_mask=use_mask,
            mask_value=mask_value,
        )

        # ----- Classifier head -----
        self.cls_head = ClassifierHead(
            in_dim=feat_dim,
            n_classes=n_classes,
            hidden=cls_hidden,
            dropout=dropout,
        )

        # ----- Reconstruction head -----
        self.rec_head = ReconstructionHead(
            in_dim=feat_dim,
            out_dim=in_dim,
            hidden=rec_hidden,
            dropout=dropout,
        )

    def forward(self, x: Tensor):
        """
        x: [B, 1, in_dim] 或 [B, in_dim]
        """
        f = self.extractor(x)       # [B, feat_dim]
        logits = self.cls_head(f)   # [B, n_classes]
        recon = self.rec_head(f)    # [B, in_dim]
        return logits, recon, f


# ============================================================
# FullModel stub（避免 import 失敗）
# ============================================================

class FullModel(nn.Module):
    """占位用 stub，之後如果有 DNN-based 版再自己實作替換掉。"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("FullModel 尚未在這份 models.py 中實作，請使用 trans_FullModel")

    def forward(self, x: Tensor):
        raise NotImplementedError