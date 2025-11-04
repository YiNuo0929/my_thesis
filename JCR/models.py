import torch
import torch.nn as nn
from torch import Tensor
import math

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=None, dropout=0.4, last_activation=False):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(dropout)]
            last = h
        if out_dim is not None:
            layers += [nn.Linear(last, out_dim)]
            if last_activation: layers += [nn.ReLU(True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class FullModel(nn.Module):
    def __init__(self, in_dim, n_classes, feat_dim=256, ext_hidden=[512,512],
                 cls_hidden=[256], rec_hidden=[128], dropout=0.4):
        super().__init__()
        self.extractor = MLP(in_dim, ext_hidden + [feat_dim], None, dropout)
        self.cls_head = MLP(feat_dim, cls_hidden, n_classes, dropout)
        self.rec_head = MLP(feat_dim, rec_hidden, in_dim, dropout)
    def forward(self, x):
        f = self.extractor(x)
        return self.cls_head(f), self.rec_head(f), f
'''
class TransformerExtractor(nn.Module):
    """
    一維 RSSI Fingerprint 的 Transformer-Encoder 特徵擷取器（無輸入正規化）。
    - 輸入:  x ∈ R^{B, N_AP}，每個 AP 值 = 一個 token
    - 遮罩:  use_mask=True 時，(x == mask_value) 會當成 key_padding_mask
    - 結構:  token Linear(1->d_model) -> (可選) [CLS] + **固定 sinusoidal pos emb**
           -> TransformerEncoder -> 池化 -> 線性投影到 feat_dim
    - 輸出:  feat ∈ R^{B, feat_dim}
    """
    def __init__(
        self,
        num_tokens: int,       # N_AP (= in_dim)
        feat_dim: int,         # 對齊你 FullModel 預設 720 或 512 等
        d_model: int,          # Transformer hidden dim
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool,   # 用 [CLS] 匯聚；關掉用 mean pooling
        use_mask: bool,        # 是否啟用缺失值 masking（預設關閉）
        mask_value: float      # 缺失值（你的 Dataset 預設 0.0 或其他）
    ):
        super().__init__()
        self.num_tokens   = num_tokens
        self.feat_dim     = feat_dim
        self.d_model      = d_model
        self.use_cls_token = use_cls_token
        self.use_mask     = use_mask
        self.mask_value   = mask_value

        # 每 token（單一 RSSI scalar）嵌入到 d_model
        self.token_embed = nn.Linear(1, d_model)

        # (可選) [CLS] token
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None

        # ===== 固定 sinusoidal 位置編碼（含 CLS 位置）=====
        n_pos = num_tokens + (1 if use_cls_token else 0)   # token 數 + CLS（若有）
        pe = torch.zeros(n_pos, d_model)                   # [n_pos, d_model]
        position = torch.arange(0, n_pos, dtype=torch.float32).unsqueeze(1)  # [n_pos, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                               # [1, n_pos, d_model]
        # 存成 buffer，不參與訓練
        self.register_buffer("pos_embed", pe)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, feat_dim)

    @staticmethod
    def _masked_mean(x: Tensor, key_padding_mask: Tensor) -> Tensor:
        """
        x: (B, T, D)
        key_padding_mask: (B, T) True=忽略
        """
        valid = (~key_padding_mask).float()                       # 1 for valid
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)     # (B,1)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom       # (B,D)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, N_AP)
        return: (B, feat_dim)
        """
        B, N = x.shape
        assert N == self.num_tokens, f"num_tokens mismatch: got {N}, expected {self.num_tokens}"

        # 缺失值遮罩（可關閉）
        key_padding_mask = (x == self.mask_value) if self.use_mask else None

        # (B,N) -> (B,N,1) -> (B,N,d_model)
        tok = self.token_embed(x.unsqueeze(-1))   # (B,N,D)

        # 加 [CLS] 與位置嵌入（sinusoidal, fixed）
        if self.use_cls_token:
            cls = self.cls_embed.expand(B, 1, self.d_model)   # (B,1,D)
            tok = torch.cat([cls, tok], dim=1)                # (B,N+1,D)
            pos = self.pos_embed[:, : tok.size(1), :]         # (1,N+1,D)
        else:
            pos = self.pos_embed[:, : tok.size(1), :]         # (1,N,D)

        tok = tok + pos                                       # (B,T,D)

        # 準備 key padding mask（若含 CLS，最前面補 False）
        src_key_padding_mask = None
        if key_padding_mask is not None:
            if self.use_cls_token:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # (B,N+1)
            else:
                src_key_padding_mask = key_padding_mask       # (B,N)

        # Encoder
        enc = self.encoder(tok, src_key_padding_mask=src_key_padding_mask)  # (B,T,D)

        # 匯聚
        if self.use_cls_token:
            pooled = enc[:, 0, :]  # (B,D)
        else:
            pooled = enc.mean(dim=1) if src_key_padding_mask is None else self._masked_mean(enc, src_key_padding_mask)

        feat = self.proj(self.out_norm(pooled))  # (B, feat_dim)
        return feat
'''
class TransformerExtractor(nn.Module):
    """
    一維 RSSI Fingerprint 的 Transformer-Encoder 特徵擷取器（無輸入正規化）。
    - 輸入:  x ∈ R^{B, N_AP}，每個 AP 值 = 一個 token
    - 遮罩:  use_mask=True 時，(x == mask_value) 會當成 key_padding_mask
    - 結構:  token Linear(1->d_model) -> (可選) [CLS] + 固定 sinusoidal pos emb
           -> TransformerEncoder -> 取 CLS / mean
           -> bottleneck MLP: d_model -> ... -> feat_dim (= z_dim)
    - 輸出:  z ∈ R^{B, feat_dim}  （明確當作 latent code 使用）
    """
    def __init__(
        self,
        num_tokens: int,       # N_AP (= in_dim)
        feat_dim: int,         # 這裡就當作 z_dim 使用
        d_model: int,          # Transformer hidden dim
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_cls_token: bool,   # 用 [CLS] 匯聚；關掉用 mean pooling
        use_mask: bool,        # 是否啟用缺失值 masking（預設關閉）
        mask_value: float,     # 缺失值（你的 Dataset 預設 0.0 或其他）
        bottleneck_hidden=None,      # 新增：bottleneck MLP 的 hidden 層，例如 [128]
        bottleneck_dropout: float = 0.4,  # 新增：bottleneck MLP dropout
    ):
        super().__init__()
        self.num_tokens    = num_tokens
        self.feat_dim      = feat_dim     # 這個就是 z_dim
        self.d_model       = d_model
        self.use_cls_token = use_cls_token
        self.use_mask      = use_mask
        self.mask_value    = mask_value

        if bottleneck_hidden is None:
            # 預設等價於原本的 Linear(d_model -> feat_dim)
            bottleneck_hidden = []

        # 每 token（單一 RSSI scalar）嵌入到 d_model
        self.token_embed = nn.Linear(1, d_model)

        # (可選) [CLS] token
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None

        # ===== 固定 sinusoidal 位置編碼（含 CLS 位置）=====
        n_pos = num_tokens + (1 if use_cls_token else 0)   # token 數 + CLS（若有）
        pe = torch.zeros(n_pos, d_model)                   # [n_pos, d_model]
        position = torch.arange(0, n_pos, dtype=torch.float32).unsqueeze(1)  # [n_pos, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                               # [1, n_pos, d_model]
        # 存成 buffer，不參與訓練
        self.register_buffer("pos_embed", pe)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(d_model)

        # 🔴 新增：bottleneck MLP，把 CLS(d_model) 壓到 feat_dim (= z_dim)
        # 使用你原本的 MLP 結構：Linear+BN+ReLU+Dropout...
        self.bottleneck = MLP(
            in_dim=d_model,
            hidden=bottleneck_hidden,
            out_dim=feat_dim,
            dropout=bottleneck_dropout,
            last_activation=False
        )

    @staticmethod
    def _masked_mean(x: Tensor, key_padding_mask: Tensor) -> Tensor:
        """
        x: (B, T, D)
        key_padding_mask: (B, T) True=忽略
        """
        valid = (~key_padding_mask).float()                       # 1 for valid
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)     # (B,1)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom       # (B,D)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, N_AP)
        return: (B, feat_dim) = (B, z_dim)
        """
        B, N = x.shape
        assert N == self.num_tokens, f"num_tokens mismatch: got {N}, expected {self.num_tokens}"

        # 缺失值遮罩（可關閉）
        key_padding_mask = (x == self.mask_value) if self.use_mask else None

        # (B,N) -> (B,N,1) -> (B,N,d_model)
        tok = self.token_embed(x.unsqueeze(-1))   # (B,N,D)

        # 加 [CLS] 與位置嵌入（sinusoidal, fixed）
        if self.use_cls_token:
            cls = self.cls_embed.expand(B, 1, self.d_model)   # (B,1,D)
            tok = torch.cat([cls, tok], dim=1)                # (B,N+1,D)
            pos = self.pos_embed[:, : tok.size(1), :]         # (1,N+1,D)
        else:
            pos = self.pos_embed[:, : tok.size(1), :]         # (1,N,D)

        tok = tok + pos                                       # (B,T,D)

        # 準備 key padding mask（若含 CLS，最前面補 False）
        src_key_padding_mask = None
        if key_padding_mask is not None:
            if self.use_cls_token:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # (B,N+1)
            else:
                src_key_padding_mask = key_padding_mask       # (B,N)

        # Encoder
        enc = self.encoder(tok, src_key_padding_mask=src_key_padding_mask)  # (B,T,D)

        # 匯聚成 CLS 或 mean
        if self.use_cls_token:
            pooled = enc[:, 0, :]  # (B,D)
        else:
            pooled = enc.mean(dim=1) if src_key_padding_mask is None else self._masked_mean(enc, src_key_padding_mask)

        # LayerNorm + bottleneck MLP → z
        pooled = self.out_norm(pooled)         # (B,D)
        z = self.bottleneck(pooled)            # (B, feat_dim) = 你的 latent code

        return z

class trans_FullModel(nn.Module):
    """
    使用 TransformerExtractor 作為特徵擷取器的完整模型。
    - 輸入:  x ∈ R^{B, in_dim}  (一維 RSSI，each AP = 1 token)
    - extractor: TransformerExtractor（支援 use_mask, [CLS]/mean 匯聚）
    - head: 與原本一致，皆吃 feat_dim
    """
    def __init__(self, in_dim, n_classes, feat_dim = 32, cls_hidden = [256,256], rec_hidden = [128], dropout: float = 0.4,
                 # Transformer 超參
                 d_model= 128, nhead = 4, num_layers = 2, dim_feedforward = 256, attn_dropout: float = 0.4,
                 use_cls_token: bool = True,   # True: 取 CLS；False: mean pooling
                 use_mask: bool = True,       # True: 對 (x == mask_value) 做 key_padding_mask
                 mask_value: float = 0.0):
        super().__init__()

        # Transformer-based extractor
        '''
        self.extractor = TransformerExtractor(
            num_tokens=in_dim,
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            use_cls_token=use_cls_token,
            use_mask=use_mask,
            mask_value=mask_value
        )
        '''
        self.extractor = TransformerExtractor(
        num_tokens=in_dim,
        feat_dim=feat_dim,       # 這個就當 z_dim 用，例如 32 / 64
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=attn_dropout,
        use_cls_token=use_cls_token,
        use_mask=use_mask,
        mask_value=mask_value,
        bottleneck_hidden=[128],       # 🔴 新增：明確有一層 hidden
        bottleneck_dropout=0.4
)

        # 與原版一致的分類與重建頭
        self.cls_head = MLP(feat_dim, cls_hidden, n_classes, dropout)
        self.rec_head = MLP(feat_dim, rec_hidden, in_dim, dropout)

    def forward(self, x: torch.Tensor):
        f = self.extractor(x)     # (B, feat_dim)
        logits = self.cls_head(f) # (B, n_classes)
        recon  = self.rec_head(f) # (B, in_dim)
        return logits, recon, f