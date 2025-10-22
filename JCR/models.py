import torch.nn as nn

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
                 cls_hidden=[256], rec_hidden=[256,256], dropout=0.4):
        super().__init__()
        self.extractor = MLP(in_dim, ext_hidden + [feat_dim], None, dropout)
        self.cls_head = MLP(feat_dim, cls_hidden, n_classes, dropout)
        self.rec_head = MLP(feat_dim, rec_hidden, in_dim, dropout)
    def forward(self, x):
        f = self.extractor(x)
        return self.cls_head(f), self.rec_head(f), f
