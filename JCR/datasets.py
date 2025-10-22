import numpy as np, torch
from torch.utils.data import Dataset

class RSSISourceDataset(Dataset):
    def __init__(self, df, ap_cols, label_col="rp_id", missing_val=0.0):
        # 已在外部 apply_scaler
        self.X = df[ap_cols].values.astype(np.float32)
        self.miss = (df[ap_cols].values == missing_val).astype(np.float32)

        # Label 轉成 index
        y_raw = df[label_col].values.astype(np.int64)
        uniq = np.sort(np.unique(y_raw[y_raw != -1]))
        self.id2idx = {rid: i for i, rid in enumerate(uniq)}
        self.idx2id = {i: rid for rid, i in self.id2idx.items()}
        self.y = np.array([self.id2idx.get(int(r), -1) for r in y_raw], np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i]),
            torch.from_numpy(self.miss[i]),
        )


class RSSITargetDataset(Dataset):
    def __init__(self, df, ap_cols, missing_val=0.0):
        self.X = df[ap_cols].values.astype(np.float32)
        self.miss = (df[ap_cols].values == missing_val).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(-1),  # target domain 無標籤
            torch.from_numpy(self.miss[i]),
        )


class TestDataset(Dataset):
    def __init__(self, X, y, m):
        self.X, self.y, self.m = X, y, m

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i]),
            torch.from_numpy(self.m[i]),
        )
