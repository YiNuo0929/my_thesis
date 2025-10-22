import numpy as np
#做標準化的
'''
def fit_scaler(train_ap: np.ndarray, missing_val=-110.0):
    mask = train_ap != missing_val
    means = np.zeros(train_ap.shape[1], np.float32)
    stds = np.ones(train_ap.shape[1], np.float32)
    for j in range(train_ap.shape[1]):
        col, m = train_ap[:, j], mask[:, j]
        if m.any():
            mu, sigma = col[m].mean(), col[m].std()
            if sigma < 1e-6: sigma = 1.0
        else:
            mu, sigma = -100.0, 10.0
        means[j], stds[j] = mu, sigma
    return means, stds

def apply_scaler(x, means, stds, missing_val=-110.0):
    x = x.copy()
    miss = (x == missing_val)
    if miss.any():
        ap_idx = np.where(miss)[1]
        x[miss] = means[ap_idx]
    return (x - means) / stds
'''

#做正規化
def fit_scaler(train_ap: np.ndarray, missing_val: float = -110.0):
    """
    為單一 domain（source 或 target）計算每個 AP 的 min、max。
    缺失值（missing_val）不參與統計。
    """
    mask = train_ap != missing_val
    n_ap = train_ap.shape[1]
    mins = np.zeros(n_ap, dtype=np.float32)
    maxs = np.zeros(n_ap, dtype=np.float32)

    for j in range(n_ap):
        col, m = train_ap[:, j], mask[:, j]
        if m.any():
            vmin, vmax = col[m].min(), col[m].max()
            # 避免 max == min 造成除以 0
            if abs(vmax - vmin) < 1e-6:
                vmin, vmax = vmin - 1.0, vmin + 1.0
        else:
            # 若該 AP 全缺，給合理的預設值（論文中未特別定義，這裡保持一致）
            vmin, vmax = -110.0, -30.0
        mins[j], maxs[j] = vmin, vmax

    return mins, maxs


def apply_scaler(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray, missing_val: float = -110.0):
    """
    依 HistLoc 設計進行 Min–Max normalization，將 RSSI 縮放至 0~1。
    缺失值補 0（相當於最小值），並忽略新出現但 source 未見的 AP。
    """
    x = x.copy()
    miss_mask = (x == missing_val)

    # Min–Max normalization
    denom = np.clip(maxs - mins, 1e-6, None)
    x = (x - mins) / denom

    # 缺失補 0
    x[miss_mask] = 0.0

    # clip 以防超出 [0, 1] 範圍
    x = np.clip(x, 0.0, 1.0)
    return x