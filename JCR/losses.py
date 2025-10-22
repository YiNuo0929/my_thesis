import torch.nn.functional as F

def masked_mse(pred, target, mask_missing, ignore_missing=True):
    if ignore_missing:
        valid = (mask_missing == 0.0).float()
        denom = valid.sum().clamp(min=1.0)
        return ((pred - target)**2 * valid).sum() / denom
    return F.mse_loss(pred, target)
