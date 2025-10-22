import pandas as pd, torch

@torch.no_grad()
def evaluate_classification(model, dl_te, device):
    model.eval(); ce = torch.nn.CrossEntropyLoss()
    total_loss = total = correct = 0
    for xb, yb, _ in dl_te:
        xb, yb = xb.to(device), yb.to(device)
        logits, _, _ = model(xb)
        if (yb != -1).any():
            loss = ce(logits[yb != -1], yb[yb != -1])
            total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += ((pred == yb) & (yb != -1)).sum().item()
        total += (yb != -1).sum().item()
    acc = correct / max(1, total)
    return total_loss / max(1, total), acc

def load_rp_map(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    mp = {int(r.rp_id):(float(r.x), float(r.y), int(r.floor)) for _,r in df.iterrows()}
    return mp
