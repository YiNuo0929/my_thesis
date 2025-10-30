import os, numpy as np, torch, matplotlib.pyplot as plt, pandas as pd
from itertools import cycle
from losses import masked_mse
from eval_utils import evaluate_classification

def train_model(model, dl_src, dl_tgt, dl_te, device, ce, lambda_recon, ignore_missing_in_recon, out_dir, epochs):
    it_tgt, best_acc = cycle(dl_tgt), -1.0  #target domain 的 dataloader 轉成無限循環迭代器。因為 target 資料通常少每輪需要重複利用。
    hist = dict(epoch=[], train_ce=[], rec_s=[], rec_t=[], val_ce=[], val_acc=[])
    opt = torch.optim.AdamW(model.parameters())

    id_maps = {"id2idx": getattr(dl_src.dataset, "id2idx", {}), "idx2id": getattr(dl_src.dataset, "idx2id", {})}

    for ep in range(1, epochs + 1):
        model.train(); rc, rs, rt = 0, 0, 0
        for xs, ys, ms in dl_src:
            xt, _, mt = next(it_tgt)
            xs, ys, ms, xt, mt = xs.to(device), ys.to(device), ms.to(device), xt.to(device), mt.to(device)
            logits_s, rec_s, _ = model(xs)
            ce_loss = ce(logits_s, ys)
            rec_loss_s = masked_mse(rec_s, xs, ms, ignore_missing_in_recon)
            _, rec_t, _ = model(xt)
            rec_loss_t = masked_mse(rec_t, xt, mt, ignore_missing_in_recon)
            loss = ce_loss + lambda_recon * (rec_loss_s + rec_loss_t)
            opt.zero_grad(); loss.backward(); opt.step()
            rc += ce_loss.item(); rs += rec_loss_s.item(); rt += rec_loss_t.item()

        val_ce, val_acc = evaluate_classification(model, dl_te, device)
        print(f"Epoch {ep:03d} | train_CE {rc/len(dl_src):.4f} | "
              f"train_Recon_s {rs/len(dl_src):.4f} | train_Recon_t {rt/len(dl_src):.4f} | "
              f"val_CE {val_ce:.4f} | val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict()}, os.path.join(out_dir, "best_recon_model.pth"))
            print(f"[Info] Saved new best (acc={best_acc:.4f})")

        hist["epoch"].append(ep); hist["train_ce"].append(rc/len(dl_src))
        hist["rec_s"].append(rs/len(dl_src)); hist["rec_t"].append(rt/len(dl_src))
        hist["val_ce"].append(val_ce); hist["val_acc"].append(val_acc)

    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "training_log.csv"), index=False, encoding="utf-8-sig")
    plt.figure(); plt.plot(hist["epoch"], hist["train_ce"], label="train_CE")
    plt.plot(hist["epoch"], hist["val_ce"], label="val_CE"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=160); plt.close()
    plt.figure(); plt.plot(hist["epoch"], hist["rec_s"], label="train_Recon_s")
    plt.plot(hist["epoch"], hist["rec_t"], label="train_Recon_t"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(out_dir, "recon_curves.png"), dpi=160); plt.close()
    plt.figure(); plt.plot(hist["epoch"], hist["val_acc"], label="val_acc"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(out_dir, "val_acc_curve.png"), dpi=160); plt.close()
    return best_acc, id_maps
