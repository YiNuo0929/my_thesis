# main.py
# 主程式入口，整合所有模組並執行訓練與測試

import argparse, os, numpy as np, pandas as pd
from torch.utils.data import DataLoader
import torch
from dataio import load_csvs, get_feature_cols, set_seed
from scaling import fit_scaler, apply_scaler
from datasets import RSSISourceDataset, RSSITargetDataset, TestDataset
from models import FullModel
from eval_utils import evaluate_classification, load_rp_map
from train_loop import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_path", type=str, required=True)
    parser.add_argument("--target_train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--rp_map_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size_src", type=int, default=256)
    parser.add_argument("--batch_size_tgt", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    '''
    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--ext_hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--cls_hidden", type=int, nargs="+", default=[256])
    parser.add_argument("--rec_hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--dropout", type=float, default=0.4)
    '''
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--missing_val", type=float, default=-110.0)
    parser.add_argument("--ignore_missing_in_recon", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./rssi_recon_ckpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # -------- Load CSVs --------
    df_src = load_csvs(args.source_train_path)
    df_tgt = load_csvs(args.target_train_path)
    df_te = load_csvs(args.test_path)
    ap_cols = get_feature_cols(df_src)

    # -------- Scaler --------
    # 各 domain 各自 fit min/max
    mins_src, maxs_src = fit_scaler(df_src[ap_cols].values.astype(np.float32), args.missing_val)
    mins_tgt, maxs_tgt = fit_scaler(df_tgt[ap_cols].values.astype(np.float32), args.missing_val)

    # 儲存各自的 scaler 參數
    np.savez(os.path.join(args.out_dir, "scaler_source_minmax.npz"), mins=mins_src, maxs=maxs_src)
    np.savez(os.path.join(args.out_dir, "scaler_target_minmax.npz"), mins=mins_tgt, maxs=maxs_tgt)

    # -------- Apply scaling before Dataset --------
    # Source domain
    df_src_proc = df_src.copy()
    df_src_proc[ap_cols] = apply_scaler(
        df_src[ap_cols].values.astype(np.float32), mins_src, maxs_src, args.missing_val
    )

    # Target domain
    df_tgt_proc = df_tgt.copy()
    df_tgt_proc[ap_cols] = apply_scaler(
        df_tgt[ap_cols].values.astype(np.float32), mins_tgt, maxs_tgt, args.missing_val
    )

    # -------- Dataset / Loader --------
    ds_src = RSSISourceDataset(df_src_proc, ap_cols, "rp_id", missing_val=0.0)
    id2idx, idx2id = ds_src.id2idx, ds_src.idx2id
    n_classes = len(id2idx)
    dl_src = DataLoader(ds_src, args.batch_size_src, shuffle=True, drop_last=True)

    ds_tgt = RSSITargetDataset(df_tgt_proc, ap_cols, missing_val=0.0)
    dl_tgt = DataLoader(ds_tgt, args.batch_size_tgt, shuffle=True, drop_last=True)

    # -------- Test set (use target's scaler) --------
    X_te_raw = df_te[ap_cols].values.astype(np.float32)
    X_te = apply_scaler(X_te_raw, mins_tgt, maxs_tgt, args.missing_val)
    miss_te = (X_te_raw == args.missing_val).astype(np.float32)
    y_te_raw = df_te["rp_id"].astype(int).values if "rp_id" in df_te.columns else np.full(len(df_te), -1, int)
    y_te_idx = np.array([id2idx.get(int(r), -1) for r in y_te_raw], np.int64)

    dl_te = DataLoader(TestDataset(X_te, y_te_idx, miss_te), batch_size=512, shuffle=False)

    # -------- Model --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(len(ap_cols), n_classes).to(device)
    ce = torch.nn.CrossEntropyLoss()

    # -------- Train --------
    best_acc, id_maps = train_model(
        model, dl_src, dl_tgt, dl_te, device, ce, args.lambda_recon,
        args.ignore_missing_in_recon, args.out_dir, args.epochs
    )

    # -------- Evaluate (MDE) --------
    from math import sqrt

    model.eval()
    preds_idx = []
    with torch.no_grad():
        for xb, yb_idx, _ in dl_te:
            xb = xb.to(device)
            logits, _, _ = model(xb)
            preds_idx.append(logits.argmax(1).cpu().numpy())
    preds_idx = np.concatenate(preds_idx) if preds_idx else np.array([])

    # 與原版一致：用訓練時的 idx2id 對應回 rpid，並用 te 的 y_te_raw/y_te_idx 當作 GT
    idx2id = id_maps["idx2id"]
    preds_rpid = np.array([idx2id.get(int(i), -999999) for i in preds_idx], dtype=int)
    gts_rpid   = y_te_raw
    eval_mask  = (y_te_idx != -1)

    acc = (preds_idx[eval_mask] == y_te_idx[eval_mask]).mean().item() if eval_mask.any() else float("nan")

    rp_map = load_rp_map(args.rp_map_path)
    mde_distances, floor_mismatch, mde_skipped_notfound = [], 0, 0
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

    # === 與原始輸出格式完全一致 ===
    print("==== Final Test Metrics ====")
    print(f"Test samples total          : {len(y_te_idx)}")
    print(f"Evaluated (label available) : {int(eval_mask.sum())}")
    print(f"Test Accuracy               : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy               : N/A")
    print(
        f"Mean Distance Error (same floor only): {avg_mde:.4f} (meters)"
        if not np.isnan(avg_mde) else
        "Mean Distance Error (same floor only): N/A"
    )
    print(f"Floor mismatches (excluded from MDE) : {floor_mismatch}")
    print(f"Skipped (rp_id not in rp_map)        : {mde_skipped_notfound}")



if __name__ == "__main__":
    main()
