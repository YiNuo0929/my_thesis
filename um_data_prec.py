import argparse
from pathlib import Path
import pandas as pd

MASK_RSSI = -110  # 缺值統一填這個


def collect_aps_from_rssis(rssis_path: Path):
    """從第一份 rssis.csv 收集所有 AP，建立 ap_id -> index 映射。"""
    ap_ids = []
    seen = set()
    with rssis_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for pair in line.split(","):
                if ":" not in pair:
                    continue
                ap_id, _ = pair.split(":", 1)
                ap_id = ap_id.strip()
                if ap_id and ap_id not in seen:
                    seen.add(ap_id)
                    ap_ids.append(ap_id)

    ap_to_idx = {ap_id: idx for idx, ap_id in enumerate(ap_ids)}
    return ap_to_idx


def build_rp_mapping_from_coords(coords_path: Path):
    """
    從 coordinates.csv 建立 (x, y, floor) -> rp_id 映射。
    rp_id 從 1 開始連續編號。
    """
    df = pd.read_csv(coords_path, header=None, names=["x", "y", "floor"])
    unique = df.drop_duplicates().reset_index(drop=True)
    rp_map = {}
    for i, row in unique.iterrows():
        key = (row["x"], row["y"], row["floor"])
        rp_map[key] = i + 1  # rp_id 從 1 開始
    return rp_map


def write_rp_id_um_csv(rp_map, out_path: Path):
    """把 rp_id mapping 輸出成 rp_id_um.csv（rp_id, x, y, floor）。"""
    rows = []
    for (x, y, floor), rp_id in rp_map.items():
        rows.append({"rp_id": rp_id, "x": x, "y": y, "floor": floor})
    df = pd.DataFrame(rows).sort_values("rp_id")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def build_fingerprint_for_folder(
    folder: Path,
    ap_to_idx: dict,
    rp_map: dict,
    allow_new_rp: bool = True,
):
    """
    用指定的 ap_to_idx, rp_map 產生 fingerprint.csv。
    若 allow_new_rp=True，遇到新座標會繼續編新的 rp_id。
    """
    coords_path = folder / "coordinates.csv"
    rssis_path = folder / "rssis.csv"

    # 讀 coordinates
    coords_df = pd.read_csv(coords_path, header=None, names=["x", "y", "floor"])

    # 讀 rssis 原始文字
    with rssis_path.open("r", encoding="utf-8") as f:
        rssi_lines = [line.strip() for line in f if line.strip()]

    if len(coords_df) != len(rssi_lines):
        raise ValueError(
            f"Row 數不一致：coordinates={len(coords_df)}, rssis={len(rssi_lines)}"
        )

    num_samples = len(coords_df)
    num_aps = len(ap_to_idx)

    # 方便新增 RP id
    if rp_map:
        next_rp_id = max(rp_map.values()) + 1
    else:
        next_rp_id = 1

    records = []

    for i in range(num_samples):
        x = coords_df.iloc[i]["x"]
        y = coords_df.iloc[i]["y"]
        floor = coords_df.iloc[i]["floor"]
        key = (x, y, floor)

        if key not in rp_map:
            if allow_new_rp:
                rp_map[key] = next_rp_id
                next_rp_id += 1
            else:
                # 若不允許新增 RP，可以選擇跳過或丟錯誤
                raise ValueError(f"在 rp_map 中找不到座標 {key}，且不允許新增")

        rp_id = rp_map[key]

        # 初始化 fingerprint vector，全填 -110
        rssi_vec = [MASK_RSSI] * num_aps

        # 解析這一 row 的 rssis
        line = rssi_lines[i]
        for pair in line.split(","):
            if ":" not in pair:
                continue
            ap_id, rssi_str = pair.split(":", 1)
            ap_id = ap_id.strip()
            rssi_str = rssi_str.strip()

            if not ap_id:
                continue
            if ap_id not in ap_to_idx:
                # 這個 AP 不在第一份資料夾的 AP 集合中，直接略過
                continue

            try:
                rssi_val = float(rssi_str)
            except ValueError:
                # 非數字就當成缺值
                rssi_val = MASK_RSSI

            idx = ap_to_idx[ap_id]
            rssi_vec[idx] = rssi_val

        # 組成一筆記錄
        rec = {
            "rp_id": rp_id,
            "x": x,
            "y": y,
            "floor": floor,
        }
        # 加上 ap0~apN
        for ap_idx in range(num_aps):
            rec[f"ap{ap_idx}"] = rssi_vec[ap_idx]

        records.append(rec)

    # 轉 DataFrame 輸出
    df_out = pd.DataFrame(records)
    out_path = folder / "fingerprint.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved fingerprint to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=False,
        default="UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-06-11",
        help="第一份資料夾 (基準資料夾)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=False,
        default="UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2020-02-19",
        help="第二份資料夾 (依照第一份設定產生 fingerprint)",
    )
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    target_dir = Path(args.target_dir)

    # ---------- 1) 以第一份為基準：建立 AP 映射與 RP 映射 ----------
    rssis_ref = ref_dir / "rssis.csv"
    coords_ref = ref_dir / "coordinates.csv"

    if not rssis_ref.exists() or not coords_ref.exists():
        raise FileNotFoundError("第一份資料夾裡找不到 rssis.csv 或 coordinates.csv")

    print(f"[INFO] Collect APs from {rssis_ref}")
    ap_to_idx = collect_aps_from_rssis(rssis_ref)
    print(f"[INFO] Found {len(ap_to_idx)} APs (ap0 ~ ap{len(ap_to_idx)-1})")

    print(f"[INFO] Build RP mapping from {coords_ref}")
    rp_map = build_rp_mapping_from_coords(coords_ref)
    print(f"[INFO] Found {len(rp_map)} RPs in reference folder")

    # ---------- 2) 在第一份資料夾產生 fingerprint.csv ＋ rp_id_um.csv ----------
    print(f"[INFO] Build fingerprint for ref_dir: {ref_dir}")
    build_fingerprint_for_folder(ref_dir, ap_to_idx, rp_map, allow_new_rp=True)

    rp_id_um_path = ref_dir / "rp_id_um.csv"
    write_rp_id_um_csv(rp_map, rp_id_um_path)
    print(f"[INFO] Saved RP mapping to {rp_id_um_path}")

    # ---------- 3) 在第二份資料夾產生 fingerprint.csv (共用 AP & RP 映射，可新增新 RP) ----------
    rssis_tgt = target_dir / "rssis.csv"
    coords_tgt = target_dir / "coordinates.csv"

    if not rssis_tgt.exists() or not coords_tgt.exists():
        raise FileNotFoundError("第二份資料夾裡找不到 rssis.csv 或 coordinates.csv")

    print(f"[INFO] Build fingerprint for target_dir: {target_dir}")
    build_fingerprint_for_folder(target_dir, ap_to_idx, rp_map, allow_new_rp=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
