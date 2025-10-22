import re
import sys
from pathlib import Path
import pandas as pd

# ==== 路徑設定（可按需修改） ====
DB_DIR = Path(r"C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\06")
RP_MAP_PATH = Path(r"C:\Users\Yinuo\Desktop\my_thesis\rp_id.csv")

TRN_OUT_DIR = DB_DIR / "train_all"   # 訓練資料輸出資料夾
TST_OUT_DIR = DB_DIR / "test_all"    # 測試資料輸出資料夾
TRN_OUT_DIR.mkdir(exist_ok=True)
TST_OUT_DIR.mkdir(exist_ok=True)

# ==== 讀取 RP 對照表：rp_id,x,y,floor ====
try:
    rp_map = pd.read_csv(RP_MAP_PATH, encoding="utf-8-sig")
except FileNotFoundError:
    print(f"[ERROR] 找不到 RP 對照表：{RP_MAP_PATH}")
    sys.exit(1)

required_cols = {"rp_id", "x", "y", "floor"}
if not required_cols.issubset(rp_map.columns):
    print(f"[ERROR] RP 對照表缺少欄位，需包含：{required_cols}，實際欄位：{list(rp_map.columns)}")
    sys.exit(1)

rp_map = rp_map.copy()
rp_map["floor"] = rp_map["floor"].astype(int)

def rounded(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["x"] = tmp["x"].round(decimals)
    tmp["y"] = tmp["y"].round(decimals)
    return tmp

rp_map_r3 = rounded(rp_map, 3)
rp_map_r2 = rounded(rp_map, 2)

# ==== 共用處理函式（split_label = 'trn' 或 'tst'） ====
def process_split(split_label: str, out_dir: Path):
    pattern_crd = re.compile(rf"^{split_label}(\d{{2}})crd\.csv$", re.IGNORECASE)

    pairs = []
    for p in DB_DIR.glob(f"{split_label}*crd.csv"):
        m = pattern_crd.match(p.name)
        if m:
            idx = m.group(1)
            rss_path = DB_DIR / f"{split_label}{idx}rss.csv"
            if rss_path.exists():
                pairs.append((idx, p, rss_path))
            else:
                print(f"[WARN] 找不到對應的 RSS：{rss_path}（略過 {p.name}）")

    if not pairs:
        print(f"[INFO] 沒有找到任何配對的 {split_label}??crd.csv 與 {split_label}??rss.csv")
        return []

    merged_paths = []
    for idx, crd_path, rss_path in sorted(pairs, key=lambda x: x[0]):
        try:
            # 讀 crd
            crd = pd.read_csv(crd_path, header=None, names=["x", "y", "floor"], encoding="utf-8-sig")
            crd["floor"] = crd["floor"].astype(int)

            # 讀 rss，將 100 → -110，並命名 ap0, ap1, ...
            rss = pd.read_csv(rss_path, header=None, encoding="utf-8-sig")
            rss = rss.replace(100, -110)
            rss.columns = [f"ap{i}" for i in range(rss.shape[1])]

            # 筆數對齊
            if len(crd) != len(rss):
                print(f"[WARN] 筆數不一致：{crd_path.name}({len(crd)}) vs {rss_path.name}({len(rss)})，以較短者為準")
                n = min(len(crd), len(rss))
                crd = crd.iloc[:n].reset_index(drop=True)
                rss = rss.iloc[:n].reset_index(drop=True)

            # 依座標對應 rp_id：exact → round(3) → round(2)
            merged = crd.merge(rp_map[["rp_id", "x", "y", "floor"]], on=["x", "y", "floor"], how="left")

            for dec, rp_ref in [(3, rp_map_r3), (2, rp_map_r2)]:
                missing = merged["rp_id"].isna()
                if missing.any():
                    tmp = crd.loc[missing, ["x", "y", "floor"]].round(dec)
                    tmp = tmp.merge(rp_ref[["rp_id", "x", "y", "floor"]],
                                    on=["x", "y", "floor"], how="left")
                    merged.loc[missing, "rp_id"] = tmp["rp_id"]

            merged["rp_id"] = merged["rp_id"].fillna(-1).astype(int)

            # 合併：crd(含 rp_id) + rss
            out_df = pd.concat([merged[["rp_id", "x", "y", "floor"]], rss], axis=1)

            out_path = out_dir / f"{split_label}{idx}_merged.csv"
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            merged_paths.append(out_path)
            print(f"[OK] 輸出：{out_path}（rows={len(out_df)}, rss_cols={rss.shape[1]}）")

        except Exception as e:
            print(f"[ERROR] 處理 {split_label}{idx} 時發生錯誤：{e}")

    # 彙整成一個大檔
    if merged_paths:
        big = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in merged_paths],
                        axis=0, ignore_index=True)

        # ★ 只在 TST 總表時，濾掉 rp_id == -1 的列；單檔保持原樣
        if split_label == "tst":
            before = len(big)
            big = big[big["rp_id"] != -1].reset_index(drop=True)
            removed = before - len(big)
            print(f"[INFO] all_tst_merged.csv 過濾 rp_id = -1 列數：{removed}")

        all_out = out_dir / (f"all_{split_label}_merged.csv" if split_label == "trn"
                             else f"all_tst_merged.csv")
        big.to_csv(all_out, index=False, encoding="utf-8-sig")
        print(f"\n✅ 已成功整合 {split_label.upper()} 所有資料，共 {len(big)} 筆，輸出：{all_out}\n")
    return merged_paths

# ==== 執行：先 TRN，再 TST ====
print("=== 處理 TRN（訓練資料） ===")
process_split("trn", TRN_OUT_DIR)

print("=== 處理 TST（測試資料） ===")
process_split("tst", TST_OUT_DIR)

print("全部完成 ✅")
