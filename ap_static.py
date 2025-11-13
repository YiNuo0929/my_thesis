import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV（路徑照你自己的來）
df = pd.read_csv("UJI_LIB_DB_v2.2/db/04/test_all/all_tst_merged.csv")

# 取得所有 AP 欄位（保留原本在 CSV 裡的順序，不做排序）
ap_cols = [c for c in df.columns if c.startswith("ap")]

# 統計每個 AP 的「非 -110」數量
# non_missing_counts[i] 對應 ap_cols[i]
non_missing_counts = [(df[c] != -110).sum() for c in ap_cols]

# --------- 找出「最後一個有非 -110」的 AP ---------
last_non_zero_idx = -1
for i in range(len(non_missing_counts) - 1, -1, -1):
    if non_missing_counts[i] > 0:
        last_non_zero_idx = i
        break

if last_non_zero_idx == -1:
    print("⚠ 這份資料裡所有 AP 的非 -110 數量都為 0（很不正常，請檢查檔案）")
else:
    print("最後一個有非 -110 資料的 AP：")
    print(f"  index  = {last_non_zero_idx}")
    print(f"  欄位名 = {ap_cols[last_non_zero_idx]}")
    print(f"  非 -110 筆數 = {non_missing_counts[last_non_zero_idx]}")

    # 確認後面是不是全部 0
    all_zero_after = all(
        v == 0 for v in non_missing_counts[last_non_zero_idx + 1 :]
    )

    if all_zero_after:
        print(
            f"➡ 從 index {last_non_zero_idx + 1}（欄位 {ap_cols[last_non_zero_idx + 1]}）"
            f" 一直到最後一個 AP（{ap_cols[-1]}），非 -110 數量都為 0"
        )
    else:
        print("❗ 注意：最後一個非 0 之後，仍然存在非 0 的 AP，"
              "代表資料不是「某一點之後全 0」，請再檢查。")

    # 額外印附近幾個 AP 幫你肉眼檢查
    print("\n附近幾個 AP 統計值（方便對照圖形）：")
    start = max(0, last_non_zero_idx - 5)
    end = min(len(ap_cols), last_non_zero_idx + 6)
    for i in range(start, end):
        print(f"  index {i:3d}  {ap_cols[i]:>6}  count = {non_missing_counts[i]}")

# --------- 畫圖 ---------
plt.figure(figsize=(20, 6))
plt.plot(range(len(ap_cols)), non_missing_counts)
plt.xlabel("AP index (same order as CSV: ap0, ap1, ...)")
plt.ylabel("Count of RSSI != -110")
plt.title("Non -110 RSSI Count for Each AP")
plt.tight_layout()

# 存圖
save_path = "ap_non_missing_count.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"\n📁 圖片已儲存為：{save_path}")
