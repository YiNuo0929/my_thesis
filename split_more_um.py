import os
import pandas as pd

# =========================
# 1. 輸入檔案路徑
# =========================
input_csv = "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2020-02-19/fingerprint.csv"

# =========================
# 2. 輸出資料夾名稱
# =========================
output_dir = "UM_/fingerprint_split"

# =========================
# 3. 要保留的比例
# =========================
ratios = [1.0, 0.8, 0.6, 0.4, 0.2]

# =========================
# 4. 隨機種子（可改）
# =========================
random_seed = 42

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 讀取原始 CSV
df = pd.read_csv(input_csv)

# 原始資料筆數
total_rows = len(df)
print(f"原始資料筆數: {total_rows}")

# 依比例產生四份資料
for ratio in ratios:
    sampled_df = df.sample(frac=ratio, random_state=random_seed)

    percent = int(ratio * 100)
    output_csv = os.path.join(output_dir, f"fingerprint_{percent}.csv")

    sampled_df.to_csv(output_csv, index=False)

    print(f"已輸出: {output_csv}")
    print(f"  保留比例: {ratio}")
    print(f"  資料筆數: {len(sampled_df)}")

print("全部完成。")