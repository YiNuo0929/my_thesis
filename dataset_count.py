import pandas as pd

# ====== 在這裡填入你的 CSV 路徑 ======
csv_path = "simulation_data/target/train.csv"
# ====================================

# 讀取資料
df = pd.read_csv(csv_path)

# fingerprint 數量（資料筆數）
num_fingerprints = len(df)

# RP 點位數量
num_rp = df["rp_id"].nunique()

# AP 欄位（假設欄位名稱是 ap0, ap1...）
ap_columns = [c for c in df.columns if c.startswith("ap")]
num_ap = len(ap_columns)

print("===== Dataset Statistics =====")
print(f"Fingerprint samples : {num_fingerprints}")
print(f"RP points           : {num_rp}")
print(f"AP count            : {num_ap}")