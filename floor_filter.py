import pandas as pd
from pathlib import Path

# === 輸入檔案路徑 ===
input_path = Path(r"C:\Users\Yinuo\Desktop\UJI_LIB_DB_v2.2\db\04\train_all\all_trn_merged.csv")

# === 讀取 CSV ===
df = pd.read_csv(input_path)

# === 檢查欄位 ===
if 'floor' not in df.columns:
    raise ValueError("找不到 'floor' 欄位，請確認 CSV 檔案內容")

# === 依樓層分割並輸出 ===
output_dir = input_path.parent
for floor_val in df['floor'].unique():
    floor_df = df[df['floor'] == floor_val]
    out_path = output_dir / f"test_floor{int(floor_val)}.csv"
    floor_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出: {out_path}, 共 {len(floor_df)} 筆資料")

print("🎯 分割完成！")
