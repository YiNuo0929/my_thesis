import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp",
        type=str,
        required=False,
        default="UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2020-02-19/fingerprint.csv",
        help="fingerprint.csv 的相對路徑",
    )
    args = parser.parse_args()

    fp_path = Path(args.fp)
    if not fp_path.exists():
        raise FileNotFoundError(f"找不到 fingerprint.csv: {fp_path}")

    print(f"[INFO] 讀取 fingerprint.csv：{fp_path}")
    df = pd.read_csv(fp_path)

    # 9:1 分割
    train_df, test_df = train_test_split(
        df,
        test_size=0.6,
        shuffle=True,
        random_state=42  # 固定 seed，可重現
    )

    # 輸出位置
    out_dir = fp_path.parent
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 已輸出：{train_path}")
    print(f"[INFO] 已輸出：{test_path}")
    print(f"[INFO] Train 筆數：{len(train_df)}")
    print(f"[INFO] Test 筆數：{len(test_df)}")

if __name__ == "__main__":
    main()
