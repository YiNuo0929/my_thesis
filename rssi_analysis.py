import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Parameters
# ==========================================
rp_id_target = 1

path_time1 = (
    "/home/mcslab/yinuo/my_thesis/"
    "UM_DSI_DB_v1.0.0_lite/"
    "UM_DSI_DB_v1.0.0_lite/data/site_surveys/"
    "2019-06-11/fingerprint.csv"
)

path_time2 = (
    "/home/mcslab/yinuo/my_thesis/"
    "UM_DSI_DB_v1.0.0_lite/"
    "UM_DSI_DB_v1.0.0_lite/data/site_surveys/"
    "2019-10-09/fingerprint.csv"
)

label_time1 = "2019-06-11"
label_time2 = "2019-10-09"

save_path = (
    f"/home/mcslab/yinuo/my_thesis/"
    f"rp_{rp_id_target}_wifi_comparison.png"
)

# ==========================================
# 2. Read CSV
# ==========================================
df1 = pd.read_csv(path_time1)
df2 = pd.read_csv(path_time2)

# ==========================================
# 3. Select AP0 ~ AP79 (first 80 APs)
# ==========================================
ap_cols = [f"ap{i}" for i in range(80)]

# 確保兩個 CSV 都存在
ap_cols = [
    col for col in ap_cols
    if col in df1.columns and col in df2.columns
]

if len(ap_cols) == 0:
    raise ValueError("找不到 AP 欄位 (ap0 ~ ap79)")

# ==========================================
# 4. Filter RP
# ==========================================
rp1 = df1[df1["rp_id"] == rp_id_target]
rp2 = df2[df2["rp_id"] == rp_id_target]

if rp1.empty:
    raise ValueError(
        f"{label_time1} 找不到 rp_id = {rp_id_target}"
    )

if rp2.empty:
    raise ValueError(
        f"{label_time2} 找不到 rp_id = {rp_id_target}"
    )

# ==========================================
# 5. Mean RSSI of same RP
# ==========================================
mean_rssi_1 = rp1[ap_cols].mean()
mean_rssi_2 = rp2[ap_cols].mean()

# ==========================================
# 6. Plot
# ==========================================
x = range(len(ap_cols))

plt.figure(figsize=(18, 6))

plt.plot(
    x,
    mean_rssi_1,
    marker="o",
    linewidth=2,
    label=label_time1
)

plt.plot(
    x,
    mean_rssi_2,
    marker="o",
    linewidth=2,
    label=label_time2
)

# ==========================================
# 7. Axis / Labels
# ==========================================
plt.title(
    f"WiFi Fingerprint Comparison at RP {rp_id_target}",
    fontsize=16
)

plt.xlabel("AP Index", fontsize=12)
plt.ylabel("RSSI (dBm)", fontsize=12)

# 每 10 個 AP 顯示一次
tick_positions = list(range(0, len(ap_cols), 10))
tick_labels = [str(i) for i in tick_positions]

plt.xticks(
    tick_positions,
    tick_labels
)

plt.xlim(0, len(ap_cols) - 1)
plt.ylim(-120, 0)

plt.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.legend()
plt.tight_layout()

# ==========================================
# 8. Save Figure
# ==========================================
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved figure: {save_path}")

plt.show()