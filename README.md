# TransJCR

## Overview

本 Repository 為碩士論文 **TransJCR** 的完整程式碼，主要研究 RSSI Fingerprinting Indoor Localization 在 Temporal Domain Shift 下的 Unsupervised Domain Adaptation。

此 Repository 主要作為實驗室交接使用，因此除了論文最終使用的模型外，也保留了研究過程中所有 Baseline、模型演進、消融實驗以及各種測試程式。若僅需重現論文結果，建議直接使用 **`3head/`** 資料夾中的程式即可。

Python 環境與套件版本請參考 `requirements.txt`。

---

# Dataset

為了方便後續維護與不同資料集共用程式碼，所有資料集皆已整理成統一格式。

每個資料集皆使用相同的 RP Mapping 格式：

```text
rp_id, x, y, floor
```

其中：

- `rp_id`：Reference Point ID
- `x`：X 座標
- `y`：Y 座標
- `floor`：樓層資訊

各資料集皆有對應的 RP Mapping 檔：

```text
rp_id_um.csv
rp_id_simulation.csv
rp_id_mall.csv
```

程式會透過這些 Mapping 將模型預測的 `rp_id` 轉換回實際座標，用於定位誤差(MDE)計算。

---

## Dataset Location

### UM Dataset

資料放置於

```text
UM_DSI_DB_v1.0.0_lite/
└── UM_DSI_DB_v1.0.0_lite/
    └── data/
        └── site_surveys/
```

論文實驗中只用到 `2019-06-11` `2019-10-09` `2020-02-19` 這三個時間點的資料，所以只有這幾個資料集內有我分好用好訓練及測試的csv檔案，其餘為dataset原檔。

---

### Simulation Office Dataset

資料放置於

```text
simulation_data/
```

---

### Mall Dataset

Mall Dataset 因資料量較大，因此**未上傳至 GitHub**。

其資料格式與 UM Dataset 相同，只需放置於 `mall_data/` 即可正常執行。

---

## Unused Dataset

Repository 中仍保留

```text
UJI_LIB_DB_v2.2
```

此資料集為研究初期測試使用，**論文最終實驗並未使用**，保留僅供日後參考。

---

# Model

Repository 中包含許多不同版本的模型，主要是研究過程中的發想、Prototype、Baseline 與消融實驗，因此資料夾數量較多。

## baseline_model/ & cr_model/

存放各種 Baseline Model。

存放研究過程中的模型演進、Prototype、消融實驗以及其他測試程式。

若有興趣了解整個研究發展過程，可自行參考此資料夾(但很雜我認為沒必要)。

---

## 3head/

此資料夾為**論文最終使用的 TransJCR Model**。

若只是希望重現論文實驗結果，建議直接使用此資料夾即可。

其餘模型大多屬於研究過程中的測試版本，詳細內容可參考論文說明。

---

# Training

## 1. UM-Mid

```bash
python 3head/JCR3_um.py \
--source_train_path "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-06-11/train.csv" \
--target_train_path "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-10-09/train.csv" \
--test_path "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-10-09/test.csv" \
--rp_map_path "rp_id_um.csv" \
--epochs 200 \
--batch_size 256 \
--lr 0.0002 \
--lambda_recon 18.0 \
--use_mask \
--column 168 \
--lambda_consist 0.05 \
--noise_std 0.1 \
--drop_prob 0.3 \
--lambda_entropy 0.03
```

---

## 2. UM-Long

```bash
python 3head/JCR3_um.py \
--source_train_path "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2019-06-11/train.csv" \
--target_train_path "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2020-02-19/train.csv" \
--test_path "UM_DSI_DB_v1.0.0_lite/UM_DSI_DB_v1.0.0_lite/data/site_surveys/2020-02-19/test.csv" \
--rp_map_path "rp_id_um.csv" \
--epochs 100 \
--batch_size 256 \
--lr 0.0002 \
--lambda_recon 8.0 \
--use_mask \
--column 168 \
--lambda_consist 0.07 \
--noise_std 0.1 \
--drop_prob 0.3 \
--lambda_entropy 0.05
```

---

## 3. Mall

```bash
python 3head/JCR3_mall.py \
--source_train_path "mall_data/Mall_1/train.csv" \
--target_train_path "mall_data/Mall_7/train.csv" \
--test_path "mall_data/Mall_7/test.csv" \
--rp_map_path "rp_id_mall.csv" \
--epochs 200 \
--batch_size 256 \
--lr 0.00005 \
--lambda_recon 3.0 \
--column 1033 \
--use_mask \
--z_dim 32
```

---

## 4. Simulation Office

```bash
python 3head/JCR3_simulation.py \
--source_train_path "simulation_data/source/train.csv" \
--target_train_path "simulation_data/source/train.csv" \
--test_path "simulation_data/source/test.csv" \
--rp_map_path "rp_id_simulation.csv" \
--epochs 300 \
--batch_size 256 \
--lr 0.0005 \
--lambda_recon 50.0 \
--column 9 \
--use_mask \
--lambda_entropy 0.01 \
--lambda_consist 0.01
```

---

# Notes

- 本 Repository 保留了研究過程中的大部分程式，因此包含許多測試版本、Prototype 、消融實驗或資料處理和實驗呈現的程式碼。
- 若僅需重現論文結果，建議直接使用 **`3head/`** 中的程式即可。
- 其他模型及實驗設計的詳細內容皆可參考論文說明。
