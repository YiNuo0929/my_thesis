import os
import importlib.util
import torch
from thop import profile, clever_format

# ==========================================
# 1. 請在這裡設定每個模型的「相對路徑」與「類別名稱」
# ==========================================
# 範例：假設 dnn.py 放在 dnn_model 資料夾下，路徑就是 "./dnn_model/dnn.py"
# 請把 "./你的XXX資料夾/..." 替換成你實際擺放程式碼的位置
MODEL_FILES = {
    "DNN (Baseline)":       {"file": "./baseline_model/dnn.py", "class": "DNNClassifier"},
    "CNN 1D (Baseline)":    {"file": "./baseline_model/cnn.py", "class": "CNN1DClassifier"},
    "DANN (Baseline)":      {"file": "./baseline_model/DANN.py", "class": "DANN"},
    "Deep CORAL (Baseline)":{"file": "./baseline_model/deep_coral.py", "class": "CORALNet"},
    "Fidora (Baseline)":    {"file": "./sota_model/fidora.py", "class": "FidoraJCRModel"},
    "iToLoc (Baseline)":    {"file": "./sota_model/itoloc.py", "class": "iToLocModel"},
    "Proposed JCR3 (Ours)": {"file": "./3head/JCR3_um.py", "class": "TransClassifier"} # <--- 已更新為正確的檔名
}

# 假設資料集參數
IN_DIM = 256      # AP 的數量 (請根據你訓練集的 AP 維度修改，例如 520)
N_CLASSES = 100   # Reference Points 數量 (請根據你的資料集修改)

# ==========================================
# 2. 動態載入模組的工具函數
# ==========================================
def load_class_from_file(filepath, class_name):
    """根據相對路徑動態載入 Python 檔案中的指定 Class"""
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"找不到檔案: {abs_path}")
        
    spec = importlib.util.spec_from_file_location("dynamic_module", abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def build_model(model_name, ModelClass):
    """根據你提供的各個原始碼架構，精準進行模型初始化"""
    if "CNN1DClassifier" in model_name or "DNNClassifier" in model_name:
        return ModelClass(in_len=IN_DIM, n_classes=N_CLASSES)
    elif "DANN" in model_name or "CORALNet" in model_name:
        return ModelClass(in_len=IN_DIM, n_classes=N_CLASSES)
    elif "FidoraJCRModel" in model_name:
        return ModelClass(in_dim=IN_DIM, n_classes=N_CLASSES)
    elif "iToLocModel" in model_name:
        # 依照你提供的 itoloc.py，只傳入 n_classes，內部會自動處理影像轉換與維度對齊
        return ModelClass(n_classes=N_CLASSES)
    elif "TransClassifier" in model_name:
        # === 這裡修正了！補齊 JCR3 (TransClassifier) 必需的所有 14 個參數 ===
        return ModelClass(
            input_dim=IN_DIM,           # 你的 AP 數量
            vit_tokens=IN_DIM,          # token 數量 (預設為 AP 數)
            n_classes=N_CLASSES,        # RP 的數量
            d_model=128,                # Transformer 隱藏層維度 (預設 128)
            nhead=4,                    # 注意力頭數 (預設 4)
            num_layers=4,               # Transformer 層數 (預設 4)
            dim_feedforward=128,        # FFN 維度 (預設 128)
            dropout=0.2,                # Dropout 機率 (預設 0.2)
            z_dim=64,                   # Bottleneck 壓縮後的維度 (預設 64)
            recon_hidden=[256],         # 重建層的隱藏層 (預設 [256])
            p_drop=0.2,                 # Predictor 的 Dropout (預設 0.2)
            use_mask=False,             # 是否使用 Mask (預設 False)
            mask_value=-1.0             # Mask 的數值 (預設 -1.0)
        )
    else:
        # 通用 Fallback
        return ModelClass(in_dim=IN_DIM, n_classes=N_CLASSES)

# ==========================================
# 3. 主程式：計算 FLOPs 與 Parameters
# ==========================================
def main():
    print("="*65)
    print(f"{'Model Benchmarking (FLOPs & Parameters)':^65}")
    print("="*65)
    print(f"{'Model Name':<25} | {'Parameters':<15} | {'FLOPs (MACs)':<15}")
    print("-" * 65)

    # 模擬實際部署時，收到 1 筆使用者的 RSSI 請求
    # 準備兩種維度以適應不同模型的 Input 需求
    dummy_input_3d = torch.randn(1, 1, IN_DIM) # [Batch, Channel, Length] (供 CNN, DANN 等使用)
    dummy_input_2d = torch.randn(1, IN_DIM)    # [Batch, Length] (供 Fidora, iToLoc, 你的 JCR3 等使用)

    for display_name, info in MODEL_FILES.items():
        file_path = info["file"]
        class_name = info["class"]

        try:
            # 1. 載入並實例化模型
            ModelClass = load_class_from_file(file_path, class_name)
            model = build_model(class_name, ModelClass)
            model.eval() # 切換至推論模式 (關閉 Dropout / BatchNorm 行為)

            # 2. 計算 FLOPs 與 Parameters
            try:
                # 優先嘗試 3D 輸入 (如 CNN, DANN, Deep CORAL 預期有 Channel 維度)
                macs, params = profile(model, inputs=(dummy_input_3d, ), verbose=False)
            except RuntimeError:
                # 若報錯，則改用 2D 輸入 (如 Fidora, iToLoc 或你的 JCR3 會自己處理)
                macs, params = profile(model, inputs=(dummy_input_2d, ), verbose=False)

            # 格式化輸出 (自動轉為 M 百萬 或 G 十億)
            macs_fmt, params_fmt = clever_format([macs, params], "%.2f")
            print(f"{display_name:<25} | {params_fmt:<15} | {macs_fmt:<15}")

        except FileNotFoundError:
            print(f"{display_name:<25} | {'File Not Found':<15} | {'N/A':<15}")
        except Exception as e:
            # 只印出簡短的錯誤訊息避免洗頻
            print(f"{display_name:<25} | {'Error':<15} | {str(e)[:15]:<15}")

    print("="*65)
    print("* 註：FLOPs 越低，代表推論計算量越小、速度越快，越適合終端/邊緣設備。")
    print("* 註：Parameters 越低，代表模型佔用記憶體越小。")

if __name__ == "__main__":
    main()