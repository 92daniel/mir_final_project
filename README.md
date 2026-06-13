# 🎤 Singer Identity Representation Learning 

這是一個基於自監督學習（Self-Supervised Learning, SSL）的歌手音色提取專題。本專案基於 ISMIR 2023 論文 [*Singer Identity Representation Learning using Self-supervised techniques*](https://arxiv.org/abs/2401.05064) 所提出的 BYOL 架構進行改良。

我們使用 **EfficientNet-B0** 作為骨幹網路，並導入了我們實作的 **正交解耦 (Orthogonal Disentanglement, OD)** 與 **非對稱局部裁切 (Asymmetric Orthogonal Disentanglement, AOD)** 機制，旨在剝離演唱技巧與雜訊，提取出最純淨的歌手物理音色。

<br>

---

## 🛠️ 環境建置 (Installation)

請打開終端機，依照以下步驟建立 `ssl-singer-identity` 虛擬環境並安裝所需套件。

> 💡 **重要環境修復提醒：**
> 由於 PyTorch Lightning 的套件依賴問題，我們必須強制降級 `pydantic`，並手動補齊 NVIDIA 編譯庫，否則在讀取 Checkpoint 或計算 GPU 頻譜圖時會發生崩潰。

```bash
# 1. 建立虛擬環境 (建議 Python 3.10)
conda create -n ssl-singer-identity python=3.10 -y

# 2. 啟動虛擬環境
conda activate ssl-singer-identity

# 3. 進入專案目錄
cd ~/ssl-singer-identity-master專題/

# 4. 安裝專案基本依賴 (請確認目錄內有 requirements.txt)
pip install -r requirements.txt

# 5. 【關鍵修復】降級 pydantic 以相容 jsonargparse (解決 load checkpoint 報錯)
pip install "pydantic<2.0.0"

# 6. 【關鍵修復】補齊 CUDA 即時編譯庫 (解決 torchaudio 算頻譜圖時 nvrtc 報錯)
pip install --force-reinstall nvidia-cuda-nvrtc-cu12
```

<br>

---

## 🗂️ 資料集準備 (Dataset Preparation)

本專案使用以下資料集進行訓練與測試，所有音檔皆統一處理為 `44.1kHz` 的 `.wav` 格式：

* **訓練集 (Training Set):** `STraDa Dataset`
  * 經過 Mel-RoFormer 人聲分離與 VAD 靜音過濾，共包含 4,529 位歌手、總長約 144 小時的高音質純淨人聲。
* **評估集 (Evaluation Sets):** * `NUS-48E` (測試同個人的說話與唱歌)、`M4Singer` (中文流行)、`VocalSet` (極端唱腔技巧)、`VCTK` (純語音基準)。

> 💡 **重要提醒：**
> 資料集檔案太大所以沒有放在final_project裡面。

<br>

---

## 🚀 執行自監督訓練 (Training)

訓練過程主要由兩個 YAML 設定檔控制：`efficientnet.yaml` (掌管資料路徑、Batch Size 與存檔邏輯) 與 `byol.yaml` (掌管模型架構、損失函數與學習率)。

請在專案根目錄下執行以下指令開始訓練：

```bash
# 指定使用第 0 張 GPU 進行訓練，避免共用伺服器時的資源衝突
python train.py \
    --config singer_identity/train_configs/efficientnet.yaml \
    --config singer_identity/train_configs/byol.yaml
```

> 💡 **訓練提示：**
> 訓練過程中，PyTorch Lightning 會自動將最佳的 Checkpoint 儲存在 `logs/byol/version_X/checkpoints/` 目錄下。我們設定了 `monitor: "loss/val"` 來確保存下表現最穩定（Loss 最低）的模型權重。

<br>

---

## 🧪 模型推論與評估 (Evaluation)

訓練完成後，我們可以直接針對特定資料集計算 **EER (Equal Error Rate)** 與 **MNR (Mean Normalized Rank)**。

以下指令將載入我們訓練好的模型權重（例如 Epoch 197 的 Checkpoint），並在 `NUS-48E_read_4s16_filtered` 測試集上進行評估：

```bash
python eval_project.py \
    --backbone efficientnet_b0 \
    -r /home/jhao/dataset \
    -d NUS-48E_read_4s16_filtered \
    -meta /home/jhao/dataset \
    -ce \
    -cr \
    --ckpt_path "/home/jhao/ssl-singer-identity-master專題/logs/byol/version_4/checkpoints/best-eer-baseline-epoch=197-step=50490.ckpt"
```

### 參數說明：
* `--backbone`: 指定評估使用的骨幹網路（此處為 `efficientnet_b0`）。
* `-r`: 測試資料集所在的根目錄 (`root`)。
* `-d`: 要測試的具體資料集資料夾名稱。
* `-meta`: 包含 `speaker_pairs.txt` 配對檔的中繼資料夾。
* `-ce`: 啟用 **EER (Equal Error Rate)** 計算。
* `-cr`: 啟用 **Rank (Mean Normalized Rank)** 計算。
* `--ckpt_path`: 指定你要載入的 `.ckpt` 權重檔絕對路徑。

<br>

---

## 📊 實驗結果紀錄 (Results)
執行完 `eval_project.py` 後，程式會自動將該資料集的 EER 與 Rank 成績匯出並附加到 `metadata_dir` 下的 CSV 檔案中（例如 `eer_123.csv` 與 `rank_123.csv`），方便您直接統整到論文或專題報告的圖表中。
