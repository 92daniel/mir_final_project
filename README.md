# 🎵 Beat Tracking Fine-tuning with FMA

這是一個基於深度學習的音樂節拍追蹤 (Beat Tracking) 作業。本專案基於最新的開源論文 [**BEAT THIS!**](https://arxiv.org/abs/2407.21658) 所提出的先進模型架構進行微調 (Fine-tuning)，使用 [**FMA**](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium?resource=download-directory&select=fma_small) 資料集進行訓練，並在 GTZAN 資料集上進行最終評估。

<br>

---

## 🛠️ 環境建置 (Installation)

請依照以下步驟進行安裝：

### 1. 建立並啟動 Conda 虛擬環境與安裝依賴套件
請打開終端機，輸入以下指令建立一個名為 `beat_this_env` 且 Python 版本為 3.10 的環境，並進入指定目錄安裝所有必備套件：

```bash
# 建立虛擬環境
conda create -n beat_this_env python=3.10 -y

# 啟動虛擬環境
conda activate beat_this_env

# 進入指定的 src 目錄
cd 61447016S/src/beat_this-main/

# 一鍵安裝所有依賴套件 (包含本地開發者模式套件)
pip install -r requirements.txt
```

<br>

---

## 🗂️ 資料集準備 (Dataset Preparation)

本專案使用 FMA (Free Music Archive) 資料集進行模型的微調訓練。安裝完環境後，請依照以下步驟下載並配置資料：

### 1. 下載資料集
我已經打包好一份放在雲端硬碟了。建議直接用下面的連結下載：

👉 **[Google Drive 下載連結 (推薦)](https://drive.google.com/file/d/1wyArzMdfOgKOWtmxBpuwug15wEnL0Yvp/view?usp=sharing)**

> 備用管道：[Kaggle 官方下載頁面](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium?resource=download-directory&select=fma_small)

### 2. 解壓縮與放置
下載完成後，請將檔案解壓縮，並將整包 `fma_small` 資料夾放置到本專案指定的路徑下：
👉 目標路徑：`61447016S/src/beat_this-main/fma_small/`

### 3. 結構檢查
請務必確認解壓縮後的位置與層級正確。`fma_small` 資料夾底下應該要包含 **158 個以數字命名的子資料夾**。目錄結構應如下所示：

```text
61447016S/src/beat_this-main/
└── fma_small/
    ├── 000/
    ├── 001/
    ├── 002/
    ...
    └── (依此類推共 158 個資料夾)
```

<br>

---

## ⚙️ 階段一：資料前處理與產生 Pseudo-labels

因為 FMA 資料集本身並沒有提供節拍 (Beat) 的標準答案，所以我寫了一個腳本，利用兩個不同的模型來幫忙「標答案」，並篩選出品質比較好的資料來做微調 (Fine-tuning)。

> ⚠️ **防呆提醒 (非常重要！)**
> 在執行以下腳本之前，請務必確認 `61447016S/src/beat_this-main/data` 這個資料夾是**空的**。我預設是空的資料夾。

### 執行萃取指令
確認環境乾淨後，請在終端機進入 `beat_this-main` 目錄並執行腳本：

```bash
# 進入腳本所在目錄
cd 61447016S/src/beat_this-main/

# 執行前處理與標註萃取
python prepare_fma_for_beat_this2.py
```

### 🧠 腳本處理流程
* **雙模型篩選 (Consensus)：** 程式會同時載入作者的 `final1` 模型以及 `madmom` 的 RNN 模型去預測 FMA 的音檔。只有當這兩個模型對同一首歌的預測結果高度一致（F1-score ≥ 0.6）時，這首歌才會被保留。這樣可以有效過濾掉節拍太模糊或太難標的壞檔。
* **切出 Train 與 Validation：** 篩選過關的 2,377 首歌，會以 9:1 的比例切成訓練集 (2,140 首) 與驗證集 (237 首)。這裡有固定亂數種子 (`random.seed(42)`) 確保每次跑的結果一致，最後會自動生出一個 `split.txt` 檔供後續訓練使用。

<br>

---

## 🎼 階段二：特徵提取與資料增強 (Feature Extraction & Augmentation)

因為模型沒辦法直接吃 MP3 音檔，所以我們要先把它們轉成頻譜圖，並順便做一下資料增強 (Data Augmentation) 來擴充訓練資料量。

### 執行預處理指令
請保持在 `beat_this-main` 目錄下，執行以下腳本：

```bash
# 執行特徵轉換與資料增強
python launch_scripts/preprocess_audio.py
```

### 腳本核心轉換邏輯
* **資料增強 (Data Augmentation)：** 程式會自動幫音檔做升降 Key（-5 到 +6 半音）以及改變節奏快慢（±20%）。這樣可以讓模型多學一點不同速度和音高的變化。
* **頻譜圖轉換 (Log Mel Spectrogram)：** 將原始音訊轉換成模型真正需要的特徵格式：對數梅爾頻譜圖 (Log Mel Spectrogram)。
* **打包壓縮 (NPZ Bundling)：** 把算好的一大堆頻譜圖特徵壓縮打包成 `.npz` 檔案。這樣之後在訓練模型時，讀取速度才會快，不會被硬碟的 I/O 速度卡住。

<br>

---

## 🧠 階段三：模型微調訓練

這部分我們使用 PyTorch Lightning 來做微調。為了節省訓練時間並提高準確率，我們直接拿原作者已經訓練好的權重當作起點 (Transfer Learning)，並開啟了 PyTorch 2.0 內建的加速功能。

### 執行訓練指令
請在終端機輸入以下指令啟動訓練：

```bash
python launch_scripts/train.py \
  --resume-checkpoint checkpoints/final1.ckpt \
  --max-epochs 150 \
  --batch-size 16 \
  --force-flash-attention \
  --val-frequency 1 \
  --name "fma_finetune" \
  --compile
```

### 參數重點說明
* **`--resume-checkpoint`：** 載入原作者用 15 個資料集訓練好的 `final1` 權重。直接從這個基礎接著訓練，模型會收斂得比較快。
* **`--max-epochs 150` 與 `--val-frequency 1`：** 總共訓練 150 輪。這裡刻意設定 `val-frequency 1` 讓程式每個 Epoch 都強制跑一次 Validation，這樣之後的腳本才有辦法準確抓出分數最高的那個 Checkpoint。
* **`--compile` 與 `--force-flash-attention`：** 開啟 PyTorch 2.0 的 `torch.compile` 和 Flash Attention。
* **`--name "fma_finetune"`：** 給這次的訓練取個名字，之後去 `lightning_logs` 找訓練紀錄才不會眼花。

<br>

---

## 🚀 階段四：推論與最終評估

訓練完成後，我們要從 150 個 Epoch 中挑出驗證集表現最好的一組權重，拿來預測 GTZAN 資料集並計算最終成績。這裡提供了「全自動」和「手動備用」兩種方式：

### 方案 A：自動抓取最佳權重 (預設方案)
這是最快的方法。腳本會自動去 `lightning_logs` 找最新一次訓練的 `metrics.csv`，抓出驗證集 F1-score 最高的那個 Epoch，然後自動載入 `checkpoints/` 裡對應的權重進行預測。

```bash
cd 61447016S/src/beat_this-main/
python gen_beat_this_multi_json.py
```

### 方案 B：手動指定權重 (備用方案)
如果自動腳本抓錯檔案或發生 bug，可以自己手動指定要測試的權重。目前在 `checkpoints/` 資料夾下，我已經幫你保留了兩個現成的權重可以直接測試：
* `prediction.ckpt`：我微調出來的權重。
* `final1.ckpt`：原作者的預訓練權重。

**操作步驟：**
1. 打開 `gen_beat_this_multi_json2.py`。
2. 找到大約第 50 行的 `CHECKPOINT_FILENAME = "..."`，把引號內改成你要指定的 `.ckpt` 檔名（例如 `"prediction.ckpt"`）並存檔。
3. 執行備用腳本：

```bash
# 確保一樣在 beat_this-main 目錄下
python gen_beat_this_multi_json2.py
```

> 💡 **改名提醒：**
> 不管你手動指定哪個權重，這支腳本吐出來的結果檔名**一律都會叫 `prediction.json`**。所以，如果你是拿 `final1.ckpt` 來測，跑完後請記得手動把產生的檔案改名為 `prediction_final1.json`。

---

### 📊 計算最終成績
不管你是用方案 A 還是方案 B，執行成功後都會在當前目錄 (`beat_this-main/`) 產生一個 `prediction.json` 檔案（這就是我們這次作業微調出來的結果！）。
最後，只要把它搬到上一層，交給老師的腳本算分數就大功告成了：

```bash
# 1. 將生成的 prediction.json 移動到上一層 (src 目錄)
mv prediction.json ../

# 2. 回到 src 目錄
cd ..

# 3. 執行課程提供的評估腳本，看我們微調後的最終成績
python eval_json.py prediction.json
```

> **💡 補充對比測試：**
> 如果你想親眼確認原作者模型的表現，專案資料夾裡也有附上原作者的預測結果 (`prediction_final1.json`)。
> 
> ```bash
> python eval_json.py prediction_final1.json
> ```

<br>

### 📊 最終成績比較表

模型比較結果如下：

| 模型版本 | 檔案名稱 | F-Measure | Cemgil | P-Score |
| :--- | :--- | :---: | :---: | :---: |
| 🥇 **原作者 Checkpoint** | `prediction_final1.json` | **0.8903** | **0.8093** | **0.8828** |
| 🥈 **微調後模型 (本次作業)** | `prediction.json` | 0.8864 | 0.7950 | 0.8779 |
| 🥉 **課程 Baseline (老師)** | `(Baseline)` | 0.8702 | 0.7851 | 0.8603 |

<br>

## 📚 參考文獻 (References)

本專案的核心模型架構、預處理管線與預訓練權重 (`final1`) 皆源自以下論文與其開源專案，特此致謝原作者們的貢獻：

* Foscarin, F., Schlüter, J., & Widmer, G. (2024). **BEAT THIS! ACCURATE BEAT TRACKING WITHOUT DBN POSTPROCESSING**. *arXiv preprint arXiv:2407.21658*. 
  * 📄 [Paper (arXiv)](https://arxiv.org/abs/2407.21658)
  * 💻 [Official GitHub Repository](https://github.com/CPJKU/beat_this)
