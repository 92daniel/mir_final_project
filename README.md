# 🎤 Singer Identity Representation Learning 

這是一個基於自監督學習（Self-Supervised Learning, SSL）的歌手音色提取專題。本專案基於 ISMIR 2024 論文 [*Singer Identity Representation Learning using Self-supervised techniques*](https://arxiv.org/abs/2401.05064) 所提出的 BYOL 架構進行改良。

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
