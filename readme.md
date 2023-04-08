# AOI 瑕疵影像檢測

## 專案介紹
本專案旨在利用深度學習技術來檢測電路板(PCB)上的電子元件是否出現瑕疵，例如電子元件位移、破損等問題。專案目標除了提高模型預測的準確率外，我們更關注**漏檢率**這個指標，即不良品被預測為良品的比例。因為這樣的錯誤是較不能容忍的，故我們的模型會以低漏檢率為目標進行設計。  


## 資料來源
本資料與台灣科技公司合作所取得，因為基於資料保密協定，故無法透露資料與產線細節。  


## 資料簡介與前處理
- 照片的種類有良品與瑕疵品兩類
- Training data 共 12224 張，其中將 12224 中拆出 10000 與 2224 分別作為 Training 與 Validation，而 Testing data 共 2418 張
- 照片尺寸沒有一致的大小，但在訓練時統一將照片轉為 224x224 
- 利用 Data augmentation 來做資料強化，包含 Guassian noise、Autocontrast、AdjustSharpness


## 模型簡介
主要嘗試的模型有：
1. 一般的 CNN 模型（5 層 convolution & 2 層 fully connected），簡稱為 My_CNN 更詳細的模型架構可至 `aoi_dd.ipynb` 內查看
2. 利用 pretrained 的 ResNet50，搭配我們的資料進行 Transfer learning，也就是利用我們的資料來 Fine-tune pretrained 好的權重
3. 利用 pretrained 的 EfficientNet，做法與 2. 相同
4. 利用 ResNet50 與 EffecientNet 做 Ensemble learning


## 模型表現
因為這個任務是一個二元分類的問題，我們稱瑕疵品為 NG(Positve)，良品為 OK(Negetive)。
| Predict \ True  | NG(Positve)   | OK(Negetive)  |
| -------------   | :-----------: | :-----------: |
| NG(Positive)    | TP            | FP            |
| OK(Negetive)    | FN            | TN            |
  
根據上方的 Confusion matrix 可以計算出準確率與漏檢率為：
- 準確率 = (TP+TN) / (TP+FP+FN+TN)  
- 漏檢率 = FN / (TP+FN)  

在專案介紹中曾提到，漏檢率這個指標對我們較為重要，因此我們將以**低漏檢率**為模型選取的依據，  
下表顯示了不同模型在 Validation 上漏檢率和準確率的表現，漏檢率越接近 0 表示模型表現越佳，而準確率越接近 1 則表示模型表現越佳。


| 模型               | 漏檢率      | 準確率   |
| ----------------- | :--------- | :------ | 
| My_CNN            | 0          | 0.394   | 
| ResNet50          | 0.00685    | 0.883   | 
| EffecientNet      | 0.00913    | 0.934   | 
| Ensemble model    | 0.00114    | 0.854   | 

由於該專案的目標是降低漏檢率，這與一般常用的提高準確率標準稍有不同。我們以往所使用的臨界值通常為 0.5，然而，為了達成更低的漏檢率，我們調整了臨界值，這可能會使得準確率表現變差，但可以達到露檢率降低的目的，這符合我們的目標。  
**NOTE**：如果臨界值設定在 0.5，Validation 上的準確率皆可以在 0.9 以上，最高可到 0.95。

根據表格中的四個模型，雖然 My_CNN 具有最低的漏檢率，但因為它的準確率太低，故將其捨棄。以 Ensemble model 做為用來 Testing 的最終模型！該模型在 Testing 的漏檢率為 0.00112（僅有一張真實為NG被預測成OK），且準確率也有 0.75600，是我們可以接受的範圍！


## 參考資料
李鴻毅2022機器學習CNN：https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php  
Data augmentation：https://pytorch.org/vision/main/transforms.html  
ResNet50：https://pytorch.org/vision/stable/models.html  
EffecientNet_V2_S：https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s  
Transfer Learning：https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html  


## 若有發現任何問題再煩請指教，感謝您的觀看！ 