# 深度強化學習 - HW3 
本專案為深度強化學習的第三次作業 (HW3) 實作，包含了解決傳統 Q-learning 容易產生的「高估問題」的 Double DQN，以及能夠更精準評估狀態價值的 Dueling DQN 架構。整體訓練流程已採用 PyTorch Lightning 進行重構，以獲得更穩定與工程化的訓練體驗。

## 執行環境與如何運行

### 環境依賴
* Python 3.8+
* PyTorch
* PyTorch Lightning
* NumPy
* Matplotlib

您可以直接透過以下指令安裝缺少的依賴套件：
```bash
pip install torch numpy matplotlib pytorch-lightning
```

### 如何執行代碼
我們的訓練腳本支援三種 GridWorld 模式：`static`、`player` 以及 `random`。
您可以透過命令列參數 `--mode` 指定訓練的模式，以及透過 `--steps` 指定要互動的環境總步數。

1. **Static 模式訓練 (預設)**：
```bash
python train.py --mode static --steps 20000
```
2. **Player 模式訓練**：
```bash
python train.py --mode player --steps 20000
```
3. **Random 模式訓練**：
```bash
python train.py --mode random --steps 50000
```

訓練結束後，程式會自動：
1. 儲存模型權重為 `dqn_model_[mode]_pl.pth`。
2. 產生並儲存該模式的 Reward 變化圖（包含原始 Reward 與 50 個 Episode 的移動平均線）至 `reward_plot_[mode]_pl.png`。

---

## 演算法理論與實作說明

本專案將標準的 DQN 升級為 **Double Dueling DQN**，具體解決了傳統 DQN 的幾項痛點：

### 1. 傳統 Q-learning 的高估問題 (Overestimation Bias) 與 Double DQN
**問題描述：** 
傳統 DQN 在計算 TD Target 時，會使用**相同的網路**來選擇下一步最佳動作與評估該動作的 Q-Value：
$Y = R + \gamma \max_a Q(s', a; \theta)$
由於 Q-Value 中包含了神經網路估計的誤差，當使用 $\max$ 操作時，這些隨機誤差往往會被往正向放大。隨著迭代進行，這會導致模型出現嚴重的高估現象（Overestimation），進而學習到次優的策略。

**Double DQN 解決方案：**
將「選擇動作」與「評估價值」解耦（Decoupling）。我們實作了兩個網路：`Main Network` 與 `Target Network`。
* **動作選擇**：交由 `Main Network` 決定（找到具有最大 Q-Value 的動作）。
* **價值評估**：利用 `Target Network` 計算出該動作的 Q-Value。
$Y = R + \gamma Q_{target}(s', \arg\max_a Q_{main}(s', a; \theta_{main}); \theta_{target})$
這樣的解耦能大幅度減少高估的風險，讓模型的收斂更為穩定。

### 2. 狀態價值評估與 Dueling DQN
**問題描述：**
傳統的 DQN 只有一個流（Stream）來輸出所有可能動作的 $Q(s,a)$。但在許多狀態下，不管採取何種行動對未來的收益影響並不大（例如遠離陷阱的安全狀態）。要求模型為每一個 $(s,a)$ pair 精準預估是不具效率的。

**Dueling DQN 解決方案：**
我們在 `model.py` 中將神經網路的最後一層拆分為兩個流（Streams）：
1. **Value Stream $V(s)$**：單獨評估處於狀態 $s$ 有多好。
2. **Advantage Stream $A(s,a)$**：評估在狀態 $s$ 下採取行動 $a$ 的相對優勢。

最後透過聚合層結合這兩個流：
$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s, a') \right)$
減去平均值的操作確保了 $A$ 的均值為 0，這解決了 $V$ 與 $A$ 的不可識別性（Unidentifiability）問題，讓模型能夠獨立且更精準地學習到狀態的基底價值。

### 3. 穩定性優化 (Stability)
* **PyTorch Lightning 重構**：將 Agent 與訓練邏輯封裝在 `pl.LightningModule` 中，標準化 `training_step` 的過程。
* **Gradient Clipping**：在 `Trainer` 中設定 `gradient_clip_val=1.0`，有效防止梯度爆炸，確保 Loss 下降更平穩。
* **Epsilon-greedy 隨機率遞減策略**：探索率 $\epsilon$ 從 1.0 開始，隨著訓練的 step 線性衰減至 0.1，使 Agent 前期能充分探索環境，後期則偏向利用已知最優策略。

---

## 實驗結果分析

根據我們在測試環境中收集到的實驗結果，以下是針對不同架構在 `player` 模式下的表現比較：

### 1. Standard DQN 的表現
在初期的實驗中，標準 DQN 雖然能夠學會抵達目標，但在 `player` 模式下，由於玩家的初始位置與陷阱（Pitfall）的分佈影響，容易陷入高估某幾個特定危險邊緣狀態的 Q-Value。這導致模型即使學到了策略，Reward 的震盪依然很大，時常在收斂後又「忘記」正確策略而掉入陷阱，學習曲線呈現非常不穩定的劇烈抖動。

### 2. Double DQN 的表現
加入 Target Network 並解耦之後（Double DQN），模型的穩定性有了顯著的提升。高估問題（Overestimation）被大幅壓制。從 Reward 圖中可以看到，Double DQN 相較於標準 DQN，達到平均正向收益所需的 Episode 數量更少，且收斂後的突發性斷崖式失敗大幅減少。它能夠確實分辨出通往目標的安全路徑。

### 3. Double Dueling DQN 的表現 (最佳表現)
將網路結構升級為 Dueling Architecture 是提升表現的關鍵一步。在 GridWorld 中，大部分的安全狀態無論採取上下左右都不會立即致死（除非走進陷阱）。Dueling DQN 的 Value Stream $V(s)$ 可以很有效地學習到「當前狀態的安全性」，而 Advantage Stream 則專注於「哪一步能更快靠近目標」。

**綜合結論**：
結合了 Double DQN 與 Dueling DQN 後的智能體，在 `player` 模式下的收斂速度最快，Reward 曲線的 50-epoch 移動平均線也是最快且最平滑地爬升至接近滿分（+10）的水準。配合 Gradient Clipping，進一步削弱了 TD Error 在面臨大額懲罰時所產生的過大梯度，使整個神經網路在訓練過程中的容錯率達到最高。
