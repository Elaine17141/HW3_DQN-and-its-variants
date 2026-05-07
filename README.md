# 深度強化學習 - HW3: DQN and its Variants

本專案為深度強化學習第三次作業 (HW3) 實作。基於 [Deep Reinforcement Learning in Action](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master) 提供的 GridWorld 環境，逐步實現從最基礎的 Naive DQN 到各種改良架構（Double DQN, Dueling DQN），並全面使用 **PyTorch Lightning** 進行現代化與工程化的重構，以完成隨機環境下的高難度訓練。

---

## 🛠️ 執行環境與如何運行 (Setup & How to Run)

### 依賴套件 (Dependencies)
本專案採用 Python 3.8+ 與 PyTorch Lightning 開發，請確保您安裝了以下套件：
```bash
pip install torch numpy matplotlib pytorch-lightning
```

### 執行方式 (How to Run)
訓練腳本 `train.py` 已經高度封裝，您可以透過改變命令列參數來自由組合「演算法」與「環境難度」。

**參數說明：**
* `--mode`: 選擇 GridWorld 模式 (`static`, `player`, `random`)
* `--algo`: 選擇演算法 (`dqn`, `double_dqn`, `dueling_dqn`, `double_dueling_dqn`)
* `--steps`: 設定與環境互動的總訓練步數 (預設 `20000`)

**執行範例：**
```bash
# HW3-1: 執行基礎 DQN 於 static 模式
python train.py --mode static --algo dqn --steps 10000

# HW3-2: 執行 Double DQN 於 player 模式
python train.py --mode player --algo double_dqn --steps 20000

# HW3-3: 執行結合所有技巧的最佳演算法於 random 模式
python train.py --mode random --algo double_dueling_dqn --steps 50000
```
訓練完成後，程式將自動把模型權重存為 `model_[algo]_[mode]_pl.pth`，並繪製出 Reward 變化趨勢圖保存為 `reward_plot_[algo]_[mode]_pl.png`。

---

## 🧠 HW3-1: Naive DQN for Static Mode [Understanding Report]

### 基本原理與實作
在最基礎的 `static` 模式中，陷阱與目標的位置是固定的，這是一個較為簡單的環境。我們在此實作了標準的 **Naive DQN** 以及 **Experience Replay Buffer**。
* **Naive DQN**：使用單一的 Q-Network 來預測每個動作的未來價值（Q-Value）。在更新時，利用同一個網路找出最高 Q-Value 的動作來計算 TD Target。
* **Experience Replay Buffer (經驗回放緩衝區)**：如果 Agent 只學習最新的一筆經驗，資料會具有高度的時間相關性，導致神經網路學習不穩定（Catastrophic Forgetting）。因此我們實作了 `ReplayBuffer` 類別，讓 Agent 先將走過的經驗 $(s, a, r, s', done)$ 存入 Buffer 中，訓練時再隨機抽樣 (Sample) 出一個 Batch 進行梯度下降，有效打破資料的相關性並提升資料利用率。

---

## ⚖️ HW3-2: Enhanced DQN Variants for Player Mode 

當環境提升至 `player` 模式時，玩家的初始位置與目標會產生變化，傳統 DQN 開始暴露其缺陷。我們為此實作並比較了兩種改良架構。

### 1. Double DQN 
* **改良核心**：解決 Q-learning 的 **高估問題 (Overestimation Bias)**。
* **原理**：傳統 DQN 在計算 TD Target: $Y = R + \gamma \max_a Q(s', a; \theta)$ 時，總是樂觀地選擇最大值。因為神經網路本來就存在估計誤差，這種操作會使正向誤差被不斷放大，導致 Q 值的嚴重高估。Double DQN 引入了 `Target Network` 將「動作選擇」與「價值評估」解耦：
  * **Main Network** 負責選出下個狀態的最好動作：$a^* = \arg\max_a Q_{main}(s', a)$
  * **Target Network** 負責評估該動作的真實價值：$Q_{target}(s', a^*)$
* **結果表現**：在 `player` 模式中，Double DQN 使得 Reward 曲線大幅穩定，不再發生原本 DQN 在收斂後突然因為過度高估某些危險狀態的 Q 值而崩潰掉入陷阱的情況。

### 2. Dueling DQN
* **改良核心**：更精準的 **狀態價值評估**。
* **原理**：在 GridWorld 許多安全狀態中，無論採取何種動作，都不會立刻產生巨大的風險或獎勵。傳統 DQN 浪費算力去精準估計每一個動作的 Q 值。Dueling DQN 將神經網路的最後一層拆分為兩個流（Streams）：
  1. **Value Stream $V(s)$**：評估「單純待在狀態 $s$ 有多安全/多好」。
  2. **Advantage Stream $A(s,a)$**：評估「在狀態 $s$ 下，採取動作 $a$ 比其他動作好多少」。
  我們透過公式 $Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s, a') \right)$ 來聚合兩者，減去均值確保了網路的穩定訓練（解決不可識別性問題）。
* **結果表現**：在 `player` 模式下，Dueling DQN 的收斂速度明顯快於普通 DQN。因為它能夠獨立學習到「遠離坑洞」的基礎狀態價值 $V(s)$，使智能體即使在未嘗試過所有動作的情況下，也能迅速辨識危險。

---

## 🔁 HW3-3: Enhance DQN for Random Mode WITH Training Tips

在最高難度的 `random` 模式中，每一次 Episode 不僅起點改變，連坑洞與目標點也會隨機重置。為了應對如此複雜的環境，我們做出了架構上的徹底升級。

### 1. 導入 PyTorch Lightning 框架
我們摒棄了傳統冗長的 `while` 迴圈，將神經網路、環境互動、Optimizer 與 Target Network 的更新邏輯全部整合進 `pl.LightningModule`。透過定義明確的 `training_step`，程式碼不僅易讀性大增，也減少了人為 Bug，更具備現代工程化的擴展性。

### 2. Training Tips (穩定學習技巧)
* **Gradient Clipping (梯度裁剪)**：在隨機模式中，碰到隨機生成的死局陷阱時會產生巨大的 TD Error 負回饋。我們在 Lightning Trainer 中設置了 `gradient_clip_val=1.0`，強制限制梯度的最大範數，避免產生梯度爆炸導致模型崩潰。
* **Epsilon Decay ($\epsilon$-greedy 衰減策略)**：將探索率 $\epsilon$ 的衰減封裝在訓練的每個 Step 中。設定 $\epsilon$ 從 1.0 開始，隨著訓練的 step 線性衰減至 0.1。這保證了模型在隨機環境的初期有充分的勇氣去探索每種可能的地圖組合，而在後期則專注於收斂並利用最優策略。

**總結**：結合 Double Dueling DQN 與上述 Training Tips，本專案完美克服了 `random` 模式的環境動態性，展現出高度的收斂穩定性。
