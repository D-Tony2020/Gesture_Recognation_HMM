# ECE5242 Project 2: 手势识别 (HMM) — 完整交接文档

> **目的**: 本文档为现场汇报准备，覆盖项目的每一个细节。假设读者对项目零了解。

---

## 一、项目是什么

用手机/手表的 **IMU 传感器数据**（3轴陀螺仪 + 3轴加速度计 = 6D 向量/时刻）识别 **6 种手臂手势**。

| 手势 | 英文 | 直觉描述 |
|------|------|---------|
| Wave | 挥手 | 左右摆手 |
| Infinity | ∞ | 空中画无穷符号 |
| Eight | 8 | 空中画数字 8 |
| Circle | 圆 | 空中画圈 |
| Beat3 | 三拍 | 三拍节奏拍打 |
| Beat4 | 四拍 | 四拍节奏拍打 |

**方法**: 每种手势训练一个独立的 Hidden Markov Model (HMM)。来了一条新数据，分别算 6 个模型的 log-likelihood，**谁最高判为谁**。

---

## 二、数据长什么样

### 2.1 文件格式

每个文件是纯文本，每行 7 列：
```
时间戳(ms)  Wx  Wy  Wz(rad/s)  Ax  Ay  Az(m/s²)
            ↑ 陀螺仪3轴（角速度）  ↑ 加速度计3轴（含重力~9.8）
```
代码里丢弃时间戳列，只用后面 **6 维向量**。

### 2.2 训练集组成

| 数据集 | 说明 | 每手势文件数 | 每文件长度 |
|-------|------|-----------|----------|
| Train Set 1 (Repeated) | 同一手势重复多次，一个文件里连续做 3-6 遍 | 5 | 1900-3500 样本 |
| Train Set 2 (Single) | 一个文件只做一次手势，格式与测试集相同 | 1 | 400-850 样本 |

**每手势共 6 条训练序列**（5 repeated + 1 single）。

### 2.3 关键决策：不分割重复手势

**我尝试过分割**：用陀螺仪能量（L2 norm of Wx,Wy,Wz）做阈值分割，寻找重复之间的"静止段"。

**结果失败**：做手势的人在重复之间几乎不停，gyro 能量始终在 0.8-2.5 范围波动，无法检测到可靠的切分点。5 个 repeated 文件只分割出 5-6 个大段，而实际应有 15-26 个单独手势。

**解决方案**：遵循项目 PDF 建议，不分割，改用 **cyclic Left-to-Right HMM**（最后一个状态可以转移回第一个状态），让模型自动匹配多次循环。

---

## 三、完整处理流程 (Pipeline)

```
原始 6D IMU 数据 (每行: Wx,Wy,Wz,Ax,Ay,Az)
       ↓
  ① K-means 向量量化 (M=70 个聚类中心)
       ↓
  离散观测序列 (如 [23, 45, 12, 67, 3, ...])
       ↓
  ② 送入 HMM 训练 (Baum-Welch / EM 算法)
       ↓
  6 个训练好的 HMM 模型 (A, B, π 矩阵)
       ↓
  ③ 新数据来了 → 量化 → 算 6 个 log-likelihood → argmax → 预测手势
```

---

## 四、向量量化 (Vector Quantization)

### 4.1 为什么要做

HMM 需要**离散**的观测值。IMU 数据是连续的 6D 向量，必须先离散化。

### 4.2 怎么做

用 **K-means 聚类**把所有训练数据的 6D 向量聚成 **M=70 个簇**。之后每个新来的 6D 向量，找到最近的簇中心，用簇编号（0-69）代替原始向量。

### 4.3 具体参数

| 参数 | 值 | 说明 |
|------|-----|------|
| M (簇数) | 70 | PDF 建议 50-100，取中间偏上 |
| 输入 | 6D 全部通道 | 未做标准化（各通道量级差异不大） |
| 训练数据 | 83,451 个 6D 向量 | 全部 36 条序列合并 |
| sklearn 参数 | random_state=42, n_init=10, max_iter=300 | 标准设置 |
| Inertia | 855,511 | K-means 目标函数值 |
| 空簇 | 0 | 所有簇都有样本 |
| 簇大小 | min=272, max=4899, mean≈1191 | 分布合理 |

### 4.4 如果被问为什么选 70

"PDF 建议 50-100。我试了 70 效果已经很好（LOOCV 97.2%），没有继续调。更大的 M 会让 B 矩阵更稀疏，训练数据可能不够填满每个状态-观测对。"

---

## 五、HMM 理论详解（老师最关注的部分）

### 5.1 HMM 是什么

HMM 有一串你**看不到的隐藏状态**（比如手势的不同阶段），每个状态按某个概率分布**发射一个你能看到的观测**（量化后的 IMU 标签）。

**三个核心参数**:

| 符号 | 维度 | 含义 | 约束 |
|------|------|------|------|
| **A** | (N×N) | 状态转移概率。A[i,j] = P(下一步到状态 j \| 当前在状态 i) | 每行和=1 |
| **B** | (N×M) | 发射概率。B[j,k] = P(发射观测 k \| 在状态 j) | 每行和=1 |
| **π** | (N,) | 初始状态概率。π[i] = P(开始时在状态 i) | 总和=1 |

**在我们的设置中**: N=15 隐状态, M=70 观测符号。

### 5.2 Left-to-Right Cyclic 拓扑

**为什么用 Left-to-Right**：手势是有时间顺序的动作（起手→中间动作→收手），LR 结构天然匹配这种从头到尾的过程。Ergodic（全连接）允许任意跳转，对手势没有物理意义。

```
状态 0 → 状态 1 → 状态 2 → ... → 状态 14
  ↻ 自环   ↻ 自环                      ↓
  ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← (cyclic 回环)
```

**约束条件**:
- **π = [1, 0, 0, ..., 0]**：永远从状态 0 开始，不需要学习
- **A 矩阵**只有三种非零位置：
  - `A[i,i]`：自环（留在当前状态）
  - `A[i,i+1]`：前进（到下一个状态）
  - `A[14,0]`：**cyclic 回环**（最后状态回到第一个）
- 其他所有位置**永远为 0**，通过 mask 在训练中强制执行

**为什么 cyclic**: 因为没分割 repeated 数据，手势在一个文件里重复了多次，需要允许从末态跳回首态来匹配下一次重复。

### 5.3 Forward 算法（计算 log-likelihood）

**作用**: 给定观测序列 O 和模型 λ=(A,B,π)，计算 P(O|λ)。这是分类的依据。

**公式**:
```
初始化: α[i, 0] = π[i] × B[i, o₀]
递推:   α[j, t] = [Σᵢ α[i, t-1] × A[i,j]] × B[j, oₜ]
结果:   P(O|λ) = Σᵢ α[i, T-1]
```

**代码实现**（矩阵化）:
```python
α[:,0] = π * B[:, obs[0]]          # 逐元素乘
α[:,t] = (A.T @ α[:,t-1]) * B[:, obs[t]]  # 矩阵乘 + 逐元素乘
```

**直觉解释**: `A.T @ α[:,t-1]` 的意思是——对每个目标状态 j，从所有可能的来源状态 i 加权汇总（权重是转移概率 A[i,j]），然后乘以在状态 j 发射当前观测的概率。用 A.T 是因为我们用行随机约定 A[i,j] = 从 i 到 j。

### 5.4 Scaling（数值下溢处理）— 老师花了很多时间讲这个

**问题**: α 是概率连乘，序列有几百到几千步，会指数衰减到 0（float64 也装不下）。

**解决**（Rabiner 论文 Section V）:
```
每一步 forward 后:
  c[t] = Σᵢ α[i,t]     (+ 1e-300 防止除零)
  α[:,t] /= c[t]        (归一化到总和=1)

最后 log-likelihood:
  log P(O|λ) = Σₜ log(c[t])
```

**原理**: 相当于把 P(O|λ) 分解成 Π c[t] 的乘积。取 log 后变成求和，避免了下溢。

**为什么 +1e-300**: 防止某一步所有状态概率都是 0 导致除以 0。理论上不该发生，但数值上偶尔会出现。

### 5.5 Backward 算法

```
初始化: β[i, T-1] = 1 / c[T-1]
递推:   β[i, t] = Σⱼ A[i,j] × B[j, o_{t+1}] × β[j, t+1]
        β[:,t] /= c[t]
```

**代码**: `β[:,t] = A @ (B[:, obs[t+1]] * β[:, t+1]) / c[t]`

**Backward 本身不用于分类**。它配合 Forward 一起算 γ 和 ξ，用于 Baum-Welch 训练。

### 5.6 Baum-Welch (EM) 训练 — 核心算法

**目标**: 调整 A 和 B，最大化训练数据的 log-likelihood。

#### E-step（对每条训练序列独立做）

**γ 计算**（状态占有概率）:
```
γ[i,t] = P(时刻 t 在状态 i | 整条序列, 模型)
       = α[i,t] × β[i,t]
       → 按列归一化使每个 t 的 γ 之和 = 1
```

**ξ 计算**（状态转移概率）:
```
ξ[i,j,t] = P(时刻 t 在 i 且 t+1 在 j | 整条序列, 模型)
          = α[i,t] × A[i,j] × B[j, o_{t+1}] × β[j, t+1]
          → 归一化使每个 t 的 ξ 总和 = 1
```

**代码**（向量化，避免三重循环）:
```python
# 原始三重循环版本（太慢）:
# for t in range(T-1):
#     for i in range(N):
#         for j in range(N):
#             xi[i,j,t] = alpha[i,t]*A[i,j]*B[j,obs[t+1]]*beta[j,t+1]

# 优化版（逐 t 步计算，累加统计量）:
for t in range(T-1):
    bj_beta = B[:, obs[t+1]] * beta[:, t+1]                    # (N,)
    xi_t = alpha[:, t:t+1] * A * bj_beta[np.newaxis, :]        # (N,N)
    xi_t /= xi_t.sum() + 1e-300
    xi_sum += xi_t  # 直接累加，不存储完整 (N,N,T-1) 数组
```

#### 多序列训练

6 条训练序列各自独立做 E-step，然后把 γ 和 ξ 的统计量**累加**，统一做 M-step。

```python
for obs in obs_sequences:     # 对每条序列
    alpha, c = forward(obs)
    beta = backward(obs, c)
    gamma = alpha * beta       # → 按列归一化
    xi_sum += ...              # 累加 ξ
    B_numer[:, obs[t]] += gamma[:, t]  # 累加 B 的分子
```

#### M-step（统一更新）

```python
# 更新 A
A_new[i,j] = Σₜ ξ[i,j,t] / Σₜ γ[i,t]     (t 从 0 到 T-2)
A_new *= A_mask                               # 强制 LR 拓扑
A_new /= A_new.sum(axis=1, keepdims=True)     # 重新归一化

# 更新 B
B_new[j,k] = Σ_{t: oₜ=k} γ[j,t] / Σₜ γ[j,t]
B_new += 1e-8                                  # 防零概率
B_new /= B_new.sum(axis=1, keepdims=True)      # 重新归一化
```

#### 收敛判断

- 每个 epoch 计算总 LL = 所有序列的 LL 之和
- **LL 必须单调不降**（下降 = 实现有 bug）
- LL 变化 < tol=0.01 时认为收敛，或最多 200 个 epoch

### 5.7 EM 局部最优问题

**现象**: seed=42 在 toy data 上陷入局部最优（两行 B 几乎一样，LL=-1383 而非最优 -707）。

**解决**: 尝试多个随机种子，选 LL 最高的。toy data 验证用 seed=4 收敛正确。

---

## 六、Toy Data 验证（证明算法正确）

**这是老师要求的 sanity check，必须能解释。**

### 6.1 构造

```python
seg1 = [0,0,...,0]     × 1000  # 状态 0: 只发射 obs=0
seg2 = [随机{0,1}]    × 1000  # 状态 1: 50% obs=0, 50% obs=1
seg3 = [0,0,...,0]     × 1000  # 状态 0: 只发射 obs=0
toy_obs = 拼接(seg1, seg2, seg3)  # 总长 3000
```

### 6.2 预期结果

```
A ≈ [[0.999, 0.001],   # 状态 0 几乎不跳出
     [0.001, 0.999]]   # 状态 1 几乎不跳出

B ≈ [[1.0, 0.0],       # 状态 0 只发射 obs=0
     [0.5, 0.5]]       # 状态 1 均匀发射 0/1
```

### 6.3 实际结果

- **seed=4, ergodic topology, N=2, M=2**
- 收敛: 19 epoch, LL = -707.50
- A = [[0.999, 0.001], [0.001, 0.999]] ✓
- B = [[1.0, 0.0], [0.48, 0.52]] ✓
- LL 严格单调不降 ✓

**注意**: toy data 必须用 **ergodic**（全连接）HMM，因为状态 0→1→0 有"回跳"，left-right 不允许。

---

## 七、训练结果

### 7.1 超参数总结

| 参数 | 值 | 说明 |
|------|-----|------|
| N (隐状态数) | 15 | 每个状态平均覆盖 30-50 样本点 |
| M (观测簇数) | 70 | K-means 聚类中心数 |
| topology | left-right-cyclic | 允许末态→首态 |
| seed | 42 | 每个手势统一 |
| max_iter | 200 | 最多 200 个 EM epoch |
| tol | 0.01 | LL 变化 < 0.01 停止 |
| B ε | 1e-8 | 防零概率平滑 |

### 7.2 各手势训练情况

| 手势 | 收敛 epoch | 最终 LL | 状态 |
|------|-----------|---------|------|
| wave | 109 | -10,451 | 收敛 |
| inf | 108 | -13,628 | 收敛 |
| eight | 45 | -13,912 | 收敛 |
| circle | 84 | -6,787 | 收敛 |
| beat3 | 85 | -10,073 | 收敛 |
| beat4 | 106 | -14,102 | 收敛 |

- **LL 全部单调不降** ✓ — 算法实现正确
- **B 行和 = 1.0** ✓ — 归一化正确
- LL 训练曲线图: `docs/figures/ll_curves_all.png`

---

## 八、LOOCV 验证结果（核心评估指标）

### 8.1 方法

6-fold Leave-One-Out Cross-Validation:
- 每手势 6 条序列，每折留出 1 条做验证
- 用剩余 5 条重新训练该手势的 HMM
- 留出的序列对 6 个手势模型计算 LL，取 argmax 分类
- 其他 5 个手势用全量训练的模型（不需要重训）

### 8.2 总体结果

```
Top-1 准确率: 35/36 = 97.2%
Top-3 准确率: 36/36 = 100.0%
```

### 8.3 每一折详细结果

| 手势 | rep1 | rep2 | rep3 | rep4 | rep5 | single | 合计 |
|------|------|------|------|------|------|--------|------|
| wave | ✓ wave | ✓ wave | ✓ wave | ✓ wave | ✓ wave | ✓ wave | 6/6 |
| inf | ✓ inf | ✓ inf | ✓ inf | ✓ inf | ✓ inf | ✓ inf | 6/6 |
| eight | ✓ eight | ✓ eight | ✓ eight | ✓ eight | ✓ eight | ✓ eight | 6/6 |
| circle | ✓ circle | ✓ circle | ✓ circle | ✓ circle | ✓ circle | ✓ circle | 6/6 |
| beat3 | ✓ beat3 | ✓ beat3 | ✓ beat3 | **✗ beat4** | ✓ beat3 | ✓ beat3 | **5/6** |
| beat4 | ✓ beat4 | ✓ beat4 | ✓ beat4 | ✓ beat4 | ✓ beat4 | ✓ beat4 | 6/6 |

### 8.4 唯一错误分析

- **beat3 的 rep4**（长 2865 样本）被误判为 beat4
- Top-3 为 [beat4, **beat3**, inf] — beat3 排第二，不丢 Top-3 分数
- beat3↔beat4 是最易混淆的手势对（两者都是拍打类，信号模式最接近）
- 混淆矩阵图: `docs/figures/loocv_confusion.png`

### 8.5 之前的验证 (E001) 为什么作废

最初验证: 用 single 文件分类，得 6/6=100%。但 **single 文件同时参与了训练**，存在数据泄漏。100% 的结果无意义，因此实施了 LOOCV 作为真正的泛化评估。

---

## 九、分类流程（测试时怎么跑）

```python
# 1. 加载测试文件
raw = np.loadtxt(test_file)
imu = raw[:, 1:]              # 丢掉时间戳，得到 (T, 6)

# 2. 向量量化
obs = kmeans.predict(imu)      # 得到离散序列 [23, 45, 12, ...]

# 3. 对 6 个 HMM 分别跑 Forward 算法
for gesture in ['wave', 'inf', 'eight', 'circle', 'beat3', 'beat4']:
    ll = forward_log_likelihood(A, B, pi, obs)

# 4. 取 LL 最大的 → 预测结果
# 5. 输出 Top-3 预测及 LL 值
```

### 运行命令

```bash
python test_classifier.py
# 修改顶部 TEST_DIR 变量指向测试数据路径即可
```

### 输出格式
```
File                          Top-1      Top-2      Top-3          LL-1         LL-2         LL-3
------------------------------------------------------------------------------------------------
test_001.txt                  wave       beat3      inf          -474.37     -3674.98     -4266.72
```

---

## 十、LL 值怎么解读（老师会问）

### 10.1 为什么是负数且很大

因为是 **log-likelihood**。概率 P(O|λ) 本身是极小的正数（几百步概率连乘），取 log 后就是很大的负数。

### 10.2 数值本身有意义吗

**没有绝对意义**。只有不同模型之间的**相对比较**有意义——越大（越接近 0）说明模型越匹配。

### 10.3 典型值范围

| 类型 | LL 范围 | 说明 |
|------|---------|------|
| 正确模型 | -300 ~ -700 | 模型很匹配 |
| 第二名（相似手势） | -600 ~ -4000 | 有一定匹配但不如正确的 |
| 其他手势 | -4000 ~ -10000 | 明显不匹配 |

### 10.4 分离度

- 大部分手势: 正确模型 LL 与第二名差距 > 3000（非常清晰的分离）
- beat3 vs beat4: 差距约 180-770（最小，但通常仍能正确区分）

---

## 十一、代码文件说明

```
gesture_hmm.ipynb      ← 主 notebook (29 cells)，包含全部探索/训练/验证
test_classifier.py     ← 独立推理脚本 (~97 行)
run_training.py        ← 分批训练脚本（LOOCV 用）

models/
  kmeans_model.pkl     ← K-means 模型 (330KB)
  wave_hmm.pkl         ← 各手势 HMM 参数 {A, B, π, N, M} (~11KB each)
  inf_hmm.pkl
  eight_hmm.pkl
  circle_hmm.pkl
  beat3_hmm.pkl
  beat4_hmm.pkl
  training_obs.pkl     ← 量化后训练数据
  training_raw.pkl     ← 原始训练数据
  checkpoints/         ← LOOCV 中间结果

data/
  Repeated_gesture/    ← 训练集 1 (每手势 5 文件)
  Single_gesture/      ← 训练集 2 (每手势 1 文件)
  Test_gesture/        ← 测试集（3/4 发布后放这里）

docs/
  figures/
    ll_curves_all.png     ← 6 个手势的 LL 训练曲线
    loocv_confusion.png   ← LOOCV 混淆矩阵
```

---

## 十二、Notebook 各 Cell 索引

| Cell | 类型 | 内容 |
|------|------|------|
| 0 | code | imports (numpy, matplotlib, sklearn, etc.) |
| 1 | md | "Loading the training data" |
| 2 | code | 加载 Repeated + Single 数据 |
| 3 | md | "Let me visualize..." |
| 4 | code | 6 个手势的原始 IMU 信号可视化 |
| 5 | md | "Splitting repeated gestures..." |
| 6 | code | 用 L2 能量做分割尝试 |
| 7 | code | 改用 gyro 能量 |
| 8 | code | segment_gesture() 函数 |
| 9 | code | 可视化分割结果 |
| 10 | code | 确认分割失败 |
| 11 | md | "Plan B: cyclic LR-HMM" |
| 12 | code | 合并为训练数据（每手势 6 序列） |
| 13 | md | "Vector Quantization with K-means" |
| 14 | code | 堆叠所有向量 |
| 15 | code | K-means 拟合 (M=70) + 簇大小可视化 |
| 16 | code | 量化所有序列 |
| 17 | code | 保存 kmeans + 训练数据 |
| 18 | md | "HMM Implementation" |
| 19 | code | **HMM class 完整实现 (~110 行)** |
| 20 | md | "Testing on toy data" |
| 21 | code | Toy data 生成 + 训练 (seed=4, ergodic) |
| 22 | code | Toy data 结果验证 + LL 曲线 |
| 23 | md | "Training gesture HMMs" + LOOCV 说明 |
| 24 | code | 训练 6 个手势 HMM (seed=42, max_iter=200) |
| 25 | code | LL 训练曲线图 (2×3 subplot) |
| 26 | code | **LOOCV 代码** (36 折交叉验证) |
| 27 | code | **混淆矩阵可视化** |
| 28 | code | 保存模型 |

---

## 十三、可能被问到的问题 & 标准回答

### 算法相关

**Q: 为什么用 Left-to-Right 而不是 Ergodic HMM？**
> 手势是有时间顺序的动作（起手→动作→收手），Left-to-Right 天然匹配这种从头到尾的结构。Ergodic 允许任意状态间跳转，对手势来说没有物理意义。老师课上也重点推荐了 LR 结构。

**Q: 为什么用离散 HMM 而不是连续 HMM（Gaussian emission）？**
> 老师课上 strongly suggest 用向量量化 + 离散 HMM，PDF 也明确建议。离散 HMM 实现简单，Baum-Welch 更稳定。

**Q: Forward 和 Backward 为什么要验证一致性？**
> Forward 算 P(O|λ) = Σ α[i,T]，Backward 也能算 P(O|λ) = Σ π[i] × B[i,o₀] × β[i,0]。两者必须相等，否则实现有 bug。我的实现已验证一致。

**Q: LL 为什么必须单调不降？**
> EM 算法的理论保证。每次 E-step + M-step 要么增大 LL，要么保持不变。如果 LL 下降了，100% 是代码写错了。

**Q: A.T @ α 是什么意思？**
> 对每个目标状态 j，汇总所有来源状态: Σᵢ A[i,j] × α[i,t]。用 A.T 转置是因为我们用行随机约定（A[i,j] = 从 i 到 j），要按列 j 汇总就需要转置。

### 参数选择

**Q: 为什么 15 个隐状态？**
> 单次手势约 400-800 采样点，15 个状态意味着每个状态平均覆盖 30-50 个点，粒度合理。太少（5）捕不到细节，太多（30）训练数据不够估计参数。PDF 建议 10-20。

**Q: 为什么不做超参数搜索？**
> 时间有限，N=15 M=70 的 LOOCV 已经 97.2%，表现足够好。如果需要进一步优化可以做 N 和 M 的网格搜索。

**Q: 为什么给 B 加 1e-8？**
> 如果训练时某个观测从未在某个状态出现，B 矩阵该位置是 0。测试时如果遇到这个观测，整条序列概率直接变 0，无法比较。加一个极小值让它变成"极不可能"而不是"不可能"。老师课上专门强调了这点。

### 结果解读

**Q: beat3 和 beat4 为什么容易混淆？**
> 两者都是拍打类手势，信号模式最相似。LOOCV 中唯一的错误就是 beat3 被误判为 beat4，但 Top-3 里 beat3 排第二，说明模型仍然捕捉到了一定的区分度。

**Q: 为什么不分割 repeated 手势？**
> 我试过了。用陀螺仪能量做阈值分割，但做手势的人在重复之间几乎不停，能量一直在波动，没有明显的"静止段"可以切分。Notebook 里 cell 6-10 有完整的分割尝试和失败过程的可视化。

**Q: LOOCV 和直接验证有什么区别？**
> 直接验证用 single 文件，但那个文件也参与了训练，所以 100% 的结果是假的（数据泄漏）。LOOCV 每折都把验证数据从训练集中排除，得到的 97.2% 才是真实的泛化性能。

---

## 十四、关键公式速查

**Forward**: `α[:,t] = (A.T @ α[:,t-1]) × B[:, oₜ]`

**Backward**: `β[:,t] = A @ (B[:, o_{t+1}] × β[:,t+1])`

**Scaling**: `c[t] = Σ α[:,t]; log P = Σ log(c[t])`

**γ**: `γ = α × β → 按列归一化`

**ξ**: `ξ[i,j,t] = α[i,t] × A[i,j] × B[j,o_{t+1}] × β[j,t+1] → 归一化`

**A 更新**: `A[i,j] = Σₜ ξ[i,j,t] / Σₜ γ[i,t]`

**B 更新**: `B[j,k] = Σ_{t:oₜ=k} γ[j,t] / Σₜ γ[j,t]`

---

## 十五、参考文献

- **Rabiner, L. R. (1989)**. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition". *IEEE Proceedings*, 77(2), 257–286. — 特别是 **Section V** (Scaling)
- **标注文档**: https://dl.icdst.org/pdfs/files3/93a4bc705f6afe148018dcfee66a9217.pdf
- **课程脉络**: Lec07 Kalman → Lec08-09 HMM; HMM = GMM + 时序马尔可夫链
