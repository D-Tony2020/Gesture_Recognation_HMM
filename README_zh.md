# ECE5242 项目2：基于HMM的手势识别

利用隐马尔可夫模型（HMM）对 IMU 传感器数据进行手势识别，可分类 6 种手势（Wave、Infinity、Eight、Circle、Beat3、Beat4），数据来源为三轴陀螺仪 + 三轴加速度计。

## 环境要求

- Python 3.8+
- numpy, scipy, scikit-learn, matplotlib

安装依赖：
```bash
pip install numpy scipy scikit-learn matplotlib
```

## 项目结构

```
├── gesture_hmm.ipynb      # 主 notebook（数据探索、训练、验证）
├── test_classifier.py     # 独立推理脚本（用于测试数据）
├── run_training.py        # 批量训练工具（种子搜索、LOOCV）
├── models/
│   ├── kmeans_model.pkl   # K-means 模型（M=70 个聚类，6D IMU 向量）
│   ├── wave_hmm.pkl       # 各手势的 HMM 模型（A、B、π 矩阵）
│   ├── inf_hmm.pkl
│   ├── eight_hmm.pkl
│   ├── circle_hmm.pkl
│   ├── beat3_hmm.pkl
│   └── beat4_hmm.pkl
├── docs/
│   ├── DEVLOG.md          # 开发日志
│   ├── experiments/       # 实验记录（量化、训练、评估）
│   └── figures/           # 训练曲线、混淆矩阵图
└── data/                  # 训练和测试数据（未包含在仓库中）
    ├── train/             # 训练集1：重复手势
    ├── train_single/      # 训练集2：单次手势
    └── Test_gesture/      # 测试集
```

## 快速推理（无需重新训练）

1. 将测试文件（`.txt`，7列：时间戳 + 6D IMU）放入 `data/Test_gesture/`

2. 运行分类器：
```bash
python test_classifier.py
```

3. 输出每个测试文件的 Top-3 预测及对数似然值：
```
File                           Top-1      Top-2      Top-3          LL-1         LL-2         LL-3
------------------------------------------------------------------------------------------------
test_001.txt                   wave       inf        eight       -1234.56     -2345.67     -3456.78
...
```

如果测试文件在其他目录，修改 `test_classifier.py` 顶部的 `TEST_DIR` 即可。

## 流程概览

1. **矢量量化**：原始 6D IMU 向量 → K-means（70 个聚类）→ 离散观测标签
2. **HMM 分类**：每种手势有一个训练好的左右循环 HMM（N=15 个状态，M=70 个观测符号）。对测试序列，计算每个模型下的对数似然 → 选最高的作为预测结果。

## 训练（可选）

notebook `gesture_hmm.ipynb` 包含完整流程：
- 数据加载与可视化
- K-means 矢量量化（M=70）
- HMM 实现（Forward/Backward/Baum-Welch，使用 Rabiner 缩放）
- 训练 6 个手势模型（N=15，seed=42，max_iter=200）
- 留一交叉验证：**97.2% Top-1 准确率**（35/36）

如需从头训练，运行 notebook 所有 cell 即可。预训练模型已保存在 `models/` 中。

## 验证结果

| 指标 | 数值 |
|------|------|
| LOOCV Top-1 准确率 | 35/36 = 97.2% |
| LOOCV Top-3 准确率 | 36/36 = 100% |
| 唯一误分类 | beat3 rep4 → beat4 |

## 数据格式

每个输入文件包含 7 列（空格分隔）：
```
时间戳(ms)  Wx  Wy  Wz(rad/s)  Ax  Ay  Az(m/s²)
```
处理时丢弃时间戳列，仅使用 6 个 IMU 通道。
