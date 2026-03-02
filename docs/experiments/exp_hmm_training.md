# HMM 训练实验记录

> 记录每次 HMM 训练的超参数组合、收敛行为与模型质量。
> 每次实验分配唯一 ID (H###)，供 DEVLOG 和评估实验表交叉引用。

## 实验表

| ID | 日期 | 手势 | N (状态数) | M (观测数) | 量化ID | HMM结构 | 初始化策略 | 收敛轮次 | 最终LL | LL单调? | 备注 |
|----|------|------|-----------|-----------|--------|---------|-----------|---------|--------|---------|------|
| H001 | 2026-03-02 | wave | 15 | 70 | Q001 | Left-Right-Cyclic | dirichlet+随机A, seed=42 | 100 (未收敛) | -10452.24 | Yes | 基线配置 |
| H002 | 2026-03-02 | inf | 15 | 70 | Q001 | Left-Right-Cyclic | dirichlet+随机A, seed=42 | 100 (未收敛) | -13629.38 | Yes | 基线配置 |
| H003 | 2026-03-02 | eight | 15 | 70 | Q001 | Left-Right-Cyclic | dirichlet+随机A, seed=42 | 45 | -13912.13 | Yes | 收敛 tol=0.01 |
| H004 | 2026-03-02 | circle | 15 | 70 | Q001 | Left-Right-Cyclic | dirichlet+随机A, seed=42 | 84 | -6787.27 | Yes | 收敛 tol=0.01 |
| H005 | 2026-03-02 | beat3 | 15 | 70 | Q001 | Left-Right-Cyclic | dirichlet+随机A, seed=42 | 85 | -10073.10 | Yes | 收敛 tol=0.01 |
| H006 | 2026-03-02 | beat4 | 15 | 70 | Q001 | Left-Right-Cyclic | dirichlet+随机A, seed=42 | 100 (未收敛) | -14102.24 | Yes | 基线配置 |

## 字段说明

- **量化ID**: 对应 `exp_quantization.md` 中的 Q### ID
- **HMM结构**: Left-to-Right / Left-to-Right-Cyclic / Ergodic
- **初始化策略**: 均匀、随机、自定义（描述具体方式）
- **收敛轮次**: Baum-Welch 迭代至收敛的 epoch 数
- **最终LL**: 收敛后的 log-likelihood 值
- **LL单调?**: Yes/No — 训练过程中 LL 是否保持单调不降（No 意味着可能有 bug）

## 训练曲线存档

> 每个关键实验的 log-likelihood per epoch 曲线图保存至 `../figures/`，
> 文件命名: `ll_curve_{ID}.png`（如 `ll_curve_H001.png`）
