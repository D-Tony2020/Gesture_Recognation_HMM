# Development Log — ECE5242 Project 2: Gesture Recognition (HMM)

> 里程碑开发日志，按时间倒序排列。
> 每条记录关注「发生了什么」「为什么」「下一步」，具体实验数据见 `experiments/` 下的对应表格。

---

<!-- 模板（复制使用）:
## [YYYY-MM-DD] 标题

- **背景**: 做了什么、为什么做
- **关键结论**: 核心发现或成果
- **问题/踩坑**: （如有）遇到的问题及解决方式
- **下一步**: 接下来做什么
- **关联实验**: experiments/exp_xxx.md#ID
-->

## [2026-03-02] 数据探索与向量量化完成

- **背景**: 加载训练数据后，首先需要理解 IMU 信号的形态，然后将连续 6D 向量离散化为 HMM 可用的观测标签。
- **动机**: 直觉上先看 raw signal 是否有明显的手势模式差异 → 有。6 种手势的 gyro/accel 波形视觉上可区分。接下来需要处理一个关键问题：重复手势文件是否需要分割为单个手势。
- **关键决策 — 放弃分割**:
  - 尝试了基于 gyroscope 能量阈值的自动分割（threshold=0.3, smooth window=15）
  - **发现**: 手势执行者在重复之间几乎不停顿，gyro 能量始终在 0.8-2.5 区间波动，无法可靠检测 "静止段"
  - 每个文件基本只得到 1 个大段（1500-3000 样本），而单次手势约 400-800 样本
  - **决定**: 遵循 PDF 建议，不分割重复数据，改用 cyclic Left-to-Right HMM（允许 last state → first state 转移）
  - **数据结构**: 每手势 6 条训练序列（5 repeated 完整文件 + 1 single 文件）
- **K-means 量化 (Q001)**:
  - M=70 clusters, 全量 83451 个 6D 向量, inertia=855511
  - 无空簇, 簇大小 min=272 / max=4899, 分布合理
  - 未做标准化（各通道量级差异不大, gyro ~0-5 rad/s, accel ~0-15 m/s²）
- **问题/踩坑**: 最初用全 6D 能量做分割，但加速度计包含重力分量 (~9.8 m/s²) 导致能量基线很高，改用仅 gyroscope 通道后好了一些但仍不够
- **下一步**: 实现 HMM 核心算法（Forward/Backward/Baum-Welch），先用 toy data 验证
- **关联实验**: experiments/exp_quantization.md#Q001

---

## [2026-03-02] 项目初始化

- **背景**: 搭建项目结构、版本控制、制定开发纲领与实验记录体系
- **关键结论**: 确定 Left-to-Right HMM + k-means 向量量化方案，分 feature 分支开发
- **下一步**: 开始数据探索与向量量化实验 (`feature/data-preprocess`)
