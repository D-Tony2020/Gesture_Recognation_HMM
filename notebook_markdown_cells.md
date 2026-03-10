# Notebook Markdown Cells — 合并指南

> **使用方法**: 以远程 notebook 为基础，按下面的位置替换/插入对应的 markdown cell 内容。
> 远程 notebook 共 29 cells (cell-0 ~ cell-28)。

---

## R-cell-1 (markdown) — 替换

```
## Loading the training data

Two training sets: one with repeated gestures (same gesture done multiple times in one recording) and one with single gestures.
```

---

## R-cell-3 (markdown) — 替换

```
## Visualize the data

I want to see the raw IMU signals for each gesture to understand the patterns. Plotting one repeated file per gesture.
```

---

## R-cell-5 (markdown) — 替换

```
## Splitting repeated gestures into individual ones

The repeated gesture files contain the same gesture done multiple times. I wanted to split them into individual gestures. So I looked at the signal energy to find the "rest" periods between repetitions.
```

---

## 在 R-cell-10 之前 — 插入新 markdown cell

> 远程 cell-10 是代码 `# hmm so the segmentation...`。在它前面插入一个 markdown cell。

```
Hmm so the segmentation basically gives one big chunk per file...the gestures are done continuously without clear rest periods between reps, the repeated files have maybe 3-6 reps but my method can't split them.
```

---

## R-cell-11 (markdown) — 替换

> 远程原文是 "Plan B"，改为本地风格。注意：远程代码(cell-12)把所有数据放入 training_raw（不拆分），markdown 需匹配。

```
## No split, use cyclic LR-HMM

Splitting the repeated gestures is not really working - the person just keeps moving without pausing between reps.

The HMM will need a cyclic left-to-right structure (last state can transition back to first state).

So my training data will be:
- Repeated gesture files: each file = 1 long sequence with the ~3-6 times repeated gesture
- Single gesture files: each file = 1 short sequence with one gesture

All 6 files per gesture go into training. I'll use leave-one-out CV later to check generalization.
```

---

## R-cell-13 (markdown) — 替换

> 远程代码 cell-14 加了归一化。markdown 需提到这一点。

```
## Vector Quantization with K-means

I need to discretize the continuous 6D IMU vectors into discrete observation labels for the HMM. Using k-means clustering.

But first I noticed the raw accel channels have way higher variance than gyro because of the gravity component (~9.8 m/s²). If I run kmeans directly the clusters would be dominated by accelerometer values. So I normalize first: subtract per-sequence mean (removes gravity bias) then divide by global std (equalizes channel contributions).

After that, k-means with M=70. I tried 70 first and it looked fine, later did a small grid search over [65, 70, 75] to confirm.
```

---

## R-cell-18 (markdown) — 保持不变

```
## HMM Implementation

Time to build the actual HMM. Following Rabiner's paper mostly - the forward/backward algorithms and Baum-Welch for training. I'll use scaling to avoid numerical underflow (section V of the paper).

The plan is:
1. Build an HMM class with forward, backward, and training
2. Test it on some toy data first to make sure it works
3. Then train on the real gesture data
```

---

## R-cell-20 (markdown) — 替换

```
## Testing on toy data

I created a 2-state, 2-observation toy problem:
- State 0 always emits observation 0
- State 1 emits 0 or 1 with equal probability
- The sequence goes: 1000 zeros, then 1000 random {0,1}, then 1000 zeros

Using ergodic topology since the toy data goes state 0 → 1 → 0 (not left-to-right).
```

---

## R-cell-23 (markdown) — 替换

> 远程原文提到 "6 training sequences"。本地更简洁。

```
## Training gesture HMMs

Now let me train one model per gesture. N=15 states, M=70 obs, left-right-cyclic.

After training I'll do leave-one-out cross-validation to check how well the models generalize.
```
