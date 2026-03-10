# ECE5242 Project 2: Gesture Recognition with Hidden Markov Models

---

## 1. Introduction

The goal of this project is to recognize 6 types of hand gestures — Wave, Infinity, Eight, Circle, Beat3, Beat4 — from IMU sensor data (3-axis gyroscope + 3-axis accelerometer, so 6D per timestep).

The basic idea is to train one HMM per gesture, and at test time, feed the unknown sequence into all 6 models, compute the log-likelihood for each, and pick the highest one. The whole pipeline has four parts: data preprocessing + vector quantization, HMM algorithm implementation, training + cross-validation, and inference.

Final result on Leave-One-Out cross-validation: **97.2% Top-1 accuracy** and **100% Top-3 accuracy**. The only mistake was between beat3 and beat4, which makes sense since they look really similar.

---

## 2. Understanding the Data and the First Big Decision: To Segment or Not

### 2.1 Data Layout

There are two sets of training data:

- **Repeated gesture**: each file has the same gesture repeated 3-6 times back to back, so the sequences are pretty long (2000-3400 timesteps)
- **Single gesture**: one gesture per file, around 400-850 timesteps, same format as the test data

Each gesture type has 5 repeated files and 1 single file. Each timestep is a 7-dim vector (timestamp + 6D IMU). I drop the timestamp column and just keep the 6 sensor channels.

### 2.2 Looking at the Signals

First thing I did was plot the raw IMU signals for all 6 gestures (Figure 1). The patterns are pretty different visually — wave has this back-and-forth oscillation, circle is smooth and periodic, beat3/beat4 have sharp pulse-like patterns. So yeah, the 6D data clearly carries enough info to tell them apart. HMM should work.

> **Figure 1**: Raw IMU signals for the 6 gesture types (one repeated file each, all 6 channels overlaid)

### 2.3 Trying to Segment — and Failing

Since the repeated files have multiple gestures in them, the obvious idea is to chop them up into individual ones before training. I tried energy-based segmentation:

1. **Full 6D energy** (L2 norm of all channels): The baseline energy was always above 10 because the accelerometer picks up gravity (~9.8 m/s²), so it never drops near zero even when the arm is still.
2. **Gyroscope-only energy** (L2 norm of first 3 channels): In theory angular velocity should be zero when still. But in practice the person barely pauses between repetitions — the gyro energy hovers around 0.8-2.5 the whole time, no reliable "quiet zone" to cut at.

The results were terrible: 5 files gave me only 5-6 big chunks (basically one chunk per file), when I expected 20+ individual gestures. So segmentation didn't work.

### 2.4 The Decision: Cyclic Left-to-Right HMM

Since I couldn't segment, I changed approach: just use the entire repeated file as one training sequence, and use a **cyclic left-to-right HMM**.

A standard left-to-right topology only lets states go forward (stay or advance one step), which models a gesture going through a series of phases from start to finish. Adding the cyclic part means the last state can jump back to the first state, representing "one repetition done, starting the next one". This way even if there's 3-6 repeats in the sequence, the HMM can handle it naturally.

This decision came straight from Tip 10 in the project PDF: if you don't split the repeated gestures, you can use a left-to-right structure that allows transition from the last state back to the first.

**Data split**: 5 repeated files go to HMM training, 1 single file (same format as test data) is held out for validation. This avoids leaking test-like data into training.

---

## 3. Vector Quantization

### 3.1 Why Quantize

I'm implementing a discrete HMM, so the emission matrix B is an N×M probability table (B[j,k] = probability of state j emitting observation symbol k). The observations need to be integers from a finite set. So I need to map the continuous 6D IMU vectors to discrete labels first.

### 3.2 K-means Clustering

Pretty straightforward: pool all 83,451 6D vectors from both training and validation sets, run K-means with M=70 clusters, then replace each timestep's 6D vector with its nearest cluster index.

Including validation data in K-means is fine and doesn't cause data leakage — K-means just defines a mapping from continuous space to discrete labels, it has nothing to do with the HMM parameter learning (A, B, π are trained completely separately).

### 3.3 Why M=70

The course recommends M between 50-100. I started with 70, checked the clustering quality: no empty clusters, smallest cluster has 272 samples, largest has 4899, distribution looks reasonable (Figure 2). Later grid search (see §6) over M∈{65, 70, 75} confirmed 70 works best. Too small loses information, too large means each cluster has too few samples and B estimation gets noisy.

> **Figure 2**: Histogram of cluster sizes for K-means with M=70

After quantization each sequence becomes an integer array, same length as before (quantization doesn't change the number of timesteps). Each gesture has 5 training sequences (from repeated files) and 1 validation sequence (from single file), stored in `training_obs` and `val_obs`.

---

## 4. HMM Implementation and Correctness Checks

### 4.1 The Core

The HMM class is about 110 lines, with four main methods: `forward()`, `backward()`, `log_likelihood()`, and `train()`. It supports both ergodic and left-to-right-cyclic topologies.

### 4.2 Numerical Stability: Rabiner Scaling

This was probably the most important technical decision. In the forward algorithm, α[i,t] is a product of probabilities. With sequences 2000-3000 steps long, α values quickly underflow to 0 (double precision bottoms out around 10⁻³⁰⁸), and then everything becomes NaN.

I went with the scaling procedure from Section V of the Rabiner paper: at each step, normalize α so it sums to 1, and save the scaling factor c[t]. The final log P(O|λ) = Σ log(c[t]) — the scaling factors themselves encode all the probability info, no need to sum α at the end. The backward pass uses the same c[t] values so forward and backward stay numerically consistent.

Why scaling instead of log-space: the scaling approach keeps the formulas looking almost the same as the original forward/backward equations, so it's easier to implement. Log-space would mean turning every multiplication into addition and every summation into log-sum-exp, and computing ξ gets messier.

### 4.3 Dealing with Zero Emission Probabilities

If some state never emits a particular observation during training, then B has a zero in that spot. At test time, if that observation shows up, the whole sequence probability goes to zero. And you can't compare two models that both give zero.

Fix: additive smoothing on B — add ε=1e-8 to every entry, then renormalize. Intuitively this means the model says "this observation is extremely unlikely" instead of "this observation is impossible". This needs to happen both at initialization and after every M-step update.

### 4.4 Parameter Initialization

**A matrix** (transition probabilities): For the left-to-right cyclic topology, only diagonal (self-loop), super-diagonal (advance), and the cyclic position (last→first) are nonzero. Self-loop probabilities are initialized to random values between 0.5-0.7 (so each state stays for a few steps on average, covering a chunk of the observation), and the advance probability is whatever's left. An A_mask matrix forces all disallowed positions to zero after every M-step, so the LR topology can't get corrupted during training. Also worth noting: positions that start at zero in A will stay zero forever through EM — this is mathematically guaranteed by the left-to-right constraint.

**B matrix** (emission probabilities): Sampled from Dirichlet(1,1,...,1) (uniform random on the M-dim simplex), then add ε=1e-8 and renormalize. I don't use uniform initialization (every row = 1/M) because symmetric initial conditions cause all states to learn the same B row, and the model basically can't learn anything useful.

**π** (initial state distribution): Fixed at [1, 0, ..., 0]. In a left-to-right model the sequence always starts at state 0, so π doesn't need learning.

### 4.5 Speed

The forward recursion core `α[:,t] = (A.T @ α[:,t-1]) * B[:,oₜ]` is done with numpy matrix multiply in one shot, replacing the naive double for-loop. With N=15 the difference isn't extreme, but over 2000+ timesteps it adds up a lot.

For ξ computation, I accumulate step by step: compute the N×N matrix at each timestep and add it directly to `xi_sum`, instead of storing the full (N, N, T-1) 3D array. For a sequence with T=3000 and N=15, this reduces memory from about 5MB to 1.8KB. This turned out to be critical when running 30+ consecutive HMM trainings during LOOCV — without it I was getting MemoryError.

### 4.6 Toy Data Sanity Check

Before touching real data, I validated the implementation on a simple problem with a known answer.

**Setup**: 3000-step sequence — first 1000 steps in state 0 (only emits obs=0), middle 1000 steps in state 1 (emits 0 or 1 with equal probability), last 1000 steps back to state 0. Using ergodic topology here (because the states go 0→1→0, which needs backward transitions).

**Expected**: A ≈ [[0.999, 0.001], [0.001, 0.999]], B ≈ [[1.0, 0.0], [0.5, 0.5]]

**Result**: Converged in 19 epochs, LL = -707.50. The learned A and B match the theoretical values closely. LL curve is strictly monotonically increasing (Figure 3), confirming the Baum-Welch implementation is correct. The fact that LL never decreases also implicitly verifies forward-backward numerical consistency — if they were inconsistent, γ = α×β normalization would go wrong and LL couldn't stay monotonic.

> **Figure 3**: Toy data training LL curve (strictly increasing, converges in 19 epochs)

**Gotcha**: With seed=42, EM got stuck in a local optimum where both states learned nearly identical B rows — they couldn't differentiate. Switched to seed=4 and it converged correctly. This confirmed that EM is sensitive to initialization and you really need to try multiple random seeds.

---

## 5. Training and Validation

### 5.1 Training Setup

Hyperparameters: N=15 hidden states, M=70 observation symbols, Left-to-Right Cyclic topology, seed=42, max 200 epochs, convergence threshold tol=1e-2 (stop when absolute LL change between consecutive epochs drops below 0.01).

I train one independent HMM for each of the 6 gestures. Each model uses that gesture's 5 repeated sequences with multi-sequence Baum-Welch: run forward/backward independently on each sequence (E-step), accumulate γ and ξ statistics across sequences, then do a single M-step to update parameters.

### 5.2 Training Behavior

All 6 models converged normally with monotonically non-decreasing LL (Figure 4). Convergence speed varied quite a bit:

| Gesture | Convergence Epoch | Final LL | Notes |
|---------|------------------|----------|-------|
| wave | 142 | -9114 | Slowest, several small jumps in the middle |
| inf | 59 | -11189 | Relatively fast |
| eight | 79 | -13222 | Medium |
| circle | 69 | -6547 | Highest LL — circle sequences are shortest and most regular |
| beat3 | 118 | -10902 | Big LL jump around epoch 55 (~+1000) |
| beat4 | 87 | -17229 | Normal convergence |

The beat3 LL curve had a big jump around epoch 55 — basically EM escaped from one local optimum to a better one. This isn't unusual. What matters is the jump is always upward (LL increases), which is consistent with the EM theoretical guarantee.

> **Figure 4**: Training LL curves for all 6 gesture HMMs (2×3 subplots, monotonicity check noted in titles)

### 5.3 Fixing Data Leakage

My initial approach used all 6 sequences (5 repeated + 1 single) for training, then validated on the single file — got 6/6 = 100%. But this had data leakage: the single file appeared in both training and validation, so the result was overly optimistic.

After the fix: training only uses the 5 repeated files, the single file is completely held out for validation and never touches HMM training. LOOCV is also done only on the 5 repeated sequences.

### 5.4 Leave-One-Out Cross-Validation

To get a real sense of generalization, I ran LOOCV: for each gesture's 5 training sequences, leave one out, retrain that gesture's HMM on the remaining 4, then score the left-out sequence against all 6 models and predict with the highest LL. The other 5 gesture HMMs don't need retraining (their training data isn't affected). Together with the 6 single-file validation samples, this gives 36 evaluation samples total.

**Results**:

- **Top-1: 35/36 = 97.2%**
- **Top-3: 36/36 = 100%**
- The only error: `beat3 [rep4] → predicted as beat4`

> **Figure 5**: LOOCV confusion matrix (6×6, only beat3→beat4 has 1 error)

**Why beat3 and beat4 get confused**: Beat3 is a 3-beat rhythm, beat4 is 4-beat. The fundamental motion is the same — up-and-down beating — just different number of repetitions. In 6D IMU space their waveforms look really similar, so this is the pair most expected to be confused. Looking at the Top-3 lists, beat3 and beat4 almost always show up in each other's Top-2.

---

## 6. Hyperparameter Sensitivity

To check if N=15 and M=70 are actually in the sweet spot, I did a small grid search: M∈{65, 75}, N∈{10, 20}, using single holdout as a quick eval metric (train on 5 repeated, validate on 1 single). The metric is correct count + average LL margin (difference between the correct gesture's LL and the runner-up).

| M | N | Correct | Avg Margin | beat3 Margin | Notes |
|---|---|---------|------------|-------------|-------|
| 70 | 15 | 6/6 | ~2500 | ~200 | baseline |
| 65 | 10 | 6/6 | 2502 | 239 | |
| 65 | 20 | 6/6 | 2604 | 161 | |
| 75 | 10 | 6/6 | 2822 | 397 | highest beat3 margin |
| 75 | 20 | 6/6 | 2846 | 38.5 | beat3 margin drops hard |

Key findings:

1. **All combos got 6/6** — the model isn't super sensitive to small N/M changes
2. **N=20 makes beat3 margin drop from ~200 to 38.5** — too many hidden states, overfitting risk, the beat3 and beat4 models start looking too similar
3. M=75 has slightly better overall margin than M=65, but not by much

**Conclusion**: Stick with N=15, M=70. Don't go too high on N (overfitting), M is fine anywhere in 65-75.

---

## 7. Final Model and Test Results

### 7.1 Full Retrain

After LOOCV looked good, I merged the repeated (5) and single (1) files, retrained each gesture HMM on all 6 sequences as the final model, to use as much data as possible. The retrained models are saved to the `models/` directory.

### 7.2 Inference Pipeline

The inference script `test_classifier.py` is separate from the notebook, about 97 lines:

1. Load K-means model (70 cluster centers) + 6 HMM models (A, B, π matrices)
2. Read test file → drop timestamp column → get 6D IMU data
3. K-means predict → quantize the 6D vector sequence into integer label sequence
4. Run scaled forward algorithm on all 6 HMMs → get 6 log-likelihoods
5. The model with highest LL gives the Top-1 prediction, also output Top-3 with their LL values

### 7.3 Test Set Results

The test set has 8 unlabeled gesture files. Ran `test_classifier.py` with the final models:

| File | Top-1 | Top-2 | Top-3 | LL-1 | LL-2 | LL-3 |
|------|-------|-------|-------|------|------|------|
| test1.txt | beat4 | beat3 | inf | -7513.72 | -7723.96 | -11088.79 |
| test2.txt | eight | beat3 | beat4 | -6156.36 | -11369.81 | -11427.72 |
| test3.txt | inf | eight | beat3 | -6304.60 | -7072.84 | -8095.79 |
| test4.txt | eight | beat4 | beat3 | -6236.71 | -11357.30 | -12036.94 |
| test5.txt | wave | beat3 | eight | -4032.47 | -5820.84 | -6373.30 |
| test6.txt | beat4 | beat3 | inf | -6930.47 | -7343.34 | -10870.21 |
| test7.txt | beat4 | beat3 | wave | -4336.43 | -4430.10 | -4750.72 |
| test8.txt | eight | beat4 | beat3 | -7426.51 | -10954.65 | -11321.89 |

**Analysis**:

- **Top-1 distribution**: eight ×3, beat4 ×3, wave ×1, inf ×1. No circle or beat3 showed up as Top-1 predictions.
- **LL margins vary a lot**: For test2/test4/test8, the gap between Top-1 and Top-2 is over 3500 — the model is very confident about those being "eight". test5 (wave) also has a solid margin of 1788.
- **beat4 files have small margins**: test1 (beat4 vs beat3 margin = 210) and test7 (beat4 vs beat3 margin = 94), consistent with what we saw in LOOCV — beat3 and beat4 are just hard to tell apart.
- **Top-3 patterns**: beat3 and beat4 keep showing up as each other's backup in Top-2/Top-3, which again confirms these two beating gestures are really similar in the IMU feature space.

---

## 8. Discussion

### 8.1 Why beat3 and beat4 Get Confused

Both gestures are basically up-and-down beating motions. The main difference in the IMU signal is just temporal — 3 beats vs 4 beats — not spatial. A fixed-state-count LR-HMM has a hard time precisely capturing the difference between "repeat 3 times" and "repeat 4 times" because that requires very fine-grained temporal modeling. If I wanted to improve this further, some options would be: (a) use more hidden states specifically for beat3/beat4; (b) add time-domain derived features on top of the 6D IMU (like first differences or short-time energy) to help differentiate; (c) try multiple random seeds and keep the best model.

### 8.2 Limitations of the Approach

Discrete HMM + K-means quantization is what the course recommends, and it's simple and works pretty well, but there are inherent limitations. K-means quantization inevitably loses information — different gestures' IMU vectors might get mapped to the same cluster. A continuous HMM (with GMM emission probabilities) could avoid this quantization loss, but it's way more complex to implement.

### 8.3 Engineering Lessons

- **Memory**: Initially Baum-Welch stored the full ξ 3D array (N, N, T-1). When running LOOCV with 30+ consecutive trainings this hit MemoryError. Switching to step-by-step accumulation of the (N, N) statistics fixed it.
- **Batch execution**: The long LOOCV training was done through a standalone Python script with checkpoint saving and real-time progress output, instead of running the whole notebook as a black box. Much more controllable.
- **Data leakage awareness**: My first attempt mixed the single file into the training set, which gave inflated validation results (100%). The corrected LOOCV (97.2%) is the real generalization estimate. Good reminder to always be careful about the train/validation boundary in ML experiments.
