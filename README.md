# ECE5242 Project 2: Gesture Recognition with HMM

IMU-based gesture recognition using Hidden Markov Models. Classifies 6 gesture types (Wave, Infinity, Eight, Circle, Beat3, Beat4) from gyroscope + accelerometer data.

## Environment

- Python 3.8+
- numpy, scipy, scikit-learn, matplotlib

Install dependencies:
```bash
pip install numpy scipy scikit-learn matplotlib
```

## Project Structure

```
├── gesture_hmm.ipynb      # Main notebook (data exploration, training, validation)
├── test_classifier.py     # Standalone inference script for test data
├── run_training.py        # Batch training utility (seed search, LOOCV)
├── models/
│   ├── kmeans_model.pkl   # K-means model (M=70 clusters, 6D IMU vectors)
│   ├── wave_hmm.pkl       # HMM for each gesture (A, B, pi matrices)
│   ├── inf_hmm.pkl
│   ├── eight_hmm.pkl
│   ├── circle_hmm.pkl
│   ├── beat3_hmm.pkl
│   └── beat4_hmm.pkl
├── docs/
│   ├── DEVLOG.md          # Development log
│   ├── experiments/       # Experiment records (quantization, training, evaluation)
│   └── figures/           # Training curves, confusion matrix plots
└── data/                  # Training and test data (not included in repo)
    ├── train/             # Train Set 1: repeated gestures
    ├── train_single/      # Train Set 2: single gestures
    └── Test_gesture/      # Test set
```

## How to Run Inference (No Retraining Needed)

1. Place test files (`.txt`, 7 columns: timestamp + 6D IMU) in `data/Test_gesture/`

2. Run the classifier:
```bash
python test_classifier.py
```

3. Output shows Top-3 predictions with log-likelihoods for each test file:
```
File                           Top-1      Top-2      Top-3          LL-1         LL-2         LL-3
------------------------------------------------------------------------------------------------
test_001.txt                   wave       inf        eight       -1234.56     -2345.67     -3456.78
...
```

If your test files are in a different directory, edit `TEST_DIR` at the top of `test_classifier.py`.

## Pipeline Overview

1. **Vector Quantization**: Raw 6D IMU vectors → K-means (70 clusters) → discrete observation labels
2. **HMM Classification**: Each gesture has a trained Left-to-Right Cyclic HMM (N=15 states, M=70 observations). For a test sequence, compute log-likelihood under each model → predict the gesture with highest log-likelihood.

## Training (Optional)

The notebook `gesture_hmm.ipynb` contains the full pipeline:
- Data loading and visualization
- K-means vector quantization (M=70)
- HMM implementation (Forward/Backward/Baum-Welch with Rabiner scaling)
- Training 6 gesture models (N=15, seed=42, max_iter=200)
- Leave-One-Out Cross-Validation: **97.2% Top-1 accuracy** (35/36)

To retrain from scratch, run all cells in the notebook. Pre-trained models are already saved in `models/`.

## Validation Results

| Metric | Value |
|--------|-------|
| LOOCV Top-1 Accuracy | 35/36 = 97.2% |
| LOOCV Top-3 Accuracy | 36/36 = 100% |
| Only misclassification | beat3 rep4 → beat4 |

## Data Format

Each input file has 7 columns (space-separated):
```
timestamp(ms)  Wx  Wy  Wz(rad/s)  Ax  Ay  Az(m/s²)
```
The timestamp column is dropped during processing; only the 6 IMU channels are used.
