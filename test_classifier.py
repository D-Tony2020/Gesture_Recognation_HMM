"""
Gesture classifier - loads trained HMM models and classifies test IMU data

Preprocessing pipeline:
  1. Load raw IMU data (drop timestamp column)
  2. Per-sequence mean removal (removes gravity / device orientation bias)
  3. Global std normalization (equalizes gyro & accel channel contributions)
  4. KMeans quantization -> discrete observation sequence
  5. Forward algorithm -> log-likelihood per gesture model
  6. Rank by LL -> Top-3 predictions
"""
import numpy as np
import pickle
import os
import glob

# ---- config ----
MODEL_DIR = 'models'
TEST_DIR = 'data/Test_gesture'  # change this to wherever the test files are

GESTURE_NAMES = ['wave', 'inf', 'eight', 'circle', 'beat3', 'beat4']

def load_models():
    # load normalization parameters
    with open(os.path.join(MODEL_DIR, 'norm_params.pkl'), 'rb') as f:
        norm_params = pickle.load(f)

    # load kmeans
    with open(os.path.join(MODEL_DIR, 'kmeans_model.pkl'), 'rb') as f:
        kmeans = pickle.load(f)

    # load HMM models
    models = {}
    for gname in GESTURE_NAMES:
        fname = os.path.join(MODEL_DIR, f'{gname}_hmm.pkl')
        with open(fname, 'rb') as f:
            models[gname] = pickle.load(f)

    return kmeans, models, norm_params


def normalize_sequence(imu, global_std):
    """Per-sequence mean removal + global std normalization"""
    centered = imu - imu.mean(axis=0)
    return centered / global_std


def forward_log_likelihood(A, B, pi, obs):
    """
    scaled forward algorithm - same as in the notebook but standalone
    returns log P(O | model)
    """
    N = A.shape[0]
    T = len(obs)
    alpha = np.zeros((N, T))
    c = np.zeros(T)

    alpha[:, 0] = pi * B[:, obs[0]]
    c[0] = alpha[:, 0].sum() + 1e-300
    alpha[:, 0] /= c[0]

    for t in range(1, T):
        alpha[:, t] = (A.T @ alpha[:, t-1]) * B[:, obs[t]]
        c[t] = alpha[:, t].sum() + 1e-300
        alpha[:, t] /= c[t]

    return np.sum(np.log(c))


def classify(obs, models):
    """compute LL for each gesture model, return sorted list"""
    results = []
    for gname in GESTURE_NAMES:
        m = models[gname]
        ll = forward_log_likelihood(m['A'], m['B'], m['pi'], obs)
        results.append((gname, ll))

    # sort by LL descending
    results.sort(key=lambda x: -x[1])
    return results


def main():
    print("Loading models...")
    kmeans, models, norm_params = load_models()
    global_std = norm_params['global_std']
    print(f"Loaded {len(models)} gesture models, kmeans with {kmeans.n_clusters} clusters")

    # find test files
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*.txt')))
    if not test_files:
        print(f"No test files found in {TEST_DIR}")
        print("Make sure to update TEST_DIR at the top of this script")
        return

    print(f"\nFound {len(test_files)} test files\n")
    print(f"{'File':<30} {'Top-1':<10} {'Top-2':<10} {'Top-3':<10} {'LL-1':>12} {'LL-2':>12} {'LL-3':>12}")
    print("-" * 96)

    for fpath in test_files:
        # load raw IMU data
        raw = np.loadtxt(fpath)
        imu = raw[:, 1:]  # drop timestamp

        # normalize: per-sequence mean removal + global std normalization
        imu_norm = normalize_sequence(imu, global_std)

        # quantize
        obs = kmeans.predict(imu_norm)

        # classify
        ranked = classify(obs, models)

        fname = os.path.basename(fpath)
        top3 = ranked[:3]
        print(f"{fname:<30} {top3[0][0]:<10} {top3[1][0]:<10} {top3[2][0]:<10} {top3[0][1]:>12.2f} {top3[1][1]:>12.2f} {top3[2][1]:>12.2f}")


if __name__ == '__main__':
    main()
