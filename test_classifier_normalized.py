import numpy as np
import pickle, os, glob

MODEL_DIR = 'models'
TEST_DIR = 'data/Test_gesture'
gestures = ['wave', 'inf', 'eight', 'circle', 'beat3', 'beat4']

# load everything
with open(f'{MODEL_DIR}/norm_params.pkl', 'rb') as f:
    global_std = pickle.load(f)['global_std']
with open(f'{MODEL_DIR}/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

models = {}
for g in gestures:
    with open(f'{MODEL_DIR}/{g}_hmm.pkl', 'rb') as f:
        models[g] = pickle.load(f)

def forward_ll(A, B, pi, obs):
    # scaled forward, returns log P(O|lambda)
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

# run on test files
test_files = sorted(glob.glob(f'{TEST_DIR}/*.txt'))
print(f"{'File':<20} {'Top-1':<10} {'Top-2':<10} {'Top-3':<10} {'LL-1':>10} {'LL-2':>10} {'LL-3':>10}")
print("-" * 80)

for fpath in test_files:
    raw = np.loadtxt(fpath)
    imu = raw[:, 1:]
    # same normalization as training
    imu = (imu - imu.mean(axis=0)) / global_std
    obs = kmeans.predict(imu)

    lls = []
    for g in gestures:
        m = models[g]
        lls.append((g, forward_ll(m['A'], m['B'], m['pi'], obs)))
    lls.sort(key=lambda x: -x[1])

    fname = os.path.basename(fpath)
    t = lls[:3]
    print(f"{fname:<20} {t[0][0]:<10} {t[1][0]:<10} {t[2][0]:<10} {t[0][1]:>10.2f} {t[1][1]:>10.2f} {t[2][1]:>10.2f}")
