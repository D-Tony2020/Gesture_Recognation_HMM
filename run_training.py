"""
Batch training script - multi-seed + LOOCV
Runs one gesture at a time for progress monitoring & checkpoint saving.

Usage:
  python run_training.py seed_search <gesture>    # Batch 1: find best seed
  python run_training.py loocv <gesture>           # Batch 2: leave-one-out CV
"""
import numpy as np
import pickle
import sys
import os
import time

# ---- load data ----
with open('models/training_obs.pkl', 'rb') as f:
    training_obs = pickle.load(f)

GESTURE_NAMES = ['wave', 'inf', 'eight', 'circle', 'beat3', 'beat4']
N = 15
M = 70
SEEDS = [42, 7, 13, 99, 55]
CHECKPOINT_DIR = 'models/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---- HMM class (same as notebook, memory-optimized) ----
class HMM:
    def __init__(self, n_states, n_obs, topology='left-right-cyclic'):
        self.N = n_states
        self.M = n_obs
        self.topology = topology

        if topology == 'ergodic':
            self.pi = np.ones(self.N) / self.N
            self.A = np.random.dirichlet(np.ones(self.N), size=self.N)
        else:
            self.pi = np.zeros(self.N)
            self.pi[0] = 1.0
            self.A = np.zeros((self.N, self.N))
            for i in range(self.N):
                stay = 0.5 + np.random.uniform(0, 0.2)
                self.A[i, i] = stay
                nxt = (i + 1) % self.N
                self.A[i, nxt] = 1.0 - stay

        self.B = np.random.dirichlet(np.ones(self.M), size=self.N)
        self.B += 1e-8
        self.B /= self.B.sum(axis=1, keepdims=True)
        self.A_mask = (self.A > 0).astype(float)

    def forward(self, obs):
        T = len(obs)
        alpha = np.zeros((self.N, T))
        c = np.zeros(T)
        alpha[:, 0] = self.pi * self.B[:, obs[0]]
        c[0] = alpha[:, 0].sum() + 1e-300
        alpha[:, 0] /= c[0]
        for t in range(1, T):
            alpha[:, t] = (self.A.T @ alpha[:, t-1]) * self.B[:, obs[t]]
            c[t] = alpha[:, t].sum() + 1e-300
            alpha[:, t] /= c[t]
        return alpha, c

    def backward(self, obs, c):
        T = len(obs)
        beta = np.zeros((self.N, T))
        beta[:, T-1] = 1.0 / c[T-1]
        for t in range(T-2, -1, -1):
            beta[:, t] = self.A @ (self.B[:, obs[t+1]] * beta[:, t+1])
            beta[:, t] /= c[t]
        return beta

    def log_likelihood(self, obs):
        _, c = self.forward(obs)
        return np.sum(np.log(c))

    def train(self, obs_sequences, max_iter=200, tol=1e-2, verbose=False):
        ll_history = []
        for epoch in range(max_iter):
            total_ll = 0
            xi_sum = np.zeros((self.N, self.N))
            gamma_denom_A = np.zeros(self.N)
            gamma_denom_B = np.zeros(self.N)
            B_numer = np.zeros((self.N, self.M))

            for obs in obs_sequences:
                alpha, c = self.forward(obs)
                beta = self.backward(obs, c)
                T = len(obs)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=0, keepdims=True) + 1e-300

                gamma_denom_B += gamma.sum(axis=1)
                for t in range(T):
                    B_numer[:, obs[t]] += gamma[:, t]

                gamma_denom_A += gamma[:, :-1].sum(axis=1)
                for t in range(T-1):
                    bj_beta = self.B[:, obs[t+1]] * beta[:, t+1]
                    xi_t = alpha[:, t:t+1] * self.A * bj_beta[np.newaxis, :]
                    xi_t /= xi_t.sum() + 1e-300
                    xi_sum += xi_t

                total_ll += np.sum(np.log(c))

            ll_history.append(total_ll)

            if epoch > 0:
                ll_change = total_ll - ll_history[-2]
                if ll_change < -0.01:
                    print(f"    WARNING: LL decreased at epoch {epoch}!")
                if abs(ll_change) < tol:
                    if verbose:
                        print(f"    converged at epoch {epoch}")
                    break

            gamma_denom_A[gamma_denom_A == 0] = 1e-300
            A_new = xi_sum / gamma_denom_A[:, np.newaxis]
            A_new *= self.A_mask
            A_new /= A_new.sum(axis=1, keepdims=True) + 1e-300

            gamma_denom_B[gamma_denom_B == 0] = 1e-300
            B_new = B_numer / gamma_denom_B[:, np.newaxis]
            B_new += 1e-8
            B_new /= B_new.sum(axis=1, keepdims=True)

            self.A = A_new
            self.B = B_new

        return ll_history


def cmd_seed_search(gesture):
    """Batch 1: try multiple seeds, save best model"""
    seqs = training_obs[gesture]
    print(f"[seed_search] {gesture}: {len(seqs)} sequences, seeds={SEEDS}")

    best_ll = -np.inf
    best_seed = None
    best_hmm = None
    best_curve = None

    for seed in SEEDS:
        t0 = time.time()
        np.random.seed(seed)
        hmm = HMM(n_states=N, n_obs=M, topology='left-right-cyclic')
        ll_curve = hmm.train(seqs, max_iter=200, tol=1e-2, verbose=True)
        elapsed = time.time() - t0
        final_ll = ll_curve[-1]
        print(f"  seed={seed}: LL={final_ll:.2f}, epochs={len(ll_curve)}, time={elapsed:.1f}s")

        if final_ll > best_ll:
            best_ll = final_ll
            best_seed = seed
            best_hmm = hmm
            best_curve = ll_curve

    print(f"  >> BEST: seed={best_seed}, LL={best_ll:.2f}")

    # save checkpoint
    ckpt = {
        'gesture': gesture, 'best_seed': best_seed, 'best_ll': best_ll,
        'll_curve': best_curve,
        'A': best_hmm.A, 'B': best_hmm.B, 'pi': best_hmm.pi,
        'N': best_hmm.N, 'M': best_hmm.M, 'topology': best_hmm.topology,
        'all_seeds_results': {}
    }
    # also save the final model
    model_data = {
        'A': best_hmm.A, 'B': best_hmm.B, 'pi': best_hmm.pi,
        'N': best_hmm.N, 'M': best_hmm.M, 'topology': best_hmm.topology
    }
    with open(f'models/{gesture}_hmm.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    with open(f'{CHECKPOINT_DIR}/seed_search_{gesture}.pkl', 'wb') as f:
        pickle.dump(ckpt, f)
    print(f"  saved: models/{gesture}_hmm.pkl + checkpoint")


def cmd_loocv(gesture):
    """Batch 2: leave-one-out CV for one gesture"""
    seqs = training_obs[gesture]
    n_seqs = len(seqs)

    # load best seed from Batch 1 checkpoint, or default to 42
    ckpt_path = f'{CHECKPOINT_DIR}/seed_search_{gesture}.pkl'
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        best_seed = ckpt['best_seed']
    else:
        best_seed = 42

    # load all other gestures' full models (for comparison)
    other_models = {}
    for gname in GESTURE_NAMES:
        if gname == gesture:
            continue
        with open(f'models/{gname}_hmm.pkl', 'rb') as f:
            other_models[gname] = pickle.load(f)

    print(f"[loocv] {gesture}: {n_seqs} folds, seed={best_seed}")

    correct = 0
    for fold in range(n_seqs):
        t0 = time.time()
        train_seqs = [seqs[i] for i in range(n_seqs) if i != fold]
        val_seq = seqs[fold]

        np.random.seed(best_seed)
        fold_hmm = HMM(n_states=N, n_obs=M, topology='left-right-cyclic')
        fold_hmm.train(train_seqs, max_iter=200, tol=1e-2)

        # classify val_seq
        lls = {}
        lls[gesture] = fold_hmm.log_likelihood(val_seq)
        for gname, mdata in other_models.items():
            tmp_hmm = HMM.__new__(HMM)
            tmp_hmm.A = mdata['A']
            tmp_hmm.B = mdata['B']
            tmp_hmm.pi = mdata['pi']
            tmp_hmm.N = mdata['N']
            tmp_hmm.M = mdata['M']
            lls[gname] = tmp_hmm.log_likelihood(val_seq)

        ranked = sorted(lls.items(), key=lambda x: -x[1])
        pred = ranked[0][0]
        is_correct = pred == gesture
        correct += int(is_correct)
        elapsed = time.time() - t0

        seq_type = "single" if fold == n_seqs - 1 else f"rep{fold+1}"
        mark = "OK" if is_correct else "WRONG"
        top3 = [r[0] for r in ranked[:3]]
        print(f"  fold {fold} [{seq_type}]: pred={pred} [{mark}]  top3={top3}  ({elapsed:.1f}s)")

    print(f"  >> {gesture}: {correct}/{n_seqs} correct")

    # save loocv result
    result = {'gesture': gesture, 'correct': correct, 'total': n_seqs}
    with open(f'{CHECKPOINT_DIR}/loocv_{gesture}.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python run_training.py <seed_search|loocv> <gesture>")
        sys.exit(1)

    cmd = sys.argv[1]
    gesture = sys.argv[2]

    if gesture not in GESTURE_NAMES:
        print(f"Unknown gesture: {gesture}. Must be one of {GESTURE_NAMES}")
        sys.exit(1)

    if cmd == 'seed_search':
        cmd_seed_search(gesture)
    elif cmd == 'loocv':
        cmd_loocv(gesture)
    else:
        print(f"Unknown command: {cmd}")
