"""Per-task 稳健性: drift γ=0.5 场景下各任务域上 PaceRouter vs 基线的质量."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, MODELS, load_matrices
from pacerouter import (train_predictors, predict_all, run_static, run_pacing,
                        calibrate_lambda, IDX_CHEAP)
from static_router import extract_features, big_task

dataset, gamma, seed = "5shot", 0.5, 42
df = pd.read_pickle(DATA_DIR / f"routerbench_{dataset}.pkl")
Q, C, _ = load_matrices(dataset)
E = np.load(DATA_DIR / f"emb_{dataset}.npy")
X = np.hstack([E, extract_features(df["prompt"], df["eval_name"].to_numpy())])
eval_names = df["eval_name"].to_numpy()
N = Q.shape[0]

idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=seed)
q_regs, c_regs = train_predictors(X[idx_tr], Q[idx_tr], C[idx_tr])

tasks_all = np.array([big_task(t) for t in eval_names])
n_cal = int(0.2 * len(idx_te))
easy_mask = np.array([big_task(t) in ("hellaswag", "winogrande", "arc") for t in eval_names[idx_te]])
rng0 = np.random.default_rng(seed)
idx_cal = rng0.choice(idx_te[easy_mask], size=n_cal, replace=False)
idx_on = idx_te[n_cal:]

Q_hat_cal = predict_all(q_regs, X[idx_cal]); C_hat_cal = np.exp(predict_all(c_regs, X[idx_cal]))
Q_hat_on = predict_all(q_regs, X[idx_on]);  C_hat_on = np.exp(predict_all(c_regs, X[idx_on]))
Q_on, C_on = Q[idx_on], C[idx_on]
tasks_on = tasks_all[idx_on]

best_qc = np.nanmax(Q[idx_cal], axis=1, keepdims=True)
tiedc = np.isclose(Q[idx_cal], best_qc, equal_nan=True)
oc = np.argmin(np.where(tiedc, C[idx_cal], np.inf), axis=1)
oracle_cal = float(np.nansum(C[idx_cal][np.arange(len(idx_cal)), oc]))
cheap_cal = float(np.nansum(C[idx_cal][:, IDX_CHEAP]))
B_cal = cheap_cal + gamma * (oracle_cal - cheap_cal)
lam_star = calibrate_lambda(Q_hat_cal, C_hat_cal, C[idx_cal], B_cal)

best_q = np.nanmax(Q_on, axis=1, keepdims=True)
tied = np.isclose(Q_on, best_q, equal_nan=True)
o_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
oracle_cost = float(np.nansum(C_on[np.arange(len(idx_on)), o_ch]))
cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
B = cheap_cost + gamma * (oracle_cost - cheap_cost)

ch_static, _, _ = run_static(Q_hat_on, C_hat_on, Q_on, C_on, lam_star)
ch_pace, _, _, _ = run_pacing(Q_hat_on, C_hat_on, Q_on, C_on, B, lam0=lam_star, chunk=500)

n = len(idx_on)
q_static = Q_on[np.arange(n), ch_static]
q_pace = Q_on[np.arange(n), ch_pace]

print(f"{'任务域':<12} {'样本数':>6} {'Static':>8} {'PaceRouter':>10} {'差值':>8}")
for t in ["hellaswag", "winogrande", "arc", "gsm8k", "mmlu", "other"]:
    m = tasks_on == t
    if m.sum() == 0:
        continue
    s, p = float(np.nanmean(q_static[m])), float(np.nanmean(q_pace[m]))
    print(f"{t:<12} {m.sum():>6} {s:>8.4f} {p:>10.4f} {p-s:>+8.4f}")
