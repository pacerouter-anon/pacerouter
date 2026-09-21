"""T2b: 窗口内累积质量曲线 (回应评审: PaceRouter 前期震荡是否损害质量).

drift γ=0.5 场景, 画 PaceRouter vs Static(GBR) 的"前 x% 窗口平均质量"曲线.
若 PaceRouter 前期有探索代价, 曲线应在早期低于 Static 后反超.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, load_matrices, MODELS
from pacerouter import train_predictors, predict_all, run_pacing, run_static, calibrate_lambda, IDX_CHEAP
from static_router import extract_features, big_task

dataset, gamma, seed = "5shot", 0.5, 42
df = pd.read_pickle(DATA_DIR / f"routerbench_{dataset}.pkl")
Q, C, _ = load_matrices(dataset)
E = np.load(DATA_DIR / f"emb_{dataset}.npy")
X = np.hstack([E, extract_features(df["prompt"], df["eval_name"].to_numpy())])
N = Q.shape[0]
idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=seed)
q_regs, c_regs = train_predictors(X[idx_tr], Q[idx_tr], C[idx_tr])
tasks_all = df["eval_name"].to_numpy()
easy = np.array([big_task(t) in ("hellaswag", "winogrande", "arc") for t in tasks_all[idx_te]])
n_cal = int(0.2 * len(idx_te))
rng0 = np.random.default_rng(seed)
idx_cal = rng0.choice(idx_te[easy], size=n_cal, replace=False)
idx_on = idx_te[n_cal:]
Q_hat_cal = predict_all(q_regs, X[idx_cal]); C_hat_cal = np.exp(predict_all(c_regs, X[idx_cal]))
best_qc = np.nanmax(Q[idx_cal], axis=1, keepdims=True)
tiedc = np.isclose(Q[idx_cal], best_qc, equal_nan=True)
oc = np.argmin(np.where(tiedc, C[idx_cal], np.inf), axis=1)
oracle_cal = float(np.nansum(C[idx_cal][np.arange(len(idx_cal)), oc]))
cheap_cal = float(np.nansum(C[idx_cal][:, IDX_CHEAP]))
B_cal = cheap_cal + gamma * (oracle_cal - cheap_cal)
lam_star = calibrate_lambda(Q_hat_cal, C_hat_cal, C[idx_cal], B_cal)
Q_hat_on = predict_all(q_regs, X[idx_on]); C_hat_on = np.exp(predict_all(c_regs, X[idx_on]))
rng = np.random.default_rng(seed)
order = np.arange(len(idx_on)); rng.shuffle(order)
Q_on, C_on = Q[idx_on][order], C[idx_on][order]
Q_hat_on, C_hat_on = Q_hat_on[order], C_hat_on[order]
best_q = np.nanmax(Q_on, axis=1, keepdims=True)
tied = np.isclose(Q_on, best_q, equal_nan=True)
o_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
oracle_cost = float(np.nansum(C_on[np.arange(len(order)), o_ch]))
cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
B = cheap_cost + gamma * (oracle_cost - cheap_cost)

ch_static, _, _ = run_static(Q_hat_on, C_hat_on, Q_on, C_on, lam_star)
ch_pace, _, lam_traj, _ = run_pacing(Q_hat_on, C_hat_on, Q_on, C_on, B, lam0=lam_star, chunk=500)

n = len(order)
x_frac = np.arange(1, n + 1) / n
q_s = np.nan_to_num(Q_on[np.arange(n), ch_static], nan=0.0)
q_p = np.nan_to_num(Q_on[np.arange(n), ch_pace], nan=0.0)
cum_static = np.cumsum(q_s) / np.arange(1, n + 1)
cum_pace = np.cumsum(q_p) / np.arange(1, n + 1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x_frac[::50], cum_static[::50], color="#c0392b", lw=1.8, label="Static (calibrated)")
ax.plot(x_frac[::50], cum_pace[::50], color="#2980b9", lw=1.8, label="PaceRouter (ours)")
ax.axhline(cum_static[-1], color="#c0392b", ls=":", lw=1, alpha=0.6)
ax.axhline(cum_pace[-1], color="#2980b9", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("window progress (fraction of queries served)")
ax.set_ylabel("cumulative average quality")
ax.set_title("Cumulative quality within the serving window (drift, $\\gamma$=0.5)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("results/fig_window_quality.png", dpi=200)
print("saved results/fig_window_quality.png")
print(f"前10%窗口: Static={cum_static[int(0.1*n)]:.4f} vs Pace={cum_pace[int(0.1*n)]:.4f}")
print(f"前30%窗口: Static={cum_static[int(0.3*n)]:.4f} vs Pace={cum_pace[int(0.3*n)]:.4f}")
print(f"全程:      Static={cum_static[-1]:.4f} vs Pace={cum_pace[-1]:.4f}")
