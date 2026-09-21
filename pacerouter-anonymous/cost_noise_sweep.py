"""T3b: 成本预测误差扫描 — 验证 Theorem 1 的 λ̄ε_c 项.

质量预测用真实 GBR; 成本预测值乘以对数正态噪声 ĉ = c · exp(N(0, σ_c)).
σ_c=0 对应完美成本预测. 期望: 质量随 σ_c 下降, 验证 prediction regret 中成本项.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, RESULTS_DIR, load_matrices, MODELS
from pacerouter import train_predictors, predict_all, run_pacing, calibrate_lambda, IDX_CHEAP
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
n = len(order)

results = {}
for sigma_c in [0.0, 0.1, 0.25, 0.5, 1.0]:
    noise = rng.normal(0, sigma_c, C_hat_on.shape)
    C_noisy = C_hat_on * np.exp(noise)  # 对数正态乘法噪声
    ch, spent, lam_traj, _ = run_pacing(Q_hat_on, C_noisy, Q_on, C_on, B, lam0=lam_star, chunk=500)
    q = float(np.nanmean(Q_on[np.arange(n), ch]))
    over = max(0.0, spent - B) / B
    unused = max(0.0, B - spent) / B
    lam_bar = float(np.mean(lam_traj))
    results[str(sigma_c)] = dict(quality=round(q, 4), overspend=round(over, 4),
                                 unused=round(unused, 4), lam_bar=round(lam_bar, 1))
    print(f"σ_c={sigma_c}: q={q:.4f} over={over:.2%} unused={unused:.2%} λ̄={lam_bar:.0f}")

out = RESULTS_DIR / "cost_noise_sweep.json"
out.write_text(json.dumps(results, indent=2))
print("saved ->", out)
