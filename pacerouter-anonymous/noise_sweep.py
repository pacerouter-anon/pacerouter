"""敏感性实验: 质量预测器精度 sigma -> PaceRouter 质量曲线.

q̂_noisy = clip(Q_true + N(0, sigma), 0, 1), sigma 越小预测越准.
证明: 瓶颈在预测器, pacing 机制随预测器升级而放大收益.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, RESULTS_DIR, load_matrices, MODELS
from pacerouter import run_pacing, train_predictors, predict_all
from static_router import extract_features
from replay import MODELS

IDX_CHEAP = MODELS.index("mistralai/mistral-7b-chat")


def main():
    dataset, gamma, seed = "5shot", 0.5, 42
    df = pd.read_pickle(DATA_DIR / f"routerbench_{dataset}.pkl")
    Q, C, _ = load_matrices(dataset)
    E = np.load(DATA_DIR / f"emb_{dataset}.npy")
    X = np.hstack([E, extract_features(df["prompt"], df["eval_name"].to_numpy())])
    N = Q.shape[0]

    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=seed)
    n_cal = int(0.2 * len(idx_te))
    idx_on = idx_te[n_cal:]

    # 成本预测器保持真实训练 (成本可预测性高)
    _, c_regs = train_predictors(X[idx_tr], Q[idx_tr], C[idx_tr])
    C_hat_on = np.exp(predict_all(c_regs, X[idx_on]))

    rng = np.random.default_rng(seed)
    order = np.arange(len(idx_on)); rng.shuffle(order)
    Q_on, C_on, C_hat_on = Q[idx_on][order], C[idx_on][order], C_hat_on[order]

    best_q = np.nanmax(Q_on, axis=1, keepdims=True)
    tied = np.isclose(Q_on, best_q, equal_nan=True)
    oc = np.argmin(np.where(tied, C_on, np.inf), axis=1)
    oracle_cost = float(np.nansum(C_on[np.arange(len(order)), oc]))
    cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
    B = cheap_cost + gamma * (oracle_cost - cheap_cost)
    print(f"B=${B:.2f}")

    results = {}
    for sigma in [0.5, 0.3, 0.2, 0.1, 0.05, 0.0]:
        noise = rng.normal(0, sigma, Q_on.shape)
        Q_hat = np.clip(np.nan_to_num(Q_on, nan=0.5) + noise, 0, 1)
        ch, spent, _, _ = run_pacing(Q_hat, C_hat_on, Q_on, C_on, B, lam0=100.0, chunk=500)
        q = float(np.nanmean(Q_on[np.arange(len(order)), ch]))
        over = max(0.0, spent - B) / B
        results[sigma] = dict(quality=round(q, 4), overspend=round(over, 4))
        print(f"sigma={sigma:>4}: quality={q:.4f} overspend={over:.2%}")

    out = RESULTS_DIR / "noise_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
