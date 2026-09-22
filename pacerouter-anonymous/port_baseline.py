"""PORT (NeurIPS 2025) 核心算法的 replay 复现 — 对比实验用.

按 fzwark/PORT 的 algo.py 逻辑忠实复现:
- 学习期: 前 eps*T 查询随机路由, 用 KNN(ANN) 估计各模型质量/成本
- 一次性优化 gamma (预算权重, L-BFGS-B): min eps*gamma·B + sum max(d*alpha - g*gamma)
- 在线段: score = d*alpha - g*gamma, argmax (gamma 固定, 无 pacing)
- 预算耗尽后强制 cheapest (窗口强制服务, 与本文所有方法一致)
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from replay import DATA_DIR, RESULTS_DIR, load_matrices, MODELS
from static_router import big_task

IDX_CHEAP = MODELS.index("mistralai/mistral-7b-chat")


def run_port(Q, C, E_on, E_tr, Q_tr, C_tr, budget, alpha=0.0001, eps=0.025, k=10):
    """PORT 核心算法复现. E_tr/Q_tr/C_tr: 历史(学习期)数据."""
    N, K = Q.shape
    # ANN: 在训练集 embedding 上建索引
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    nn.fit(E_tr)

    def estimate(Xq):
        """KNN 均值估计质量/成本 [n, K]."""
        _, idxs = nn.kneighbors(Xq)
        q_est = np.nanmean(np.where(np.isfinite(Q_tr[idxs]), Q_tr[idxs], np.nan), axis=1)
        c_est = np.nanmean(np.where(np.isfinite(C_tr[idxs]), C_tr[idxs], np.nan), axis=1)
        return q_est, c_est  # [n,K]

    # 学习期: 前 eps*N 查询
    n_learn = int(np.ceil(eps * N))
    choices = np.zeros(N, dtype=int)
    spent = 0.0

    # 学习期随机路由 + 收集估计
    rng = np.random.default_rng(42)
    for t in range(n_learn):
        i = int(rng.integers(0, K))
        choices[t] = i
        spent += C[t, i]

    # 一次性优化 gamma (PORT 的 F 目标)
    d_hist, g_hist = estimate(E_on[:n_learn])
    B_vec = np.full(K, budget / K, dtype=np.float64)  # split=uniform 近似

    def F(gamma):
        term1 = eps * np.dot(gamma, B_vec)
        scores = d_hist * alpha - g_hist * gamma[None, :]
        term2 = np.max(scores, axis=1).sum()
        return term1 + term2

    res = minimize(F, np.full(K, 1.0 / K), method="L-BFGS-B", bounds=[(0.0, 1.0)] * K)
    gamma = res.x

    # 在线段: 静态打分 (gamma 固定)
    d_on, c_on = estimate(E_on[n_learn:])
    for t in range(n_learn, N):
        if spent >= budget:
            choices[t] = IDX_CHEAP
            spent += C[t, IDX_CHEAP]
            continue
        s = d_on[t - n_learn] * alpha - c_on[t - n_learn] * gamma
        i = int(np.argmax(s))
        choices[t] = i
        spent += C[t, i]
    return choices, spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="5shot", choices=["5shot", "0shot"])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--stream", default="drift", choices=["shuffle", "drift"])
    ap.add_argument("--alpha", type=float, default=0.0001)
    ap.add_argument("--eps", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_pickle(DATA_DIR / f"routerbench_{args.dataset}.pkl")
    Q, C, _ = load_matrices(args.dataset)
    E = np.load(DATA_DIR / f"emb_{args.dataset}.npy")
    N = Q.shape[0]
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=args.seed)
    n_cal = int(0.2 * len(idx_te))
    idx_cal, idx_on = idx_te[:n_cal], idx_te[n_cal:]
    if args.stream == "drift":
        tasks_all = df["eval_name"].to_numpy()
        easy = np.array([big_task(t) in ("hellaswag", "winogrande", "arc") for t in tasks_all[idx_te]])
        rng0 = np.random.default_rng(args.seed)
        idx_cal = rng0.choice(idx_te[easy], size=n_cal, replace=False)

    rng = np.random.default_rng(args.seed)
    order = np.arange(len(idx_on)); rng.shuffle(order)
    Q_on, C_on = Q[idx_on][order], C[idx_on][order]
    E_on = E[idx_on][order]

    # PORT 的历史数据 = 标定段 (与我们静态基线同口径)
    E_tr, Q_tr, C_tr = E[idx_cal], Q[idx_cal], C[idx_cal]

    best_q = np.nanmax(Q_on, axis=1, keepdims=True)
    tied = np.isclose(Q_on, best_q, equal_nan=True)
    o_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
    oracle_cost = float(np.nansum(C_on[np.arange(len(order)), o_ch]))
    cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
    B = cheap_cost + args.gamma * (oracle_cost - cheap_cost)

    ch, spent = run_port(Q_on, C_on, E_on, E_tr, Q_tr, C_tr, B,
                         alpha=args.alpha, eps=args.eps)
    n = len(order)
    q = float(np.nanmean(Q_on[np.arange(n), ch]))
    over = max(0.0, spent - B) / B
    unused = max(0.0, B - spent) / B
    print(f"PORT[{args.dataset} {args.stream} γ={args.gamma} α={args.alpha} eps={args.eps}]: "
          f"q={q:.4f} spent=${spent:.2f} overspend={over:.2%} unused={unused:.2%}")
    out = RESULTS_DIR / f"port_{args.dataset}_g{args.gamma}_{args.stream}.json"
    out.write_text(json.dumps(dict(quality=round(q, 4), overspend=round(over, 4),
                                   unused=round(unused, 4)), indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
