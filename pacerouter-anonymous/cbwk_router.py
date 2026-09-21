"""CBwK-Router: 上下文 BwK 增强版路由 (CCF-B 增强原型).

升级点 vs v1 (PaceRouter):
- 质量/成本预测器: GBR -> LinUCB (岭回归 + 置信椭球, 质量 UCB / 成本 LCB)
- 在线学习: 在线段每查询服务后用真实 (q,c) 更新预测器 (lifelong)
- 决策分数: UCB_q - λ_t * LCB_c  (λ_t 沿用比例反馈 pacing + 末端安全网)

用法: python cbwk_router.py --dataset 5shot --gamma 0.5 --stream drift --alpha 1.0
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, RESULTS_DIR, MODELS, load_matrices
from pacerouter import run_static, calibrate_lambda, IDX_CHEAP
from static_router import extract_features, big_task


class EnsemblePool:
    """GBR 集成不确定性: 5 个不同种子 GBR, 均值+标准差提供乐观探索信号."""

    def __init__(self, n_models, n_members=5, alpha=1.0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.K, self.M, self.alpha = n_models, n_members, alpha
        self.HistGBR = HistGradientBoostingRegressor

    def fit(self, X_tr, Y_tr, log_target=False):
        self.regs_ = []
        for k in range(self.K):
            y = Y_tr[:, k]
            ok = np.isfinite(y)
            if log_target:
                ok = ok & (y > 0)
            members = []
            for s in range(self.M):
                r = self.HistGBR(max_iter=200, learning_rate=0.08, random_state=42 + s)
                yy = np.log(y[ok]) if log_target else y[ok]
                r.fit(X_tr[ok], yy)
                members.append(r)
            self.regs_.append(members)

    def ucb_scores(self, X):
        """返回 (集成均值 [N,K], 集成标准差×alpha [N,K]) 作为 (估计, 不确定性)."""
        K, M = self.K, self.M
        preds = np.stack([[self.regs_[k][m].predict(X) for m in range(M)]
                          for k in range(K)], axis=2)  # [M,N,K]
        return preds.mean(axis=0), self.alpha * preds.std(axis=0)


class LinUCBPool:
    """每模型一组 LinUCB 参数; Sherman-Morrison 在线更新 A^{-1}."""

    def __init__(self, n_models, dim, alpha=1.0):
        self.K, self.d, self.alpha = n_models, dim, alpha
        self.A_inv = np.stack([np.eye(dim) for _ in range(n_models)])  # [K,d,d]
        self.b = np.zeros((n_models, dim))

    def theta(self):
        return np.einsum("kij,kj->ki", self.A_inv, self.b)

    def ucb_scores(self, X):
        """返回 (均值估计 [N,K], UCB 宽度 [N,K])."""
        th = self.theta()                              # [K,d]
        mean = X @ th.T                                # [N,K]
        # 宽度: sqrt(x^T A_inv x) 逐模型
        Ax = np.einsum("kij,nj->nki", self.A_inv, X)   # [N,K,d]
        width = np.sqrt(np.einsum("nki,ni->nk", Ax, X))
        return mean, self.alpha * width

    def update(self, k, x, y):
        A_inv = self.A_inv[k]
        Ax = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[k] = A_inv - np.outer(Ax, Ax) / denom
        self.b[k] += y * x


def run_cbwk(Q, C, X, pool_q, pool_c, budget, lam0, chunk=500,
             online=True, cheap_idx=IDX_CHEAP):
    """上下文 BwK 路由: UCB_q - λ·LCB_c + pacing + 安全网 + 在线更新."""
    N = Q.shape[0]
    choices = np.zeros(N, dtype=int)
    lam = lam0
    spent = 0.0
    mean_c, _ = pool_c.ucb_scores(X[:200])
    cheap_est = float(np.exp(np.median(mean_c[:, cheap_idx])))
    lam_traj, spend_traj = [], []
    chunk_spent = 0.0
    # 不在线更新时(如 ensemble): 一次性批量计算全部 UCB/LCB, 循环内查表
    if not online:
        mq_all, wq_all = pool_q.ucb_scores(X)
        mc_all, wc_all = pool_c.ucb_scores(X)
        UCB_Q = mq_all + wq_all
        LCB_C = np.exp(np.maximum(mc_all - wc_all, np.log(1e-9)))
    for t in range(N):
        x = X[t]
        if online:
            mq, wq = pool_q.ucb_scores(x[None, :])
            mc, wc = pool_c.ucb_scores(x[None, :])
            ucb_q = (mq + wq)[0]
            lcb_c = np.exp(np.maximum((mc - wc)[0], np.log(1e-9)))
        else:
            ucb_q = UCB_Q[t]
            lcb_c = LCB_C[t]
        remaining_budget = budget - spent
        remaining_queries = N - t
        if remaining_budget <= remaining_queries * cheap_est:
            i = cheap_idx
        else:
            i = int(np.argmax(ucb_q - lam * lcb_c))
        choices[t] = i
        c_real = C[t, i]
        spent += c_real
        chunk_spent += c_real
        if online and np.isfinite(Q[t, i]) and np.isfinite(c_real) and c_real > 0:
            pool_q.update(i, x, float(Q[t, i]))
            pool_c.update(i, x, float(np.log(c_real)))
        if (t + 1) % chunk == 0 or t == N - 1:
            chunk_len = chunk if (t + 1) % chunk == 0 else (t + 1) % chunk
            quota = budget * chunk_len / N
            lam = lam * float(np.clip(chunk_spent / max(quota, 1e-12), 0.5, 2.0))
            lam_traj.append(float(lam)); spend_traj.append(float(spent))
            chunk_spent = 0.0
    return choices, spent, lam_traj, spend_traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="5shot", choices=["5shot", "0shot"])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--stream", default="drift", choices=["shuffle", "drift"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="linucb", choices=["linucb", "ensemble"],
                    help="linucb=线性置信上界(可在线更新); ensemble=GBR集成方差探索")
    ap.add_argument("--no-online", action="store_true")
    args = ap.parse_args()

    df = pd.read_pickle(DATA_DIR / f"routerbench_{args.dataset}.pkl")
    Q, C, _ = load_matrices(args.dataset)
    E = np.load(DATA_DIR / f"emb_{args.dataset}.npy")
    X_shallow = extract_features(df["prompt"], df["eval_name"].to_numpy())
    X_all = np.hstack([E, X_shallow])
    N, K = Q.shape

    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=args.seed)
    n_cal = int(0.2 * len(idx_te))
    idx_cal, idx_on = idx_te[:n_cal], idx_te[n_cal:]
    if args.stream == "drift":
        tasks_all = df["eval_name"].to_numpy()
        easy = np.array([big_task(t) in ("hellaswag", "winogrande", "arc") for t in tasks_all[idx_te]])
        rng0 = np.random.default_rng(args.seed)
        idx_cal = rng0.choice(idx_te[easy], size=min(n_cal, int(easy.sum())), replace=False)

       # 预测器离线预训练
    d = X_all.shape[1]
    if args.mode == "linucb":
        pool_q = LinUCBPool(K, d, alpha=args.alpha)
        pool_c = LinUCBPool(K, d, alpha=args.alpha)
        for idx in idx_tr:
            x = X_all[idx]
            for k in range(K):
                if np.isfinite(Q[idx, k]):
                    pool_q.update(k, x, float(Q[idx, k]))
                if np.isfinite(C[idx, k]) and C[idx, k] > 0:
                    pool_c.update(k, x, float(np.log(C[idx, k])))
    else:
        pool_q = EnsemblePool(K, alpha=args.alpha); pool_q.fit(X_all[idx_tr], Q[idx_tr])
        pool_c = EnsemblePool(K, alpha=args.alpha); pool_c.fit(X_all[idx_tr], C[idx_tr], log_target=True)

    # 标定段估计静态 λ* (供对照和热启动)
    mq, _ = pool_q.ucb_scores(X_all[idx_cal]); mc, _ = pool_c.ucb_scores(X_all[idx_cal])
    Q_hat_cal, C_hat_cal = mq, np.exp(mc)
    best_qc = np.nanmax(Q[idx_cal], axis=1, keepdims=True)
    tiedc = np.isclose(Q[idx_cal], best_qc, equal_nan=True)
    oc = np.argmin(np.where(tiedc, C[idx_cal], np.inf), axis=1)
    oracle_cal = float(np.nansum(C[idx_cal][np.arange(len(idx_cal)), oc]))
    cheap_cal = float(np.nansum(C[idx_cal][:, IDX_CHEAP]))
    B_cal = cheap_cal + args.gamma * (oracle_cal - cheap_cal)
    lam_star = calibrate_lambda(Q_hat_cal, C_hat_cal, C[idx_cal], B_cal)

    # 在线段
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(idx_on)); rng.shuffle(order)
    Q_on, C_on, X_on = Q[idx_on][order], C[idx_on][order], X_all[idx_on][order]

    best_q = np.nanmax(Q_on, axis=1, keepdims=True)
    tied = np.isclose(Q_on, best_q, equal_nan=True)
    o_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
    oracle_cost = float(np.nansum(C_on[np.arange(len(order)), o_ch]))
    oracle_q = float(np.nanmean(Q_on[np.arange(len(order)), o_ch]))
    cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
    B = cheap_cost + args.gamma * (oracle_cost - cheap_cost)
    print(f"在线段 N={len(order)} | oracle q={oracle_q:.4f} ${oracle_cost:.2f} | 预算 B=${B:.2f} | λ*={lam_star:.1f}")

    t0 = time.time()
    ch, spent, lam_traj, _ = run_cbwk(Q_on, C_on, X_on, pool_q, pool_c, B,
                                      lam0=lam_star, chunk=args.chunk,
                                      online=(args.mode == "linucb") and (not args.no_online))
    q = float(np.nanmean(Q_on[np.arange(len(order)), ch]))
    over = max(0.0, spent - B) / B
    unused = max(0.0, B - spent) / B
    print(f"CBwK-Router[{args.mode} {args.dataset} {args.stream} γ={args.gamma} α={args.alpha} online={not args.no_online}]: "
          f"q={q:.4f} spent=${spent:.2f} overspend={over:.2%} unused={unused:.2%} "
          f"| λ: {lam_traj[0]:.0f}->{lam_traj[-1]:.0f} | {time.time()-t0:.0f}s")

    tag = f"cbwk_{args.dataset}_a{args.alpha}_g{args.gamma}_{args.stream}"
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(
        dict(quality=round(q, 4), spent=round(spent, 2), overspend=round(over, 4),
             unused=round(unused, 4), lam_start=round(lam_traj[0], 2),
             lam_end=round(lam_traj[-1], 2)), indent=2))
    print("saved ->", RESULTS_DIR / f"{tag}.json")


if __name__ == "__main__":
    main()
