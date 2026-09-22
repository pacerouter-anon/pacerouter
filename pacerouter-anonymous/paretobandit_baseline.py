"""ParetoBandit (arXiv:2604.00136) 的 BudgetPacer 机制复现 — 窗口预算协议对比.

按官方包 pareto_bandit/budget_pacer.py 的机制忠实复现:
- 逐请求加法对偶更新: λ ← clip(λ + lr·(c_ema/target − 1), 0, λ_max)
- EMA 成本平滑: c_ema ← (1−α)c_ema + α·c_t
- ADAPTIVE 模式: 硬顶(λ 大时排除贵模型, 动态上限 c_max/(1+λ)) + 软罚(λ·c̃ 入 UCB 分数)
- 质量预测: LinUCB (与官方 DisjointLinUCB 同族), log 归一化成本进软罚
窗口协议: target = B/N (窗口预算均摊为单请求目标), 耗尽后转 cheapest.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, RESULTS_DIR, load_matrices, MODELS
from static_router import extract_features, big_task
from cbwk_router import LinUCBPool

IDX_CHEAP = MODELS.index("mistralai/mistral-7b-chat")


def run_pareto(Q, C, X, budget, lr=0.1, alpha_ema=0.1, lam_max=10.0,
               chunk=1, seed=42):
    """ParetoBandit BudgetPacer(ADAPTIVE) 复现. 逐请求更新(chunk=1)."""
    N, K = Q.shape
    d = X.shape[1]
    pool_q = LinUCBPool(K, d, alpha=1.0)
    target = budget / N  # per-request 目标成本
    lam = 0.0
    c_ema = target  # 官方: warm-start 于 target
    choices = np.zeros(N, dtype=int)
    spent = 0.0
    # log 归一化成本的边界(按数据集成本范围)
    all_c = C[np.isfinite(C) & (C > 0)]
    c_floor, c_ceil = np.percentile(all_c, 1), np.percentile(all_c, 99)
    c_max = float(np.nanmax(all_c))

    for t in range(N):
        x = X[t]
        mq, wq = pool_q.ucb_scores(x[None, :])
        ucb = (mq + wq)[0]
        # 成本(log 归一化)
        cvec = np.where(np.isfinite(C[t]) & (C[t] > 0), C[t], c_floor)
        c_norm = np.clip((np.log(cvec) - np.log(c_floor)) /
                         (np.log(c_ceil) - np.log(c_floor)), 0, 1)
        # 硬顶: 动态上限
        ceiling = c_max / (1.0 + lam) if lam > 0 else np.inf
        eligible = cvec <= ceiling
        if not np.any(eligible):
            eligible = np.ones(K, dtype=bool)
        scores = np.where(eligible, ucb - lam * c_norm, -np.inf)
        i = int(np.argmax(scores))
        # 耗尽兜底
        if spent >= budget:
            i = IDX_CHEAP
        choices[t] = i
        c_real = C[t, i] if np.isfinite(C[t, i]) else c_floor
        spent += c_real
        # 更新: EMA + 加法对偶
        c_ema = (1 - alpha_ema) * c_ema + alpha_ema * c_real
        c_norm_ema = c_ema / target
        lam = min(lam_max, max(0.0, lam + lr * (c_norm_ema - 1.0)))
        # LinUCB 在线更新(官方行为)
        if np.isfinite(Q[t, i]):
            pool_q.update(i, x, float(Q[t, i]))
    return choices, spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="5shot", choices=["5shot", "0shot"])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--stream", default="drift", choices=["shuffle", "drift"])
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_pickle(DATA_DIR / f"routerbench_{args.dataset}.pkl")
    Q, C, _ = load_matrices(args.dataset)
    E = np.load(DATA_DIR / f"emb_{args.dataset}.npy")
    X = np.hstack([E, extract_features(df["prompt"], df["eval_name"].to_numpy())])
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
    order = np.arange(len(idx_on))
    if args.stream == "drift":
        # 纯在线学习无标定期: drift 体现为在线段中途分布突变(难任务涌入)
        tasks_on = np.array([big_task(t) for t in df["eval_name"].to_numpy()[idx_on]])
        hard = np.isin(tasks_on, ["gsm8k", "mmlu"])
        rng.shuffle(order)
        order = np.concatenate([order[~hard], order[hard]])  # 先易后难突发
    else:
        rng.shuffle(order)
    Q_on, C_on, X_on = Q[idx_on][order], C[idx_on][order], X[idx_on][order]

    best_q = np.nanmax(Q_on, axis=1, keepdims=True)
    tied = np.isclose(Q_on, best_q, equal_nan=True)
    o_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
    oracle_cost = float(np.nansum(C_on[np.arange(len(order)), o_ch]))
    cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
    B = cheap_cost + args.gamma * (oracle_cost - cheap_cost)

    ch, spent = run_pareto(Q_on, C_on, X_on, B, lr=args.lr)
    n = len(order)
    q = float(np.nanmean(Q_on[np.arange(n), ch]))
    over = max(0.0, spent - B) / B
    unused = max(0.0, B - spent) / B
    print(f"ParetoBandit[{args.dataset} {args.stream} γ={args.gamma} lr={args.lr}]: "
          f"q={q:.4f} spent=${spent:.2f} overspend={over:.2%} unused={unused:.2%}")
    out = RESULTS_DIR / f"pb_{args.dataset}_g{args.gamma}_{args.stream}.json"
    out.write_text(json.dumps(dict(quality=round(q, 4), overspend=round(over, 4),
                                   unused=round(unused, 4)), indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
