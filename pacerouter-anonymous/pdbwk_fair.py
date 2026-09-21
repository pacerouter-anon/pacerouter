"""T4b: 忠实版 PD-BwK (Badanidiyuru et al. Algorithm 2) 公平对照.

忠实实现: 臂级(无上下文) UCB/LCB + 乐观性价比比率 argmax u_x/EstCost_x
+ 乘性权重更新 v ← v·(1+ε)^ℓ (单调不减). 耗尽后强制 cheapest 完成窗口服务.
对比 v1 双向比例反馈, 同场景同预算同预测信息源(臂级经验均值).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, load_matrices, MODELS
from static_router import big_task

IDX_CHEAP = MODELS.index("mistralai/mistral-7b-chat")
C_RAD = 1.0  # Hoeffding 半径常数


def rad(n, T):
    return np.sqrt(C_RAD * np.log(T) / np.maximum(n, 1))


def run_pdbwk(Q, C, budget, eps):
    """忠实 PD-BwK (单资源). 耗尽后强制 cheapest."""
    N, K = Q.shape
    # 归一化成本到 [0,1]: 用每查询最大成本
    cmax = np.nanmax(C) + 1e-12
    # 臂级统计
    sum_q = np.zeros(K); sum_c = np.zeros(K); cnt = np.zeros(K)
    # 初始化: 每臂拉一次
    choices = np.zeros(N, dtype=int)
    spent = 0.0
    lam = 1.0  # 单资源价格标量 v
    stopped_at = N
    for t in range(N):
        if t < K:
            i = t  # 初始化轮流拉
        else:
            u = (sum_q + 1e-9) / np.maximum(cnt, 1) + rad(cnt, N)         # UCB 质量
            u = np.where(cnt > 0, u, 1.0)
            lcb = np.clip((sum_c / np.maximum(cnt, 1)) - rad(cnt, N) * (sum_c / np.maximum(cnt, 1)), 1e-9, None)
            lcb = np.where(cnt > 0, lcb, 1e-9)                            # LCB 成本
            est_cost = lcb * lam
            i = int(np.argmax(u / est_cost))
        if spent >= budget:
            i = IDX_CHEAP  # 耗尽后强制 cheapest (窗口强制服务)
            if stopped_at == N:
                stopped_at = t
        choices[t] = i
        q_real = Q[t, i] if np.isfinite(Q[t, i]) else 0.0
        c_real = C[t, i]
        spent += c_real
        cnt[i] += 1; sum_q[i] += q_real; sum_c[i] += c_real
        lam = lam * (1 + eps) ** (c_real / (budget / N))  # 乘性更新, ℓ=归一化消耗
    return choices, spent, stopped_at / N


def main():
    dataset, gamma, seed = "5shot", 0.5, 42
    df = pd.read_pickle(DATA_DIR / f"routerbench_{dataset}.pkl")
    Q, C, _ = load_matrices(dataset)
    N = Q.shape[0]
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=seed)

    for stream in ["shuffle", "drift"]:
        n_cal = int(0.2 * len(idx_te))
        idx_on = idx_te[n_cal:]
        rng = np.random.default_rng(seed)
        order = np.arange(len(idx_on))
        if stream == "drift":
            # 难任务(gsm8k/mmlu)在前, 易任务在后 — 分布漂移突发
            tasks = np.array([big_task(t) for t in df["eval_name"].to_numpy()[idx_on]])
            hard = np.isin(tasks, ["gsm8k", "mmlu"])
            order = np.concatenate([order[hard], order[~hard]])
        else:
            rng.shuffle(order)
        Q_on, C_on = Q[idx_on][order], C[idx_on][order]
        best_q = np.nanmax(Q_on, axis=1, keepdims=True)
        tied = np.isclose(Q_on, best_q, equal_nan=True)
        o_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
        oracle_cost = float(np.nansum(C_on[np.arange(len(order)), o_ch]))
        cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
        B = cheap_cost + gamma * (oracle_cost - cheap_cost)
        n = len(order)
        for eps in [0.1, 0.5, 1.0]:
            ch, spent, stop = run_pdbwk(Q_on, C_on, B, eps)
            q = float(np.nanmean(Q_on[np.arange(n), ch]))
            over = max(0.0, spent - B) / B
            unused = max(0.0, B - spent) / B
            print(f"{stream} eps={eps}: q={q:.4f} spent=${spent:.2f} over={over:.2%} "
                  f"unused={unused:.2%} 耗尽点={stop:.0%}")


if __name__ == "__main__":
    main()
