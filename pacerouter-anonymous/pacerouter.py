"""PaceRouter v0: 预算节奏控制路由 vs 静态基线 的在线对比实验.

实验协议 (论文核心故事):
  测试集 -> 前 20% 作"历史标定段" (静态基线在此把 lambda 标定到恰好花完预算),
            后 80% 作"在线评估段" (模拟真实流量, 所有方法在线跑).
  预算 B = gamma * oracle 在在线段的成本.

基线:
  B3 静态阈值 (lambda 历史标定)          -- 预期: 流量漂移后超支或浪费
  B4 静态阈值 + 硬截断 (花完转 cheapest)  -- 工程补救: 不超支但后段质量崩
  B5 PaceRouter (lambda_t 乘性更新 + per-query 成本预测)
  B6 oracle-pacing (真实质量, 上界参考)

用法: python pacerouter.py --dataset 5shot --gamma 0.5 --stream drift
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, RESULTS_DIR, MODELS, load_matrices
from static_router import extract_features

IDX_CHEAP = MODELS.index("mistralai/mistral-7b-chat")  # 最便宜模型 (硬截断兜底)


def train_predictors(X_tr, Q_tr, C_tr):
    """质量预测器 (q̂) + 成本预测器 (log ĉ), 每模型各一个."""
    q_regs, c_regs = [], []
    for k in range(len(MODELS)):
        ok_q = np.isfinite(Q_tr[:, k])
        rq = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42)
        rq.fit(X_tr[ok_q], Q_tr[ok_q, k])
        q_regs.append(rq)
        ok_c = np.isfinite(C_tr[:, k]) & (C_tr[:, k] > 0)
        rc = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42)
        rc.fit(X_tr[ok_c], np.log(C_tr[ok_c, k]))
        c_regs.append(rc)
    return q_regs, c_regs


class KNNPredictor:
    """KNN 回归 (RouterBench 最优配置: k=40, cosine). 直接对 embedding 操作."""

    def __init__(self, E_tr, Y_tr, k=40, log_target=False):
        from sklearn.neighbors import NearestNeighbors
        self.k = k
        self.log_target = log_target
        self.Y_tr = Y_tr
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
        self.nn.fit(E_tr)

    def predict(self, E_te):
        _, idx = self.nn.kneighbors(E_te)
        vals = self.Y_tr[idx]  # [N_te, k, K]
        pred = np.nanmean(vals, axis=1)
        if self.log_target:
            pred = np.exp(pred)
        return pred


def train_knn_predictors(E_tr, Q_tr, C_tr):
    """KNN 版质量/成本预测: 一次索引, 两表查询."""
    Q_f = np.where(np.isfinite(Q_tr), Q_tr, np.nan)
    C_log = np.where(np.isfinite(C_tr) & (C_tr > 0), np.log(C_tr), np.nan)
    return (KNNPredictor(E_tr, Q_f, k=40, log_target=False),
            KNNPredictor(E_tr, C_log, k=40, log_target=True))


def predict_all(regs, X):
    return np.stack([r.predict(X) for r in regs], axis=1)


def run_static(Q_hat, C_hat, Q, C, lam, budget=None):
    """静态阈值 (可选硬截断). 返回 (choices, spent, hit_stop_ratio)."""
    N = Q.shape[0]
    choices = np.zeros(N, dtype=int)
    spent = 0.0
    stop_at = N  # 预算耗尽点
    for t in range(N):
        if budget is not None and spent >= budget:
            choices[t] = IDX_CHEAP
            continue
        score = Q_hat[t] - lam * C_hat[t]
        i = int(np.argmax(score))
        choices[t] = i
        spent += C[t, i]
    if budget is not None:
        hits = np.where(np.cumsum(C[np.arange(N), choices]) >= budget)[0]
        stop_at = int(hits[0]) if len(hits) else N
    return choices, spent, stop_at / N


def run_cascade(Q, C, budget, judge_thresh=0.5):
    """FrugalGPT 式级联 + 预算: 模型按均价升序逐级调用, judge(质量分<阈值)不合格则升级.
    预算耗尽后强制 cheapest. replay 下用记录中的真实质量分模拟 judge."""
    N = Q.shape[0]
    order_models = np.argsort(np.nanmean(C, axis=0))  # 价格升序
    choices = np.zeros(N, dtype=int)
    spent = 0.0
    for t in range(N):
        if spent >= budget:
            choices[t] = IDX_CHEAP
            spent += C[t, IDX_CHEAP]
            continue
        chosen = order_models[0]
        acc = 0.0
        for m in order_models:
            acc += C[t, m] if np.isfinite(C[t, m]) else 0.0
            qm = Q[t, m]
            chosen = m
            if np.isfinite(qm) and qm >= judge_thresh:
                break  # judge 通过, 交付
        # 若预算不足以级联则只保留第一次调用成本
        choices[t] = chosen
        spent += acc if spent + acc <= budget else C[t, order_models[0]]
    return choices, spent


def run_pacing(Q_hat, C_hat, Q, C, budget, lam0=100.0, chunk=500,
               cheap_idx=IDX_CHEAP, safety_margin=1.0, update="prop", eps=0.5,
               rho_min=0.5, rho_max=2.0, return_triggers=False):
    """PaceRouter: 影子价格 pacing + 末端安全网.

    update="prop": 比例反馈  λ *= clip(chunk_spent/quota, rho_min, rho_max)
    update="mw":   乘性权重 (PD-BwK 形式)  λ *= (1+eps)^(chunk_spent/quota)
    safety_margin: 安全网阈值系数 (1+δ), δ 灵敏度由调用方扫描
    return_triggers=True 时额外返回安全网触发时刻列表
    """
    N = Q.shape[0]
    choices = np.zeros(N, dtype=int)
    lam = lam0
    spent = 0.0
    cheap_est = float(np.nanmean(C_hat[:, cheap_idx]))
    lam_traj, spend_traj = [], []
    chunk_spent = 0.0
    triggers = []
    for t in range(N):
        remaining_budget = budget - spent
        remaining_queries = N - t
        if remaining_budget <= remaining_queries * cheap_est * safety_margin:
            i = cheap_idx
            if return_triggers and (not triggers or triggers[-1] != t - 1):
                triggers.append(t)
        else:
            score = Q_hat[t] - lam * C_hat[t]
            i = int(np.argmax(score))
        choices[t] = i
        spent += C[t, i]
        chunk_spent += C[t, i]
        if (t + 1) % chunk == 0 or t == N - 1:
            chunk_len = chunk if (t + 1) % chunk == 0 else (t + 1) % chunk
            quota = budget * chunk_len / N
            ratio = chunk_spent / max(quota, 1e-12)
            if update == "mw":
                lam = lam * (1.0 + eps) ** ratio
            else:
                lam = lam * float(np.clip(ratio, rho_min, rho_max))
            lam_traj.append(float(lam))
            spend_traj.append(float(spent))
            chunk_spent = 0.0
    if return_triggers:
        return choices, spent, lam_traj, spend_traj, triggers
    return choices, spent, lam_traj, spend_traj


def calibrate_lambda(Q_hat_c, C_hat_c, C_c, budget_c):
    """在历史标定段上二分搜索 lambda, 使静态路由恰好花完 budget_c."""
    lo, hi = 1e-2, 1e5
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        _, spent, _ = run_static(Q_hat_c, C_hat_c, np.zeros_like(Q_hat_c), C_c, mid)
        # 用预测口径标定 (真实场景拿不到真实质量): 以预测成本累计估算花费
        score = Q_hat_c - mid * C_hat_c
        ch = np.argmax(score, axis=1)
        spent_pred = float(np.nansum(C_c[np.arange(len(ch)), ch]))
        if spent_pred > budget_c:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="5shot", choices=["5shot", "0shot"])
    ap.add_argument("--gamma", type=float, default=0.5, help="预算 = gamma × oracle 成本")
    ap.add_argument("--stream", default="shuffle",
                    choices=["shuffle", "burst", "burst_hard_first", "drift"])
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qpred", default="gbr", choices=["gbr", "knn"],
                    help="质量/成本预测器: gbr=梯度提升树, knn=KNN余弦回归(RouterBench式)")
    ap.add_argument("--cost-mode", default="predict", choices=["predict", "mean"],
                    help="消融: predict=per-query成本预测(本文), mean=模型级均值(现状)")
    ap.add_argument("--lam0", type=float, default=None,
                    help="pacing 初始影子价格; 缺省=用历史标定值 lam_star(热启动)")
    args = ap.parse_args()

    df = pd.read_pickle(DATA_DIR / f"routerbench_{args.dataset}.pkl")
    Q, C, _ = load_matrices(args.dataset)
    X_shallow = extract_features(df["prompt"], df["eval_name"].to_numpy())
    emb_path = DATA_DIR / f"emb_{args.dataset}.npy"
    E = np.load(emb_path) if emb_path.exists() else None
    if E is not None:
        X = np.hstack([E, X_shallow])
        print(f"特征: embedding(384) + 浅层(12) = {X.shape[1]} 维")
    else:
        X = X_shallow
        print(f"特征: 浅层 {X.shape[1]} 维 (无 embedding 缓存)")
    if args.qpred == "knn" and E is None:
        raise SystemExit("knn 预测器需要 embedding 缓存, 先跑 embed_prompts.py")
    N, K = Q.shape

    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=args.seed)
    if args.qpred == "gbr":
        q_regs, c_regs = train_predictors(X[idx_tr], Q[idx_tr], C[idx_tr])
        predict_q = lambda idxs: predict_all(q_regs, X[idxs])
        predict_c = lambda idxs: np.exp(predict_all(c_regs, X[idxs]))
    else:
        knn_q, knn_c = train_knn_predictors(E[idx_tr], Q[idx_tr], C[idx_tr])
        predict_q = lambda idxs: knn_q.predict(E[idxs])
        predict_c = lambda idxs: knn_c.predict(E[idxs])

    # 标定段 (历史) / 在线段
    n_cal = int(0.2 * len(idx_te))
    idx_cal, idx_on = idx_te[:n_cal], idx_te[n_cal:]
    if args.stream == "drift":
        # 分布漂移: 标定期只有简单任务流量(平时), 在线期混入大量难题(期末周)
        from static_router import big_task
        tasks_all = df["eval_name"].to_numpy()
        easy_mask = np.array([big_task(t) in ("hellaswag", "winogrande", "arc")
                              for t in tasks_all[idx_te]])
        easy_pool = idx_te[easy_mask]
        rng0 = np.random.default_rng(args.seed)
        idx_cal = rng0.choice(easy_pool, size=min(n_cal, len(easy_pool)), replace=False)
    Q_hat_cal = predict_q(idx_cal); C_hat_cal = predict_c(idx_cal)
    Q_hat_on = predict_q(idx_on);  C_hat_on = predict_c(idx_on)
    if args.cost_mode == "mean":
        # 消融: 退回"模型级平均成本"(RouterBench/RouteLLM 现状)
        c_mean = np.nanmean(C[idx_tr], axis=0)
        C_hat_cal = np.tile(c_mean, (len(idx_cal), 1))
        C_hat_on = np.tile(c_mean, (len(idx_on), 1))
    Q_on, C_on = Q[idx_on], C[idx_on]

    # 在线流量顺序
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(idx_on))
    if args.stream in ("shuffle", "drift"):
        rng.shuffle(order)
    elif args.stream in ("burst", "burst_hard_first"):
        # 突发流: burst=简单在前难在后; burst_hard_first=难在前简单在后
        from static_router import big_task
        tasks = np.array([big_task(e) for e in df["eval_name"].to_numpy()[idx_on]])
        hard = np.isin(tasks, ["gsm8k", "mmlu"])
        if args.stream == "burst":
            order = np.concatenate([order[~hard], order[hard]])
        else:
            order = np.concatenate([order[hard], order[~hard]])

    def perm(A): return A[order]
    Q_on, C_on, Q_hat_on, C_hat_on = map(perm, (Q_on, C_on, Q_hat_on, C_hat_on))

    # oracle 与 cheapest 在在线段的成本 -> 可行域预算: B = cheap + gamma*(oracle - cheap)
    best_q = np.nanmax(Q_on, axis=1, keepdims=True)
    tied = np.isclose(Q_on, best_q, equal_nan=True)
    oracle_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
    oracle_cost = float(np.nansum(C_on[np.arange(len(order)), oracle_ch]))
    oracle_q = float(np.nanmean(Q_on[np.arange(len(order)), oracle_ch]))
    cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
    cheap_q = float(np.nanmean(Q_on[:, IDX_CHEAP]))
    B = cheap_cost + args.gamma * (oracle_cost - cheap_cost)
    print(f"在线段 N={len(order)} | oracle q={oracle_q:.4f} ${oracle_cost:.2f} | "
          f"cheapest q={cheap_q:.4f} ${cheap_cost:.2f} | 预算 B=${B:.2f} (gamma={args.gamma})")

    # 静态 lambda 标定: 历史段按同口径预算
    best_qc = np.nanmax(Q[idx_cal], axis=1, keepdims=True)
    tiedc = np.isclose(Q[idx_cal], best_qc, equal_nan=True)
    oc = np.argmin(np.where(tiedc, C[idx_cal], np.inf), axis=1)
    oracle_cal_cost = float(np.nansum(C[idx_cal][np.arange(len(idx_cal)), oc]))
    cheap_cal_cost = float(np.nansum(C[idx_cal][:, IDX_CHEAP]))
    B_cal = cheap_cal_cost + args.gamma * (oracle_cal_cost - cheap_cal_cost)
    lam_star = calibrate_lambda(Q_hat_cal, C_hat_cal, C[idx_cal], B_cal)
    print(f"历史段标定: oracle=${oracle_cal_cost:.2f} B_cal=${B_cal:.2f} -> lambda*={lam_star:.2f}")

    def metrics(choices, spent):
        q = float(np.nanmean(Q_on[np.arange(len(order)), choices]))
        over = max(0.0, spent - B) / B
        unused = max(0.0, B - spent) / B
        return dict(avg_quality=round(q, 4), spent=round(spent, 2),
                    overspend=round(over, 4), unused=round(unused, 4))

    results = {}
    t0 = time.time()

    # 简单基线
    n_on = len(order)
    rng_b = np.random.default_rng(args.seed + 1)
    ch = rng_b.integers(0, K, size=n_on)
    results["B1_random"] = metrics(ch, float(np.nansum(C_on[np.arange(n_on), ch])))
    ch = np.full(n_on, IDX_CHEAP)
    results["B2_always_cheapest"] = metrics(ch, float(np.nansum(C_on[:, IDX_CHEAP])))
    idx_gpt4 = MODELS.index("gpt-4-1106-preview")
    ch = np.full(n_on, idx_gpt4)
    results["B2_always_gpt4"] = metrics(ch, float(np.nansum(C_on[:, idx_gpt4])))

    ch, spent, _ = run_static(Q_hat_on, C_hat_on, Q_on, C_on, lam_star)
    results["B3_static"] = metrics(ch, spent)

    ch, spent, stop_ratio = run_static(Q_hat_on, C_hat_on, Q_on, C_on, lam_star, budget=B)
    r = metrics(ch, spent); r["hard_stop_at"] = round(stop_ratio, 3)
    results["B4_static_hardstop"] = r

    # FrugalGPT 式级联 + 预算
    ch, spent = run_cascade(Q_on, C_on, B)
    results["B4b_cascade"] = metrics(ch, spent)

    # KNN 预测器版静态路由 (RouterBench 式)
    if E is not None:
        knn_q, knn_c = train_knn_predictors(E[idx_tr], Q[idx_tr], C[idx_tr])
        Q_hat_cal_k = knn_q.predict(E[idx_cal]); C_hat_cal_k = knn_c.predict(E[idx_cal])
        lam_k = calibrate_lambda(Q_hat_cal_k, C_hat_cal_k, C[idx_cal], B_cal)
        Q_hat_on_k = knn_q.predict(E[idx_on])[order]; C_hat_on_k = knn_c.predict(E[idx_on])[order]
        ch, spent, _ = run_static(Q_hat_on_k, C_hat_on_k, Q_on, C_on, lam_k)
        results["B3b_static_knn"] = metrics(ch, spent)

    lam0_used = args.lam0 if args.lam0 is not None else lam_star
    ch, spent, lam_traj, spend_traj = run_pacing(Q_hat_on, C_hat_on, Q_on, C_on, B,
                                                 lam0=lam0_used, chunk=args.chunk)
    results["B5_pacerouter"] = metrics(ch, spent)
    results["B5_pacerouter"]["lam0_used"] = round(lam0_used, 2)
    results["B5_pacerouter"]["lam_start"] = round(lam_traj[0], 2)
    results["B5_pacerouter"]["lam_end"] = round(lam_traj[-1], 2)

    # oracle-pacing: 用真实质量, 展示 pacing 机制上界
    ch, spent, _, _ = run_pacing(Q_on, C_on, Q_on, C_on, B, lam0=lam_star, chunk=args.chunk)
    results["B6_oracle_pacing"] = metrics(ch, spent)

    print(f"\n=== stream={args.stream} gamma={args.gamma} ===")
    for k, v in results.items():
        print(f"  {k:22s} q={v['avg_quality']:.4f} spent=${v['spent']:.2f} "
              f"overspend={v['overspend']:.2%} unused={v['unused']:.2%}"
              + (f" stop@{v['hard_stop_at']:.0%}" if "hard_stop_at" in v else ""))
    print(f"(耗时 {time.time()-t0:.0f}s)")

    tag = f"{args.dataset}_{args.qpred}-{args.cost_mode}_g{args.gamma}_{args.stream}"
    (RESULTS_DIR / f"pacing_{tag}.json").write_text(json.dumps(results, indent=2))
    np.save(RESULTS_DIR / f"lam_traj_{tag}.npy", np.array(lam_traj))
    np.save(RESULTS_DIR / f"spend_traj_{tag}.npy", np.array(spend_traj))
    print(f"saved -> results/pacing_{tag}.json (+trajectory npy)")


if __name__ == "__main__":
    main()
