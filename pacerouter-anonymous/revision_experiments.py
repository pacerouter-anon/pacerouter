"""大修综合实验包: S1 安全网统计+灵敏度 / S3 对抗震荡 / S5 clip 扫描 / 标定样本量对照.
共享一次预测器训练. 全部结果写 results/revision_pack.json + 打印.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, RESULTS_DIR, load_matrices, MODELS
from pacerouter import train_predictors, predict_all, run_pacing, run_static, calibrate_lambda, IDX_CHEAP
from static_router import extract_features, big_task


def setup(dataset, seed=42):
    df = pd.read_pickle(DATA_DIR / f"routerbench_{dataset}.pkl")
    Q, C, _ = load_matrices(dataset)
    E = np.load(DATA_DIR / f"emb_{dataset}.npy")
    X = np.hstack([E, extract_features(df["prompt"], df["eval_name"].to_numpy())])
    return df, Q, C, X


def split_online(df, Q, C, X, idx_tr, idx_te, stream, seed):
    """构造标定段/在线段. drift: 标定段只含易任务."""
    n_cal = int(0.2 * len(idx_te))
    idx_cal, idx_on = idx_te[:n_cal], idx_te[n_cal:]
    if stream == "drift":
        tasks_all = df["eval_name"].to_numpy()
        easy = np.array([big_task(t) in ("hellaswag", "winogrande", "arc") for t in tasks_all[idx_te]])
        rng0 = np.random.default_rng(seed)
        idx_cal = rng0.choice(idx_te[easy], size=min(n_cal, int(easy.sum())), replace=False)
    return idx_cal, idx_on


def budget_of(Q_on, C_on, gamma):
    best_q = np.nanmax(Q_on, axis=1, keepdims=True)
    tied = np.isclose(Q_on, best_q, equal_nan=True)
    o_ch = np.argmin(np.where(tied, C_on, np.inf), axis=1)
    oracle_cost = float(np.nansum(C_on[np.arange(len(o_ch)), o_ch]))
    cheap_cost = float(np.nansum(C_on[:, IDX_CHEAP]))
    return cheap_cost + gamma * (oracle_cost - cheap_cost)


def metrics_of(ch, spent, Q_on, B):
    n = len(ch)
    q = float(np.nanmean(Q_on[np.arange(n), ch]))
    return dict(q=round(q, 4), over=round(max(0.0, spent - B) / B, 4),
                unused=round(max(0.0, B - spent) / B, 4))


def main():
    seed = 42
    results = {}

    # ============ S1a: 安全网触发统计 (12 格点) ============
    print("=== S1a 安全网触发统计 ===")
    s1a = {}
    for dataset in ["5shot", "0shot"]:
        df, Q, C, X = setup(dataset)
        N = Q.shape[0]
        idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=seed)
        q_regs, c_regs = train_predictors(X[idx_tr], Q[idx_tr], C[idx_tr])
        for stream in ["shuffle", "drift"]:
            idx_cal, idx_on = split_online(df, Q, C, X, idx_tr, idx_te, stream, seed)
            Q_hat_cal = predict_all(q_regs, X[idx_cal]); C_hat_cal = np.exp(predict_all(c_regs, X[idx_cal]))
            lam_star = calibrate_lambda(Q_hat_cal, C_hat_cal, C[idx_cal],
                                        budget_of(Q[idx_cal], C[idx_cal], 0.5))
            Q_hat_on = predict_all(q_regs, X[idx_on]); C_hat_on = np.exp(predict_all(c_regs, X[idx_on]))
            rng = np.random.default_rng(seed)
            order = np.arange(len(idx_on)); rng.shuffle(order)
            Q_on, C_on = Q[idx_on][order], C[idx_on][order]
            Qh, Ch = Q_hat_on[order], C_hat_on[order]
            for gamma in [0.25, 0.5, 0.75]:
                B = budget_of(Q_on, C_on, gamma)
                ch, spent, _, _, trig = run_pacing(Qh, Ch, Q_on, C_on, B,
                                                   lam0=lam_star, chunk=500, return_triggers=True)
                m = metrics_of(ch, spent, Q_on, B)
                trig_ratio = len(trig) / max(len(order) / 500, 1)  # 触发占比近似
                first_trig = (trig[0] / len(order)) if trig else None
                s1a[f"{dataset}_{stream}_g{gamma}"] = dict(
                    **m, n_trigger_segments=len(trig),
                    first_trigger_at=(round(first_trig, 3) if first_trig else None))
                print(f"  {dataset} {stream} γ={gamma}: q={m['q']} 触发段数={len(trig)} "
                      f"首次触发={first_trig if first_trig else '无'}")
    results["S1a_trigger_stats"] = s1a

    # ============ 以下实验固定在 5shot, drift, γ=0.5 ============
    df, Q, C, X = setup("5shot")
    N = Q.shape[0]
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=seed)
    q_regs, c_regs = train_predictors(X[idx_tr], Q[idx_tr], C[idx_tr])
    idx_cal, idx_on = split_online(df, Q, C, X, idx_tr, idx_te, "drift", seed)
    Q_hat_cal = predict_all(q_regs, X[idx_cal]); C_hat_cal = np.exp(predict_all(c_regs, X[idx_cal]))
    lam_star = calibrate_lambda(Q_hat_cal, C_hat_cal, C[idx_cal], budget_of(Q[idx_cal], C[idx_cal], 0.5))
    Q_hat_on = predict_all(q_regs, X[idx_on]); C_hat_on = np.exp(predict_all(c_regs, X[idx_on]))
    rng = np.random.default_rng(seed)
    order = np.arange(len(idx_on)); rng.shuffle(order)
    Q_on, C_on = Q[idx_on][order], C[idx_on][order]
    Qh, Ch = Q_hat_on[order], C_hat_on[order]
    B = budget_of(Q_on, C_on, 0.5)

    # ============ S1b: 安全网阈值灵敏度 ============
    print("\n=== S1b 安全网阈值 δ 灵敏度 (drift γ=0.5) ===")
    s1b = {}
    for delta in [-0.2, 0.0, 0.2, 0.5]:
        ch, spent, _, _ = run_pacing(Qh, Ch, Q_on, C_on, B, lam0=lam_star, chunk=500,
                                     safety_margin=1.0 + delta)
        s1b[str(delta)] = metrics_of(ch, spent, Q_on, B)
        print(f"  δ={delta:+.1f}: {s1b[str(delta)]}")
    results["S1b_margin_sensitivity"] = s1b

    # ============ S3: 对抗震荡流量 ============
    print("\n=== S3 对抗震荡流量 (easy/hard 交替) ===")
    tasks_on = np.array([big_task(t) for t in df["eval_name"].to_numpy()[idx_on]])
    hard_mask = np.isin(tasks_on, ["gsm8k", "mmlu"])
    s3 = {}
    for K_period in [2, 5, 20]:  # 周期(单位: micro-batch=500 查询)
        n_chunk = K_period * 500
        hard_seq = np.where(hard_mask)[0]; easy_seq = np.where(~hard_mask)[0]
        # 交替拼接: hard 块 + easy 块循环
        order_adv = []
        hi, ei = 0, 0
        flip = True
        while hi < len(hard_seq) or ei < len(easy_seq):
            if flip and hi < len(hard_seq):
                order_adv.extend(hard_seq[hi:hi + n_chunk]); hi += n_chunk
            elif not flip and ei < len(easy_seq):
                order_adv.extend(easy_seq[ei:ei + n_chunk]); ei += n_chunk
            flip = not flip
        order_adv = np.array(order_adv)
        Q_a, C_a = Q_on[order_adv], C_on[order_adv]
        Qh_a, Ch_a = Qh[order_adv], Ch[order_adv]
        B_a = budget_of(Q_a, C_a, 0.5)
        ch, spent, lam_traj, _ = run_pacing(Qh_a, Ch_a, Q_a, C_a, B_a, lam0=lam_star, chunk=500)
        m = metrics_of(ch, spent, Q_a, B_a)
        lam_arr = np.array(lam_traj)
        lam_osc = float(np.mean(np.abs(np.diff(np.sign(np.diff(lam_arr))))) / 2)  # 方向翻转率
        s3[str(K_period)] = dict(**m, lam_final=round(lam_traj[-1], 1),
                                 lam_oscillation=round(lam_osc, 3))
        print(f"  K={K_period}: {m} λ震荡率={lam_osc:.2f}")
    results["S3_adversarial"] = s3

    # ============ S5: clip 参数扫描 ============
    print("\n=== S5 clip 参数扫描 (drift γ=0.5) ===")
    s5 = {}
    for rho_min in [0.8, 0.9, 0.95]:
        for rho_max in [1.05, 1.1, 1.25, 2.0]:
            ch, spent, _, _ = run_pacing(Qh, Ch, Q_on, C_on, B, lam0=lam_star, chunk=500,
                                         rho_min=rho_min, rho_max=rho_max)
            s5[f"{rho_min}_{rho_max}"] = metrics_of(ch, spent, Q_on, B)
            print(f"  ρmin={rho_min} ρmax={rho_max}: {s5[f'{rho_min}_{rho_max}']}")
    results["S5_clip_sweep"] = s5

    # ============ 标定样本量对照 (反向: 静态基线给更多历史是否变好) ============
    print("\n=== 标定样本量对照 (drift γ=0.5) ===")
    calib_study = {}
    for cal_ratio in [0.1, 0.2, 0.4]:
        n_cal_r = int(cal_ratio * len(idx_te))
        idx_cal_r = idx_te[:n_cal_r]
        idx_on_r = idx_te[n_cal_r:]
        # drift 设定: 标定段只从 easy 抽
        tasks_te = df["eval_name"].to_numpy()[idx_te]
        easy_idx = np.array([big_task(t) in ("hellaswag", "winogrande", "arc") for t in tasks_te])
        rng1 = np.random.default_rng(seed)
        idx_cal_r = rng1.choice(idx_te[easy_idx], size=min(n_cal_r, int(easy_idx.sum())), replace=False)
        Q_hat_cal_r = predict_all(q_regs, X[idx_cal_r]); C_hat_cal_r = np.exp(predict_all(c_regs, X[idx_cal_r]))
        lam_r = calibrate_lambda(Q_hat_cal_r, C_hat_cal_r, C[idx_cal_r],
                                 budget_of(Q[idx_cal_r], C[idx_cal_r], 0.5))
        rng2 = np.random.default_rng(seed)
        order_r = np.arange(len(idx_on_r)); rng2.shuffle(order_r)
        Q_r, C_r = Q[idx_on_r][order_r], C[idx_on_r][order_r]
        Qh_r = predict_all(q_regs, X[idx_on_r])[order_r]
        Ch_r = np.exp(predict_all(c_regs, X[idx_on_r]))[order_r]
        B_r = budget_of(Q_r, C_r, 0.5)
        ch_s, spent_s, _ = run_static(Qh_r, Ch_r, Q_r, C_r, lam_r)
        ch_p, spent_p, _, _ = run_pacing(Qh_r, Ch_r, Q_r, C_r, B_r, lam0=lam_r, chunk=500)
        calib_study[str(cal_ratio)] = dict(
            n_calib=int(len(idx_cal_r)),
            static=metrics_of(ch_s, spent_s, Q_r, B_r),
            pacing=metrics_of(ch_p, spent_p, Q_r, B_r))
        print(f"  标定比例={cal_ratio}: static={calib_study[str(cal_ratio)]['static']} pacing={calib_study[str(cal_ratio)]['pacing']}")
    results["calibration_size"] = calib_study

    out = RESULTS_DIR / "revision_pack.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
