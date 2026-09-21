"""静态阈值基线: 浅特征质量预测器 + WTP 打分路由 (RouterBench/RouteLLM 式).

- 特征: prompt 浅层统计 + 任务类型 one-hot (零外部依赖, 不需 embedding 模型)
- 预测器: 每模型一个 HistGradientBoosting 回归器, 预测质量 q_hat
- 路由: i* = argmax_i [ q_hat_i - lambda * c_bar_i ], 扫 lambda 得质量-成本曲线
用法: python static_router.py --dataset 5shot
"""
import argparse
import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from replay import DATA_DIR, RESULTS_DIR, MODELS, load_matrices, evaluate


def normalize_prompt(p) -> str:
    """prompt 列常是字符串化的 list, 解析后拼接为纯文本."""
    if isinstance(p, str) and p.startswith("[") and p.endswith("]"):
        try:
            parts = ast.literal_eval(p)
            if isinstance(parts, list):
                return "\n".join(str(x) for x in parts)
        except (ValueError, SyntaxError):
            pass
    return str(p)


def big_task(eval_name: str) -> str:
    e = str(eval_name)
    if "hellaswag" in e:
        return "hellaswag"
    if "grade-school" in e:
        return "gsm8k"
    if "arc" in e:
        return "arc"
    if "winogrande" in e:
        return "winogrande"
    if e.startswith("mmlu"):
        return "mmlu"
    return "other"


TASKS = ["hellaswag", "gsm8k", "arc", "winogrande", "mmlu", "other"]


def extract_features(prompts: pd.Series, eval_names: np.ndarray) -> np.ndarray:
    feats = []
    tasks = [big_task(e) for e in eval_names]
    for p, t in zip(prompts, tasks):
        text = normalize_prompt(p)
        n = len(text)
        n_words = len(text.split())
        n_digit = len(re.findall(r"\d", text))
        n_cjk = len(re.findall(r"[一-鿿]", text))
        n_lines = text.count("\n") + 1
        has_q = float("?" in text or "？" in text)
        row = [
            np.log1p(n), np.log1p(n_words), n_digit / max(n, 1),
            n_cjk / max(n, 1), np.log1p(n_lines), has_q,
        ]
        row += [float(t == tk) for tk in TASKS]
        feats.append(row)
    return np.asarray(feats, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="5shot", choices=["5shot", "0shot"])
    args = ap.parse_args()

    df = pd.read_pickle(DATA_DIR / f"routerbench_{args.dataset}.pkl")
    Q, C, _ = load_matrices(args.dataset)
    eval_names = df["eval_name"].to_numpy()
    N, K = Q.shape
    print(f"dataset={args.dataset} N={N} K={K}; 提取浅特征...")

    X = extract_features(df["prompt"], eval_names)
    print(f"特征维度: {X.shape[1]}")

    idx_tr, idx_te = train_test_split(
        np.arange(N), test_size=0.3, random_state=42
    )
    X_tr, X_te = X[idx_tr], X[idx_te]
    Q_tr, Q_te, C_te = Q[idx_tr], Q[idx_te], C[idx_te]

    # 每模型训练一个质量回归器 (剔除该模型缺失的样本)
    Q_hat_te = np.zeros_like(Q_te)
    t0 = time.time()
    for k, m in enumerate(MODELS):
        y_k = Q_tr[:, k]
        ok = np.isfinite(y_k)
        reg = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_depth=None, random_state=42
        )
        reg.fit(X_tr[ok], y_k[ok])
        Q_hat_te[:, k] = reg.predict(X_te)
    print(f"11 个质量预测器训练完成, 耗时 {time.time()-t0:.1f}s")

    # 预测质量 (测试集上的预测器本身精度, 供笔记)
    corr = []
    for k in range(K):
        ok = np.isfinite(Q_te[:, k])
        c = float(np.corrcoef(Q_hat_te[ok, k], Q_te[ok, k])[0, 1]) if ok.sum() > 10 else float("nan")
        corr.append(c)
    print("各模型预测-真实质量相关系数:",
          {m.split('/')[-1]: round(c, 3) for m, c in zip(MODELS, corr)})

    # 静态路由: 成本用模型级均值 (对齐现状, 也是论文批评点)
    c_bar = np.nanmean(C[idx_tr], axis=0)
    lambdas = np.logspace(-2, 4.5, 30)  # WTP 参数扫描 (美元/单位质量)
    points = []
    for lam in lambdas:
        score = Q_hat_te - lam * c_bar[None, :]
        choices = np.argmax(score, axis=1)
        m = evaluate(choices, Q_te, C_te)
        m["lambda"] = float(lam)
        points.append(m)

    out = RESULTS_DIR / f"static_router_{args.dataset}.json"
    out.write_text(json.dumps(points, indent=2))
    # 打印几个代表工作点
    print("\n静态路由质量-成本工作点 (节选):")
    for p in points[::6]:
        print(f"  lambda={p['lambda']:>9.4f}  quality={p['avg_quality']:.4f}  "
              f"per_query=${p['cost_per_query']:.6f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
