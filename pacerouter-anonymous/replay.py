"""RouterBench replay 评估框架 + 傻瓜基线验证.

数据: data/routerbench_{0,5}shot.pkl  (宽表, 每行一个查询, 含 11 模型的质量/成本/输出)
用法: python replay.py --dataset 5shot
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# 11 个可路由模型 (与 RouterBench 论文一致; 排除 RAG 专用的 You.com/Perplexity)
MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]
SHORT = {m: m.split("/")[-1] for m in MODELS}


def load_matrices(dataset: str):
    """加载 pkl, 返回质量矩阵 Q [N,K]、成本矩阵 C [N,K]、eval_name 数组."""
    df = pd.read_pickle(DATA_DIR / f"routerbench_{dataset}.pkl")
    Q = np.stack([df[m].to_numpy(dtype=np.float64) for m in MODELS], axis=1)
    C = np.stack(
        [df[f"{m}|total_cost"].to_numpy(dtype=np.float64) for m in MODELS], axis=1
    )
    eval_names = df["eval_name"].to_numpy()
    return Q, C, eval_names


def evaluate(choices: np.ndarray, Q: np.ndarray, C: np.ndarray) -> dict:
    """choices: 每个查询选中的模型索引 [N]. 返回指标."""
    n = len(choices)
    q = Q[np.arange(n), choices]
    c = C[np.arange(n), choices]
    return {
        "avg_quality": float(np.nanmean(q)),
        "total_cost": float(np.nansum(c)),
        "cost_per_query": float(np.nanmean(c)),
    }


# ---------------- 基线策略 ----------------

def policy_random(K, rng):
    return lambda n: rng.integers(0, K, size=n)


def policy_always(idx):
    return lambda n: np.full(n, idx)


def policy_oracle(Q, C):
    """每查询选质量最高者; 并列取最便宜. 理论上界."""

    def fn(n):
        best_q = np.nanmax(Q, axis=1, keepdims=True)
        tied = np.isclose(Q, best_q, equal_nan=True)
        cost_masked = np.where(tied, C, np.inf)
        return np.argmin(cost_masked, axis=1)

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="5shot", choices=["5shot", "0shot"])
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    Q, C, eval_names = load_matrices(args.dataset)
    N, K = Q.shape
    print(f"dataset={args.dataset}  N={N}  K={K}")

    rng = np.random.default_rng(42)
    idx_gpt4 = MODELS.index("gpt-4-1106-preview")
    idx_cheap = int(np.argmin(np.nanmean(C, axis=0)))  # 平均成本最低模型

    strategies = {
        "random": policy_random(K, rng),
        "always_gpt4": policy_always(idx_gpt4),
        f"always_cheapest({SHORT[MODELS[idx_cheap]]})": policy_always(idx_cheap),
        "oracle": policy_oracle(Q, C),
    }

    results = {}
    for name, fn in strategies.items():
        t0 = time.time()
        metrics = evaluate(fn(N), Q, C)
        metrics["wall_time_s"] = round(time.time() - t0, 2)
        results[name] = metrics
        print(
            f"{name:34s} quality={metrics['avg_quality']:.4f}  "
            f"total_cost=${metrics['total_cost']:.2f}  "
            f"per_query=${metrics['cost_per_query']:.6f}"
        )

    out = RESULTS_DIR / f"naive_baselines_{args.dataset}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
