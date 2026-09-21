"""T3a: 输出长度真实分布证据 (直方图 + CCDF + 偏度 + P99/P50).

RouterBench 记录含 model_response 文本; 用字符数/4 近似 token 数 (GPT 系英文约 4 字符/token),
中文按字符计. 目的: 支撑论文 "right-skewed generation-cost distribution" 立论.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from replay import DATA_DIR, MODELS

df = pd.read_pickle(DATA_DIR / "routerbench_5shot.pkl")

lengths = {}
for m in MODELS:
    col = f"{m}|model_response"
    if col in df.columns:
        resp = df[col].fillna("").astype(str)
        lengths[m] = (resp.str.len() / 4.0).to_numpy()  # 字符数/4 ≈ token 数

# 汇总统计
print(f"{'模型':<28} {'均值':>7} {'P50':>6} {'P99':>7} {'P99/P50':>8} {'偏度':>7}")
stats_rows = []
for m, arr in lengths.items():
    arr = arr[arr > 0]
    p50, p99 = np.percentile(arr, 50), np.percentile(arr, 99)
    sk = float(stats.skew(arr))
    short = m.split("/")[-1]
    print(f"{short:<28} {arr.mean():>7.0f} {p50:>6.0f} {p99:>7.0f} {p99/max(p50,1):>8.1f} {sk:>7.1f}")
    stats_rows.append(dict(model=short, mean=float(arr.mean()), p50=float(p50),
                           p99=float(p99), p99_p50=float(p99/max(p50, 1)), skew=sk))

# 图: 左直方图(对数轴), 右 CCDF(对数轴)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
colors = plt.cm.tab20(np.linspace(0, 1, len(lengths)))
for (m, arr), c in zip(lengths.items(), colors):
    arr = arr[arr > 0]
    ax1.hist(arr, bins=80, histtype="step", lw=1.2, density=True, color=c)
for (m, arr), c in zip(lengths.items(), colors):
    arr = np.sort(arr[arr > 0])
    ccdf = 1 - np.arange(len(arr)) / len(arr)
    ax2.plot(arr, ccdf, lw=1.4, color=c, label=m.split("/")[-1])
ax1.set_yscale("log"); ax1.set_xlabel("output length (tokens, approx.)"); ax1.set_ylabel("density (log)")
ax1.set_title("Output-length distribution by model")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("output length (tokens, approx.)"); ax2.set_ylabel("CCDF")
ax2.set_title("CCDF (log-log) — right-skew evidence")
ax2.legend(fontsize=6, ncol=2)
fig.tight_layout()
out = Path("results/fig_length_dist.png")
fig.savefig(out, dpi=200)
print("saved ->", out)
Path("results/length_stats.json").write_text(json.dumps(stats_rows, indent=2))
