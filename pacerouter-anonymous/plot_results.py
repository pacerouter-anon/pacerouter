"""汇总 pacing 实验结果并绘图 (Figure 2 主结果 + Figure 4 λ 轨迹)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
METHODS = ["B3_static", "B4_static_hardstop", "B5_pacerouter", "B6_oracle_pacing"]
LABELS = ["Static", "Static+HardStop", "PaceRouter (ours)", "Oracle-Pacing (upper)"]
COLORS = ["#c0392b", "#e67e22", "#2980b9", "#7f8c8d"]


def collect(dataset, stream, qpred_tag="gbr-predict"):
    rows = []
    for f in sorted(RESULTS.glob(f"pacing_{dataset}_{qpred_tag}_g*_{stream}.json")):
        gamma = float(f.stem.split("_g")[-1].split("_")[0])
        r = json.loads(f.read_text())
        rows.append((gamma, r))
    return sorted(rows)


def plot_quality_and_overspend(dataset, stream):
    rows = collect(dataset, stream)
    if not rows:
        return
    gammas = [g for g, _ in rows]
    x = np.arange(len(gammas))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (m, lb, c) in enumerate(zip(METHODS, LABELS, COLORS)):
        qs = [r[m]["avg_quality"] for _, r in rows]
        ax1.bar(x + i * width, qs, width, label=lb, color=c)
        # 预算执行偏差 = 超支 + 浪费 (越低越好), 一张图同时容纳两种失败模式
        dev = [r[m]["overspend"] * 100 + r[m]["unused"] * 100 for _, r in rows]
        ax2.bar(x + i * width, dev, width, label=lb, color=c)
    ax1.set_xticks(x + 1.5 * width); ax1.set_xticklabels([f"{g}" for g in gammas])
    ax1.set_xlabel("budget ratio $\\gamma$"); ax1.set_ylabel("avg quality")
    ax1.set_title(f"Quality under budget ({dataset}, {stream})")
    ax1.legend(fontsize=8)
    ax2.set_xticks(x + 1.5 * width); ax2.set_xticklabels([f"{g}" for g in gammas])
    ax2.set_xlabel("budget ratio $\\gamma$"); ax2.set_ylabel("budget deviation %")
    ax2.set_title("Budget deviation (overspend+unused, lower is better)")
    fig.tight_layout()
    out = RESULTS_DIR_PLOT / f"fig_main_{dataset}_{stream}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def plot_lambda_traj(dataset, gamma, stream, qpred_tag="gbr-predict"):
    tag = f"{dataset}_{qpred_tag}_g{gamma}_{stream}"
    lam_f = RESULTS / f"lam_traj_{tag}.npy"
    sp_f = RESULTS / f"spend_traj_{tag}.npy"
    js_f = RESULTS / f"pacing_{tag}.json"
    if not lam_f.exists():
        return
    lam, spend = np.load(lam_f), np.load(sp_f)
    # 从结果反推预算 B 用于画参考线
    B = None
    if js_f.exists():
        v = json.loads(js_f.read_text())["B5_pacerouter"]
        if v["overspend"] > 0:
            B = v["spent"] / (1 + v["overspend"])
        elif v["unused"] > 0:
            B = v["spent"] / (1 - v["unused"])
        else:
            B = v["spent"]
    # 修正: x 轴为真实时间进度 (chunk 记录点映射到 (j+1)/J, 末点=1.0)
    t = (np.arange(len(lam)) + 1) / len(lam)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(t, lam, color="#2980b9", lw=1.8, label="shadow price $\\lambda_t$")
    ax1.set_xlabel("time progress (fraction of serving window)")
    ax1.set_ylabel("$\\lambda_t$", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax2 = ax1.twinx()
    ax2.plot(t, spend, color="#c0392b", lw=1.8, label="cumulative spend")
    if B is not None:
        ax2.axhline(B, color="black", ls="--", lw=1, label="budget $B$")
        ax2.plot([0, 1], [0, B], color="gray", ls=":", lw=1, label="ideal even pace")
        ax2.set_xlim(0, 1.0)
        ax2.annotate(f"final spend ${spend[-1]:.2f} / budget ${B:.2f}",
                     xy=(0.55, 0.08), xycoords="axes fraction", fontsize=8, color="#c0392b")
    ax2.set_ylabel("cumulative spend ($)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax1.set_title(f"PaceRouter dynamics ({dataset}, {stream}, $\\gamma$={gamma})")
    fig.tight_layout()
    out = RESULTS_DIR_PLOT / f"fig_lambda_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


RESULTS_DIR_PLOT = RESULTS

if __name__ == "__main__":
    for stream in ["shuffle", "burst", "burst_hard_first", "drift"]:
        plot_quality_and_overspend("5shot", stream)
        for g in [0.25, 0.5]:
            plot_lambda_traj("5shot", g, stream)
