"""绘制 PaceRouter 系统框架图 (Figure 1), 300dpi."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "results" / "fig_framework.png"

fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

def box(x, y, w, h, text, fc, ec="black", fs=9, lw=1.2, style="round,pad=0.25", tc="black", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(x1, y1, x2, y2, color="black", lw=1.6, ls="-", label=None, lab_dx=0, lab_dy=1.2, conn="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls, connectionstyle=conn))
    if label:
        ax.text((x1 + x2) / 2 + lab_dx, (y1 + y2) / 2 + lab_dy, label,
                fontsize=8, color=color, ha="center")

# ---- 离线训练 (左上虚线区) ----
box(2, 44, 30, 13, "Offline (historical data)\nTrain quality & cost\npredictors", "#f2f2f2", ls="--", fs=8)

# ---- 在线主链路 ----
box(2, 26, 15, 10, "User query\n$x_t$", "#eaf2fb", fs=10)
box(24, 32, 22, 9, "Quality predictor\n$\\hat{q}_i(x_t)$", "#d5f5e3", fs=9)
box(24, 20, 22, 9, "Cost predictor\n$\\hat{c}_i(x_t)$  (per-query)", "#d5f5e3", fs=9)
box(53, 25, 21, 11, "Router scoring\n$i^* = \\arg\\max_i [\\hat{q}_i - \\lambda_t \\hat{c}_i]$", "#aed6f1", fs=9)
box(80, 25, 18, 11, "LLM pool\n$K$ models\ncheap $\\leftrightarrow$ strong", "#fdf2e9", fs=9)

# 箭头: 主链路
arrow(17, 31, 24, 35.5, conn="arc3,rad=-0.15")
arrow(17, 29.5, 24, 24.5, conn="arc3,rad=0.15")
arrow(46, 36.5, 53, 33, conn="arc3,rad=-0.1")
arrow(46, 24.5, 53, 28.5, conn="arc3,rad=0.1")
arrow(74, 30.5, 80, 30.5, label="serve $x_t$", lab_dy=1.5)
# 离线 -> 在线 (虚线)
arrow(17, 44, 35, 41.2, color="#7f8c8d", ls="--", conn="arc3,rad=-0.2")

# ---- 底部: 预算节奏器 + 安全网 ----
box(30, 3, 34, 11, "Budget pacer (shadow price)\n$\\lambda \\leftarrow \\lambda \\cdot \\mathrm{clip}(\\Delta B_j / quota)$\nremaining budget $R_t$, time progress", "#fdebd0", fs=8.5)
box(70, 3, 27, 11, "Safety net\nif $R_t \\leq (T-t)\\cdot \\tilde{c}$:\nforce cheapest model", "#fadbd8", fs=8.5)

# 反馈回路: 模型 -> 结算 -> pacer (走模型池右侧外缘, 避开 safety net 框)
arrow(98.3, 28, 98.3, 1.2, color="#c0392b", conn="arc3,rad=0")
arrow(98.3, 1.2, 47, 1.2, color="#c0392b", conn="arc3,rad=0")
arrow(47, 1.2, 47, 2.9, color="#c0392b", conn="arc3,rad=0")
ax.text(72, 2.3, "realized cost $c_{i^*}(x_t)$", fontsize=7.5, color="#c0392b", ha="center")
# pacer -> scoring (更新 lambda)
arrow(47, 14, 60, 25, color="#e67e22", lw=2.0, label="update $\\lambda_t$", lab_dx=-7, lab_dy=0, conn="arc3,rad=-0.25")
# 安全网 -> scoring (红色虚线)
arrow(70, 14, 68, 25, color="#c0392b", ls="--", conn="arc3,rad=0.2")

ax.text(50, 58, "PaceRouter: online serving loop with budget feedback", ha="center", fontsize=11, weight="bold")

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
