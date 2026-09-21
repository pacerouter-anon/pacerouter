"""从 results/*.json 自动生成论文 Table 素材 (Markdown)."""
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
METHODS = ["B1_random", "B2_always_cheapest", "B2_always_gpt4", "B4b_cascade",
           "B3b_static_knn", "B3_static", "B4_static_hardstop", "B5_pacerouter", "B6_oracle_pacing"]
NAMES = {"B1_random": "Random", "B2_always_cheapest": "Always-Cheapest",
         "B2_always_gpt4": "Always-GPT4", "B4b_cascade": "Cascade",
         "B3b_static_knn": "Static (KNN)", "B3_static": "Static (GBR)",
         "B4_static_hardstop": "Static+HardStop", "B5_pacerouter": "PaceRouter",
         "B6_oracle_pacing": "Oracle-Pacing"}

def load(dataset, stream):
    rows = []
    for f in sorted(RESULTS.glob(f"pacing_{dataset}_gbr-predict_g*_{stream}.json")):
        g = float(f.stem.split("_g")[-1].split("_")[0])
        rows.append((g, json.loads(f.read_text())))
    return sorted(rows)

lines = ["# 主实验结果汇总（Table 素材）\n"]
for ds in ["5shot", "0shot"]:
    for st in ["shuffle", "drift"]:
        rows = load(ds, st)
        if not rows:
            continue
        lines.append(f"\n## {ds} / {st}\n")
        lines.append("| γ | 方法 | 质量 | 超支% | 浪费% |")
        lines.append("|---|---|---|---|---|")
        for g, r in rows:
            for m in METHODS:
                if m not in r:
                    continue
                v = r[m]
                name = NAMES[m] + (" (ours)" if m == "B5_pacerouter" else "")
                lines.append(f"| {g} | {name} | {v['avg_quality']:.4f} | "
                             f"{v['overspend']*100:.2f} | {v['unused']*100:.2f} |")
        lines.append("")

out = RESULTS / "主结果汇总.md"
out.write_text("\n".join(lines))
print("saved ->", out)
print("\n".join(lines)[:600])
