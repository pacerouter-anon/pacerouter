"""T2a: 多种子方差实验. 5 种子 × {shuffle,drift} × γ∈{0.25,0.5,0.75}, 对 Static(GBR) 与 PaceRouter 收集均值±std."""
import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")
out_path = RESULTS / "multi_seed_0shot.json"

# 断点续跑: 读已有结果, 跳过已完成配置
all_runs = {}
if out_path.exists():
    all_runs = json.loads(out_path.read_text())

combos = []
for seed in [42, 43, 44, 45, 46]:
    for stream in ["shuffle", "drift"]:
        for gamma in [0.25, 0.5, 0.75]:
            combos.append((seed, stream, gamma))

for seed, stream, gamma in combos:
    tag = f"ms_{seed}_{stream}_{gamma}"
    if tag in all_runs and all_runs[tag]:
        print(f"skip {tag} (cached)", flush=True)
        continue
    r = subprocess.run(
        ["../.venv/bin/python", "pacerouter.py", "--dataset", "0shot",
         "--gamma", str(gamma), "--stream", stream, "--seed", str(seed)],
        capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("  B")]
    entry = {}
    for l in lines:
        parts = l.split()
        name = parts[0]
        q = float(parts[1].split("=")[1])
        over = float(parts[3].split("=")[1].rstrip("%"))
        unused = float(parts[4].split("=")[1].rstrip("%"))
        entry[name] = dict(q=q, overspend=over, unused=unused)
    all_runs[tag] = entry
    out_path.write_text(json.dumps(all_runs, indent=2))  # 每个配置落盘
    print(f"done {tag}", flush=True)

out_path.write_text(json.dumps(all_runs, indent=2))
print("saved ->", out_path)

# 汇总
import numpy as np
print("\n=== 汇总 (mean ± std over 5 seeds) ===")
for stream in ["shuffle", "drift"]:
    for gamma in [0.25, 0.5, 0.75]:
        for method in ["B3_static", "B5_pacerouter"]:
            qs, overs, unuseds = [], [], []
            for seed in [42, 43, 44, 45, 46]:
                e = all_runs[f"ms_{seed}_{stream}_{gamma}"].get(method)
                if e:
                    qs.append(e["q"]); overs.append(e["overspend"]); unuseds.append(e["unused"])
            print(f"{stream:8s} γ={gamma} {method:15s} q={np.mean(qs):.4f}±{np.std(qs):.4f} "
                  f"over={np.mean(overs):.2f}±{np.std(overs):.2f}% unused={np.mean(unuseds):.2f}±{np.std(unuseds):.2f}%")
