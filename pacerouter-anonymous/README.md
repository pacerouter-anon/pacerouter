# PaceRouter — Anonymous Reproduction Package

Anonymous code release for the paper *"Budget-Paced Routing for Large Language Model Serving: A Primal-Dual Framework with Regret Analysis"*.

## Contents

| File | Role |
|---|---|
| `replay.py` | Replay evaluation framework + naive baselines (random / always-extremes / oracle) |
| `static_router.py` | Static threshold baseline (shallow features + WTP scoring) |
| `pacerouter.py` | **PaceRouter main method** (bidirectional pacing + per-query cost prediction + safety net) |
| `cbwk_router.py` | Predictor-family variants (LinUCB with online updates, GBR-ensemble) |
| `pdbwk_fair.py` | Faithful PD-BwK baseline (UCB/LCB, monotone multiplicative update) |
| `noise_sweep.py` | E4: quality-predictor noise sweep |
| `cost_noise_sweep.py` | Cost-predictor noise sweep (verification of the λ̄ε_c term) |
| `revision_experiments.py` | Safety-net telemetry, adversarial oscillating traffic, clip sweep, calibration-size control |
| `multi_seed.py` / `multi_seed_0shot.py` | Multi-seed runs (5 seeds × 2 splits × 3 budget levels) |
| `per_task.py`, `window_quality.py`, `length_dist.py` | Per-task robustness, within-window quality curve, output-length distribution |
| `plot_results.py`, `plot_framework.py`, `summarize_results.py` | Figure/table generation |
| `embed_prompts.py` | Prompt embedding precomputation (all-MiniLM-L12-v2) |
| `results/` | All experiment outputs as JSON (every number in the paper is reproducible from these) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib scipy sentence-transformers
```

## Data

Download RouterBench from Hugging Face (`withmartian/routerbench`):
`routerbench_0shot.pkl` and `routerbench_5shot.pkl`, place under `data/`.
Then run `python embed_prompts.py` once (CPU, ~35 min per split) to build `data/emb_*.npy`.

## Quick reproduction of the main result (drift, γ=0.5, 5-shot)

```bash
python pacerouter.py --dataset 5shot --gamma 0.5 --stream drift
```

Expected output (seed 42): PaceRouter quality ≈ 0.6548, overspend ≈ 0%, unused ≈ 1.4%;
Static baseline quality ≈ 0.5700 with ≈ 35% of the budget unused.

## License

MIT (for review purposes; authors anonymous during double-blind review).
