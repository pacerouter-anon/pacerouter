"""真实生成核验实验 (测试机自包含版).

本机 CPU 用 Qwen2.5-0.5B-Instruct 对抽样查询真实生成, 记录 token 数,
核验成本模型 c = p_in*l_in + p_out*l_out 的结构与长度分布右偏性.
用法: python3 live_validation_remote.py  (在 deploy/ 目录下, 模型在 ./model)
"""
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

BASE = Path(__file__).parent
MAX_NEW = 256


def normalize_prompt(p) -> str:
    if isinstance(p, str) and p.startswith("[") and p.endswith("]"):
        try:
            parts = ast.literal_eval(p)
            if isinstance(parts, list):
                return "\n".join(str(x) for x in parts)
        except (ValueError, SyntaxError):
            pass
    return str(p)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_csv(BASE / "prompts.csv")
    prompts = df["prompt"].tolist()
    tasks = df["task"].tolist()
    print(f"共 {len(prompts)} 条查询")

    tok = AutoTokenizer.from_pretrained(str(BASE / "model"))
    model = AutoModelForCausalLM.from_pretrained(str(BASE / "model"), dtype=torch.float32)
    model.eval()
    torch.set_num_threads(min(64, os.cpu_count() or 16)) if (os := __import__("os")) else None

    records = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        text = normalize_prompt(p)
        msgs = [{"role": "user", "content": text[:4000]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"]
        n_in = ids.shape[1]
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)
        n_out = out.shape[1] - n_in
        records.append(dict(idx=i, task=tasks[i], n_in=int(n_in), n_out=int(n_out)))
        if (i + 1) % 5 == 0:
            dt = time.time() - t0
            print(f"{i+1}/{len(prompts)} 完成, 平均 {dt/(i+1):.1f}s/条", flush=True)

    df_r = pd.DataFrame(records)
    df_r.to_csv(BASE / "live_gen_records.csv", index=False)
    arr = df_r["n_out"].to_numpy(dtype=float)
    result = dict(
        n=len(arr), mean=float(arr.mean()), p50=float(np.percentile(arr, 50)),
        p99=float(np.percentile(arr, 99)), skew=float(stats.skew(arr)),
        p99_p50=float(np.percentile(arr, 99) / max(np.percentile(arr, 50), 1)),
        max=int(arr.max()),
    )
    for tk in set(tasks):
        a = df_r[df_r.task == tk]["n_out"].to_numpy(dtype=float)
        result[tk] = dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                          p99=float(np.percentile(a, 99)), skew=float(stats.skew(a)))
    (BASE / "live_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("saved -> live_validation.json + live_gen_records.csv")


if __name__ == "__main__":
    main()
