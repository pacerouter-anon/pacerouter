"""离线计算全量 prompt embedding 并缓存 (all-MiniLM-L12-v2, 与 RouterBench 论文一致)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from replay import DATA_DIR
from static_router import normalize_prompt

for ds in ["5shot", "0shot"]:
    out = DATA_DIR / f"emb_{ds}.npy"
    if out.exists():
        print(f"{out} 已存在, 跳过")
        continue
    df = pd.read_pickle(DATA_DIR / f"routerbench_{ds}.pkl")
    texts = [normalize_prompt(p) for p in df["prompt"]]
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L12-v2", device="cpu")
    emb = model.encode(texts, batch_size=256, show_progress_bar=False,
                       normalize_embeddings=True)
    np.save(out, emb.astype(np.float32))
    print(f"{ds}: {emb.shape} 耗时 {time.time()-t0:.0f}s -> {out}")
