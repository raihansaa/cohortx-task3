
from __future__ import annotations

# Determinism / offline: must be set before torch is imported.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
DICT_NAME = "mimic-iv_icd-10_dict.xlsx"

# (hub id, local dir name, pooling, batch size, output file)
SPECS = [
    ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
     "SapBERT-from-PubMedBERT-fulltext", "cls", 256, "sapbert_icd_titles.npy"),
    ("FremyCompany/BioLORD-2023",
     "BioLORD-2023", "mean", 128, "biolord_icd_titles.npy"),
]
MAX_LEN = 64


def find_dict() -> Path:
    """Locate the competition dictionary xlsx. TASK3_DATA_DIR overrides."""
    roots = []
    env = os.environ.get("TASK3_DATA_DIR")
    if env:
        roots += [Path(x) for x in env.split(os.pathsep) if x]
    roots += [ROOT, ROOT / "cohort-x-task-3", ROOT / "data", ROOT.parent,
              ROOT.parent / "cohort-x-task-3"]
    for r in roots:
        cand = r / DICT_NAME
        if cand.exists():
            return cand
    for r in roots:
        if r.exists():
            for hit in r.rglob(DICT_NAME):
                return hit
    sys.exit(
        f"Could not find {DICT_NAME}. Download it from the competition data tab and place "
        f"it in {ROOT / 'cohort-x-task-3'}/, or set TASK3_DATA_DIR to its folder.")


def find_model(hub_id: str, local_name: str) -> str:
    """Prefer a local models/ copy so the build runs fully offline; else the hub id."""
    for base in (ROOT / "models", ROOT.parent / "models"):
        if (base / local_name / "config.json").exists():
            return str(base / local_name)
    return hub_id


@torch.no_grad()
def encode_all(titles: list[str], model_ref: str, pooling: str, batch: int) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_ref)
    mdl = AutoModel.from_pretrained(model_ref)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    print(f"  device={device} pooling={pooling} batch={batch}")

    outs = []
    t0 = time.time()
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc)
        if pooling == "cls":
            h = out.last_hidden_state[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        outs.append(h.cpu().numpy())
        if (i // batch) % 40 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(titles) - done) / max(rate, 1e-9)
            print(f"  {done:,}/{len(titles):,}  ({rate:,.0f} titles/s, ETA {eta / 60:.1f} min)")
    print(f"  encoded in {time.time() - t0:.1f}s")
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the .npy already exists")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    CACHE.mkdir(exist_ok=True)

    dict_xlsx = find_dict()
    print(f"[dict] {dict_xlsx}")
    df = pd.read_excel(dict_xlsx)
    df.columns = [c.strip() for c in df.columns]
    titles = df["long_title"].astype(str).tolist()
    print(f"[dict] {len(titles):,} titles\n")

    for hub_id, local_name, pooling, batch, out_name in SPECS:
        out_path = CACHE / out_name
        if out_path.exists() and not args.force:
            emb = np.load(out_path, mmap_mode="r")
            print(f"=== {out_name}: already present, shape={emb.shape} (use --force to rebuild)\n")
            continue

        model_ref = find_model(hub_id, local_name)
        print(f"=== {out_name}  <-  {model_ref}")
        emb = encode_all(titles, model_ref, pooling, batch)
        assert emb.shape[0] == len(titles), f"row mismatch: {emb.shape[0]} vs {len(titles)}"
        np.save(out_path, emb)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        print(f"  saved {out_path}  shape={emb.shape}  sha256={digest}\n")

    print("Done. Next: python scripts/precompute_neural.py  (rebuilds query_emb.npz + ce_scores.json)")


if __name__ == "__main__":
    main()
