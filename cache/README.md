# cache/ — precomputed neural artifacts

The pipeline reads its neural features from this folder instead of running the encoders at
inference time. This is what lets the submission notebook run **without loading a single
model** — `transformers` is never imported on that path. It reads these arrays and does
deterministic NumPy dot products, so the output does not depend on which `torch` /
`transformers` build the evaluation machine happens to have installed.

## Committed to git (small)

| File | Size | Contents |
|---|---|---|
| `query_emb.npz` | 143 KB | SapBERT + BioLORD embeddings of the 23 expanded test queries |
| `ce_scores.json` | 10 KB | MedCPT cross-encoder max-probability per (condition, borderline category) |

## Not committed (large, regenerable)

| File | Size | Contents |
|---|---|---|
| `sapbert_icd_titles.npy` | 299 MB | SapBERT CLS embeddings of all 97,441 dictionary titles |
| `biolord_icd_titles.npy` | 299 MB | BioLORD-2023 mean-pooled embeddings of the same titles |

These exceed GitHub's 100 MB per-file limit. Two ways to get them:

**A. Download** — attached as assets to the repository's latest
[Release](../../releases/latest). Drop both `.npy` into this folder.

**B. Rebuild** (~1h40m on one CPU thread, no GPU needed):

```bash
python scripts/export_models.py           # once, needs internet — fetches the 3 encoders
python scripts/build_dict_embeddings.py   # offline from here on

# ~2.6x faster, verified bit-identical on a 512-title slice:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/build_dict_embeddings.py
```

The rebuild is verified to reproduce the shipped arrays **bit-for-bit** (`max|diff| = 0.0`
over a 512-title slice against the released arrays), because the pooling recipe, tokenizer
`max_length`, batch sizes and dtype are pinned in the script. Row *i* of each array
corresponds to row *i* of `mimic-iv_icd-10_dict.xlsx`.

`query_emb.npz` and `ce_scores.json` are themselves derived from the two `.npy` files and
can be regenerated with `python scripts/precompute_neural.py` once those are present.
