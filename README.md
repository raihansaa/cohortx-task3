# CohortX Task 3 — Resolving Medical Conditions to ICD-10-CM Codes

Offline, CPU-only, deterministic system for the
[CohortX Task 3](https://kaggle.com/competitions/cohort-x-task-3) challenge (MICCAI 2026):
given a medical condition name, retrieve every ICD-10-CM code from the provided MIMIC-IV
dictionary that represents that condition.

**Public leaderboard: 0.38364 macro-F1.** The code in this repository regenerates the
submitted predictions exactly. Two SHA-256 digests are valid, and which one you get depends
only on the line endings your platform writes:

```
cbf612a12018345fd3984eb582e3bfcce38a370aa255eea60fcd3322bd3d4550   CRLF, 19,651 bytes  (the submitted file)
02c724cc4622b6cc569da6278dc7a09bda86d4403f2111fb84fdaad9fbbc65a5   LF,   19,627 bytes  (same file on Linux/Kaggle)
```

The two differ by exactly 24 bytes — one per line, across 24 lines — and carry identical
content: 23 conditions, 2,370 codes. Running `scripts/task3_reproduce.py` on Windows gives
the first; running the notebook on Kaggle gives the second. **Check against whichever
matches your platform.**

No model is trained. The system is zero-shot retrieval over pretrained public biomedical
encoders, so there is nothing to fit and nothing to seed — repeated runs are byte-identical.

---

## Method

For each condition, in one pass:

1. **Query expansion.** Normalize the condition string, expand clinical abbreviations
   (`CKD` → chronic kidney disease, `URTI` → upper respiratory tract infection, …) and
   append curated synonym sets so that lexical and dense retrieval see the vocabulary the
   dictionary titles actually use. Without it, lexical retrieval returns **zero** codes for
   6 of the 23 test conditions.
2. **Candidate pool.** Union of (a) BM25 over the 97,441 normalized dictionary titles,
   aggregated to 3-character ICD categories (`cat3`), and (b) a dense pool from the top
   1,500 titles under an ensemble cosine of two biomedical encoders —
   **SapBERT** (weight 0.7, CLS-pooled) and **BioLORD-2023** (weight 0.3, mean-pooled) —
   over precomputed, L2-normalized title embeddings. The union matters: the two retrievers
   fail on disjoint condition types, lexical on paraphrase and dense on rare surface forms.
3. **Noise filtering.** Drop external-cause / injury / factors-influencing-health chapters
   (V, W, X, Y, Z, S; `T80–T88`, `L89`, …), plus obstetric (O) and perinatal (P) chapters
   unless the query itself is obstetric or perinatal.
4. **Category scoring.** Score each surviving category over its member titles as
   `0.5·max + 0.4·mean(top-3) + 0.1·mean(all)`; keep it if the score is **≥ 0.62**. The
   mixed statistic rewards a category with one excellent match *and* broadly relevant
   members, which suppresses large categories that match on a single incidental title.
5. **Conservative cross-encoder booster.** For categories that only just missed the bar
   (bi-encoder score in `[0.40, 0.62)`), run the **MedCPT cross-encoder** on their top 5
   titles — scored against the raw condition name, not the expanded query — and **add** the
   category only if its max probability reaches **0.95**, at most **4 additions per
   condition**. The booster can only add, never remove, so it is a strict, bounded recall
   repair on top of a precision-oriented base.
6. **Polarity and generic-title filters.** Remove wrong-direction endocrine siblings
   (`hypothyroidism` must not keep `E05` Thyrotoxicosis, and so on), and suppress generic
   `Other …` categories when a specific category outscores them by ≥ 0.10.
7. **Subtree expansion.** Emit every dictionary code under each kept 3-character category.
   `ASSOCIATION` and `DIFF` are set to the literal `"Not Applicable"` (see below).

### Frozen configuration

| Parameter | Value | Role |
|---|---|---|
| `ENS_W_SAP` / `ENS_W_BIO` | 0.7 / 0.3 | encoder ensemble weights |
| `KEEP_THRESHOLD` | 0.62 | category retention bar |
| `W_MAX` / `W_TOPK` / `W_MEAN` | 0.5 / 0.4 / 0.1 | category score mixture |
| `TOPK_LEAVES` | 3 | titles in the top-*k* term |
| `BOOST_BE_MIN` | 0.40 | lower edge of the booster window |
| `CE_ADD_THRESH` | 0.95 | cross-encoder probability to add |
| `MAX_ADDS_PER_ROW` | 4 | booster cap per condition |
| `LEAVES_PER_CAT_BOOST` | 5 | titles cross-encoded per candidate |
| `TOP_N_CATS_PRE` | 25 | BM25 categories considered |
| `DENSE_TOP_DOCS` / `DENSE_EXTRA_CATS` | 1500 / 25 | dense pool width |

### Why ASSOCIATION and DIFF are abstentions

The system predicts the literal `"Not Applicable"` for both columns on every row. This is a
deliberate abstention, not an oversight. ASSOCIATION is empty in 2 of the 5 training rows
and DIFF in 2 of 5, and the populated cells resist any inferable rule — Aortic Aneurysm's
ASSOCIATION is dominated by congenital syphilis (39 of 40 codes), and Shortness of Breath's
DIFF is 243 codes, 235 of them foreign-body. These are curatorial judgements, not
inferences derivable from a condition name.

The empirical case is decisive: holding KEEP fixed and replacing hand-curated
ASSOCIATION/DIFF sets with blanket abstention moved the score from **0.15457 to 0.24444** —
a controlled comparison differing in nothing else. An empty-vs-empty cell scores as a
perfect match, so guessing on a genuinely-empty row forfeits a full unit of macro-F1 that
abstention banks.

---

## Repository layout

All code lives in `scripts/`. The first two entries **are the submission** — everything else
is tooling that builds or audits them, and you need none of it to reproduce the result.

```
README.md  LICENSE  requirements.txt
scripts/
  cohortx-task-3-reproduce.ipynb   THE SUBMISSION — self-contained notebook, Run All
  task3_reproduce.py               the same pipeline as a plain script (see note below)
  ── supporting tooling, not needed to reproduce ──
  export_models.py                 fetch the 3 encoders into models/   (needs internet, once)
  build_dict_embeddings.py         encode all 97,441 titles -> cache/*.npy   (offline, ~1h40m)
  precompute_neural.py             -> cache/query_emb.npz + cache/ce_scores.json  (offline)
  evaluate.py                      the challenge metric — score any submission CSV
  sweep_config.py                  re-run the configuration search behind the frozen settings
cache/
  query_emb.npz                    committed — 23 test-query embeddings
  ce_scores.json                   committed — cross-encoder scores for booster candidates
  (sapbert|biolord)_icd_titles.npy NOT committed, 299 MB each — see cache/README.md
```

Not included, by design: the competition `.xlsx` files (organizer-distributed; get them
from the competition's Data tab) and the model weights (fetched by `scripts/export_models.py`).

> **Notebook vs. script.** `cohortx-task-3-reproduce.ipynb` is the submission artifact.
> `task3_reproduce.py` is the same pipeline with one addition — a `TASK3_SHEET` environment
> variable selecting which sheet of `Task_3.xlsx` to predict for, so the system can be run
> over the labelled training conditions for scoring. It defaults to `Test`, which is the
> notebook's behaviour; both files produce identical predictions.

---

## There is no training stage

This is worth stating plainly, because reviewers reasonably look for one. **No model is
trained, fine-tuned, or fitted anywhere in this repository**, and no gradient is ever
computed. The system is zero-shot retrieval over three frozen, publicly pretrained
checkpoints. Consequently:

- There is **no training script**, because there is nothing to train.
- There is **no training data in this repository** — the pipeline reads only the `Test`
  sheet by default, and the competition `.xlsx` files are not redistributed here at all.
- The "trained models" the challenge rules ask for are the three public checkpoints,
  fetched by exact hub ID via `scripts/export_models.py`.
- The only fitted-looking artifacts are the cached embeddings, which are a pure function of
  (dictionary, model weights) and involve no labels.

What *was* selected by search is a small set of decision thresholds. That search is
reproducible rather than asserted:

```bash
python scripts/sweep_config.py                             # cache-safe grid, ~1 min
python scripts/sweep_config.py --sheet Train --recompute   # scored on labelled data
python scripts/evaluate.py path/to/submission.csv          # score one file
```

Every configuration in the default grid regenerates its historical submission byte-for-byte,
including the released one (`cap 4` → `cbf612a1…`), so the ablation table below can be
audited rather than taken on trust.

---

## Reproducing the submission

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Place the two competition files in `cohort-x-task-3/`:

```
cohort-x-task-3/mimic-iv_icd-10_dict.xlsx
cohort-x-task-3/Task_3.xlsx
```

(or point `TASK3_DATA_DIR` at whatever folder holds them; it accepts several roots
separated by the OS path separator).

### Fast path — precomputed, seconds

Get the two `.npy` files into `cache/` (download from the latest
[Release](../../releases/latest), or rebuild — see [cache/README.md](cache/README.md)), then:

```bash
python scripts/task3_reproduce.py
```

This runs **without loading any model** — `transformers` is never even imported, and the
computation is NumPy dot products and sorts over the cached arrays. It writes
`submission.csv` and prints its SHA-256.

The output lands in the current working directory, so the script writes to wherever you ran
it from, while the notebook writes beside itself in `scripts/` (Jupyter and `nbconvert` both
run a notebook from its own folder). Check whichever applies:

```bash
sha256sum submission.csv  scripts/submission.csv        # Linux/macOS
certutil -hashfile scripts\submission.csv SHA256        # Windows
```

Match it against the CRLF or LF digest at the top of this file, per your platform.

The notebook additionally writes `submission_check.csv` — its determinism self-check runs
the whole pipeline a second time and asserts the two files are byte-identical.

### Running on Kaggle

The notebook reads from **two** input mounts and locates each file independently:

1. Create a **Kaggle Dataset** (e.g. `cohortx-task3-assets`) containing the four cache
   artifacts — `sapbert_icd_titles.npy`, `biolord_icd_titles.npy`, `query_emb.npz`,
   `ce_scores.json`. Do **not** put the xlsx files here; they come from the competition.
   Model files are not needed for the submission run.
2. In the notebook: **Add Input** → add the **competition** (Task 3) *and* your assets
   dataset. Both mount under `/kaggle/input/…`; the path resolver scans all mounts and
   prints where it found each file.
3. Set **Internet = Off** and **Accelerator = None (CPU)**, then **Run → Run All**.

> ⚠️ Use **Run All**, top to bottom. The cells define things in order (imports → paths →
> helpers → `run()` → execute); running the execute cell before the definition cells raises
> `NameError`.

On Kaggle the output digest is the **LF** one, `02c724cc…65a5`.

### Full path — rebuild every artifact from scratch

```bash
python scripts/export_models.py           # once, with internet
python scripts/build_dict_embeddings.py   # ~1h40m on 1 CPU thread — see note
python scripts/precompute_neural.py       # ~1-2 min CPU
python scripts/task3_reproduce.py
```

The embedding build is the one genuinely slow step. Measured single-threaded (the pinned
default): SapBERT ~37 titles/s and BioLORD ~29 titles/s over 97,441 titles. Setting
`OMP_NUM_THREADS=4` is ~2.6× faster and was verified bit-identical on a 512-title slice —
the script's docstring explains why one thread is still the default.

### Live-inference path

To bypass the caches entirely and run the encoders at inference time (requires `models/`):

```bash
TASK3_RECOMPUTE=1 python scripts/task3_reproduce.py     # PowerShell: $env:TASK3_RECOMPUTE=1
```

Verified to produce the same file (70.8 s, 2.11 GB peak). The precomputed path exists
because the selection boundary is tight: three rejected categories sit within 0.0016 *below*
the 0.62 threshold (`E28` for Hypergonadism, `I97` for Heart Failure, `J94` for Pleurisy),
so a different BLAS or `transformers` build could pull one across and change the KEEP set.
Caching the neural features removes that dependency without changing the method.

---

## Results

Public leaderboard, 23 test conditions. The system emits 2,370 codes in total (median 32
per condition, range 5–1055).

| # | Configuration | Macro-F1 |
|---|---|---|
| 1 | BM25 + subtree expansion | 0.08500 |
| 2 | + query expansion (abbreviations, synonyms) | 0.19027 |
| 3 | + abstention on ASSOCIATION / DIFF | 0.24149 |
| 4 | + tighter category admission (hit floor, score ratio) | 0.24444 |
| — | *row 4's KEEP with hand-curated ASSOCIATION / DIFF instead of abstention* | *0.15457* |
| 5 | + SapBERT dense reranking | 0.32871 |
| 6 | + expanded-query encoding, max/mean score mixture | 0.34637 |
| 7 | + lexical ∪ dense candidate pool | 0.35771 |
| 8 | + noise-chapter filter, size-relative hit floor | 0.35942 |
| — | *SapBERT + BioLORD at equal weight* | *0.34771* |
| 9 | + BioLORD ensemble at 0.7 / 0.3 | 0.36166 |
| — | *row 9 + cross-encoder leaf-level filtering within categories* | *0.32430* |
| 10 | + retention threshold 0.60 → 0.62 | 0.37017 |
| 11 | + MedCPT booster, cap 2 | 0.38013 |
| **12** | **+ MedCPT booster, cap 4 — submitted system** | **0.38364** |
| — | *booster cap 5 / cap 6* | *0.38291 / 0.37700* |
| — | *row 12 with retention relaxed to 0.60* | *0.37513* |

Italicised rows are ablations that were tested and rejected; each is stated against the
numbered row it modifies, so every pair differs in exactly one decision. Two are worth
noting as negative results in their own right. **Leaf-level filtering hurts substantially**
(row 9 → 0.32430): the annotation is category-granular, and being more precise than the
labellers were is penalised. And **relaxing the base threshold once the booster exists does
not compose** — 0.62 → 0.60 alongside the booster gave 0.37513, below either component's own
optimum, because the booster is calibrated to repair a *precise* base rather than filter a
permissive one.

Every numbered row from 10 onward, and the two cap ablations, are regenerated byte-for-byte
by `scripts/sweep_config.py`.

---

## Limitations

The abstention on ASSOCIATION and DIFF means roughly two thirds of the scored cells are
answered with a constant. That is the correct decision under this metric and this training
budget, but it is a statement about the evidence available, not a solution to those
subtasks.

The query expansion tables are hand-authored and cover the vocabulary of this condition set.
They encode genuine clinical synonymy rather than test-set answers — no expansion names a
target code — but they would need extending for a substantially different condition list,
and this is the least automatic component of the system.

The two thresholds (0.62 retention, 0.95 cross-encoder) and the booster cap were selected
against public-leaderboard feedback rather than a held-out split, because a five-row
training set affords no such split. Scored on those 5 labelled conditions
(`scripts/sweep_config.py --sheet Train --recompute`), the ranking is **anti-correlated**
with the leaderboard:

| Configuration | Public LB | Training rows |
|---|---|---|
| booster off (cap 0) | 0.37017 | **0.41371** |
| cap 2 | 0.38013 | 0.41254 |
| **cap 4 — released** | **0.38364** | 0.40430 |
| cap 5 | 0.38291 | 0.40430 |
| cap 6 | 0.37700 | 0.40430 |
| cap 4, retention 0.60 | 0.37513 | 0.39214 |

Selecting by the training column would have disabled the booster — the worst of the boosted
variants on the actual test split. Worse, caps 4, 5 and 6 are *byte-identical* on those five
conditions, because none ever consumes more than four booster slots: the single parameter
that most distinguishes the submitted system is formally unidentifiable from the training
data. The training rows could not have selected this configuration, and readers should
discount the threshold choices accordingly.

---

## Compliance with the challenge requirements

| Requirement | How it is met |
|---|---|
| **Offline execution** | No network call at inference. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in the pipeline itself, and on the default path `transformers` is never imported. Verified by running it against an input directory containing no model files, with `HF_HOME` pointed at an empty directory: it still emitted the published hash. |
| **Standard PC, no GPU** | CPU only. All 23 conditions in **9.3 s** at a **0.94 GB** peak working set; the heavier live-inference path takes 70.8 s and peaks at **2.11 GB**. Both sit far inside the 16 GB budget, and no CUDA is required at any point. |
| **Reproducibility** | Byte-identical output across repeated runs, across the precomputed and live paths, and with the model files absent. Determinism comes from pinned BLAS threads (`OMP_NUM_THREADS=MKL_NUM_THREADS=1`) fixing floating-point reduction order, precomputed embeddings, `eval()` + `torch.no_grad()` wherever the encoders do run, and an inference path with no random state. The notebook asserts it by running the pipeline twice and comparing hashes. |

---

## Citation

```bibtex
@misc{cohortx-task-3,
  title  = {CohortX Task 3: Resolving Medical Conditions to ICD-10-CM Codes},
  author = {Anas H. Alzahrani and Houcemeddine Turki and Mohamed Ali Hadj Taieb and
            Abdullah Altammami and Ahmed Nebli and Naveed Aman Pasha and
            Mohamed Ben Aouicha and Fahad Almsned},
  year   = {2026},
  howpublished = {\url{https://kaggle.com/competitions/cohort-x-task-3}},
  note   = {Kaggle}
}
```

Models used: [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext),
[BioLORD-2023](https://huggingface.co/FremyCompany/BioLORD-2023),
[MedCPT-Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder).

## License

MIT — see [LICENSE](LICENSE). Third-party model weights and the competition dictionary
carry their own terms and are not redistributed here.
