"""
CohortX Task 3 — offline reproduction.

METHOD (zero-shot retrieval; no model is trained):
  1. BM25 lexical retrieval over MIMIC-IV ICD-10 dictionary titles -> cat3 pool.
  2. Dense retrieval pool from an ensemble of two local biomedical encoders:
     SapBERT (0.7) + BioLORD-2023 (0.3), cosine over precomputed title embeddings.
  3. Noise-chapter filtering (external-cause / injury / factors chapters).
  4. Per-cat3 score = 0.5*max + 0.4*mean(top-3) + 0.1*mean(all); keep if >= 0.62.
  5. Conservative MedCPT cross-encoder booster: for borderline cat3s (bi-encoder
     score in [0.40, 0.62)), ADD the cat3 iff CE max-prob > 0.95, capped 4 per row.
  6. Polarity filter (drop wrong-direction hyper/hypo siblings); "Other..." suppression.
  7. Expand kept cat3s to their full dictionary subtree. ASSOCIATION/DIFF = "Not Applicable".

DETERMINISM: eval() + no_grad forward passes (no dropout/sampling), pinned BLAS
threads, precomputed dictionary embeddings shipped as .npy. Repeated runs are
byte-identical.

"""
from __future__ import annotations


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)


def _all_roots():
    roots = []
    env = os.environ.get("TASK3_DATA_DIR")
    if env:
        roots += [Path(x) for x in env.split(os.pathsep) if x]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        roots += [d for d in sorted(kaggle_input.iterdir()) if d.is_dir()]
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    roots.append(here)
    # This file lives in scripts/, so cache/ and cohort-x-task-3/ sit one level up.
    # Harmless on Kaggle, where /kaggle/input mounts above already resolve everything.
    roots.append(here.parent)
    # de-dup preserving order
    seen, out = set(), []
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s); out.append(r)
    return out


def _find_file(name):
    """Locate a file by name anywhere under any input root (fully recursive)."""
    # fast structured paths first
    subdirs = ("", "cache", "data", "input", "cohort-x-task-3", "models")
    for root in _all_roots():
        for sub in subdirs:
            cand = (root / sub / name) if sub else (root / name)
            if cand.exists():
                return cand
    # bulletproof fallback: recursive search under every root (Kaggle nests unpredictably)
    for root in _all_roots():
        if root.exists():
            for hit in root.rglob(name):
                return hit
    return None


def _find_model(hub_id, *names):
    """Locate a model dir (contains config.json) anywhere under any input root, else hub id."""
    for root in _all_roots():
        if not root.exists():
            continue
        for nm in names:
            for base in (root, root / "models"):
                if (base / nm / "config.json").exists():
                    return str(base / nm)
    # recursive fallback: any dir whose name matches and which holds a config.json
    wanted = set(names)
    for root in _all_roots():
        if root.exists():
            for cfg in root.rglob("config.json"):
                if cfg.parent.name in wanted:
                    return str(cfg.parent)
    return hub_id  # fall back to HF cache (offline) if a local copy isn't shipped


def _list_inputs():
    """Print what is actually mounted, to diagnose attach/upload problems on Kaggle."""
    kin = Path("/kaggle/input")
    print("[inputs] roots being searched:")
    for r in _all_roots():
        print(f"   - {r}  (exists={r.exists()})")
    if kin.exists():
        print("[inputs] files under /kaggle/input:")
        for p in sorted(kin.rglob("*")):
            if p.is_file():
                print(f"     {p}")


def _require(x, what):
    if x is None:
        _list_inputs()
        raise SystemExit(
            f"Could not locate '{what}'. Check the [inputs] listing above: attach your "
            f"assets dataset containing cache/{what} (and the Task 3 competition for the xlsx).")
    return x


if Path("/kaggle/input").exists():
    _list_inputs()
DICT_XLSX = _require(_find_file("mimic-iv_icd-10_dict.xlsx"), "mimic-iv_icd-10_dict.xlsx")
TASK_XLSX = _require(_find_file("Task_3.xlsx"), "Task_3.xlsx")
SAP_NPY = _require(_find_file("sapbert_icd_titles.npy"), "sapbert_icd_titles.npy")
BIO_NPY = _require(_find_file("biolord_icd_titles.npy"), "biolord_icd_titles.npy")
SAP_MODEL = _find_model("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "SapBERT-from-PubMedBERT-fulltext", "sapbert")
BIO_MODEL = _find_model("FremyCompany/BioLORD-2023", "BioLORD-2023", "biolord")
CE_MODEL = _find_model("ncbi/MedCPT-Cross-Encoder", "MedCPT-Cross-Encoder", "medcpt")
print(f"[paths] dict={DICT_XLSX}\n[paths] task={TASK_XLSX}\n[paths] sap_npy={SAP_NPY}\n[paths] bio_npy={BIO_NPY}")
print(f"[paths] sap_model={SAP_MODEL}\n[paths] bio_model={BIO_MODEL}\n[paths] ce_model={CE_MODEL}")
NA = "Not Applicable"
OUT_PATH = Path(os.environ.get("TASK3_OUT", "submission.csv"))


RECOMPUTE_NEURAL = os.environ.get("TASK3_RECOMPUTE", "0") == "1"

EVAL_SHEET = os.environ.get("TASK3_SHEET", "Test")
QUERY_EMB_NPZ = _find_file("query_emb.npz")
CE_SCORES_JSON = _find_file("ce_scores.json")
USE_PRECOMPUTED = (not RECOMPUTE_NEURAL) and (QUERY_EMB_NPZ is not None) and (CE_SCORES_JSON is not None)
print(f"[mode] {'PRECOMPUTED neural outputs (deterministic, env-independent)' if USE_PRECOMPUTED else 'LIVE model inference'}")


ENS_W_SAP, ENS_W_BIO = 0.7, 0.3
KEEP_THRESHOLD = 0.62
BOOST_BE_MIN = 0.40
CE_ADD_THRESH = 0.95
LEAVES_PER_CAT_BOOST = 5
MAX_ADDS_PER_ROW = 4
TOP_N_CATS_PRE = 25
DENSE_TOP_DOCS = 1500
DENSE_EXTRA_CATS = 25
TOPK_LEAVES = 3
W_MAX, W_TOPK, W_MEAN = 0.5, 0.4, 0.1

# ---------------------------------------------------------------------------
# Query expansion (inlined verbatim from task3_bm25_baseline.py)
# ---------------------------------------------------------------------------
SPELLFIX = {"breadth": "breath"}
SYNONYMS = {
    "dermatomycosis": "dermatomycosis tinea dermatophytosis ringworm fungal skin infection mycosis",
    "hypergonadism": "hypergonadism precocious puberty ovarian hyperfunction testicular hyperfunction",
    "epistaxis": "epistaxis nosebleed nasal hemorrhage",
    "pleurisy": "pleurisy pleurisy with effusion pleural effusion pleuritis",
    "thyroiditis": "thyroiditis autoimmune thyroiditis hashimoto subacute thyroiditis",
    "hypothyroidism": "hypothyroidism myxedema thyroid hormone deficiency",
    "hyperthyroidism": "hyperthyroidism thyrotoxicosis graves disease",
    "hypoparathyroidism": "hypoparathyroidism parathyroid hormone deficiency",
    "hyperparathyroidism": "hyperparathyroidism parathyroid hyperfunction",
    "gout": "gout idiopathic gout gouty arthropathy uric acid hyperuricemia tophus",
    "hematemesis": "hematemesis vomiting blood gastrointestinal hemorrhage upper gi bleed melena other diseases of digestive system K920 K921 K922",
    "enlarged mediastinum": "mediastinum mediastinal mass lesion",
    "bronchitis": "bronchitis acute chronic bronchitis",
    "pneumonia": "pneumonia bacterial viral lobar bronchopneumonia",
    "ckd": "chronic kidney disease renal failure nephropathy",
    "uti": "urinary tract infection cystitis pyelonephritis",
    "diabetes": "diabetes mellitus type 1 type 2 type 11 type 10 insulin dependent non insulin dependent hyperglycemia diabetic ketoacidosis E11 E10 E13",
    "intracranial pressure": "intracranial pressure cerebral edema disorders of brain hydrocephalus",
    "latent adrenal insufficiency": "adrenal insufficiency adrenocortical hypofunction addison",
    "nasopharyngeal carcinoma": "nasopharyngeal carcinoma malignant neoplasm nasopharynx",
    "interstitial lung disease": "interstitial lung disease pulmonary fibrosis pneumonitis alveolitis",
    "heart failure": "heart failure congestive cardiac failure",
}
ABBREV = {
    "urti": "upper respiratory tract infection acute nasopharyngitis sinusitis pharyngitis tonsillitis laryngitis",
    "lrti": "lower respiratory tract infection",
    "uti": "urinary tract infection",
    "htn": "hypertension",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "dm1": "type 1 diabetes mellitus",
    "ckd": "chronic kidney disease",
    "copd": "chronic obstructive pulmonary disease",
    "mi": "myocardial infarction",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "cva": "cerebrovascular accident stroke",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "gerd": "gastroesophageal reflux disease",
    "sob": "shortness of breath dyspnea respiratory failure",
    "ihd": "ischemic heart disease angina pectoris myocardial infarction",
}
CONTEXT_BOOST = {
    "ischemic heart disease": "angina pectoris myocardial infarction coronary",
    "aortic aneurysm": "aortic aneurysm dissection",
    "stroke": "cerebral infarction intracerebral hemorrhage subarachnoid hemorrhage transient ischemic attack cerebrovascular",
    "shortness of breath": "dyspnea respiratory failure pneumonia bronchitis pulmonary",
    "urti": "upper respiratory infection nasopharyngitis pharyngitis laryngitis sinusitis tonsillitis influenza",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    out = []
    for tok in text.split():
        tok = SPELLFIX.get(tok, tok)
        out.append(ABBREV.get(tok, tok))
    return " ".join(out)


def expand_query(condition: str) -> str:
    base = normalize(condition)
    for key, extra in SYNONYMS.items():
        if key in base:
            base = base + " " + extra
    for key, extra in CONTEXT_BOOST.items():
        if key in base:
            base = base + " " + extra
    return base


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(normalize(text))


# ---------------------------------------------------------------------------
# Noise filter + polarity (inlined verbatim from task3_reranked.py)
# ---------------------------------------------------------------------------
POLARITY_EXCLUSIONS = [
    ("hypothyroid", {"E05"}),
    ("hyperthyroid", {"E03"}),
    ("hypopara", {"E21"}),
    ("hyperpara", {"E20"}),
    ("hypogon", {"E27", "E28"}),
    ("hypergon", {"E23"}),
]
NOISE_CHAPTER_LETTERS = {"V", "W", "X", "Y", "Z", "S"}
NOISE_CAT3 = {"L89", "T67", "T70", "T80", "T81", "T82", "T83", "T84", "T85", "T86", "T87", "T88"}
PREGNANCY_KEYWORDS = {"pregnan", "gestat", "obstet", "puerper", "matern"}
PERINATAL_KEYWORDS = {"newborn", "neonat", "perinat", "infant", "fetal", "fetus"}


def is_noise_cat3(c: str, query_text: str) -> bool:
    if not c:
        return True
    if c[0] in NOISE_CHAPTER_LETTERS:
        return True
    if c in NOISE_CAT3:
        return True
    q = query_text.lower()
    if c[0] == "O" and not any(kw in q for kw in PREGNANCY_KEYWORDS):
        return True
    if c[0] == "P" and not any(kw in q for kw in PERINATAL_KEYWORDS):
        return True
    return False


# ---------------------------------------------------------------------------
# BM25 (inlined verbatim from rank_bm25 0.2.2 — so the notebook needs no extra
# package beyond numpy/pandas, which Kaggle ships. Identical formulas => identical
# scores to the reference implementation.)
# ---------------------------------------------------------------------------
import math


class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)
            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1
            self.corpus_size += 1
        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                     (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score


MIN_CAT_HITS = 5
SCORE_RATIO = 0.30


def load_kb() -> pd.DataFrame:
    df = pd.read_excel(DICT_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df["icd_code"] = df["icd_code"].astype(str).str.strip()
    df["long_title"] = df["long_title"].astype(str)
    df["cat3"] = df["icd_code"].str[:3]
    df["norm_title"] = df["long_title"].map(normalize)
    return df


def build_bm25(kb: pd.DataFrame) -> BM25Okapi:
    corpus = [tokenize(t) for t in kb["norm_title"].tolist()]
    return BM25Okapi(corpus)


def predict_for_query(query_text, kb, bm25, top_k_docs=2000, top_n_cats=TOP_N_CATS_PRE):
    q = TOKEN_RE.findall(query_text)
    scores = bm25.get_scores(q)
    top_idx = scores.argsort()[::-1][:top_k_docs]
    cat_score = defaultdict(float)
    cat_hits = defaultdict(int)
    cats = kb["cat3"].values
    for i in top_idx:
        s = scores[i]
        if s <= 0:
            continue
        c = cats[i]
        cat_score[c] += s
        cat_hits[c] += 1
    cat_sizes = kb.groupby("cat3").size().to_dict()
    ranked = sorted(cat_score.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return [], ranked
    top_score = ranked[0][1]
    picked = []
    for c, s in ranked[:top_n_cats]:
        size = cat_sizes.get(c, 1)
        min_hits = max(1, min(MIN_CAT_HITS, size // 4 + 1))
        if cat_hits[c] < min_hits:
            continue
        if s < SCORE_RATIO * top_score:
            break
        picked.append(c)
    if not picked and ranked:
        picked = [ranked[0][0]]
    pred_codes = kb.loc[kb["cat3"].isin(picked), "icd_code"].tolist()
    return pred_codes, ranked


# ---------------------------------------------------------------------------
# Encoders (inlined from task3_medcpt.py). transformers is imported lazily inside
# ---------------------------------------------------------------------------
def get_model(name, kind):
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(name)
    if kind == "encoder":
        mdl = AutoModel.from_pretrained(name)
    else:
        mdl = AutoModelForSequenceClassification.from_pretrained(name)
    mdl.to("cpu").eval()
    return tok, mdl, "cpu"


@torch.no_grad()
def encode_query_sap(text, tok, mdl, device):
    enc = tok([text], padding=True, truncation=True, max_length=64, return_tensors="pt")
    h = mdl(**enc).last_hidden_state[:, 0, :]
    h = torch.nn.functional.normalize(h, p=2, dim=1)
    return h.cpu().numpy().astype(np.float32)[0]


@torch.no_grad()
def encode_query_bio(text, tok, mdl, device):
    enc = tok([text], padding=True, truncation=True, max_length=64, return_tensors="pt")
    out = mdl(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    h = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    h = torch.nn.functional.normalize(h, p=2, dim=1)
    return h.cpu().numpy().astype(np.float32)[0]


@torch.no_grad()
def medcpt_score_pairs(query, titles, tok, mdl, device, batch=64):
    scores = []
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        pairs = [[query, t] for t in chunk]
        enc = tok(pairs, padding=True, truncation=True, max_length=128, return_tensors="pt")
        logits = mdl(**enc).logits.squeeze(-1)
        scores.append(logits.cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run():
    print("Loading KB + BM25...")
    kb = load_kb()
    bm25 = build_bm25(kb)

    print("Loading precomputed embeddings...")
    emb_sap = np.load(SAP_NPY)
    emb_bio = np.load(BIO_NPY)
    assert emb_sap.shape[0] == len(kb) == emb_bio.shape[0], "embedding/KB row mismatch"
    print(f"  SapBERT {emb_sap.shape}, BioLORD {emb_bio.shape}")

    cats_arr = kb["cat3"].values
    tmp = defaultdict(list)
    for i, c in enumerate(cats_arr):
        tmp[c].append(i)
    cat_to_idx = {c: np.array(lst, dtype=np.int64) for c, lst in tmp.items()}

    pre_sap = pre_bio = pre_ce = None
    if USE_PRECOMPUTED:
        print(f"Loading precomputed query embeddings + CE scores (no model inference)...")
        z = np.load(QUERY_EMB_NPZ, allow_pickle=True)
        _conds = list(z["conditions"])
        pre_sap = {c: z["sap"][i] for i, c in enumerate(_conds)}
        pre_bio = {c: z["bio"][i] for i, c in enumerate(_conds)}
        with open(CE_SCORES_JSON, encoding="utf-8") as f:
            pre_ce = json.load(f)
        sap_tok = sap_mdl = bio_tok = bio_mdl = ce_tok = ce_mdl = dev = None
    else:
        print("Loading models (SapBERT / BioLORD / MedCPT)...")
        sap_tok, sap_mdl, dev = get_model(SAP_MODEL, "encoder")
        bio_tok, bio_mdl, _ = get_model(BIO_MODEL, "encoder")
        ce_tok, ce_mdl, _ = get_model(CE_MODEL, "classifier")

    test = pd.read_excel(TASK_XLSX, sheet_name=EVAL_SHEET)
    if USE_PRECOMPUTED:
        missing = [c for c in test["Condition"] if c not in pre_sap]
        if missing:
            raise SystemExit(
                f"The precomputed cache covers the Test conditions only, but sheet "
                f"'{EVAL_SHEET}' contains {len(missing)} condition(s) absent from it "
                f"(e.g. {missing[:3]}). Re-run with TASK3_RECOMPUTE=1 and models/ present.")
    print(f"Processing {len(test)} conditions from sheet '{EVAL_SHEET}'...\n")

    rows = []
    for _, r in test.iterrows():
        cond = r["Condition"]
        cond_lower = cond.lower()

        _, ranked_bm = predict_for_query(expand_query(cond), kb, bm25, top_n_cats=TOP_N_CATS_PRE)
        bm_cats = [c for c, _ in ranked_bm[:TOP_N_CATS_PRE]]

        q_text = expand_query(cond)
        if USE_PRECOMPUTED:
            q_sap = pre_sap[cond]
            q_bio = pre_bio[cond]
        else:
            q_sap = encode_query_sap(q_text, sap_tok, sap_mdl, dev)
            q_bio = encode_query_bio(q_text, bio_tok, bio_mdl, dev)
        sims_all = ENS_W_SAP * (emb_sap @ q_sap) + ENS_W_BIO * (emb_bio @ q_bio)

        dense_top_idx = np.argpartition(-sims_all, DENSE_TOP_DOCS)[:DENSE_TOP_DOCS]
        dense_top_idx = dense_top_idx[np.argsort(-sims_all[dense_top_idx])]
        dense_cat_count = {}
        for i in dense_top_idx:
            c = cats_arr[i]
            dense_cat_count[c] = dense_cat_count.get(c, 0) + 1
        dense_cats = [c for c, _ in sorted(dense_cat_count.items(), key=lambda x: x[1], reverse=True)[:DENSE_EXTRA_CATS]]

        seen = set()
        cand_cats = []
        for c in bm_cats + dense_cats:
            if c in seen or is_noise_cat3(c, q_text):
                continue
            seen.add(c)
            cand_cats.append(c)

        if not cand_cats:
            rows.append({"Condition": cond, "KEEP": kb["icd_code"].iloc[0], "ASSOCIATION": NA, "DIFF": NA})
            continue

        cat_scores = []
        for c in cand_cats:
            idxs = cat_to_idx.get(c)
            if idxs is None or len(idxs) == 0:
                continue
            sims = sims_all[idxs]
            k = min(TOPK_LEAVES, len(sims))
            topk_mean = float(np.partition(-sims, k - 1)[:k].mean() * -1)
            score = float(W_MAX * sims.max() + W_TOPK * topk_mean + W_MEAN * sims.mean())
            cat_scores.append((c, score))

        kept = [c for c, s in cat_scores if s >= KEEP_THRESHOLD]
        if not kept and cat_scores:
            cat_scores.sort(key=lambda x: x[1], reverse=True)
            kept = [cat_scores[0][0]]
        kept_set = set(kept)

        booster_pool = [(c, s) for c, s in cat_scores
                        if c not in kept_set and BOOST_BE_MIN <= s < KEEP_THRESHOLD]
        booster_pool.sort(key=lambda x: -x[1])

        added_by_booster = []
        if booster_pool:
            ce_max_by_cat = defaultdict(float)
            if USE_PRECOMPUTED:
                cond_ce = pre_ce.get(cond, {})
                for c, _ in booster_pool:
                    ce_max_by_cat[c] = float(cond_ce.get(c, 0.0))
            else:
                all_titles, pair_owner = [], []
                for c, _ in booster_pool:
                    idxs = cat_to_idx[c]
                    cat_sims = sims_all[idxs]
                    order = np.argsort(-cat_sims)[:LEAVES_PER_CAT_BOOST]
                    for li in idxs[order]:
                        all_titles.append(str(kb["long_title"].iloc[li]))
                        pair_owner.append(c)

                ce_logits = medcpt_score_pairs(cond, all_titles, ce_tok, ce_mdl, dev, batch=64)
                ce_probs = sigmoid(ce_logits)
                for c, p in zip(pair_owner, ce_probs):
                    if p > ce_max_by_cat[c]:
                        ce_max_by_cat[c] = float(p)

            ranked = sorted(
                [(c, ce_max_by_cat.get(c, 0.0), be_s) for c, be_s in booster_pool],
                key=lambda x: (x[1], x[2]), reverse=True,
            )
            for c, ce_max, _ in ranked:
                if len(added_by_booster) >= MAX_ADDS_PER_ROW:
                    break
                if ce_max >= CE_ADD_THRESH:
                    kept.append(c)
                    kept_set.add(c)
                    added_by_booster.append((c, ce_max))

        for substr, exclude in POLARITY_EXCLUSIONS:
            if substr in cond_lower:
                kept = [c for c in kept if c not in exclude]

        if len(kept) >= 2:
            score_lookup = dict(cat_scores)
            kept_with_titles = []
            for kk in kept:
                rows_k = kb.loc[kb["icd_code"] == kk, "long_title"]
                title = str(rows_k.iloc[0]).lower() if len(rows_k) else ""
                kept_with_titles.append((kk, title, score_lookup.get(kk, 0.0)))
            non_generic_max = max((s for _, t, s in kept_with_titles if not t.startswith("other")), default=None)
            if non_generic_max is not None:
                kept = [kk for kk, t, s in kept_with_titles
                        if not (t.startswith("other") and non_generic_max - s >= 0.10)]

        keep_pred = kb.loc[kb["cat3"].isin(kept), "icd_code"].tolist()
        if not keep_pred:
            keep_pred = [kb["icd_code"].iloc[0]]

        msg = f"  {cond}: kept={len(kept)} codes={len(keep_pred)}"
        if added_by_booster:
            msg += f"  +boosted {[(c, round(p, 2)) for c, p in added_by_booster]}"
        print(msg)

        rows.append({"Condition": cond, "KEEP": "; ".join(keep_pred), "ASSOCIATION": NA, "DIFF": NA})

    out = pd.DataFrame(rows, columns=["Condition", "KEEP", "ASSOCIATION", "DIFF"])
    out.to_csv(OUT_PATH, index=False)
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"\nWrote {OUT_PATH}  (SHA256 {digest})")
    return out, digest


if __name__ == "__main__":
    run()
