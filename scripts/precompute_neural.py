
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.environ["TASK3_DATA_DIR"] = str(ROOT)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "t3", Path(__file__).resolve().parent / "task3_reproduce.py")
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)

CACHE = ROOT / "cache"

kb = t3.load_kb(); bm25 = t3.build_bm25(kb)
emb_sap = np.load(t3.SAP_NPY); emb_bio = np.load(t3.BIO_NPY)
cats_arr = kb["cat3"].values
tmp = defaultdict(list)
for i, c in enumerate(cats_arr):
    tmp[c].append(i)
cat_to_idx = {c: np.array(lst, dtype=np.int64) for c, lst in tmp.items()}

sap_tok, sap_mdl, dev = t3.get_model(t3.SAP_MODEL, "encoder")
bio_tok, bio_mdl, _ = t3.get_model(t3.BIO_MODEL, "encoder")
ce_tok, ce_mdl, _ = t3.get_model(t3.CE_MODEL, "classifier")

test = pd.read_excel(t3.TASK_XLSX, sheet_name="Test")
conditions = list(test["Condition"])
q_sap = np.zeros((len(conditions), emb_sap.shape[1]), dtype=np.float32)
q_bio = np.zeros((len(conditions), emb_bio.shape[1]), dtype=np.float32)
ce_scores = {}

for row_i, cond in enumerate(conditions):
    q = t3.expand_query(cond)
    qs = t3.encode_query_sap(q, sap_tok, sap_mdl, dev)
    qb = t3.encode_query_bio(q, bio_tok, bio_mdl, dev)
    q_sap[row_i] = qs; q_bio[row_i] = qb
    sims = t3.ENS_W_SAP * (emb_sap @ qs) + t3.ENS_W_BIO * (emb_bio @ qb)

    # reproduce candidate pool + booster pool to know which CE pairs to score
    _, ranked_bm = t3.predict_for_query(q, kb, bm25, top_n_cats=t3.TOP_N_CATS_PRE)
    bm_cats = [c for c, _ in ranked_bm[:t3.TOP_N_CATS_PRE]]
    dti = np.argpartition(-sims, t3.DENSE_TOP_DOCS)[:t3.DENSE_TOP_DOCS]; dti = dti[np.argsort(-sims[dti])]
    dcount = {}
    for i in dti:
        dcount[cats_arr[i]] = dcount.get(cats_arr[i], 0) + 1
    dense_cats = [c for c, _ in sorted(dcount.items(), key=lambda x: x[1], reverse=True)[:t3.DENSE_EXTRA_CATS]]
    seen = set(); cand = []
    for c in bm_cats + dense_cats:
        if c in seen or t3.is_noise_cat3(c, q):
            continue
        seen.add(c); cand.append(c)
    cs = []
    for c in cand:
        idx = cat_to_idx.get(c)
        if idx is None or len(idx) == 0:
            continue
        s = sims[idx]; k = min(t3.TOPK_LEAVES, len(s))
        tk = float(np.partition(-s, k - 1)[:k].mean() * -1)
        cs.append((c, float(t3.W_MAX * s.max() + t3.W_TOPK * tk + t3.W_MEAN * s.mean())))
    kept_set = {c for c, s in cs if s >= t3.KEEP_THRESHOLD}
    if not kept_set and cs:
        kept_set = {max(cs, key=lambda x: x[1])[0]}
    pool = [(c, s) for c, s in cs if c not in kept_set and t3.BOOST_BE_MIN <= s < t3.KEEP_THRESHOLD]

    cond_ce = {}
    if pool:
        titles, owner = [], []
        for c, _ in pool:
            idx = cat_to_idx[c]; s = sims[idx]; order = np.argsort(-s)[:t3.LEAVES_PER_CAT_BOOST]
            for li in idx[order]:
                titles.append(str(kb["long_title"].iloc[li])); owner.append(c)
        ce = t3.sigmoid(t3.medcpt_score_pairs(cond, titles, ce_tok, ce_mdl, dev, batch=64))
        cemax = defaultdict(float)
        for c, p in zip(owner, ce):
            if p > cemax[c]:
                cemax[c] = float(p)
        cond_ce = {c: float(cemax[c]) for c in cemax}
    ce_scores[cond] = cond_ce
    print(f"  {cond}: pool={len(pool)} ce_keys={len(cond_ce)}")

np.savez(CACHE / "query_emb.npz", conditions=np.array(conditions, dtype=object), sap=q_sap, bio=q_bio)
with open(CACHE / "ce_scores.json", "w", encoding="utf-8") as f:
    json.dump(ce_scores, f, indent=1)
print(f"\nWrote {CACHE/'query_emb.npz'} and {CACHE/'ce_scores.json'}")
