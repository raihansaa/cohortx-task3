
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

SPECS = [
    ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "SapBERT-from-PubMedBERT-fulltext", "encoder"),
    ("FremyCompany/BioLORD-2023", "BioLORD-2023", "encoder"),
    ("ncbi/MedCPT-Cross-Encoder", "MedCPT-Cross-Encoder", "classifier"),
]

for hub_id, local_name, kind in SPECS:
    dest = MODELS / local_name
    print(f"\n=== {hub_id} -> {dest} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(hub_id)
    if kind == "encoder":
        mdl = AutoModel.from_pretrained(hub_id)
    else:
        mdl = AutoModelForSequenceClassification.from_pretrained(hub_id)
    tok.save_pretrained(dest)
    mdl.save_pretrained(dest, safe_serialization=True)
    print(f"  saved: {[p.name for p in dest.iterdir()]}", flush=True)

print("\nALL MODELS EXPORTED to ./models/")
