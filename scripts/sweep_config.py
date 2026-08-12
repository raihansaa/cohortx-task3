
from __future__ import annotations

import os
import io
import sys
import argparse
import contextlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


GRID = [
    ("booster off (cap 0)",      {"MAX_ADDS_PER_ROW": 0},                         0.37017),
    ("cap 2",                    {"MAX_ADDS_PER_ROW": 2},                         0.38013),
    ("cap 4  [RELEASED]",        {"MAX_ADDS_PER_ROW": 4},                         0.38364),
    ("cap 5",                    {"MAX_ADDS_PER_ROW": 5},                         0.38291),
    ("cap 6",                    {"MAX_ADDS_PER_ROW": 6},                         0.37700),
    ("cap 4, keep thresh 0.60",  {"MAX_ADDS_PER_ROW": 4, "KEEP_THRESHOLD": 0.60}, 0.37513),
]
# Require live inference; excluded from the default run.
GRID_RECOMPUTE_ONLY = [
    ("cap 4, encoders 0.5/0.5", {"MAX_ADDS_PER_ROW": 4, "ENS_W_SAP": 0.5, "ENS_W_BIO": 0.5}, None),
    ("cap 4, encoders 0.8/0.2", {"MAX_ADDS_PER_ROW": 4, "ENS_W_SAP": 0.8, "ENS_W_BIO": 0.2}, None),
    ("cap 4, keep thresh 0.64", {"MAX_ADDS_PER_ROW": 4, "KEEP_THRESHOLD": 0.64},             None),
]


CACHED_KEEP_THRESHOLD = 0.62
CACHED_BOOST_BE_MIN = 0.40

SCORE_CHANGING = {"ENS_W_SAP", "ENS_W_BIO", "W_MAX", "W_TOPK", "W_MEAN", "TOPK_LEAVES",
                  "TOP_N_CATS_PRE", "DENSE_TOP_DOCS", "DENSE_EXTRA_CATS"}

TUNABLE = ["MAX_ADDS_PER_ROW", "KEEP_THRESHOLD", "BOOST_BE_MIN", "CE_ADD_THRESH",
           "ENS_W_SAP", "ENS_W_BIO", "W_MAX", "W_TOPK", "W_MEAN", "TOPK_LEAVES",
           "TOP_N_CATS_PRE", "DENSE_TOP_DOCS", "DENSE_EXTRA_CATS"]


def cache_safe(overrides: dict) -> tuple[bool, str]:
    """True iff this config's booster window is a SUBSET of the cached [0.40, 0.62)
    window and the bi-encoder scores are unchanged."""
    changed = SCORE_CHANGING & overrides.keys()
    if changed:
        return False, f"changes bi-encoder scoring ({', '.join(sorted(changed))})"
    if overrides.get("KEEP_THRESHOLD", CACHED_KEEP_THRESHOLD) > CACHED_KEEP_THRESHOLD:
        return False, (f"keep threshold {overrides['KEEP_THRESHOLD']} > cached "
                       f"{CACHED_KEEP_THRESHOLD}, widening the booster window")
    if overrides.get("BOOST_BE_MIN", CACHED_BOOST_BE_MIN) < CACHED_BOOST_BE_MIN:
        return False, (f"booster floor {overrides['BOOST_BE_MIN']} < cached "
                       f"{CACHED_BOOST_BE_MIN}, widening the booster window")
    return True, ""


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default="Test",
                    help="Task_3.xlsx sheet to predict for (default: Test). "
                         "'Train' is the only labelled sheet and needs --recompute.")
    ap.add_argument("--all", action="store_true",
                    help="include configurations that require live inference")
    ap.add_argument("--recompute", action="store_true",
                    help="run the encoders live instead of reading cache/ (needs models/)")
    ap.add_argument("--outdir", default=str(ROOT / "sweep_out"),
                    help="where to write one submission CSV per configuration")
    args = ap.parse_args()

    if args.sheet != "Test" and not args.recompute:
        sys.exit(f"--sheet {args.sheet} requires --recompute: the shipped cache covers the "
                 f"Test conditions only. Re-run with --recompute and models/ present.")
    if args.all and not args.recompute:
        sys.exit("--all requires --recompute: those configurations need cross-encoder scores "
                 "that are not in the shipped cache. Drop --all to sweep the cache-safe grid.")

   
    if args.recompute:
        os.environ["TASK3_RECOMPUTE"] = "1"
    os.environ["TASK3_SHEET"] = args.sheet

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):          # the pipeline is chatty on import
        t3 = load_module(here / "task3_reproduce.py", "t3")
    ev = load_module(here / "evaluate.py", "ev")

    import pandas as pd
    gold = pd.read_excel(ev.find_task_xlsx(None), sheet_name=args.sheet)
    labelled = any(gold[c].notna().any() for c in ev.COLUMNS)
    print(f"[mode]  {'live model inference' if not t3.USE_PRECOMPUTED else 'precomputed cache'}"
          f"   [sheet] {args.sheet} ({len(gold)} conditions,"
          f" {'labelled' if labelled else 'unlabelled'})\n")

    defaults = {k: getattr(t3, k) for k in TUNABLE}
    grid = GRID + (GRID_RECOMPUTE_ONLY if args.all else [])
    results, skipped = [], []

    for name, overrides, observed_lb in grid:
        if not args.recompute:
            ok, why = cache_safe(overrides)
            if not ok:
                skipped.append((name, why))
                continue

        for k, v in defaults.items():              
            setattr(t3, k, v)
        for k, v in overrides.items():
            setattr(t3, k, v)

        slug = (name.replace("[RELEASED]", "").strip()
                .replace(" ", "_").replace(",", "").replace(".", ""))
        out_path = outdir / f"{args.sheet.lower()}_{slug}.csv"
        t3.OUT_PATH = out_path
        with contextlib.redirect_stdout(buf):
            _, digest = t3.run()

        pred = pd.read_csv(out_path)
        score = None
        if labelled:
            by_cond = {str(r["Condition"]).strip(): r for _, r in pred.iterrows()}
            cells = [ev.cell_f1(ev.parse_cell(by_cond[c][col]), ev.parse_cell(g))
                     for _, row in gold.iterrows()
                     for c, col, g in [(str(row["Condition"]).strip(), col, row[col])
                                       for col in ev.COLUMNS]
                     if c in by_cond]
            score = sum(cells) / len(cells) if cells else None

        n_codes = sum(len(str(v).split(";")) for v in pred["KEEP"])
        results.append((name, observed_lb, score, n_codes))
        print(f"  ran {name:<26} codes={n_codes:>5}  sha={digest[:8]}")

    label = f"{args.sheet} rows" if labelled else "(unlabelled)"
    print(f"\n{'configuration':<26}{'public LB':>12}{label:>14}{'KEEP codes':>12}")
    print("-" * 64)
    for name, lb, score, n_codes in results:
        lb_s = f"{lb:.5f}" if lb is not None else "not submitted"
        s_s = f"{score:.5f}" if score is not None else "n/a"
        print(f"{name:<26}{lb_s:>12}{s_s:>14}{n_codes:>12}")

    if not labelled:
        print(f"\nSheet '{args.sheet}' carries no gold labels, so no offline score is "
              f"possible — the public-LB column is the only signal here. For a scored run: "
              f"python scripts/sweep_config.py --sheet Train --recompute")

    if skipped:
        print(f"\nSkipped {len(skipped)} configuration(s) — not evaluable from the shipped "
              f"cache without silently zeroing the booster. Re-run with --recompute:")
        for name, why in skipped:
            print(f"  - {name}: {why}")

    print(f"\nSubmissions written to {outdir}")


if __name__ == "__main__":
    main()
