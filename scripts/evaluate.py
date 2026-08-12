
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["KEEP", "ASSOCIATION", "DIFF"]
NA_TOKENS = {"not applicable", "n/a", "na", "none", "nan", ""}


def parse_cell(value) -> set[str]:
    """A submission cell -> set of codes. 'Not Applicable' and blanks -> empty set."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    text = str(value).strip()
    if text.lower() in NA_TOKENS:
        return set()
    return {tok.strip() for tok in text.split(";") if tok.strip()}


def cell_f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def find_task_xlsx(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for cand in (ROOT / "cohort-x-task-3" / "Task_3.xlsx", ROOT / "Task_3.xlsx",
                 ROOT.parent / "cohort-x-task-3" / "Task_3.xlsx"):
        if cand.exists():
            return cand
    sys.exit("Could not find Task_3.xlsx. Pass --task <path>.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", help="submission CSV to score")
    ap.add_argument("--task", default=None, help="path to Task_3.xlsx")
    ap.add_argument("--sheet", default="Train", help="labelled sheet name (default: Train)")
    ap.add_argument("--quiet", action="store_true", help="print only the final score")
    args = ap.parse_args()

    gold_df = pd.read_excel(find_task_xlsx(args.task), sheet_name=args.sheet)
    pred_df = pd.read_csv(args.submission)
    for frame, label in ((gold_df, args.sheet), (pred_df, args.submission)):
        missing = [c for c in ["Condition"] + COLUMNS if c not in frame.columns]
        if missing:
            sys.exit(f"{label} is missing required column(s): {missing}")

    pred_by_cond = {str(r["Condition"]).strip(): r for _, r in pred_df.iterrows()}

    per_column: dict[str, list[float]] = {c: [] for c in COLUMNS}
    rows, unmatched = [], []

    for _, g in gold_df.iterrows():
        cond = str(g["Condition"]).strip()
        p = pred_by_cond.get(cond)
        if p is None:
            unmatched.append(cond)
            continue
        scores = {}
        for col in COLUMNS:
            gold, pred = parse_cell(g[col]), parse_cell(p[col])
            f1 = cell_f1(pred, gold)
            per_column[col].append(f1)
            scores[col] = (f1, len(pred), len(gold))
        rows.append((cond, scores))

    if not rows:
        sys.exit("No conditions in the submission matched the labelled sheet.")

    all_cells = [f for col in COLUMNS for f in per_column[col]]
    overall = sum(all_cells) / len(all_cells)

    if not args.quiet:
        print(f"submission : {args.submission}")
        print(f"gold sheet : {args.sheet}  ({len(rows)} conditions scored)\n")
        head = f"{'Condition':<26}" + "".join(f"{c:>26}" for c in COLUMNS)
        print(head)
        print("-" * len(head))
        for cond, scores in rows:
            line = f"{cond[:25]:<26}"
            for col in COLUMNS:
                f1, np_, ng = scores[col]
                line += f"{f1:>8.3f}  (p={np_:<4d} g={ng:<4d})"[:26].rjust(26)
            print(line)
        print("-" * len(head))
        means = "".join(f"{sum(per_column[c]) / len(per_column[c]):>26.3f}" for c in COLUMNS)
        print(f"{'mean per category':<26}{means}")
        if unmatched:
            print(f"\nWARNING: {len(unmatched)} labelled condition(s) absent from the "
                  f"submission and NOT scored: {unmatched}")
        print()

    print(f"macro-F1 = {overall:.5f}   ({len(all_cells)} cells = "
          f"{len(rows)} conditions x {len(COLUMNS)} categories)")


if __name__ == "__main__":
    main()
