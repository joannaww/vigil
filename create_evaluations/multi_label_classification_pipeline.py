"""Evaluate every grid-search pipeline run against GT and produce a wide-format
CSV mirroring the table style used in the auto-qa report:

    Threshold,Margin,Boxes,Padding,Cars,Clothes,Cosmetics,Electronics,Furniture
    0.2,0.0,False,0.5,0.31,0.27,0.34,0.36,0.31
    ...

Each category column = macro-F1 on the CALIBRATION SET (the other 4 categories,
excluding this one). Macro-F1 = mean over hallucination types of binary
TP/FP/FN-derived F1, then mean over the 4 included categories.

GT comes from data/<cat>/annotations.json. Pipeline runs are auto-discovered
from data/<cat>/pipeline_<run_name>.json (one row per run_name; rows are
skipped if any category file is missing).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

CATEGORIES = ['cars', 'clothes', 'cosmetics', 'electronics', 'furniture']
HALLUCINATION_TYPES = ['objects', 'background', 'position_logic', 'physical', 'object_omission']

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "evaluations"
DEFAULT_OUTPUT = EVAL_DIR / "PIPELINE_CLASSIFICATION_GRID_SEARCH.csv"

RUN_NAME_RE = re.compile(
    r"^run_(?P<idx>\d+)_th(?P<threshold>\d+\.\d+)_mg(?P<margin>\d+\.\d+)_"
    r"(?P<boxes>boxes|noboxes)_pad(?P<padding>\d+\.\d+)(?:_(?P<version>v\d+))?$"
)


def parse_run_name(run_name: str) -> Dict[str, object]:
    """Extract grid params from a run name like
    `run_47_th0.5_mg0.2_boxes_pad0.5` or `run_6_th0.2_mg0.1_noboxes_pad1.0_v2`.
    Returns dict with idx/threshold/margin/boxes/padding/version, or all-None
    if the name doesn't match (still usable)."""
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return {"idx": None, "threshold": None, "margin": None,
                "boxes": None, "padding": None, "version": None}
    return {
        "idx": int(m["idx"]),
        "threshold": float(m["threshold"]),
        "margin": float(m["margin"]),
        "boxes": m["boxes"] == "boxes",
        "padding": float(m["padding"]),
        "version": m["version"] or "v1",
    }


def calculate_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def per_category_f1(gt_path: Path, pred_path: Path) -> Tuple[float, float, float]:
    """Macro F1/Prec/Rec averaged over HALLUCINATION_TYPES for one category."""
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_d = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_d = json.load(f)

    pred_map = {item["generated_photo"]: item.get("hallucination", {}) for item in pred_d}

    f1s, ps, rs = [], [], []
    for ht in HALLUCINATION_TYPES:
        tp = fp = fn = 0
        for g_item in gt_d:
            photo = g_item["generated_photo"]
            g_h = bool(g_item.get("hallucination", {}).get(ht))
            p_h = bool(pred_map.get(photo, {}).get(ht))
            if p_h and g_h:
                tp += 1
            elif p_h and not g_h:
                fp += 1
            elif not p_h and g_h:
                fn += 1
        prec, rec, f1 = calculate_f1(tp, fp, fn)
        ps.append(prec)
        rs.append(rec)
        f1s.append(f1)
    n = len(HALLUCINATION_TYPES)
    return sum(ps) / n, sum(rs) / n, sum(f1s) / n


def discover_runs() -> List[str]:
    """Find run_names that have a pipeline_*.json in EVERY category dir."""
    cars_runs = sorted(
        p.stem.replace("pipeline_", "")
        for p in (DATA_DIR / CATEGORIES[0]).glob("pipeline_*.json")
    )
    runs = []
    for r in cars_runs:
        if all((DATA_DIR / c / f"pipeline_{r}.json").exists() for c in CATEGORIES):
            runs.append(r)
        else:
            missing = [c for c in CATEGORIES if not (DATA_DIR / c / f"pipeline_{r}.json").exists()]
            print(f"  [skip] {r}: missing in {missing}")
    return runs


def evaluate_run(run_name: str) -> Dict[str, Dict[str, float]]:
    """Return {cat: {f1, p, r}} for one run across all categories."""
    cat_stats = {}
    for cat in CATEGORIES:
        gt_path = DATA_DIR / cat / "annotations.json"
        pred_path = DATA_DIR / cat / f"pipeline_{run_name}.json"
        if not gt_path.exists() or not pred_path.exists():
            return {}
        p, r, f1 = per_category_f1(gt_path, pred_path)
        cat_stats[cat] = {"p": p, "r": r, "f1": f1}
    return cat_stats


def build_row(run_name: str, cat_stats: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    params = parse_run_name(run_name)
    row: Dict[str, object] = {
        "Run": run_name,
        "Version": params.get("version") or "v1",
        "Threshold": params["threshold"],
        "Margin": params["margin"],
        "Boxes": params["boxes"],
        "Padding": params["padding"],
    }
    # For each category, pair: F1 on the 4 OTHER categories (calibration set)
    # and F1 on this category alone (held-out / test set).
    for cat in CATEGORIES:
        cap = cat.capitalize()
        others = [v["f1"] for c, v in cat_stats.items() if c != cat]
        row[f"{cap}_4excl"] = sum(others) / len(others) if others else 0.0
        row[f"{cap}_Only"] = cat_stats[cat]["f1"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--decimals", type=int, default=4,
        help="Round F1 scores to this many decimals in the CSV (default: 4)",
    )
    args = parser.parse_args()

    runs = discover_runs()
    if not runs:
        raise SystemExit("No grid runs found — looked in data/<cat>/pipeline_*.json")
    print(f"Found {len(runs)} runs covering all {len(CATEGORIES)} categories.")

    rows = []
    for run_name in runs:
        cat_stats = evaluate_run(run_name)
        if not cat_stats:
            print(f"  [skip] {run_name}: missing GT or pipeline file in some category")
            continue
        rows.append(build_row(run_name, cat_stats))

    df = pd.DataFrame(rows)

    # Sort by grid params for readability (Threshold, Margin, Boxes, Padding)
    df = df.sort_values(by=["Threshold", "Margin", "Boxes", "Padding", "Version"]).reset_index(drop=True)

    score_cols = [
        f"{c.capitalize()}_{suffix}"
        for c in CATEGORIES for suffix in ("4excl", "Only")
    ]
    for col in score_cols:
        df[col] = df[col].round(args.decimals)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, sep=";", encoding="utf-8-sig")

    print(f"\nSaved report to: {args.output}")
    calib_cols = [f"{c.capitalize()}_4excl" for c in CATEGORIES]
    df["MeanF1_4excl"] = df[calib_cols].mean(axis=1)
    print(f"\nTop 5 rows by mean 4excl F1 (calibration sets):")
    print(df.sort_values("MeanF1_4excl", ascending=False).head(5)[
        ["Run", "Version", "Threshold", "Margin", "Boxes", "Padding"] + score_cols + ["MeanF1_4excl"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
