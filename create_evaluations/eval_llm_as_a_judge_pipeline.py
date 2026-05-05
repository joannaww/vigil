"""LLM-as-a-judge evaluation of one (or a few) grid-search pipeline runs
against GT, using the Gemini Batch API for safe, asynchronous, free-tier-
friendly use.

For each (category, datapoint, hallucination_type) where either GT or
prediction text is non-empty, the judge counts TP/FN/FP. Per-category
macro F1 is then computed by averaging F1 over hallucination types.

For each requested run, writes a single-row wide-format summary CSV at
evaluations/judge_summaries/PIPELINE_JUDGE_<run>.csv with both the
calibration F1 (4 other categories) and held-out F1 (only this category):

    Run,Threshold,Margin,Boxes,Padding,
    Cars_4excl,Cars_Only,...,Furniture_4excl,Furniture_Only

Usage:
  python3 create_evaluations/eval_llm_as_a_judge_pipeline.py \\
      --run run_12_th0.2_mg0.2_boxes_pad1.0 --yes

  # judge multiple runs at once:
  --run run_12_...,run_8_...

Safeguards:
  * Per-run cache (evaluations/judge_cache/<run>.jsonl). Re-running an
    already-judged run is a no-op — only re-emits the summary CSV.
  * Pre-flight stats printed before any API call. --dry-run exits there.
  * Retry-with-backoff on submit failures. Waves of submissions for
    concurrent-batch safety.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import dotenv  # type: ignore
    dotenv.load_dotenv()
except ImportError:
    pass

CATEGORIES = ['cars', 'clothes', 'cosmetics', 'electronics', 'furniture']
HALLUCINATION_TYPES = ['objects', 'background', 'position_logic', 'physical', 'object_omission']

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "evaluations"
CACHE_DIR = EVAL_DIR / "judge_cache"
SUMMARY_DIR = EVAL_DIR / "judge_summaries"

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
JUDGE_PACK_SIZE = int(os.environ.get("JUDGE_PACK_SIZE", "30"))
# Free-tier Gemini caps concurrent batch jobs around 2; keep waves small.
JUDGE_WAVE_SIZE = int(os.environ.get("JUDGE_WAVE_SIZE", "2"))
JUDGE_POLL_INTERVAL_S = int(os.environ.get("JUDGE_POLL_INTERVAL_S", "30"))
# Quota windows for free-tier Files/Batch API are minute-based; aggressive
# 1-16s retries don't help. Use longer backoffs so we ride out a full window.
JUDGE_SUBMIT_MAX_RETRIES = int(os.environ.get("JUDGE_SUBMIT_MAX_RETRIES", "8"))
JUDGE_SUBMIT_BACKOFF_CAP_S = int(os.environ.get("JUDGE_SUBMIT_BACKOFF_CAP_S", "120"))
# Stagger consecutive submits inside a wave so quota windows don't collide.
JUDGE_INTER_SUBMIT_DELAY_S = int(os.environ.get("JUDGE_INTER_SUBMIT_DELAY_S", "5"))

JUDGE_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}

RUN_NAME_RE = re.compile(
    r"^run_(?P<idx>\d+)_th(?P<threshold>\d+\.\d+)_mg(?P<margin>\d+\.\d+)_"
    r"(?P<boxes>boxes|noboxes)_pad(?P<padding>\d+\.\d+)$"
)


JUDGE_PROMPT_TEMPLATE = """You are an expert auditor evaluating AI hallucination detection.

Hallucination Type: {hallucination_type}

Human Ground Truth (list of real errors):
"{ground_truth_text}"

AI Pipeline Output (list of detected errors):
"{pipeline_text}"

Task:
Break down BOTH descriptions into individual distinct errors/issues and count:

1. TP (True Positives): How many specific errors from Ground Truth did the AI correctly detect?
   - Synonyms count as matches (e.g., "missing leg" == "leg not visible")
   - Similar descriptions of the same error count as TP

2. FN (False Negatives): How many specific errors from Ground Truth did the AI MISS completely?
   - Count each distinct error that's in Ground Truth but NOT in AI output

3. FP (False Positives): How many NEW errors did the AI report that are NOT in Ground Truth?
   - Count each distinct error that's in AI output but NOT in Ground Truth

Rules:
- Treat each distinct object/issue as a separate error instance
- If multiple errors are described in one sentence, count them separately
- Be precise and conservative in counting

Examples:
- Ground Truth ="Object omission: missing red car and blue truck" has 2 errors, not 1
- Ground Truth ="Object mutation: the back and sides of the car do not correspond to the reference" has 1 error
- For instance:
    - "Object mutation: The jeans have different details"
    - "Object mutation: The jeans show incorrect stitching patterns and colors"
    Both descriptions refer to the same visual inconsistency and should be treated as a single TP.

Output ONLY valid JSON in this exact format:
{{
    "tp": <integer>,
    "fn": <integer>,
    "fp": <integer>
}}
"""

# Wrapper prepended to a packed Batch API request that contains many tasks.
# Each task internally instructs the model to "Output ONLY valid JSON ..." —
# but with N tasks per request we need a JSON ARRAY of {id, tp, fn, fp}, so
# this header rebinds the per-task format to one element of the array.
PACKED_HEADER = ('Respond with a JSON array of objects. Each object MUST have: '
                 '"id" (string), "tp" (int), "fn" (int), "fp" (int). '
                 "Do not include any reasoning or extra text.\n\n")


def parse_run_name(run_name: str) -> Dict[str, object]:
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return {"idx": None, "threshold": None, "margin": None, "boxes": None, "padding": None}
    return {
        "idx": int(m["idx"]),
        "threshold": float(m["threshold"]),
        "margin": float(m["margin"]),
        "boxes": m["boxes"] == "boxes",
        "padding": float(m["padding"]),
    }


def calculate_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def discover_runs() -> List[str]:
    cars_runs = sorted(
        p.stem.replace("pipeline_", "")
        for p in (DATA_DIR / CATEGORIES[0]).glob("pipeline_*.json")
    )
    return [
        r for r in cars_runs
        if all((DATA_DIR / c / f"pipeline_{r}.json").exists() for c in CATEGORIES)
    ]


def build_tasks_for_run(run_name: str) -> List[Dict[str, str]]:
    """One judge task per (cat, dp, ht) where GT or pred text is non-empty."""
    tasks: List[Dict[str, str]] = []
    for cat in CATEGORIES:
        gt_path = DATA_DIR / cat / "annotations.json"
        pi_path = DATA_DIR / cat / f"pipeline_{run_name}.json"
        if not gt_path.exists() or not pi_path.exists():
            continue
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_d = json.load(f)
        with open(pi_path, "r", encoding="utf-8") as f:
            pi_d = json.load(f)
        pi_map = {it["generated_photo"]: it.get("hallucination", {}) for it in pi_d}
        for gt_it in gt_d:
            photo = gt_it["generated_photo"]
            if photo not in pi_map:
                continue
            for ht in HALLUCINATION_TYPES:
                gt_txt = (gt_it.get("hallucination") or {}).get(ht, "") or ""
                pi_txt = (pi_map[photo] or {}).get(ht, "") or ""
                if not gt_txt and not pi_txt:
                    continue
                tasks.append({
                    "id": f"{cat}__{photo}__{ht}",
                    "cat": cat,
                    "dp_photo": photo,
                    "ht": ht,
                    "gt_text": gt_txt,
                    "pred_text": pi_txt,
                })
    return tasks


def _build_packed_prompt(pack: List[Dict[str, str]]) -> str:
    items = []
    for t in pack:
        gt = t["gt_text"] if t["gt_text"] else "[EMPTY - no errors]"
        pr = t["pred_text"] if t["pred_text"] else "[EMPTY - no errors detected]"
        body = JUDGE_PROMPT_TEMPLATE.format(
            hallucination_type=t["ht"], ground_truth_text=gt, pipeline_text=pr,
        )
        items.append(f"CASE ID: {t['id']}\n{body}\n---")
    return PACKED_HEADER + "\n".join(items)


def make_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Add to .env or export."
        )
    from google import genai  # noqa: WPS433
    return genai.Client(api_key=api_key)


def submit_judge_batch_for_run(
    client, run_name: str, tasks: List[Dict[str, str]],
) -> Tuple[Optional[str], List[List[Dict[str, str]]]]:
    if not tasks:
        return None, []

    from google.genai import types  # noqa: WPS433

    packs = [tasks[i:i + JUDGE_PACK_SIZE] for i in range(0, len(tasks), JUDGE_PACK_SIZE)]

    requests_list = []
    for pack_idx, pack in enumerate(packs):
        prompt = _build_packed_prompt(pack)
        requests_list.append({
            "key": f"PACK_{pack_idx}",
            "request": {
                "contents": [{"parts": [{"text": prompt}]}],
                "generation_config": {
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            },
        })

    display_name = f"vigil_judge_{run_name}_{int(time.time())}"
    jsonl_path = SCRIPT_DIR / f".batch_{display_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for req in requests_list:
            f.write(json.dumps(req) + "\n")

    last_err: Optional[Exception] = None
    for attempt in range(JUDGE_SUBMIT_MAX_RETRIES + 1):
        try:
            uploaded = client.files.upload(
                file=str(jsonl_path),
                config=types.UploadFileConfig(
                    display_name=display_name, mime_type="application/jsonl"
                ),
            )
            batch_job = client.batches.create(
                model=JUDGE_MODEL,
                src=uploaded.name,
                config={"display_name": display_name},
            )
            try:
                os.remove(jsonl_path)
            except OSError:
                pass
            return batch_job.name, packs
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= JUDGE_SUBMIT_MAX_RETRIES:
                break
            wait = min(2 ** attempt, JUDGE_SUBMIT_BACKOFF_CAP_S)
            err_short = str(e)[:120]
            print(f"    [submit] {run_name}: attempt {attempt + 1} failed ({err_short}...); retry in {wait}s")
            time.sleep(wait)

    try:
        os.remove(jsonl_path)
    except OSError:
        pass
    print(f"    [submit] FAILED {run_name}: {last_err}")
    return None, packs


def wait_for_jobs(client, job_names: List[str]) -> Dict[str, Any]:
    pending = set(job_names)
    finished: Dict[str, Any] = {}
    while pending:
        for jn in list(pending):
            current = client.batches.get(name=jn)
            state = current.state.name
            if state in JUDGE_TERMINAL_STATES:
                finished[jn] = current
                pending.remove(jn)
                print(f"    [wait] [{time.strftime('%H:%M:%S')}] {jn}: {state}")
        if pending:
            print(
                f"    [wait] [{time.strftime('%H:%M:%S')}] "
                f"{len(pending)}/{len(job_names)} still running, sleep {JUDGE_POLL_INTERVAL_S}s..."
            )
            time.sleep(JUDGE_POLL_INTERVAL_S)
    return finished


def parse_judge_response(client, job, packs: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"    [parse] {job.name}: ended in {job.state.name}; default zeros")
        return [_default_verdict(t) for pack in packs for t in pack]

    output_content = client.files.download(file=job.dest.file_name)

    id_to_verdict: Dict[str, Dict[str, int]] = {}
    for line in output_content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if "response" not in data or not data["response"]:
            continue
        try:
            raw = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
            arr = json.loads(cleaned)
            if not isinstance(arr, list):
                continue
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                cid = str(obj.get("id", "")).strip()
                if cid:
                    id_to_verdict[cid] = {
                        "tp": int(obj.get("tp", 0) or 0),
                        "fp": int(obj.get("fp", 0) or 0),
                        "fn": int(obj.get("fn", 0) or 0),
                    }
        except Exception as e:
            print(f"    [parse] error in {data.get('key', '?')}: {e}")

    verdicts: List[Dict[str, Any]] = []
    for pack in packs:
        for t in pack:
            v = id_to_verdict.get(t["id"], {"tp": 0, "fp": 0, "fn": 0})
            verdicts.append({
                "cat": t["cat"], "dp_photo": t["dp_photo"], "ht": t["ht"], **v,
            })
    return verdicts


def _default_verdict(task: Dict[str, str]) -> Dict[str, Any]:
    return {"cat": task["cat"], "dp_photo": task["dp_photo"], "ht": task["ht"],
            "tp": 0, "fp": 0, "fn": 0}


def cache_path(run_name: str) -> Path:
    return CACHE_DIR / f"{run_name}.jsonl"


def write_run_cache(run_name: str, verdicts: List[Dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(run_name)
    with open(p, "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")


def read_run_cache(run_name: str) -> Optional[List[Dict[str, Any]]]:
    p = cache_path(run_name)
    if not p.exists():
        return None
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def aggregate_verdicts(verdicts: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    counts = {c: {ht: {"tp": 0, "fp": 0, "fn": 0} for ht in HALLUCINATION_TYPES}
              for c in CATEGORIES}
    for v in verdicts:
        c = v["cat"]; ht = v["ht"]
        if c in counts and ht in counts[c]:
            counts[c][ht]["tp"] += int(v.get("tp", 0))
            counts[c][ht]["fp"] += int(v.get("fp", 0))
            counts[c][ht]["fn"] += int(v.get("fn", 0))

    out: Dict[str, Dict[str, Any]] = {}
    for cat, type_counts in counts.items():
        fs, ps, rs = [], [], []
        for ht in HALLUCINATION_TYPES:
            tc = type_counts[ht]
            p, r, f1 = calculate_f1(tc["tp"], tc["fp"], tc["fn"])
            fs.append(f1); ps.append(p); rs.append(r)
        n = len(HALLUCINATION_TYPES)
        out[cat] = {"f1": sum(fs) / n, "p": sum(ps) / n, "r": sum(rs) / n}
    return out


def build_csv_row(run_name: str, cat_stats: Dict[str, Dict[str, Any]]) -> Dict[str, object]:
    params = parse_run_name(run_name)
    row: Dict[str, object] = {
        "Run": run_name,
        "Threshold": params["threshold"],
        "Margin": params["margin"],
        "Boxes": params["boxes"],
        "Padding": params["padding"],
    }
    for cat in CATEGORIES:
        cap = cat.capitalize()
        others = [v["f1"] for c, v in cat_stats.items() if c != cat]
        row[f"{cap}_4excl"] = sum(others) / len(others) if others else 0.0
        row[f"{cap}_Only"] = cat_stats[cat]["f1"]
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", action="append", required=True,
        help="Run name(s) to judge (repeat or comma-separate). "
             "Example: run_12_th0.2_mg0.2_boxes_pad1.0",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Pre-flight stats only, no API calls.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation.")
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even if cache exists (overwrites).")
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR,
                        help=f"Where per-run summary CSVs land "
                             f"(default: {SUMMARY_DIR}).")
    parser.add_argument("--decimals", type=int, default=4,
                        help="Round F1 scores to this many decimals (default: 4).")
    return parser.parse_args()


def main():
    args = parse_args()

    requested: List[str] = []
    for r in args.run:
        for n in r.split(","):
            n = n.strip()
            if n:
                requested.append(n)

    all_runs = set(discover_runs())
    missing = [r for r in requested if r not in all_runs]
    if missing:
        print(f"WARN: requested runs not found in data/: {missing}")
    requested = [r for r in requested if r in all_runs]
    if not requested:
        raise SystemExit("No valid runs to process.")

    pending = requested if args.force else [r for r in requested if not cache_path(r).exists()]
    cached_already = [r for r in requested if cache_path(r).exists() and not args.force]

    print(f"Requested runs:    {len(requested)}")
    print(f"  Cached (skip):   {len(cached_already)}")
    print(f"  To judge:        {len(pending)}")
    if cached_already:
        print(f"  (--force to re-judge cached runs)")

    if pending:
        tasks_by_run: Dict[str, List[Dict[str, str]]] = {r: build_tasks_for_run(r) for r in pending}
        total_tasks = sum(len(t) for t in tasks_by_run.values())
        total_packed = sum((len(t) + JUDGE_PACK_SIZE - 1) // JUDGE_PACK_SIZE
                           for t in tasks_by_run.values())

        print(f"\nPre-flight:")
        print(f"  Judge tasks:        {total_tasks}")
        print(f"  Packed requests:    {total_packed} (pack size {JUDGE_PACK_SIZE})")
        print(f"  Free-tier daily:    10000 req/day → using ~{100 * total_packed // 10000}%")

        if args.dry_run:
            print("\n--dry-run set; exiting without API calls.")
            return

        if not args.yes:
            try:
                ans = input("\nProceed with API submission? [y/N] ").strip().lower()
            except EOFError:
                ans = ""
            if ans not in {"y", "yes"}:
                print("Aborted.")
                return

        client = make_genai_client()
        runs_to_submit = [r for r in pending if tasks_by_run[r]]

        for wave_start in range(0, len(runs_to_submit), JUDGE_WAVE_SIZE):
            wave = runs_to_submit[wave_start: wave_start + JUDGE_WAVE_SIZE]
            wave_idx = wave_start // JUDGE_WAVE_SIZE + 1
            n_waves = (len(runs_to_submit) + JUDGE_WAVE_SIZE - 1) // JUDGE_WAVE_SIZE
            print(f"\n--- Wave {wave_idx}/{n_waves}: submitting {len(wave)} batch jobs ---")

            job_to_run: Dict[str, str] = {}
            packs_by_job: Dict[str, List[List[Dict[str, str]]]] = {}
            for i, r in enumerate(wave):
                if i > 0 and JUDGE_INTER_SUBMIT_DELAY_S > 0:
                    time.sleep(JUDGE_INTER_SUBMIT_DELAY_S)
                tasks = tasks_by_run[r]
                job_name, packs = submit_judge_batch_for_run(client, r, tasks)
                if job_name:
                    job_to_run[job_name] = r
                    packs_by_job[job_name] = packs
                    n_packs = (len(tasks) + JUDGE_PACK_SIZE - 1) // JUDGE_PACK_SIZE
                    print(f"  [submit] {r}: job={job_name} ({len(tasks)} tasks → {n_packs} packs)")
                else:
                    print(f"  [submit] {r}: SKIPPED")

            if not job_to_run:
                continue

            print(f"\n--- Wave {wave_idx}/{n_waves}: waiting for {len(job_to_run)} jobs ---")
            finished = wait_for_jobs(client, list(job_to_run.keys()))

            for jn, job in finished.items():
                r = job_to_run[jn]
                verdicts = parse_judge_response(client, job, packs_by_job[jn])
                write_run_cache(r, verdicts)
                cat_stats = aggregate_verdicts(verdicts)
                macro = sum(c["f1"] for c in cat_stats.values()) / len(cat_stats)
                print(f"  [cached] {r}: macro F1 = {macro:.4f} (verdicts={len(verdicts)})")

    # Write a per-run summary CSV for every requested run that now has a
    # cache file (newly judged or already cached). Each CSV is a single-row
    # wide-format file you can later concatenate with pandas if you want a
    # cross-run table.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_cols = [f"{c.capitalize()}_{s}" for c in CATEGORIES for s in ("4excl", "Only")]

    print()
    for run_name in requested:
        if not cache_path(run_name).exists():
            print(f"  [skip] {run_name}: no cache (judge probably failed)")
            continue
        verdicts = read_run_cache(run_name) or []
        cat_stats = aggregate_verdicts(verdicts)
        row = build_csv_row(run_name, cat_stats)
        for col in score_cols:
            row[col] = round(row[col], args.decimals)
        df = pd.DataFrame([row])
        out_path = args.output_dir / f"PIPELINE_JUDGE_{run_name}.csv"
        df.to_csv(out_path, index=False, sep=";", encoding="utf-8-sig")
        macro = sum(c["f1"] for c in cat_stats.values()) / len(cat_stats)
        print(f"  [csv] {out_path.name}: macro F1 = {macro:.4f}")


if __name__ == "__main__":
    main()
