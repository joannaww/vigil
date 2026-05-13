#!/usr/bin/env python3
"""
Postprocess pipeline grid-search outputs into baseline-format per-category JSONs.

Reads outputs/grid_search/<run_name>/final_results.json (mixed-category list)
and writes per-category files mirroring data/<cat>/baseline_<model>.json:

    data/<category>/pipeline_<run_name>.json

Each output entry has the same shape as a baseline file:
    generated_photo, prompt, objects (list), background, hallucination
where hallucination is {objects, background, position_logic, physical, object_omission}.

Category is read from the file path inside the pipeline result (segment
`/data/<category>/data/...`), not from the `id` field — more robust if id
schemes ever change.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import dotenv  # type: ignore
    dotenv.load_dotenv()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GRID_DIR_DEFAULT = REPO_ROOT / "outputs" / "grid_search"
DATA_DIR = REPO_ROOT / "data"

CLEAN_BG_EXACT = {
    "", "n/a", "none", "no issues", "no issue", "no hallucination",
    "no hallucinations", "no inconsistency", "no inconsistencies",
    "everything looks fine", "looks fine", "correct", "ok",
    "no visible inconsistencies", "no visible issues",
    "no visible hallucinations", "no visible errors",
    "no visible error", "no errors", "no error",
    # VLM sometimes literally echoes the example empty-string marker:
    "\"\"", "''", "<empty>", "<empty string>", "empty string", "empty",
}

CLEAN_BG_PREFIXES = (
    "no visible error",
    "no visible issue",
    "no visible hallucination",
    "no visible inconsistenc",
    "no visible change",
    "no visible difference",
    "no error",
    "no issue",
    "no inconsistenc",
    "no hallucination",
    "no change",
    "no difference",
    "everything looks fine",
    "everything is consistent",
    "the background is consistent",
    "the background appears consistent",
    "the background looks consistent",
    "the background is identical",
    "the background remains",
    "background is consistent",
    "background appears consistent",
    "image 2 is consistent",
    "image 2 appears consistent",
)

# Substring patterns for false-positive bg mutations the VLM tends to invent
# despite the tightened prompt. These are subtle texture/coloration descriptors
# that humans never flag but the VLM hallucinates as "Background Mutation: ...".
# If a text contains one of these AND nothing more substantive, treat as clean.
CLEAN_BG_NOISE_SUBSTRINGS = (
    "more uniformly",
    "more uniform",
    "less varied",
    "more varied",
    "appearing smoother",
    "appears smoother",
    "appearing slightly",
    "appears slightly",
    "subtly different",
    "slightly different texture",
    "slightly different coloration",
    "slightly more saturated",
    "slightly less saturated",
    "noticeably different texture",
    "different texture and coloration",
    "smoother and more uniform",
    "smoother, more uniform",
)


def is_clean_background(text: str) -> bool:
    """Heuristic: True if the background_evaluation text indicates no hallucination."""
    if not text:
        return True
    s = text.strip().lower()
    if not s:
        return True
    if s.rstrip(".") in CLEAN_BG_EXACT:
        return True
    if s.startswith(CLEAN_BG_PREFIXES):
        return True
    # If the entire mutation description boils down to one of the subtle-noise
    # phrasings AND is short (<=2 sentences), treat as clean false positive.
    if len(s) < 400 and any(phrase in s for phrase in CLEAN_BG_NOISE_SUBSTRINGS):
        return True
    return False


def category_and_dp_id_from_path(gen_path: str) -> Optional[Tuple[str, str]]:
    """Extract (category, dp_id) from a generated image path of the form
    .../data/<category>/data/<dp_id>/<dp_id>generated.png"""
    parts = Path(gen_path).parts
    for i, p in enumerate(parts):
        if p == "data" and i + 3 < len(parts) and parts[i + 2] == "data":
            return parts[i + 1], parts[i + 3]
    return None


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
JUDGE_POLL_INTERVAL_S = int(os.environ.get("JUDGE_POLL_INTERVAL_S", "30"))
JUDGE_PACK_SIZE = int(os.environ.get("JUDGE_PACK_SIZE", "15"))
JUDGE_WAVE_SIZE = int(os.environ.get("JUDGE_WAVE_SIZE", "10"))
JUDGE_SUBMIT_MAX_RETRIES = int(os.environ.get("JUDGE_SUBMIT_MAX_RETRIES", "5"))


JUDGE_BG_PACKED_PROMPT_HEADER = """You are a strict text classifier.

For each numbered "Text" below, decide whether it indicates that the BACKGROUND of a generated image has NO hallucinations.

Be conservative: return clean=true ONLY if the text clearly states there are no hallucinations, no inconsistencies, or is otherwise an OK/CLEAN-like message about background consistency. If the text describes ANY visible mutation, swap, change, distortion, or issue, return clean=false.

Respond with a JSON ARRAY of objects. Each object MUST have:
- "id" (string, the CASE ID exactly as given)
- "clean" (boolean)

Output ONLY the JSON array. No preamble, no commentary, no markdown fences.

"""


def _build_packed_prompt(items: List[Tuple[str, str]]) -> str:
    """Pack N (case_id, text) items into one prompt asking for a JSON array."""
    parts = []
    for case_id, text in items:
        parts.append(f"CASE ID: {case_id}\nText:\n\"\"\"{text}\"\"\"\n---")
    return JUDGE_BG_PACKED_PROMPT_HEADER + "\n".join(parts)


JUDGE_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


def make_genai_client():
    """Create a Gemini client. Raises if API key is missing."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. "
            "Add it to .env or export it; required for --use-gemini-judge."
        )
    from google import genai  # noqa: WPS433 (lazy import)
    return genai.Client(api_key=api_key)


def submit_judge_batch(client, texts_by_id: Dict[str, str], display_name: str) -> Optional[str]:
    """Upload one JSONL of packed requests and create a batch job. Returns job_name.

    Each request packs `JUDGE_PACK_SIZE` items so we send ~N/15 requests per
    batch instead of N — keeps us well below free-tier 10K/day quota.

    Retries up to `JUDGE_SUBMIT_MAX_RETRIES` times on transient errors with
    exponential backoff (1s, 2s, 4s, ... capped at 30s).
    Returns None if texts_by_id is empty.
    """
    if not texts_by_id:
        return None

    from google.genai import types  # noqa: WPS433

    items = list(texts_by_id.items())
    packs = [items[i:i + JUDGE_PACK_SIZE] for i in range(0, len(items), JUDGE_PACK_SIZE)]

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
            return batch_job.name
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= JUDGE_SUBMIT_MAX_RETRIES:
                break
            wait = min(2 ** attempt, 30)
            print(f"    [submit] {display_name}: attempt {attempt + 1} failed ({e}); retry in {wait}s")
            time.sleep(wait)

    try:
        os.remove(jsonl_path)
    except OSError:
        pass
    print(f"    [submit] FAILED {display_name} after {JUDGE_SUBMIT_MAX_RETRIES + 1} attempts: {last_err}")
    return None


def wait_for_jobs(client, job_names: List[str], poll_interval: int = JUDGE_POLL_INTERVAL_S) -> Dict[str, Any]:
    """Poll all jobs in a single loop until each reaches a terminal state.
    Returns {job_name: final_job_object}."""
    pending = set(job_names)
    finished: Dict[str, Any] = {}

    while pending:
        for jn in list(pending):
            current = client.batches.get(name=jn)
            state = current.state.name
            if state in JUDGE_TERMINAL_STATES:
                finished[jn] = current
                pending.remove(jn)
                print(f"    [judge] [{time.strftime('%H:%M:%S')}] {jn}: {state}")
        if pending:
            print(
                f"    [judge] [{time.strftime('%H:%M:%S')}] "
                f"{len(pending)}/{len(job_names)} still running, sleeping {poll_interval}s..."
            )
            time.sleep(poll_interval)
    return finished


def download_judge_result(client, job) -> Dict[str, bool]:
    """Download a finished batch job's output and parse {case_id: is_clean}.

    Each line in the JSONL output corresponds to one packed request whose
    response is a JSON array of {id, clean} objects. We flatten across all
    packs into a single dict.
    """
    state = job.state.name
    if state != "JOB_STATE_SUCCEEDED":
        print(f"    [judge] ERROR: batch job {job.name} ended in {state}: {getattr(job, 'error', None)}")
        return {}

    output_content = client.files.download(file=job.dest.file_name)
    results: Dict[str, bool] = {}
    for line in output_content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        key = str(data.get("key", ""))
        if "response" not in data or not data["response"]:
            continue
        try:
            raw = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
            arr = json.loads(cleaned)
            if not isinstance(arr, list):
                print(f"    [judge] {key}: expected JSON array, got {type(arr).__name__}")
                continue
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                cid = str(obj.get("id", "")).strip()
                if cid:
                    results[cid] = bool(obj.get("clean", False))
        except Exception as e:
            print(f"    [judge] parse error for key={key}: {e}")
    return results


def convert_item(item: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Convert one pipeline item to (category, baseline-format entry).
    Returns None if the path cannot be parsed."""
    gen_data = item.get("generated_result") or {}
    gen_path = gen_data.get("path", "")
    parsed = category_and_dp_id_from_path(gen_path)
    if not parsed:
        return None
    category, dp_id = parsed

    generated_photo = Path(gen_path).name
    prompt_filename = f"{dp_id}prompt.txt"
    background_filename = f"{dp_id}background.png"

    object_filenames = []
    for ctx in item.get("contextual_results") or []:
        ctx_path = ctx.get("path")
        if ctx_path:
            object_filenames.append(Path(ctx_path).name)

    hallucination = {
        "objects": "",
        "background": "",
        "position_logic": "",
        "physical": "",
        "object_omission": "",
    }

    pairings = item.get("object_pairings") or []

    # objects: union of paired hallucination_check descriptions
    obj_findings = []
    seen = set()
    for p in pairings:
        hc = p.get("hallucination_check") or {}
        if not hc.get("hallucination_detected"):
            continue
        desc = (hc.get("description") or "").strip()
        if not desc or desc.lower() in {"correct", "none"}:
            continue
        label = p.get("contextual_label") or "Unknown"
        finding = f"{label}: {desc}"
        if finding not in seen:
            seen.add(finding)
            obj_findings.append(finding)
    if obj_findings:
        hallucination["objects"] = " ".join(obj_findings)

    # background
    bg = ((item.get("background_evaluation") or {}).get("qwen_analysis") or "").strip()
    if bg and not is_clean_background(bg):
        hallucination["background"] = bg

    # object_omission: contextual objects that did not pair
    ctx_results = item.get("contextual_results") or []
    all_ctx_ids = {
        obj.get("object_id")
        for r in ctx_results
        for obj in (r.get("objects") or [])
        if obj.get("object_id")
    }
    id_to_label = {
        obj.get("object_id"): obj.get("label") or "Unknown"
        for r in ctx_results
        for obj in (r.get("objects") or [])
    }
    paired_ctx_ids = {
        p.get("contextual_object_id")
        for p in pairings
        if p.get("paired") and p.get("contextual_object_id")
    }
    omitted = sorted({
        id_to_label[i] for i in (all_ctx_ids - paired_ctx_ids) if i in id_to_label
    })
    if omitted:
        verb = "was" if len(omitted) == 1 else "were"
        hallucination["object_omission"] = (
            f"Object omission: {', '.join(omitted)} {verb} not pasted into the image."
        )

    # physical (already merged into a single text by physical_evaluator)
    phys = ((item.get("physical_evaluation") or {}).get("text") or "").strip()
    if phys:
        hallucination["physical"] = phys

    # position_logic (already merged into a single text by position_logic_evaluator)
    pos = ((item.get("position_logic_evaluation") or {}).get("text") or "").strip()
    if pos:
        hallucination["position_logic"] = pos

    entry = {
        "generated_photo": generated_photo,
        "prompt": prompt_filename,
        "objects": object_filenames,
        "background": background_filename,
        "hallucination": hallucination,
    }
    return category, entry


def convert_run(run_dir: Path) -> Optional[Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], int]]:
    """Read one run's final_results.json and convert to baseline-format entries.

    Returns (by_category, texts_by_id, skipped):
      - by_category: {cat: [entry, ...]} with `_judge_id` set on entries that
        have a non-empty `hallucination.background` text (so the caller can
        later clear them based on judge results).
      - texts_by_id: {run_dir.name::cat::idx: bg_text} ready for Gemini judge.
      - skipped: number of items whose path could not be parsed.

    Returns None if final_results.json is missing.
    """
    final_results = run_dir / "final_results.json"
    if not final_results.exists():
        print(f"  [skip] {run_dir.name}: no final_results.json")
        return None

    with open(final_results, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for item in data:
        result = convert_item(item)
        if result is None:
            skipped += 1
            continue
        cat, entry = result
        by_category[cat].append(entry)

    texts_by_id: Dict[str, str] = {}
    for cat, entries in by_category.items():
        for idx, e in enumerate(entries):
            bg = (e["hallucination"].get("background") or "").strip()
            if bg:
                cid = f"{run_dir.name}::{cat}::{idx}"
                texts_by_id[cid] = bg
                e["_judge_id"] = cid

    return by_category, texts_by_id, skipped


def apply_clean_map(by_category: Dict[str, List[Dict[str, Any]]],
                    clean_map: Dict[str, bool]) -> int:
    """Clear `hallucination.background` for entries the judge marked clean.
    Pops the temporary `_judge_id` key from every entry. Returns count cleared."""
    cleared = 0
    for entries in by_category.values():
        for e in entries:
            cid = e.pop("_judge_id", None)
            if cid is None:
                continue
            if clean_map.get(cid, False):
                e["hallucination"]["background"] = ""
                cleared += 1
    return cleared


def write_run_outputs(run_dir: Path,
                      by_category: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    """Write per-category JSONs for one run. Returns {cat: count}."""
    counts: Dict[str, int] = {}
    for cat, entries in by_category.items():
        out_dir = DATA_DIR / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pipeline_{run_dir.name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        counts[cat] = len(entries)
    return counts


def is_run_already_processed(run_name: str) -> bool:
    """True if at least one data/<cat>/pipeline_<run_name>.json already exists.
    Used to skip already-postprocessed runs on retry after a partial timeout."""
    return bool(list(DATA_DIR.glob(f"*/pipeline_{run_name}.json")))


def strip_judge_ids(by_category: Dict[str, List[Dict[str, Any]]]) -> None:
    """Pop _judge_id from every entry (in place)."""
    for entries in by_category.values():
        for e in entries:
            e.pop("_judge_id", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir", type=Path, default=GRID_DIR_DEFAULT,
        help="Directory holding run_<...>/final_results.json subdirs.",
    )
    parser.add_argument(
        "--run", action="append", default=None,
        help="Specific run name(s) to process (repeat or comma-separate).",
    )
    parser.add_argument(
        "--use-gemini-judge", action="store_true",
        help="After heuristic filtering, send remaining non-empty background "
             "texts to Gemini-as-judge via the asynchronous Batch API. Items "
             "are packed (default 15/request) and batches submitted in waves "
             "(default 10) to stay under free-tier 10K/day. "
             "Requires GEMINI_API_KEY/GOOGLE_API_KEY (auto-loaded from .env). "
             "Tunable via env: JUDGE_MODEL, JUDGE_POLL_INTERVAL_S, "
             "JUDGE_PACK_SIZE, JUDGE_WAVE_SIZE, JUDGE_SUBMIT_MAX_RETRIES.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process runs whose data/<cat>/pipeline_<run>.json files already "
             "exist. Default is to skip them (idempotent retry after timeout).",
    )
    args = parser.parse_args()

    if not args.grid_dir.exists():
        raise SystemExit(f"Grid dir not found: {args.grid_dir}")

    if args.run:
        runs = []
        for r in args.run:
            for n in r.split(","):
                n = n.strip()
                if not n:
                    continue
                p = args.grid_dir / n
                if p.is_dir():
                    runs.append(p)
                else:
                    print(f"WARNING: requested run not found: {p}")
    else:
        runs = sorted(
            p for p in args.grid_dir.iterdir()
            if p.is_dir() and (p / "final_results.json").exists()
        )

    if not runs:
        raise SystemExit("No runs to process.")

    # Skip runs whose data/<cat>/pipeline_<run>.json files already exist (any cat).
    # This makes retries after a partial-timeout idempotent: previously-finished
    # runs are not re-judged, only the missing ones are. Use --force to override.
    if not args.force:
        already = [r for r in runs if is_run_already_processed(r.name)]
        runs = [r for r in runs if not is_run_already_processed(r.name)]
        if already:
            print(f"Skipping {len(already)} runs already postprocessed "
                  f"(use --force to re-process):")
            for r in already[:10]:
                print(f"  [skip-existing] {r.name}")
            if len(already) > 10:
                print(f"  ... and {len(already) - 10} more")

    if not runs:
        print("All runs already postprocessed. Nothing to do.")
        return

    print(f"Processing {len(runs)} runs from {args.grid_dir}")
    print("-" * 80)

    # Phase 1: convert every run (purely local, no API)
    converted: List[Tuple[Path, Dict[str, List[Dict[str, Any]]], Dict[str, str], int]] = []
    for run in runs:
        result = convert_run(run)
        if result is None:
            continue
        by_category, texts_by_id, skipped = result
        converted.append((run, by_category, texts_by_id, skipped))
        n_total = sum(len(v) for v in by_category.values())
        n_pending = len(texts_by_id)
        print(f"  [converted] {run.name}: {n_total} entries, "
              f"{n_pending} bg-texts pending judge"
              + (f"  [skipped {skipped}]" if skipped else ""))

    # Index converted entries by run name so per-wave handlers can find them.
    converted_by_name: Dict[str, Tuple[Path, Dict[str, List[Dict[str, Any]]], Dict[str, str], int]] = {
        run.name: (run, by_cat, tb, sk) for run, by_cat, tb, sk in converted
    }
    written: Set[str] = set()
    grand_total = defaultdict(int)

    def _flush_run(run_name: str) -> None:
        """Strip judge ids and write per-cat JSONs for one run. Idempotent."""
        if run_name in written:
            return
        run, by_cat, _, skipped = converted_by_name[run_name]
        strip_judge_ids(by_cat)
        counts = write_run_outputs(run, by_cat)
        written.add(run_name)
        for c, n in counts.items():
            grand_total[c] += n
        cat_summary = ", ".join(f"{c}={n}" for c, n in sorted(counts.items()))
        extra = f"  [skipped {skipped}]" if skipped else ""
        print(f"  [write] {run_name}: {sum(counts.values())} entries  ({cat_summary}){extra}")

    # Phase 2: if --use-gemini-judge, submit batches in waves of JUDGE_WAVE_SIZE.
    # After each wave's polling completes, immediately apply judge results AND
    # write per-cat JSONs for those runs — so a later TIMEOUT cannot lose
    # already-judged work. Runs without pending bg-texts (heuristic-only) are
    # written in Phase 3 below.
    if args.use_gemini_judge:
        runs_with_pending = [(run, t) for run, _, t, _ in converted if t]
        any_pending = sum(len(t) for _, t in runs_with_pending)
        if any_pending == 0:
            print("\nNo non-empty bg-texts after heuristic; skipping Gemini judge.")
        else:
            client = make_genai_client()
            print(f"\nSubmitting {len(runs_with_pending)} batch jobs "
                  f"({any_pending} total bg-texts → "
                  f"~{(any_pending + JUDGE_PACK_SIZE - 1) // JUDGE_PACK_SIZE} packed requests) "
                  f"in waves of {JUDGE_WAVE_SIZE}...")

            for wave_start in range(0, len(runs_with_pending), JUDGE_WAVE_SIZE):
                wave = runs_with_pending[wave_start:wave_start + JUDGE_WAVE_SIZE]
                wave_idx = wave_start // JUDGE_WAVE_SIZE + 1
                n_waves = (len(runs_with_pending) + JUDGE_WAVE_SIZE - 1) // JUDGE_WAVE_SIZE
                print(f"\n--- Wave {wave_idx}/{n_waves}: submitting {len(wave)} batch jobs ---")

                wave_jobs: List[str] = []
                wave_job_to_run: Dict[str, Path] = {}
                for run, texts_by_id in wave:
                    display = f"vigil_{run.name}_{int(time.time())}"
                    job_name = submit_judge_batch(client, texts_by_id, display)
                    if job_name:
                        wave_job_to_run[job_name] = run
                        wave_jobs.append(job_name)
                        n_packs = (len(texts_by_id) + JUDGE_PACK_SIZE - 1) // JUDGE_PACK_SIZE
                        print(f"  [submit] {run.name}: job={job_name} "
                              f"({len(texts_by_id)} bg-texts → {n_packs} packed requests)")
                    else:
                        print(f"  [submit] {run.name}: SKIPPED (submit failed permanently)")

                if wave_jobs:
                    print(f"\n--- Wave {wave_idx}/{n_waves}: waiting for {len(wave_jobs)} jobs ---")
                    finished = wait_for_jobs(client, wave_jobs)
                    for jn, job in finished.items():
                        run = wave_job_to_run[jn]
                        clean_map = download_judge_result(client, job)
                        _, by_cat, tb, _ = converted_by_name[run.name]
                        cleared = apply_clean_map(by_cat, clean_map)
                        print(f"  [judge] {run.name}: cleared {cleared}/{len(tb)}")

                # Write outputs for every run in this wave (whether submit/judge
                # succeeded or not — failed submits still get heuristic-only data
                # written so the file exists and skip-existing works on retry).
                for run, _ in wave:
                    _flush_run(run.name)
                print(f"--- Wave {wave_idx}/{n_waves}: wrote {len(wave)} runs ---")

    # Phase 3: write any runs that were not written in Phase 2 (heuristic-only,
    # or non-judge mode entirely).
    print("\nWriting per-category JSONs for remaining runs...")
    for run, _, _, _ in converted:
        _flush_run(run.name)

    print("-" * 80)
    print("Summary across all runs:")
    for c, n in sorted(grand_total.items()):
        print(f"  {c}: {n} entries")


if __name__ == "__main__":
    main()
