import json
import argparse
import os
import re
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from tqdm import tqdm

CATEGORIES = ['cars', 'clothes', 'cosmetics', 'electronics', 'furniture']
HALLUCINATION_TYPES = ['objects', 'background', 'position_logic', 'physical', 'object_omission']
MODELS = ['qwen8b', 'gemini', 'gemma12b']

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "evaluations"

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
MAX_WORKERS = int(os.environ.get("JUDGE_MAX_WORKERS", "4"))
MIN_INTERVAL_S = float(os.environ.get("JUDGE_MIN_INTERVAL_S", "0.0"))
MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "5"))

_throttle_lock = threading.Lock()
_last_call_ts = 0.0


def _throttle():
    if MIN_INTERVAL_S <= 0:
        return
    global _last_call_ts
    with _throttle_lock:
        elapsed = time.time() - _last_call_ts
        if elapsed < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - elapsed)
        _last_call_ts = time.time()


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


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'F1-score': round(f1, 4), 'Precision': round(precision, 4), 'Recall': round(recall, 4)}


def llm_count_errors_batch(model, items):
    if not items:
        return []

    batch_tasks = []
    for it in items:
        gt_display = it['gt_text'] if it['gt_text'] else '[EMPTY - no errors]'
        pred_display = it['pred_text'] if it['pred_text'] else '[EMPTY - no errors detected]'
        task_str = JUDGE_PROMPT_TEMPLATE.format(
            hallucination_type=it['hallucination_type'],
            ground_truth_text=gt_display,
            pipeline_text=pred_display,
        )
        batch_tasks.append(f"CASE ID: {it['id']}\n{task_str}\n---")

    final_prompt = (
        "Respond with a JSON array of objects. Each object MUST have: \"id\" (string), "
        "\"tp\" (int), \"fn\" (int), \"fp\" (int). "
        "Do not include any reasoning or extra text.\n\n" + "\n".join(batch_tasks)
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            _throttle()
            response = model.generate_content(
                final_prompt,
                generation_config={
                    "temperature": 0.0 if attempt == 0 else 0.2,
                    "response_mime_type": "application/json",
                },
            )
            raw_text = response.text.strip()
            clean_json = re.sub(r"^```json\s*|^```\s*|```$", "", raw_text, flags=re.MULTILINE).strip()
            return json.loads(clean_json)
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
                continue
            print(f"\n[!] Error after {MAX_RETRIES} attempts (first ID: {items[0]['id']}): {e}")
            return [{'id': it['id'], 'tp': 0, 'fn': 0, 'fp': 0} for it in items]


def run_evaluation(baseline_name, batch_size):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in env / .env")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(JUDGE_MODEL)

    all_cat_stats = {}

    for cat in CATEGORIES:
        gt_f = DATA_DIR / cat / "annotations.json"
        pred_f = DATA_DIR / cat / f"baseline_{baseline_name}.json"
        if not gt_f.exists() or not pred_f.exists():
            print(f"  [skip] {baseline_name}/{cat}: missing {gt_f.name} or {pred_f.name}")
            continue

        with open(gt_f) as f:
            gt_data = json.load(f)
        with open(pred_f) as f:
            pred_data = {item['generated_photo']: item['hallucination'] for item in json.load(f)}

        cat_counts = {ht: {'tp': 0, 'fp': 0, 'fn': 0} for ht in HALLUCINATION_TYPES}
        queue = []
        for idx, gt_item in enumerate(gt_data):
            photo = gt_item['generated_photo']
            if photo not in pred_data:
                continue
            for ht in HALLUCINATION_TYPES:
                gt_txt = gt_item['hallucination'].get(ht, '')
                pred_txt = pred_data[photo].get(ht, '')
                if gt_txt or pred_txt:
                    queue.append({
                        'id': f"{cat}_{idx}_{ht}",
                        'hallucination_type': ht,
                        'gt_text': gt_txt,
                        'pred_text': pred_txt,
                        'ht_key': ht,
                    })

        batches = [queue[i:i + batch_size] for i in range(0, len(queue), batch_size)]
        desc = f"Judging {baseline_name}/{cat}"

        if MAX_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {ex.submit(llm_count_errors_batch, model, b): b for b in batches}
                for fut in tqdm(as_completed(futures), total=len(batches), desc=desc):
                    batch = futures[fut]
                    results = fut.result()
                    res_dict = {str(r['id']): r for r in results if 'id' in r}
                    for it in batch:
                        res = res_dict.get(it['id'], {'tp': 0, 'fp': 0, 'fn': 0})
                        cat_counts[it['ht_key']]['tp'] += int(res.get('tp', 0))
                        cat_counts[it['ht_key']]['fp'] += int(res.get('fp', 0))
                        cat_counts[it['ht_key']]['fn'] += int(res.get('fn', 0))
        else:
            for batch in tqdm(batches, desc=desc):
                results = llm_count_errors_batch(model, batch)
                res_dict = {str(r['id']): r for r in results if 'id' in r}
                for it in batch:
                    res = res_dict.get(it['id'], {'tp': 0, 'fp': 0, 'fn': 0})
                    cat_counts[it['ht_key']]['tp'] += int(res.get('tp', 0))
                    cat_counts[it['ht_key']]['fp'] += int(res.get('fp', 0))
                    cat_counts[it['ht_key']]['fn'] += int(res.get('fn', 0))

        all_cat_stats[cat] = cat_counts

    return all_cat_stats


def format_model_section(stats, baseline_name):
    n_types = len(HALLUCINATION_TYPES)
    cat_metrics = {}
    for cat, s in stats.items():
        per_type = []
        f1_list = []
        p_list = []
        r_list = []
        for ht in HALLUCINATION_TYPES:
            m = calculate_metrics(s[ht]['tp'], s[ht]['fp'], s[ht]['fn'])
            f1_list.append(m['F1-score'])
            p_list.append(m['Precision'])
            r_list.append(m['Recall'])
            per_type.append((ht, m['Precision'], m['Recall'], m['F1-score'],
                             s[ht]['tp'], s[ht]['fp'], s[ht]['fn']))
        cat_metrics[cat] = {
            'macro_f1': sum(f1_list) / n_types,
            'macro_p': sum(p_list) / n_types,
            'macro_r': sum(r_list) / n_types,
            'per_type': per_type,
        }

    lines = [f"\n{'=' * 100}",
             f"  LLM-AS-A-JUDGE REPORT: {baseline_name.upper()}",
             f"{'=' * 100}\n"]

    if cat_metrics:
        all_macro_f1 = sum(m['macro_f1'] for m in cat_metrics.values()) / len(cat_metrics)
        all_macro_p = sum(m['macro_p'] for m in cat_metrics.values()) / len(cat_metrics)
        all_macro_r = sum(m['macro_r'] for m in cat_metrics.values()) / len(cat_metrics)
        lines.append(
            f"OVERALL (Macro across {len(cat_metrics)} cats x {n_types} types):  "
            f"P={all_macro_p:.4f}  R={all_macro_r:.4f}  F1={all_macro_f1:.4f}"
        )
    else:
        lines.append("OVERALL: no data")
        return "\n".join(lines), {}

    lines.append("\n" + "-" * 100)
    lines.append(f"{'CATEGORY':<15} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'STAB. (4-excl) F1':<18}")
    lines.append("-" * 100)
    for cat in CATEGORIES:
        if cat not in cat_metrics:
            continue
        others = [c for c in CATEGORIES if c != cat and c in cat_metrics]
        excl_f1 = sum(cat_metrics[o]['macro_f1'] for o in others) / len(others) if others else 0
        m = cat_metrics[cat]
        lines.append(
            f"{cat.upper():<15} | {m['macro_p']:<8.4f} | {m['macro_r']:<8.4f} | "
            f"{m['macro_f1']:<8.4f} | {excl_f1:<18.4f}"
        )

    lines.append("\n" + "-" * 100)
    lines.append("PER HALLUCINATION TYPE (aggregated across all categories):")
    lines.append("-" * 100)
    lines.append(f"{'TYPE':<18} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'PREC':<8} | {'REC':<8} | {'F1':<8}")
    lines.append("-" * 100)
    type_totals = {ht: {'tp': 0, 'fp': 0, 'fn': 0} for ht in HALLUCINATION_TYPES}
    for cat in cat_metrics:
        for ht in HALLUCINATION_TYPES:
            for c, _, _, _, tp, fp, fn in cat_metrics[cat]['per_type']:
                if c == ht:
                    type_totals[ht]['tp'] += tp
                    type_totals[ht]['fp'] += fp
                    type_totals[ht]['fn'] += fn
    for ht in HALLUCINATION_TYPES:
        t = type_totals[ht]
        m = calculate_metrics(t['tp'], t['fp'], t['fn'])
        lines.append(
            f"{ht:<18} | {t['tp']:>5} | {t['fp']:>5} | {t['fn']:>5} | "
            f"{m['Precision']:<8.4f} | {m['Recall']:<8.4f} | {m['F1-score']:<8.4f}"
        )

    summary = {
        'macro_p': all_macro_p,
        'macro_r': all_macro_r,
        'macro_f1': all_macro_f1,
    }
    return "\n".join(lines), summary


def format_overall_ranking(model_summaries):
    if not model_summaries:
        return ""
    rows = sorted(model_summaries.items(), key=lambda kv: kv[1]['macro_f1'], reverse=True)
    out = [f"\n{'=' * 100}", "  OVERALL RANKING (Macro F1 across all categories x hallucination types)",
           f"{'=' * 100}",
           f"{'RANK':<5} | {'MODEL':<15} | {'PREC':<8} | {'REC':<8} | {'F1':<8}",
           "-" * 100]
    for i, (m, s) in enumerate(rows, 1):
        out.append(f"{i:<5} | {m.upper():<15} | {s['macro_p']:<8.4f} | {s['macro_r']:<8.4f} | {s['macro_f1']:<8.4f}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='all',
                        help=f"Model name (one of {MODELS}) or 'all'.")
    parser.add_argument('--batch_size', type=int, default=15,
                        help="How many (datapoint, hallucination_type) cases per Gemini request.")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    targets = MODELS if args.baseline == 'all' else [args.baseline]

    sections = []
    summaries = {}
    raw_stats = {}

    for baseline in targets:
        print(f"\n>>> Evaluating baseline: {baseline}")
        stats = run_evaluation(baseline, args.batch_size)
        section, summary = format_model_section(stats, baseline)
        sections.append(section)
        if summary:
            summaries[baseline] = summary
        raw_stats[baseline] = stats

    out_name = "JUDGE_REPORT_ALL.txt" if args.baseline == 'all' else f"JUDGE_REPORT_{args.baseline}.txt"
    out_path = EVAL_DIR / out_name

    with open(out_path, "w") as f:
        f.write(f"LLM-as-a-Judge evaluation (judge model: {JUDGE_MODEL})\n")
        f.write(f"Categories: {CATEGORIES}\n")
        f.write(f"Hallucination types: {HALLUCINATION_TYPES}\n")
        f.write(format_overall_ranking(summaries) + "\n")
        for s in sections:
            f.write(s + "\n")

    raw_path = EVAL_DIR / out_name.replace(".txt", "_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_stats, f, indent=2)

    print(f"\n[+] Report:    {out_path}")
    print(f"[+] Raw stats: {raw_path}")


if __name__ == "__main__":
    main()
