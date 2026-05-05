import json
import pandas as pd
from pathlib import Path

CATEGORIES = ['cars', 'clothes', 'cosmetics', 'electronics', 'furniture']
HALLUCINATION_TYPES = ['objects', 'background', 'position_logic', 'physical', 'object_omission']

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "evaluations"
MODELS = ['qwen8b', 'gemini', 'gemma12b']

def calculate_all_metrics(tp, fp, fn):
    """Counts Precision, Recall and F1 from raw TP/FP/FN."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

def format_ranking(df, title, sort_col):
    """Generates a readable ranking table."""
    res = f"\n{'='*100}\n"
    res += f"  RANKING: {title.upper()}\n"
    res += f"{'='*100}\n"
    res += f"{'RANK':<5} | {'MODEL':<20} | {'PREC':<10} | {'REC':<10} | {'F1':<10}\n"
    res += f"{'-'*100}\n"
    
    sorted_df = df.sort_values(by=sort_col, ascending=False)
    for i, (idx, row) in enumerate(sorted_df.iterrows(), 1):
        prefix = sort_col.replace('F1', '')
        p = row.get(f'{prefix}Prec', 0)
        r = row.get(f'{prefix}Rec', 0)
        f = row[sort_col]
        res += f"{i:<5} | {row['Model']:<20} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f}\n"
    return res

def run_comparison_report():
    data = []

    for model_name in MODELS:
        cat_stats = {}
        
        for cat in CATEGORIES:
            gt_path = DATA_DIR / cat / 'annotations.json'
            pred_path = DATA_DIR / cat / f'baseline_{model_name}.json'
            
            if not gt_path.exists() or not pred_path.exists():
                print(f"Missing files for {model_name} in category {cat}. Skipping.")
                continue

            with open(gt_path, 'r', encoding='utf-8') as f: gt_d = json.load(f)
            with open(pred_path, 'r', encoding='utf-8') as f: pred_d = json.load(f)

            pred_map = {item['generated_photo']: item['hallucination'] for item in pred_d}
            
            type_metrics = {ht: {'tp': 0, 'fp': 0, 'fn': 0} for ht in HALLUCINATION_TYPES}
            
            for gt_item in gt_d:
                photo = gt_item['generated_photo']
                gt_h = gt_item['hallucination']
                
                pred_h = pred_map.get(photo, {ht: "" for ht in HALLUCINATION_TYPES})
                
                for ht in HALLUCINATION_TYPES:
                    g_val = bool(gt_h.get(ht))
                    p_val = bool(pred_h.get(ht))
                    
                    if p_val and g_val: type_metrics[ht]['tp'] += 1
                    elif p_val and not g_val: type_metrics[ht]['fp'] += 1
                    elif not p_val and g_val: type_metrics[ht]['fn'] += 1
            
            f1_list, p_list, r_list = [], [], []
            for ht in HALLUCINATION_TYPES:
                p, r, f1 = calculate_all_metrics(type_metrics[ht]['tp'], type_metrics[ht]['fp'], type_metrics[ht]['fn'])
                p_list.append(p)
                r_list.append(r)
                f1_list.append(f1)
            
            n_types = len(HALLUCINATION_TYPES)
            cat_stats[cat] = {
                'f1': sum(f1_list) / n_types,
                'p': sum(p_list) / n_types,
                'r': sum(r_list) / n_types,
                'per_type': {
                    ht: {'p': p_list[i], 'r': r_list[i], 'f1': f1_list[i]}
                    for i, ht in enumerate(HALLUCINATION_TYPES)
                },
            }

        if not cat_stats: continue

        row = {'Model': model_name.upper()}
        
        row['ALL_Macro_Prec'] = sum(c['p'] for c in cat_stats.values()) / len(cat_stats)
        row['ALL_Macro_Rec'] = sum(c['r'] for c in cat_stats.values()) / len(cat_stats)
        row['ALL_Macro_F1'] = sum(c['f1'] for c in cat_stats.values()) / len(cat_stats)

        for cat in CATEGORIES:
            if cat not in cat_stats: continue
            
            row[f'Only_{cat}_F1'] = cat_stats[cat]['f1']
            row[f'Only_{cat}_Prec'] = cat_stats[cat]['p']
            row[f'Only_{cat}_Rec'] = cat_stats[cat]['r']
            
            o4 = [c for name, c in cat_stats.items() if name != cat]
            if o4:
                row[f'4excl_{cat}_Macro_F1'] = sum(c['f1'] for c in o4) / len(o4)
                row[f'4excl_{cat}_Macro_Prec'] = sum(c['p'] for c in o4) / len(o4)
                row[f'4excl_{cat}_Macro_Rec'] = sum(c['r'] for c in o4) / len(o4)
            
        data.append(row)

    df = pd.DataFrame(data)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_file = EVAL_DIR / "BASELINES_CLASSIFICATION_4PLUS1_EVALUATION.txt"
    with open(report_file, "w", encoding='utf-8') as f:
        f.write("COMPARISON REPORT: " + " vs ".join(m.upper() for m in MODELS) + "\n")
        f.write(format_ranking(df, "Macro Average (All 5 Categories)", "ALL_Macro_F1"))

        for cat in CATEGORIES:
            f.write(f"\n\n --- CATEGORY ANALYSIS: {cat.upper()} ---")
            f.write(format_ranking(df, f"Only {cat}", f"Only_{cat}_F1"))
            f.write(format_ranking(df, f"Stability: Macro without {cat}", f"4excl_{cat}_Macro_F1"))

    print(f"Saved report to: {report_file}")

if __name__ == "__main__":
    run_comparison_report()
