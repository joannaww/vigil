import os
import json
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
import google.generativeai as genai
from google.api_core import exceptions as gax_exc

SEED = 42
random.seed(SEED)

_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env. "
        "Get a key at https://aistudio.google.com/app/apikey"
    )
genai.configure(api_key=_API_KEY)

# Concurrency + rate-limit knobs (override via env).
#   MAX_WORKERS       : how many parallel API calls in flight
#   MIN_INTERVAL_S    : min spacing between request *starts* across all workers
#                       (set to ~6.5 to stay under 10 RPM; 0 = no throttle, rely on backoff)
#   MAX_RETRIES       : retry attempts on 429 / 503 / timeouts
MAX_WORKERS = int(os.environ.get("GEMINI_MAX_WORKERS", "8"))
MIN_INTERVAL_S = float(os.environ.get("GEMINI_MIN_INTERVAL_S", "0.0"))
MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "8"))

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

class HallucinationBaseline(BaseModel):
    objects: str = Field(description="Semantic description of object hallucinations.")
    background: str = Field(description="Semantic description of background hallucinations.")
    position_logic: str = Field(description="Semantic description of position logic hallucination")
    physical: str = Field(description="Semantic description of physical hallucination")
    object_omission: str = Field(description="Semantic description of object omission hallucinations.")

BASE_PROMPT_TEMPLATE = """
You are an image re-contextualization hallucination inspector.
You will be given:
- An instruction prompt (what the generator was asked to do).
- A background image (background_image).
- Reference object image(s) (one or two) - object1_image, object2_image.
- The generated image (generated_image) - result of re-contextualization.

Your task: compare the generated_image to the references and to the instruction, and produce THREE SHORT SEMANTIC DESCRIPTIONS (1-3 sentences each) answering the following categories. If no issues or hallucinations are detected in a category, return an empty string ("") for that category.

Categories:
1) objects: Object Visual Fidelity - texture/shape/color identity mismatches, mutations, identity loss, reference bleeding. Example: "Feature Mutation: sofa's color changed from dark green to black; Identity Loss: inserted cabinet is metallic vs wicker."
2) background: Background Fidelity - background mutations, background detail loss, context swap. Example: "Background Mutation: wall color changed; Context Swap: bedroom replaced by living room."
3) position_logic: Spatial and Instructional Fidelity - misplacement of inserted object, replacement failure (the item that should have been replaces remain in the image). Example: "Misplacement: The green chair is on the right side of the bed instead of the left side."
4) physical: Physical and Integration Fidelity - lighting/shadow incoherence, perspective/scale issues, artifacts (unnatural phenomena). Example: "Shadow Incoherence: The is no shadow under the inserted chair."
5) object_omission: Object Omission - missing required objects from the instruction that should have been pasted from object image. Example: "Object Omission: green cabinet missing."

Instruction Prompt:
{instruction_prompt}

Analyze the provided images now.
"""

def load_existing_results(output_path: str):
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = {r["generated_photo"] for r in results if "hallucination" in r}
            return results, processed
        except Exception as e:
            print(f"  [!] Error reading {output_path}: {e}")
    return [], set()

def analyze_example(instruction_prompt: str, background_path: str, object1_path: str, generated_path: str, object2_path: Optional[str] = None):
    images = []
    try:
        images.append(Image.open(background_path))
        images.append(Image.open(object1_path))
        if object2_path:
            images.append(Image.open(object2_path))
        images.append(Image.open(generated_path))
    except Exception as e:
        return {"_error": f"File error: {e}"}

    text_prompt = BASE_PROMPT_TEMPLATE.format(instruction_prompt=instruction_prompt)
    model = genai.GenerativeModel('gemini-2.5-flash')

    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            response = model.generate_content(
                [text_prompt] + images,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=HallucinationBaseline,
                ),
            )
            return json.loads(response.text)
        except (gax_exc.ResourceExhausted, gax_exc.TooManyRequests,
                gax_exc.ServiceUnavailable, gax_exc.DeadlineExceeded) as e:
            wait = min(60, (2 ** attempt) + random.random())
            print(f"  [rate/transient] {type(e).__name__}: sleep {wait:.1f}s "
                  f"(attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
        except Exception as e:
            return {"_error": f"Gemini error: {str(e)}"}
    return {"_error": "Gemini error: max retries exceeded (rate limit)"}

def _resolve_paths(annotation, data_dir):
    """Return (annotation, prompt_text, bg, obj1, gen, obj2) or None if missing files."""
    photo_name = annotation["generated_photo"]
    example_id = photo_name[:4]
    base_path = os.path.join(data_dir, example_id)
    prompt_path = os.path.join(base_path, f"{example_id}prompt.txt")
    if not os.path.exists(prompt_path):
        return None
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    actual_gen_path = None
    for name in [f"{example_id}generated.png", f"{example_id}generated_01.png", photo_name]:
        test_path = os.path.join(base_path, name)
        if os.path.exists(test_path):
            actual_gen_path = test_path
            break
    if not actual_gen_path:
        return None

    obj2_path = os.path.join(base_path, f"{example_id}object2.png")
    return (
        annotation,
        prompt_text,
        os.path.join(base_path, f"{example_id}background.png"),
        os.path.join(base_path, f"{example_id}object1.png"),
        actual_gen_path,
        obj2_path if os.path.exists(obj2_path) else None,
    )

def process_dataset(dataset_info):
    input_file = dataset_info["input"]
    output_file = dataset_info["output"]
    data_dir = dataset_info["data_dir"]

    results, processed_photos = load_existing_results(output_file)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            all_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        return

    to_process = [a for a in all_annotations if a["generated_photo"] not in processed_photos]

    print(f"\nCategory: {os.path.basename(input_file)}")
    print(f"  - Total in final: {len(all_annotations)}")
    print(f"  - Already processed: {len(processed_photos)}")
    print(f"  - To be processed: {len(to_process)}")
    print(f"  - Workers: {MAX_WORKERS}, min interval: {MIN_INTERVAL_S}s, max retries: {MAX_RETRIES}")

    if not to_process:
        print("  - Status: Kompletne.")
        return

    jobs = [j for j in (_resolve_paths(a, data_dir) for a in to_process) if j is not None]
    skipped = len(to_process) - len(jobs)
    if skipped:
        print(f"  - Skipped (missing files): {skipped}")

    save_lock = threading.Lock()
    save_every = max(1, int(os.environ.get("GEMINI_SAVE_EVERY", "5")))

    def _worker(job):
        ann, prompt_text, bg, obj1, gen, obj2 = job
        res = analyze_example(prompt_text, bg, obj1, gen, obj2)
        return ann, res

    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Gemini parallel"):
            try:
                ann, res = fut.result()
            except Exception as e:
                print(f"  [!] Worker crash: {e}")
                continue
            if "_error" in res:
                print(f"  [!] {ann['generated_photo']}: {res['_error']}")
                continue
            with save_lock:
                results.append({**ann, "hallucination": res})
                completed_since_save += 1
                if completed_since_save >= save_every:
                    results.sort(key=lambda r: r["generated_photo"])
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    completed_since_save = 0

    # Final flush
    with save_lock:
        results.sort(key=lambda r: r["generated_photo"])
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    DATA_ROOT = "/net/pr2/projects/plgrid/plggrecontext/joanna/vigil/vigil/data"
    CATEGORIES = ["cars", "clothes", "cosmetics", "electronics", "furniture"]

    DATASETS = [
        {
            "input": os.path.join(DATA_ROOT, cat, "annotations.json"),
            "data_dir": os.path.join(DATA_ROOT, cat, "data"),
            "output": os.path.join(DATA_ROOT, cat, "baseline_gemini.json"),
        }
        for cat in CATEGORIES
    ]

    for ds in DATASETS:
        process_dataset(ds)

if __name__ == "__main__":
    main()
