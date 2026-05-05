import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

from engines.qwen import QwenEngine
from configs.config_schema import PositionLogicEvaluatorConfig
from utils.visualization import save_visualization

logger = logging.getLogger(__name__)

_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s*(.+)$")


class PositionLogicEvaluator:
    """
    Detects position_logic hallucinations in two passes:
      1. distiller: text-only Qwen extracts atomic spatial constraints from the prompt.
      2. scene-pass: VLM compares background vs annotated generated, verifies the
         distilled checklist, and scans for visible leftovers — all in a single
         call per datapoint that returns structured JSON.
    """

    def __init__(self, engine: QwenEngine, config: PositionLogicEvaluatorConfig):
        self.engine = engine
        self.config = config
        logger.info(
            f"PositionLogicEvaluator initialized "
            f"(scene_resolution={config.scene_resolution}, "
            f"distiller_batch={config.distiller_batch_size}, "
            f"scene_batch={config.scene_batch_size})"
        )

    def distill_constraints_batch(
        self, prompts: List[str]
    ) -> List[Dict[str, Any]]:
        if not prompts:
            return []
        formatted = [self.config.distiller_prompt.format(prompt=p) for p in prompts]
        try:
            responses = self.engine.generate(formatted)
        except Exception as e:
            logger.error(f"Distiller batch failed: {e}")
            return [{"raw": "", "items": []} for _ in prompts]
        return [self._parse_distiller(r) for r in responses]

    def verify_scene_batch(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not jobs:
            return []
        prompts = [
            self.config.scene_prompt.format(constraint_checklist=j["checklist"])
            for j in jobs
        ]
        images = [[j["bg_img"], j["gen_img"]] for j in jobs]
        try:
            raw_responses = self.engine.generate(prompts, images=images)
        except Exception as e:
            logger.error(f"Scene verification batch failed: {e}")
            return [self._empty_scene_result() for _ in jobs]
        return [self._parse_scene(r) for r in raw_responses]

    def draw_scene(
        self, image: Image.Image, boxes: List[Tuple[List[int], str]]
    ) -> Image.Image:
        annotated = image.copy()
        if not boxes:
            return annotated
        draw = ImageDraw.Draw(annotated)
        color = tuple(self.config.bbox_color_rgb)
        thick = self.config.bbox_thickness
        for bbox, _label in boxes:
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thick)
        return annotated

    def _parse_distiller(self, text: str) -> Dict[str, Any]:
        s = (text or "").strip()
        if not s or s.upper() == "NONE":
            return {"raw": "", "items": []}
        items: List[str] = []
        for line in s.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = _NUMBERED_LINE.match(line)
            items.append(m.group(1).strip() if m else line)
        return {"raw": s, "items": items}

    def _empty_scene_result(self) -> Dict[str, Any]:
        return {
            "constraints": [],
            "leftover_detected": False,
            "leftover_description": "",
            "raw": "",
            "parse_ok": False,
        }

    def _parse_scene(self, text: str) -> Dict[str, Any]:
        if not text:
            return self._empty_scene_result()
        raw = text.strip()
        s = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            obj = json.loads(s)
        except Exception:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if not m:
                return {**self._empty_scene_result(), "raw": raw}
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return {**self._empty_scene_result(), "raw": raw}

        constraints_out = []
        for c in obj.get("constraints", []) or []:
            if not isinstance(c, dict):
                continue
            entry = {
                "constraint": str(c.get("constraint", "")).strip(),
                "satisfied": bool(c.get("satisfied", True)),
                "violation": str(c.get("violation", "")).strip(),
            }
            constraints_out.append(entry)

        return {
            "constraints": constraints_out,
            "leftover_detected": bool(obj.get("leftover_detected", False)),
            "leftover_description": str(obj.get("leftover_description", "")).strip(),
            "raw": raw,
            "parse_ok": True,
        }

    @staticmethod
    def merge_scene_text(scene: Dict[str, Any]) -> str:
        parts = []
        for c in scene.get("constraints", []):
            if not c.get("satisfied", True):
                v = (c.get("violation") or "").strip()
                if v:
                    parts.append(v)
        if scene.get("leftover_detected") and scene.get("leftover_description"):
            parts.append(f"Leftover: {scene['leftover_description']}")
        return "; ".join(parts)

    @staticmethod
    def _resize(img: Image.Image, max_dim: int) -> Image.Image:
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return img


def process_position_logic_evaluation(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    evaluator: PositionLogicEvaluator,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Pipeline Step 6: Position Logic Check.
    Adds 'position_logic_evaluation' to each datapoint with:
      - constraints: distilled atomic spatial constraints from the prompt
      - scene_analysis: structured VLM verdict (per-constraint + leftover)
      - text: single string violation summary for downstream comparison with GT
    """

    cfg = evaluator.config
    n = len(input_data)

    for dp in input_data:
        dp.setdefault(
            "position_logic_evaluation",
            {
                "constraints": {"raw": "", "items": []},
                "scene_analysis": evaluator._empty_scene_result(),
                "text": "",
            },
        )

    prompts_to_distill = [dp.get("text_prompt") or "" for dp in input_data]
    bs = cfg.distiller_batch_size
    for i in range(0, len(prompts_to_distill), bs):
        batch = prompts_to_distill[i : i + bs]
        results = evaluator.distill_constraints_batch(batch)
        for j, res in enumerate(results):
            input_data[i + j]["position_logic_evaluation"]["constraints"] = res

    scene_buffer: List[Dict[str, Any]] = []
    scene_buffer_idx: List[int] = []

    def flush_scenes():
        if not scene_buffer:
            return
        results = evaluator.verify_scene_batch(scene_buffer)
        for dp_idx, scene_obj in zip(scene_buffer_idx, results):
            input_data[dp_idx]["position_logic_evaluation"]["scene_analysis"] = scene_obj
        scene_buffer.clear()
        scene_buffer_idx.clear()

    for dp_idx, dp in enumerate(input_data):
        gen_data = dp.get("generated_result")
        if not gen_data or not gen_data.get("path"):
            continue

        gen_path = gen_data["path"]
        parent_dir = Path(gen_path).parent
        bg_candidates = (
            list(parent_dir.glob("*background*.png"))
            + list(parent_dir.glob("*background*.jpg"))
        )
        if not bg_candidates:
            logger.debug(f"No background image for {dp.get('id')}")
            continue
        bg_path = bg_candidates[0]

        paired_gen_ids = {
            p["best_generated_object_id"]
            for p in dp.get("object_pairings", [])
            if p.get("paired") and p.get("best_generated_object_id")
        }
        boxes = [
            (o["bbox"], o.get("label", ""))
            for o in gen_data.get("objects", [])
            if o.get("object_id") in paired_gen_ids and o.get("bbox")
        ]
        if not boxes:
            continue

        try:
            bg_img = Image.open(bg_path).convert("RGB")
            gen_img = Image.open(gen_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not load images for {dp.get('id')}: {e}")
            continue

        annotated = evaluator.draw_scene(gen_img, boxes)
        bg_img = evaluator._resize(bg_img, cfg.scene_resolution)
        annotated = evaluator._resize(annotated, cfg.scene_resolution)

        constraints = dp["position_logic_evaluation"]["constraints"]
        checklist_text = constraints.get("raw") or "NONE — no spatial constraints extracted from prompt."

        scene_buffer.append(
            {"checklist": checklist_text, "bg_img": bg_img, "gen_img": annotated}
        )
        scene_buffer_idx.append(dp_idx)

        if cfg.save_viz:
            viz_dir = f"{cfg.output_dir}/{dp.get('id', dp_idx)}/{cfg.viz_dir}/position_logic"
            Path(viz_dir).mkdir(parents=True, exist_ok=True)
            save_visualization(annotated, f"{viz_dir}/scene_with_bboxes.jpg")

        if len(scene_buffer) >= cfg.scene_batch_size:
            flush_scenes()

    flush_scenes()

    for dp in input_data:
        ple = dp["position_logic_evaluation"]
        scene = ple.get("scene_analysis") or evaluator._empty_scene_result()
        ple["text"] = evaluator.merge_scene_text(scene)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(input_data, f, indent=4, default=str)

    logger.info(f"Position-logic evaluation done for {n} datapoints.")
    return input_data
