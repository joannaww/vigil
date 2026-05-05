import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

from engines.qwen import QwenEngine
from configs.config_schema import PhysicalEvaluatorConfig
from utils.visualization import save_visualization

logger = logging.getLogger(__name__)


class PhysicalEvaluator:
    """
    Detects physical-realism hallucinations (lighting, shadows, scale, gravity,
    artifacts) using Qwen3-VL in two passes:
      1. per-object: padded crop of each paired generated object
      2. scene-level: full generated image with red bboxes around paired objects
    """

    def __init__(self, engine: QwenEngine, config: PhysicalEvaluatorConfig):
        self.engine = engine
        self.config = config
        logger.info(
            f"PhysicalEvaluator initialized "
            f"(padding_ratio={config.padding_ratio}, "
            f"resolution={config.resolution}, scene_resolution={config.scene_resolution})"
        )

    def extract_padded_crop(
        self, image_path: str, bbox: List[int]
    ) -> Optional[Image.Image]:
        try:
            if not Path(str(image_path)).exists():
                logger.warning(f"Image not found: {image_path}")
                return None
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

        if not bbox or len(bbox) != 4:
            return self._resize(image, self.config.resolution)

        x1, y1, x2, y2 = bbox
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        pad = int(round(self.config.padding_ratio * max(bbox_w, bbox_h)))

        w, h = image.size
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad)
        y2p = min(h, y2 + pad)

        if x2p <= x1p or y2p <= y1p:
            return None

        crop = image.crop((x1p, y1p, x2p, y2p))
        return self._resize(crop, self.config.resolution)

    def draw_scene(
        self, image_path: str, boxes: List[Tuple[List[int], str]]
    ) -> Optional[Image.Image]:
        try:
            if not Path(str(image_path)).exists():
                return None
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load scene image {image_path}: {e}")
            return None

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        color = tuple(self.config.bbox_color_rgb)
        thick = self.config.bbox_thickness

        for bbox, _label in boxes:
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thick)

        return self._resize(annotated, self.config.scene_resolution)

    def evaluate_per_object_batch(
        self, jobs: List[Dict[str, Any]]
    ) -> List[str]:
        if not jobs:
            return []
        prompts = [
            self.config.per_object_prompt.format(label=j.get("label", "object"))
            for j in jobs
        ]
        images = [j["crop"] for j in jobs]
        try:
            return [self._clean(r) for r in self.engine.generate(prompts, images=images)]
        except Exception as e:
            logger.error(f"Per-object physical batch failed: {e}")
            return ["" for _ in jobs]

    def evaluate_scene_batch(self, jobs: List[Dict[str, Any]]) -> List[str]:
        if not jobs:
            return []
        prompts = [self.config.scene_prompt for _ in jobs]
        images = [j["img"] for j in jobs]
        try:
            return [self._clean(r) for r in self.engine.generate(prompts, images=images)]
        except Exception as e:
            logger.error(f"Scene-level physical batch failed: {e}")
            return ["" for _ in jobs]

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        s = text.strip()
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            s = s[1:-1].strip()
        low = s.lower()
        if low in {"none", "n/a", "no issues", "looks fine", "correct", "ok"}:
            return ""
        return s

    @staticmethod
    def _resize(img: Image.Image, max_dim: int) -> Image.Image:
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return img


def process_physical_evaluation(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    evaluator: PhysicalEvaluator,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Pipeline Step 5: Physical Realism Check.
    Adds 'physical_evaluation' to each datapoint, with per-object findings,
    a scene-level finding, and a merged single text for downstream comparison.
    """

    cfg = evaluator.config
    n = len(input_data)

    for dp in input_data:
        dp.setdefault(
            "physical_evaluation",
            {"scene_level": "", "per_object": [], "text": ""},
        )

    obj_buffer: List[Dict[str, Any]] = []
    obj_buffer_meta: List[Tuple[int, str, str]] = []  # (dp_idx, gen_id, label)

    def flush_objects():
        if not obj_buffer:
            return
        results = evaluator.evaluate_per_object_batch(obj_buffer)
        for (dp_idx, gen_id, label), issue in zip(obj_buffer_meta, results):
            input_data[dp_idx]["physical_evaluation"]["per_object"].append(
                {"object_id": gen_id, "label": label, "issue": issue}
            )
        obj_buffer.clear()
        obj_buffer_meta.clear()

    for dp_idx, dp in enumerate(input_data):
        gen_data = dp.get("generated_result")
        if not gen_data or not gen_data.get("path"):
            continue

        gen_obj_map = {
            o["object_id"]: o for o in gen_data.get("objects", []) if "object_id" in o
        }

        for pair in dp.get("object_pairings", []):
            if not pair.get("paired"):
                continue
            gen_id = pair.get("best_generated_object_id")
            obj = gen_obj_map.get(gen_id)
            if not obj or not obj.get("bbox"):
                continue

            crop = evaluator.extract_padded_crop(gen_data["path"], obj["bbox"])
            if crop is None:
                continue

            label = obj.get("label", "object")
            obj_buffer.append({"crop": crop, "label": label})
            obj_buffer_meta.append((dp_idx, gen_id, label))

            if cfg.save_viz:
                viz_dir = f"{cfg.output_dir}/{dp.get('id', dp_idx)}/{cfg.viz_dir}/physical"
                Path(viz_dir).mkdir(parents=True, exist_ok=True)
                save_visualization(
                    crop, f"{viz_dir}/object_{gen_id}_padded.jpg"
                )

            if len(obj_buffer) >= cfg.batch_size:
                flush_objects()

    flush_objects()

    scene_buffer: List[Dict[str, Any]] = []
    scene_buffer_idx: List[int] = []

    def flush_scenes():
        if not scene_buffer:
            return
        results = evaluator.evaluate_scene_batch(scene_buffer)
        for dp_idx, scene_text in zip(scene_buffer_idx, results):
            input_data[dp_idx]["physical_evaluation"]["scene_level"] = scene_text
        scene_buffer.clear()
        scene_buffer_idx.clear()

    for dp_idx, dp in enumerate(input_data):
        gen_data = dp.get("generated_result")
        if not gen_data or not gen_data.get("path"):
            continue

        paired_gen_ids = {
            p["best_generated_object_id"]
            for p in dp.get("object_pairings", [])
            if p.get("paired") and p.get("best_generated_object_id")
        }
        if not paired_gen_ids:
            continue

        boxes = [
            (o["bbox"], o.get("label", ""))
            for o in gen_data.get("objects", [])
            if o.get("object_id") in paired_gen_ids and o.get("bbox")
        ]
        if not boxes:
            continue

        annotated = evaluator.draw_scene(gen_data["path"], boxes)
        if annotated is None:
            continue

        scene_buffer.append({"img": annotated})
        scene_buffer_idx.append(dp_idx)

        if cfg.save_viz:
            viz_dir = f"{cfg.output_dir}/{dp.get('id', dp_idx)}/{cfg.viz_dir}/physical"
            Path(viz_dir).mkdir(parents=True, exist_ok=True)
            save_visualization(annotated, f"{viz_dir}/scene_with_bboxes.jpg")

        if len(scene_buffer) >= cfg.scene_batch_size:
            flush_scenes()

    flush_scenes()

    for dp in input_data:
        pe = dp["physical_evaluation"]
        parts = []
        scene = (pe.get("scene_level") or "").strip()
        if scene:
            parts.append(f"Scene: {scene}")
        for obj_eval in pe.get("per_object", []):
            issue = (obj_eval.get("issue") or "").strip()
            if issue:
                parts.append(f"{obj_eval.get('label', 'object')}: {issue}")
        pe["text"] = "; ".join(parts)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(input_data, f, indent=4, default=str)

    logger.info(f"Physical evaluation done for {n} datapoints.")
    return input_data
