"""
SAM3 Segmentation Module

Uses SAM3 (Segment Anything 3) from Meta for text-prompted instance segmentation.
Returns masks and bounding boxes (XYXY format, pixel coordinates).
"""

from typing import List, Dict, Any, Union
from collections import defaultdict
import logging
import os

import dotenv
import torch
from torchvision.ops import nms
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model


dotenv.load_dotenv()
hf_token = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)


class SAM3Segmenter:
    """
    Segments objects in images using SAM3 with text prompts.

    Uses post_process_instance_segmentation() which returns both
    masks and tight bounding boxes derived from the masks.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        """
        Initialize SAM3 segmenter.

        Args:
            model_name: HuggingFace model identifier for SAM3
            dtype: Model precision (bfloat16 recommended for A100)
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        logger.info(f"Loading {model_name}...")
        self.processor = Sam3Processor.from_pretrained(model_name, token=hf_token)
        self.model = Sam3Model.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if device == "cuda" else None,
            token=hf_token,
        )
        self.model.eval()
        logger.info(f"SAM3 loaded successfully on {device}")

    def segment(
        self,
        image: Union[str, Image.Image],
        labels: List[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Segment objects in an image using text labels.

        Args:
            image: PIL Image or path to image file
            labels: List of entity labels to segment (e.g., ["dog", "red car"])
            threshold: Detection confidence threshold
            mask_threshold: Mask binarization threshold
            iou_threshold: IoU threshold for filtering duplicate detections (0-1).
                          Detections with IoU > threshold are considered duplicates.

        Returns:
            List of dicts containing:
                - object_id: Unique identifier (label_index)
                - label: Entity label
                - bbox: Bounding box [x1, y1, x2, y2] in XYXY pixel format
                - mask: Binary mask as numpy array
                - score: Confidence score
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image_width, image_height = image.size

        # SAM3 processor tokenizes with max_length=32 and no truncation, so we
        # must truncate labels ourselves to avoid a ValueError on long prompts.
        max_tokens = 32
        truncated_labels = [
            self.processor.tokenizer.decode(
                self.processor.tokenizer.encode(label, truncation=True, max_length=max_tokens),
                skip_special_tokens=True,
            )
            for label in labels
        ]
        if truncated_labels != labels:
            truncated = [o for o, t in zip(labels, truncated_labels) if o != t]
            logger.warning(
                "Labels truncated to %d tokens for SAM3: %s", max_tokens, truncated
            )

        inputs = self.processor(
            images=[image] * len(truncated_labels),
            text=truncated_labels,
            return_tensors="pt",
        ).to(self.device)

        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        original_sizes = [[image_height, image_width]] * len(labels)

        with torch.no_grad():
            outputs = self.model(**inputs)

        all_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=original_sizes,
        )

        results = []
        label_counters = defaultdict(int)

        for _, (label, label_results) in enumerate(zip(labels, all_results)):
            masks = label_results.get("masks", [])
            boxes = label_results.get("boxes", [])
            scores = label_results.get("scores", [])

            for _, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                mask_np = (
                    mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
                )
                box_list = box.cpu().tolist() if torch.is_tensor(box) else list(box)
                score_val = score.item() if torch.is_tensor(score) else float(score)

                bbox = [int(coord) for coord in box_list[:4]]

                obj_idx = label_counters[label]
                object_id = f"{label}_{obj_idx}"
                label_counters[label] += 1

                results.append(
                    {
                        "object_id": object_id,
                        "label": label,
                        "bbox": bbox,  # XYXY format, pixels
                        "mask": mask_np.astype(bool),
                        "score": score_val,
                    }
                )

        results = self._filter_duplicate_detections(results, iou_threshold)

        return results

    @staticmethod
    def _filter_duplicate_detections(
        results: List[Dict[str, Any]], iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter out duplicate detections using IoU-based NMS.
        Removes overlapping detections regardless of label, keeping the highest confidence one.

        Args:
            results: List of detection results
            iou_threshold: IoU threshold above which detections are considered duplicates

        Returns:
            Filtered list of detection results
        """
        if not results:
            return results

        boxes = torch.tensor([r["bbox"] for r in results], dtype=torch.float)
        scores = torch.tensor([r["score"] for r in results], dtype=torch.float)

        keep_indices = nms(boxes, scores, iou_threshold=iou_threshold)

        return [results[i] for i in keep_indices]
