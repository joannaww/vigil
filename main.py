#!/usr/bin/env python3

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


import argparse
import json
import time
import logging
import yaml
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

from transformers.utils import logging as hf_logging

from pipelines.segment_pipeline import SegmentationPipeline
from engines.qwen import QwenEngine
from tasks.background_evaluator import (
    BackgroundDescriptionEvaluator,
    process_background_evaluation,
)
from tasks.paired_object_evaluator import (
    PairedObjectEvaluator,
    process_object_consistency,
)
from tasks.physical_evaluator import (
    PhysicalEvaluator,
    process_physical_evaluation,
)
from tasks.position_logic_evaluator import (
    PositionLogicEvaluator,
    process_position_logic_evaluation,
)
from configs.config_schema import PipelineConfig


hf_logging.disable_progress_bar()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def apply_sharding(
    all_items: List[Tuple[Path, str]], shard_id: int, num_shards: int
) -> List[Tuple[Path, str]]:
    """
    Slices the (root, ID) list for distributed processing.
    """
    if num_shards <= 1:
        return all_items

    total = len(all_items)
    chunk_size = total // num_shards

    start_idx = shard_id * chunk_size
    end_idx = total if shard_id == num_shards - 1 else start_idx + chunk_size

    my_items = all_items[start_idx:end_idx]

    logger.info(f"--- SHARDING INFO ---")
    logger.info(f"Job: {shard_id + 1}/{num_shards}")
    logger.info(f"Processing range: [{start_idx}:{end_idx}] (Count: {len(my_items)})")
    logger.info(f"---------------------")

    return my_items


def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    """Loads YAML and validates it against Pydantic schema."""
    with open(yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if "data_root" in raw_config:
        if isinstance(raw_config["data_root"], str):
            raw_config["data_root"] = [raw_config["data_root"]]

    return PipelineConfig(**raw_config)


def resolve_data_ids(config: PipelineConfig) -> List[Tuple[Path, str]]:
    """
    Parses data_ids to return specific (Root, ID) tuples.
    """
    found_items = []

    if isinstance(config.data_ids, dict):
        for root_str, ids in config.data_ids.items():
            root = Path(root_str)
            if not root.exists():
                logger.warning(f"Root not found: {root}")
                continue

            for d_id in ids:
                if (root / d_id).exists():
                    found_items.append((root, d_id))
                else:
                    logger.warning(f"ID {d_id} not found in {root}")

        return found_items

    data_roots = (
        config.data_root if isinstance(config.data_root, list) else [config.data_root]
    )

    if str(config.data_ids).lower() == "all":
        found_items = []

        for root_str in data_roots:
            root = Path(root_str)
            if not root.exists():
                logger.warning(f"Data root not found: {root}")
                continue

            logger.info(f"Auto-discovering data in {root}...")

            root_ids = []
            for folder in root.iterdir():
                if folder.is_dir() and folder.name.isdigit():
                    root_ids.append(folder.name)

            root_ids.sort()

            for r_id in root_ids:
                found_items.append((root, r_id))

            if not root_ids:
                logger.warning(f"No numeric data directories found in {root}")
            else:
                logger.info(f"Found {len(root_ids)} items in {root.name}")

        logger.info(f"Total auto-discovered datapoints: {len(found_items)}")
        return found_items

    return found_items


def discover_datapoint(data_root: Path, dp_id: str) -> Dict:
    """
    Finds prompt and images.
    """
    dp_dir = data_root / dp_id
    if not dp_dir.exists():
        return None

    try:
        category_prefix = data_root.parent.name
        unique_id = f"{category_prefix}_{dp_id}"

        prompt_files = list(dp_dir.glob("*prompt.txt"))
        if not prompt_files:
            return None

        with open(prompt_files[0], "r") as f:
            prompt = f.read().strip()

        files = list(dp_dir.glob("*.png")) + list(dp_dir.glob("*.jpg"))

        obj_pat = re.compile(r".*object.*", re.IGNORECASE)
        gen_pat = re.compile(r".*generated.*", re.IGNORECASE)
        bg_pat = re.compile(r".*background.*", re.IGNORECASE)

        object_images = [f for f in files if obj_pat.match(f.name)]
        generated_candidates = [f for f in files if gen_pat.match(f.name)]
        bg_candidates = [f for f in files if bg_pat.match(f.name)]

        if not generated_candidates or not bg_candidates:
            return None

        return {
            "id": unique_id,  # because there are duplicates across categories
            "original_id": dp_id,
            "source_root": str(data_root),
            "prompt": prompt,
            "object_images": object_images,
            "background_image": bg_candidates[0],
            "generated_image": generated_candidates[0],
        }

    except Exception as e:
        logger.error(f"Error reading {dp_id}: {e}")
        return None

    except Exception as e:
        logger.error(f"Error reading {dp_id}: {e}")
        return None


def clean_output(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Recursively removes technical fields (masks, scores, tensors)
    from the output dictionary to make it human-readable.
    """
    to_remove = {
        "mask",
        "score",
        "bbox",
        "embedding_tensor",
        "image_dimensions",
    }
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k in to_remove:
                continue
            new_dict[k] = clean_output(v)
        return new_dict

    elif isinstance(data, list):
        return [clean_output(item) for item in data]

    else:
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--shard-id", type=int, default=None, help="Override config shard_id"
    )
    parser.add_argument(
        "--num-shards", type=int, default=None, help="Override config num_shards"
    )

    args = parser.parse_args()

    try:
        cfg = load_pipeline_config(args.config)

        logger.info(f"Loaded configuration from {args.config}")
        if args.shard_id is not None:
            cfg.shard_id = args.shard_id
        if args.num_shards is not None:
            cfg.num_shards = args.num_shards
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        return 1

    import random
    import numpy as np
    import torch

    seed = cfg.qwen.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeded Python/numpy/torch with seed={seed} (cfg.qwen.seed)")

    all_target_items = resolve_data_ids(cfg)

    my_target_items = apply_sharding(all_target_items, cfg.shard_id, cfg.num_shards)

    if not my_target_items:
        logger.warning(f"Shard {cfg.shard_id} has no items. Exiting.")
        return 0

    datapoints = []

    for root_path, dp_id in my_target_items:
        dp = discover_datapoint(root_path, dp_id)
        if dp:
            datapoints.append(dp)

    logger.info(f"Ready to process {len(datapoints)} datapoints.")

    if not datapoints:
        logger.error("No valid datapoints found. Exiting.")
        return 1

    global_start_time = time.time()

    engine = QwenEngine(config=cfg.qwen)

    step_start = time.time()
    logger.info("Starting Steps 1 & 2: Segmentation...")

    pipeline = SegmentationPipeline(
        qwen_engine=engine,
        config=cfg.segmentation,
        entity_extractor_config=cfg.entity,
    )

    outputs = pipeline.run_batch(
        datapoints=datapoints,
        pair_objects=True,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_part_{cfg.shard_id}" if cfg.num_shards > 1 else ""

    seg_duration = time.time() - step_start
    step_start = time.time()

    logger.info("Starting Step 3: Background Consistency...")

    bg_vlm_eval = BackgroundDescriptionEvaluator(engine=engine, config=cfg.background)

    bg_eval_outputs = process_background_evaluation(
        input_data=outputs,
        qwen_evaluator=bg_vlm_eval,
        output_path=output_dir / f"background_evaluation_results{suffix}.json",
    )

    bg_duration = time.time() - step_start
    step_start = time.time()

    logger.info("Starting Step 4: Object Consistency...")

    obj_eval = PairedObjectEvaluator(engine=engine, config=cfg.object_eval)

    obj_results = process_object_consistency(
        input_data=bg_eval_outputs,
        evaluator=obj_eval,
        output_path=output_dir / f"object_consistency_results{suffix}.json",
    )

    obj_duration = time.time() - step_start
    step_start = time.time()

    logger.info("Starting Step 5: Physical Realism...")

    phys_eval = PhysicalEvaluator(engine=engine, config=cfg.physical)

    phys_results = process_physical_evaluation(
        input_data=obj_results,
        evaluator=phys_eval,
        output_path=output_dir / f"physical_evaluation_results{suffix}.json",
    )

    phys_duration = time.time() - step_start
    step_start = time.time()

    logger.info("Starting Step 6: Position Logic...")

    pos_eval = PositionLogicEvaluator(engine=engine, config=cfg.position_logic)

    final_results = process_position_logic_evaluation(
        input_data=phys_results,
        evaluator=pos_eval,
        output_path=output_dir / f"position_logic_evaluation_results{suffix}.json",
    )

    pos_duration = time.time() - step_start
    total_duration = time.time() - global_start_time

    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info(f"• Segmentation:    {seg_duration:.2f}s")
    logger.info(f"• Background:      {bg_duration:.2f}s")
    logger.info(f"• Object Eval:     {obj_duration:.2f}s")
    logger.info(f"• Physical:        {phys_duration:.2f}s")
    logger.info(f"• Position Logic:  {pos_duration:.2f}s")
    logger.info(f"• Total Time:      {total_duration:.2f}s")
    logger.info("=" * 60)

    logger.info("Cleaning output for final report...")

    human_readable_data = clean_output(final_results)

    final_output_path = output_dir / f"final_results{suffix}.json"

    with open(final_output_path, "w") as f:
        json.dump(human_readable_data, f, indent=4)

    logger.info(f"✓ Final report saved to: {final_output_path}")


if __name__ == "__main__":
    main()
