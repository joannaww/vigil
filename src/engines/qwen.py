import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from typing import List, Optional
from PIL import Image
from vllm import LLM, SamplingParams

import logging

from configs.config_schema import QwenEngineConfig

logger = logging.getLogger(__name__)


class QwenEngine:
    """
    Singleton-style wrapper to hold the heavy model in memory once.
    """

    def __init__(
        self,
        config: QwenEngineConfig,
    ):
        self.config = config
        self.dtype = config.dtype

        logger.info(f"Loading {config.model_name} into VRAM...")
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.processor.tokenizer.padding_side = "left"
        self.llm = LLM(model=self.config.model_name, **config.vllm.dict())

        logger.info("Model loaded successfully with vLLM.")

    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
    ) -> List[str]:
        """
        Generic generation method handling batching and formatting.
        """

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )
        inputs = []

        for i, prompt in enumerate(prompts):
            current_images = []
            content = []

            if images and i < len(images) and images[i] is not None:
                img_input = images[i]
                if isinstance(img_input, list):
                    current_images.extend(img_input)
                    # Add placeholder dicts for the processor
                    for img in img_input:
                        content.append({"type": "image", "image": img})
                else:
                    current_images.append(img_input)
                    content.append({"type": "image", "image": img_input})

            content.append({"type": "text", "text": prompt})

            message = [{"role": "user", "content": content}]
            prompt_text = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )

            if current_images:
                inputs.append(
                    {
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": current_images},
                    }
                )
            else:
                inputs.append(
                    {
                        "prompt": prompt_text,
                    }
                )

        outputs = self.llm.generate(
            inputs, sampling_params=sampling_params, use_tqdm=False
        )

        generated_texts = [output.outputs[0].text for output in outputs]

        return generated_texts
