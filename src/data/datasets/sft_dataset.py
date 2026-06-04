"""SFT Dataset with proper assistant-only loss masking for Qwen3-VL.

Uses reasoning_content / content split for Qwen3-Thinking chat template.
Masks out system/user prompt tokens; only assistant response tokens compute loss.
"""

import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from .image_loader import load_image

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning of visual primitive outputs.

    Each sample is expected to have:
        image: PIL.Image
        prompt: str (user text)
        reasoning: str (thinking content with visual primitives)
        answer: str (final answer, e.g. "2" or r"\\boxed{True}")
        task_type: str (optional)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        processor: AutoProcessor,
        max_length: int = 2048,
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _build_messages(self, sample: Dict[str, Any]) -> tuple:
        """Build prompt-only and full conversation messages.

        Handles both image-based and text-only samples.
        Returns:
            (prompt_messages, full_messages)
        """
        has_image = "image" in sample and sample["image"] is not None

        if has_image:
            user_content = [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["prompt"]},
            ]
        else:
            user_content = sample["prompt"]

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful visual reasoning assistant. "
                    "Think step by step and use visual primitives when needed."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": f"The answer is {sample['answer']}.",
                "reasoning_content": sample["reasoning"],
            },
        ]

        return prompt_messages, full_messages

    def _compute_labels(
        self,
        full_input_ids: torch.Tensor,
        prompt_messages: list,
        full_messages: list,
        image,
    ) -> torch.Tensor:
        """Create labels that mask everything except assistant response.

        Strategy:
            1. Build prompt_text and full_text via apply_chat_template
            2. Process both with processor (no padding) to get actual token counts
            3. Since prompt_text is a strict prefix of full_text and both use
               the same image, prompt_len = assistant_start
            4. Mask positions [0, assistant_start) in full_input_ids
        """
        labels = full_input_ids.clone()

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # Safety check: prompt_text must be a prefix of full_text
        if not full_text.startswith(prompt_text):
            logger.warning(
                "prompt_text is not a prefix of full_text, using fallback masking"
            )
            # Fallback: keep last 25% of tokens unmasked
            mask_len = int(len(full_input_ids) * 0.75)
            labels[:mask_len] = -100
            return labels

        # Process prompt with same image (no padding) to get actual token count
        if image is not None:
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
        else:
            prompt_inputs = self.processor(
                text=[prompt_text],
                return_tensors="pt",
                padding=False,
            )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Full sequence length (after padding)
        full_len = full_input_ids.shape[0]

        # Sanity check
        if prompt_len >= full_len:
            logger.warning(
                f"prompt_len ({prompt_len}) >= full_len ({full_len}), "
                "using fallback masking"
            )
            mask_len = max(0, full_len - 10)
            labels[:mask_len] = -100
            return labels

        labels[:prompt_len] = -100
        return labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        image = load_image(sample["image"]) if "image" in sample and sample["image"] else None

        prompt_messages, full_messages = self._build_messages(sample)

        # Process full text with images (padded to max_length)
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if image is not None:
            inputs = self.processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        else:
            inputs = self.processor(
                text=[full_text],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels with assistant-only masking
        labels = self._compute_labels(
            inputs["input_ids"],
            prompt_messages,
            full_messages,
            image,
        )

        # Mask padding tokens
        if "attention_mask" in inputs:
            pad_mask = inputs["attention_mask"] == 0
            labels[pad_mask] = -100

        inputs["labels"] = labels
        return inputs
