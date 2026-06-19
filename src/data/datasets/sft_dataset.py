"""SFT Dataset with proper assistant-only loss masking for Qwen3-VL.

Uses reasoning_content / content split for Qwen3-Thinking chat template.
Masks out system/user prompt tokens; only assistant response tokens compute loss.
"""

import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from ...utils.conversation_builder import ConversationBuilder
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
        format_token_weight: float = 5.0,
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.format_token_weight = format_token_weight
        self._format_token_ids = self._build_format_token_ids()
        self._conv_builder = ConversationBuilder("sft")

    def _build_format_token_ids(self) -> set:
        """Token IDs that should receive higher loss weight."""
        from ...utils.constants import BOX_CLOSE, BOX_OPEN, POINT_CLOSE, POINT_OPEN, REF_CLOSE, REF_OPEN

        tokens = [
            BOX_OPEN,
            BOX_CLOSE,
            POINT_OPEN,
            POINT_CLOSE,
            REF_OPEN,
            REF_CLOSE,
            "<think>",
            "</think>",
        ]
        ids = set()
        for tok in tokens:
            try:
                ids.add(self.processor.tokenizer.convert_tokens_to_ids(tok))
            except Exception:
                continue
        # Remove possible UNK id
        unk_id = getattr(self.processor.tokenizer, "unk_token_id", None)
        if unk_id is not None:
            ids.discard(unk_id)
        return ids

    def __len__(self):
        return len(self.data)

    def _build_messages(self, sample: Dict[str, Any]) -> tuple:
        """Build prompt-only and full conversation messages.

        Delegates to ConversationBuilder for unified message construction.
        Returns:
            (prompt_messages, full_messages)
        """
        return self._conv_builder.build_sft(sample)

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
        # NOTE: must include image to get correct image token positions in prompt_len
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

        # Process full text with images (no padding; collator will pad per-batch)
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
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            inputs = self.processor(
                text=[full_text],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
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

        # Per-token loss weights: up-weight format tokens (<|box|>, </think>, etc.)
        # inside the assistant output so the model learns the syntax faster.
        loss_weight = torch.ones_like(labels, dtype=torch.float)
        if self._format_token_ids:
            fmt_ids_tensor = torch.tensor(
                list(self._format_token_ids),
                dtype=inputs["input_ids"].dtype,
                device=inputs["input_ids"].device,
            )
            is_format = (inputs["input_ids"].unsqueeze(1) == fmt_ids_tensor).any(dim=1)
            is_assistant = labels != -100
            loss_weight[is_assistant & is_format] = self.format_token_weight
        # Prompt and padding positions should contribute zero to the weighted sum.
        loss_weight[labels == -100] = 0.0

        inputs["labels"] = labels
        inputs["loss_weight"] = loss_weight
        return inputs
