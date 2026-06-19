"""Batched model.generate helper for reward / filtering code paths.

The main use case is difficulty filtering in Stage 4/5, where we need many
on-policy rollouts. This helper keeps the generation logic in one place and
falls back to single-sample generation if batched generation fails.
"""

import logging
from typing import Dict, List, Tuple

import torch

from ..data.datasets.image_loader import load_image
from .conversation_builder import ConversationBuilder


logger = logging.getLogger(__name__)

# Shared builder for batched / single generation (GRPO-style system message).
_conv_builder = ConversationBuilder("grpo")


def batch_generate_completions(
    model,
    processor,
    samples: List[Dict],
    num_generations: int,
    max_completion_length: int,
    temperature: float = 0.7,
) -> Tuple[torch.Tensor, int]:
    """Generate completions for a batch of samples.

    Args:
        model: The policy model.
        processor: The processor/tokenizer.
        samples: List of data samples.
        num_generations: Number of rollouts per prompt.
        max_completion_length: Max new tokens per rollout.
        temperature: Sampling temperature.

    Returns:
        (outputs, input_len) where outputs is a tensor of shape
        (len(samples) * num_generations, total_seq_len).

    Raises:
        RuntimeError: If batched generation fails and should be retried as singles.
    """
    texts = []
    images = []
    for sample in samples:
        image = load_image(sample.get("image"))
        messages = _conv_builder.build_prompt(sample["prompt"], image)
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(prompt_text)
        images.append(image)

    all_have_images = all(img is not None for img in images)
    all_text_only = all(img is None for img in images)

    if all_have_images:
        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
    elif all_text_only:
        inputs = processor(text=texts, return_tensors="pt", padding=True)
    else:
        raise RuntimeError("Mixed image/text batch; caller should fall back to singles.")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Repeat each prompt num_generations times.
    g = num_generations
    repeated_inputs = {}
    for k, v in inputs.items():
        if k in ("pixel_values", "image_grid_thw", "image_rotary_emb"):
            repeated_inputs[k] = v.repeat_interleave(g, dim=0)
        else:
            repeated_inputs[k] = v.repeat_interleave(g, dim=0)

    with torch.inference_mode():
        outputs = model.generate(
            **repeated_inputs,
            max_new_tokens=max_completion_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = repeated_inputs["input_ids"].shape[1]
    return outputs, input_len


def generate_single_completion(
    model,
    processor,
    sample: Dict,
    max_completion_length: int,
    temperature: float = 0.7,
) -> Tuple[torch.Tensor, int]:
    """Generate one completion for a single sample."""
    image = load_image(sample.get("image"))
    messages = _conv_builder.build_prompt(sample["prompt"], image)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text],
        images=[image] if image is not None else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return outputs, input_len
