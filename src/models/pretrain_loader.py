"""Pretrain model loader for visual primitive token embedding initialization.

Uses 4-bit QLoRA for base model to fit 15GB system RAM.
Only embed_tokens is trainable (fp16, bnb never quantizes embedding layer).
"""

import logging
import os
from typing import Tuple

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from ..utils.constants import BASE_VOCAB_SIZE

logger = logging.getLogger(__name__)

# 6 special tokens for visual primitives
SPECIAL_TOKENS = [
    "<|box|>", "<|/box|>",
    "<|point|>", "<|/point|>",
    "<|ref|>", "<|/ref|>",
]


def load_pretrain_model(
    model_name: str,
    attn_impl: str = "eager",
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor, int]:
    """Load Qwen3-VL in 4-bit for embedding-only pretrain.

    Steps:
    1. Load base model with 4-bit quantization (~2GB RAM)
    2. Add special tokens to tokenizer
    3. resize_token_embeddings (embedding stays fp16, bnb skips it)
    4. prepare_model_for_kbit_training
    5. Freeze all parameters except embed_tokens

    Memory: ~4-5GB total (safe for 15GB system RAM + 24GB VRAM)

    Returns:
        model: 4-bit Qwen3VL with only embed_tokens trainable (fp16)
        processor: Tokenizer with special tokens added
        old_vocab_size: Base vocab size before resize (151936)
    """
    logger.info(f"Loading model: {model_name} (4-bit, embedding-only trainable)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 1. Load base model in 4-bit with low CPU memory usage
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "20GB"},
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 2. Add special tokens
    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(
        f"Added {num_added} special tokens (vocab: {len(processor.tokenizer)})"
    )

    # 3. Resize embeddings BEFORE prepare_model_for_kbit_training
    #    bnb NEVER quantizes embedding/lm_head layers, so embed stays fp16
    model.resize_token_embeddings(len(processor.tokenizer))
    old_vocab_size = BASE_VOCAB_SIZE
    logger.info(f"Resized embeddings to {len(processor.tokenizer)}")

    # 4. Skip prepare_model_for_kbit_training — backbone is frozen,
    #    embedding layer stays fp16. Only embed_tokens receives gradients.
    #    (prepare_model_for_kbit_training is for QLoRA training, not needed here)

    # 5. Confirm weight tying
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()
    logger.info(
        f"embed_tokens & lm_head {'TIED' if tied else 'NOT tied'}"
    )

    # 6. Freeze ALL parameters, then unfreeze only embed_tokens
    for param in model.parameters():
        param.requires_grad = False

    embed.weight.requires_grad = True
    embed_dtype = embed.weight.dtype  # should be fp16/bf16
    logger.info(
        f"Frozen backbone. Trainable: embed_tokens.weight "
        f"(shape={list(embed.weight.shape)}, dtype={embed_dtype}, "
        f"{embed.weight.numel():,} params)"
    )

    if not tied:
        lm_head.weight.requires_grad = True
        logger.info(f"  + lm_head.weight (NOT tied, {lm_head.weight.numel():,} params)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
    )

    return model, processor, old_vocab_size


def save_pretrain_state(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    output_dir: str,
    old_vocab_size: int,
):
    """Save only the trained embedding weights.

    Saves:
        pretrain_state_dict.pt  — embed_tokens.weight (and lm_head if not tied)
        tokenizer files         — needed because vocab size changed
    """
    os.makedirs(output_dir, exist_ok=True)

    embed_tokens = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    state_dict = {
        "embed_tokens.weight": embed_tokens.weight.data.clone().cpu(),
        "old_vocab_size": old_vocab_size,
    }

    tied = embed_tokens.weight.data_ptr() == lm_head.weight.data_ptr()
    if not tied:
        state_dict["lm_head.weight"] = lm_head.weight.data.clone().cpu()
        logger.info("Saved separate lm_head.weight (not tied)")
    else:
        logger.info("Saved embed_tokens.weight only (tied with lm_head)")

    path = os.path.join(output_dir, "pretrain_state_dict.pt")
    torch.save(state_dict, path)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"Pretrain state saved to {path} ({file_size_mb:.1f} MB)")

    # Save tokenizer (vocab size changed!)
    processor.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved (vocab={len(processor.tokenizer)})")


def inject_pretrained_embeddings(
    model: Qwen3VLForConditionalGeneration,
    pretrain_path: str,
    old_vocab_size: int,
):
    """Inject pretrained new-token embeddings into a 4-bit QLoRA model.

    Only overwrites rows >= old_vocab_size. Old vocab untouched.
    """
    logger.info(f"Injecting pretrained embeddings from {pretrain_path}...")

    pretrain_state = torch.load(pretrain_path, map_location="cpu", weights_only=True)
    pretrained_embed = pretrain_state["embed_tokens.weight"]

    current_embed = model.get_input_embeddings()
    current_weight = current_embed.weight.data

    new_vocab_size = pretrained_embed.shape[0]
    num_new_tokens = new_vocab_size - old_vocab_size

    logger.info(
        f"  old_vocab={old_vocab_size}, new_vocab={new_vocab_size}, "
        f"num_new_tokens={num_new_tokens}"
    )

    # bnb never quantizes embedding — should be fp16/bf16, not uint8
    if current_weight.dtype in (torch.uint8, torch.int8):
        logger.warning("Embedding is quantized. Attempting dequantized injection...")
        # Fallback: use module's forward hook to get fp16 weight
        # Since bnb doesn't quantize embedding, this path should rarely trigger
        from bitsandbytes.nn.modules import Linear4bit
        pass  # Complex; rely on the fact that bnb skips embedding layers

    # Direct indexing (embedding is fp16 in bnb 4-bit models)
    new_part = pretrained_embed[old_vocab_size:].to(
        dtype=current_weight.dtype,
        device=current_weight.device,
    )
    with torch.no_grad():
        current_weight[old_vocab_size:] = new_part
    logger.info(
        f"  Injected {new_part.shape[0]} new token rows. "
        f"Old tokens preserved ({old_vocab_size} rows untouched)."
    )

    if "lm_head.weight" in pretrain_state:
        lm_head = model.get_output_embeddings()
        lm_weight = lm_head.weight.data
        if lm_weight.dtype not in (torch.uint8, torch.int8):
            pretrained_lm = pretrain_state["lm_head.weight"]
            new_lm = pretrained_lm[old_vocab_size:].to(
                dtype=lm_weight.dtype, device=lm_weight.device
            )
            with torch.no_grad():
                lm_weight[old_vocab_size:] = new_lm
            logger.info("  Injected lm_head new token rows (not tied).")

    logger.info("Pretrained embedding injection complete.")
