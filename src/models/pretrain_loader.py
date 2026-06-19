"""Pretrain model loader for visual primitive token initialization.

Uses 4-bit QLoRA for base model to fit 15GB system RAM.
Trains embed_tokens, the LM head, and the last two decoder layers so that
Stage 1 learns not only the embeddings but also the conditional pattern of
emitting visual primitives inside the thinking chain.
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
    num_trainable_layers: int = 2,
    unfreeze_vit_layers: int = 0,
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor, int]:
    """Load Qwen3-VL in 4-bit for lightweight format pretrain.

    Steps:
    1. Load base model with 4-bit quantization (~2GB RAM)
    2. Add special tokens to tokenizer
    3. resize_token_embeddings (embedding stays fp16, bnb skips it)
    4. Freeze all parameters except embed_tokens, LM head, and the last
       ``num_trainable_layers`` decoder layers.
    5. Optionally unfreeze last ``unfreeze_vit_layers`` ViT blocks + merger.

    Memory: ~4-5GB total (safe for 15GB system RAM + 24GB VRAM)

    Returns:
        model: 4-bit Qwen3VL with embedding/LM head + top decoder layers trainable
        processor: Tokenizer with special tokens added
        old_vocab_size: Base vocab size before resize (151936)
    """
    logger.info(f"Loading model: {model_name} (4-bit, embedding + top-{num_trainable_layers} layers trainable)")

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

    # 2. Record original tokenizer length BEFORE adding special tokens
    old_vocab_size = len(processor.tokenizer)

    # 3. Add special tokens
    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    new_tokenizer_len = len(processor.tokenizer)
    logger.info(
        f"Added {num_added} special tokens (tokenizer: {old_vocab_size} → {new_tokenizer_len})"
    )

    # 4. Resize embeddings ONLY if tokenizer grew beyond current embedding size.
    #    NEVER shrink — the model may have more embeddings than tokenizer tokens.
    current_embed_size = model.get_input_embeddings().num_embeddings
    if new_tokenizer_len > current_embed_size:
        model.resize_token_embeddings(new_tokenizer_len)
        logger.info(f"Resized embeddings: {current_embed_size} → {new_tokenizer_len}")
    else:
        logger.info(
            f"No resize needed: embedding ({current_embed_size}) already covers "
            f"tokenizer ({new_tokenizer_len})"
        )

    # 5. Confirm weight tying
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()
    logger.info(
        f"embed_tokens & lm_head {'TIED' if tied else 'NOT tied'}"
    )

    # 6. Freeze ALL parameters, then unfreeze embed_tokens + LM head + the
    #    last two decoder layers. Training a few top layers lets the model
    #    learn the conditional pattern of emitting visual primitives inside
    #    the thinking chain, not just the token embeddings in isolation.
    #    This aligns Stage 1 with the paper's pretraining objective
    #    (foundational visual primitive generation) under single-GPU constraints.
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

    # Unfreeze the last N language model decoder layers.
    # Qwen3-VL model structure: model.model.layers[i]
    decoder_layers = getattr(model.model, "layers", None)
    if decoder_layers is not None and len(decoder_layers) >= num_trainable_layers:
        for layer in decoder_layers[-num_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(
            f"  + last {num_trainable_layers} decoder layers "
            f"(layers {len(decoder_layers)-num_trainable_layers}-{len(decoder_layers)-1})"
        )
    else:
        logger.warning("Could not locate decoder layers; only embedding/LM head will be trained.")

    # 7. Optionally unfreeze the last N ViT blocks + merger (projection).
    #    ViT layers learn with very low LR (e.g. 1e-6) while LLM layers use normal LR.
    #    This experimentally tests whether fine-grained visual features improve
    #    coordinate precision (paper-style approach).
    if unfreeze_vit_layers > 0:
        visual = getattr(model, "visual", None)
        if visual is not None:
            blocks = getattr(visual, "blocks", None)
            if blocks is not None and len(blocks) >= unfreeze_vit_layers:
                for block in blocks[-unfreeze_vit_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
                n_params = sum(
                    p.numel() for b in blocks[-unfreeze_vit_layers:]
                    for p in b.parameters()
                )
                logger.info(
                    f"  + last {unfreeze_vit_layers} ViT blocks "
                    f"(blocks {len(blocks)-unfreeze_vit_layers}-{len(blocks)-1}, "
                    f"{n_params:,} params)"
                )
            merger = getattr(visual, "merger", None)
            if merger is not None:
                for param in merger.parameters():
                    param.requires_grad = True
                n_params = sum(p.numel() for p in merger.parameters())
                logger.info(f"  + ViT merger (vision→language projection, {n_params:,} params)")

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
    Handles both normal (expanded) and shrunk embedding states.
    """
    logger.info(f"Injecting pretrained embeddings from {pretrain_path}...")

    pretrain_state = torch.load(pretrain_path, map_location="cpu", weights_only=True)
    pretrained_embed = pretrain_state["embed_tokens.weight"]
    saved_old_vocab = pretrain_state.get("old_vocab_size", old_vocab_size)

    current_embed = model.get_input_embeddings()
    current_weight = current_embed.weight.data

    embed_shape = pretrained_embed.shape[0]

    # Detect shrunk embedding state (old bug where resize shrank instead of expanded)
    if saved_old_vocab >= embed_shape:
        effective_old = embed_shape - len(SPECIAL_TOKENS)
        logger.warning(
            f"Shrunk embedding detected (saved_old={saved_old_vocab} > shape={embed_shape}). "
            f"Assuming {len(SPECIAL_TOKENS)} new tokens at end. effective_old={effective_old}"
        )
    else:
        effective_old = saved_old_vocab

    num_new_tokens = embed_shape - effective_old
    logger.info(
        f"  effective_old={effective_old}, embed_shape={embed_shape}, "
        f"num_new_tokens={num_new_tokens}"
    )

    # bnb never quantizes embedding — should be fp16/bf16, not uint8
    if current_weight.dtype in (torch.uint8, torch.int8):
        logger.warning("Embedding is quantized. Attempting dequantized injection...")
        pass

    # Direct indexing (embedding is fp16 in bnb 4-bit models)
    new_part = pretrained_embed[effective_old:].to(
        dtype=current_weight.dtype,
        device=current_weight.device,
    )
    with torch.no_grad():
        current_weight[effective_old:effective_old + new_part.shape[0]] = new_part
    logger.info(
        f"  Injected {new_part.shape[0]} new token rows. "
        f"Old tokens preserved ({effective_old} rows untouched)."
    )

    if "lm_head.weight" in pretrain_state:
        lm_head = model.get_output_embeddings()
        lm_weight = lm_head.weight.data
        if lm_weight.dtype not in (torch.uint8, torch.int8):
            pretrained_lm = pretrain_state["lm_head.weight"]
            new_lm = pretrained_lm[effective_old:effective_old + new_part.shape[0]].to(
                dtype=lm_weight.dtype, device=lm_weight.device
            )
            with torch.no_grad():
                lm_weight[effective_old:effective_old + new_part.shape[0]] = new_lm
            logger.info("  Injected lm_head new token rows (not tied).")

    logger.info("Pretrained embedding injection complete.")
