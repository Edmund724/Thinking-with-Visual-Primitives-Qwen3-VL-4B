"""QLoRA model loader for Qwen3-VL-4B-Thinking with Blackwell/5090D compatibility."""

import logging
import os
from pathlib import Path
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from ..utils.constants import (
    BASE_MODEL_NAME,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    SPECIAL_TOKENS,
)

logger = logging.getLogger(__name__)


def _is_adapter_checkpoint(path: str) -> bool:
    """Check if path contains a saved PEFT adapter."""
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_config.json"))
        or os.path.exists(os.path.join(path, "adapter_model.safetensors"))
    )


def _load_base_model_and_processor(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    attn_impl: str,
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """Load base model + processor from local path or Hub."""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return model, processor


def load_qlora_model(
    model_name: str = BASE_MODEL_NAME,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: list | None = None,
    use_gradient_checkpointing: bool = True,
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen3-VL with QLoRA (4-bit NF4, double quantization).

    Handles two scenarios:
      1. model_name is a base model path (no adapter) -> add new LoRA adapter.
      2. model_name is a saved adapter checkpoint -> load existing adapter.

    Correct order for 4-bit + token resize:
        1. Load base model
        2. Add special tokens to tokenizer
        3. resize_token_embeddings (before prepare_model_for_kbit_training!)
        4. prepare_model_for_kbit_training
        5. add LoRA or load existing adapter
    """
    logger.info(f"Loading model: {model_name}")

    # Detect if model_name itself is an adapter checkpoint
    is_adapter = _is_adapter_checkpoint(model_name)
    base_model_path = model_name
    if is_adapter:
        # Read base model name from adapter config
        import json
        adapter_config_path = os.path.join(model_name, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path", BASE_MODEL_NAME)
        logger.info(
            f"Detected adapter checkpoint. Base model: {base_model_path}"
        )

    # Blackwell/5090D: try flash-attn, fallback to eager
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Using flash_attention_2")
    except ImportError:
        attn_impl = "eager"
        logger.warning("flash-attn not available, falling back to eager attention")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model and processor
    model, processor = _load_base_model_and_processor(
        base_model_path, bnb_config, attn_impl
    )

    # If loading from adapter checkpoint, also load its tokenizer
    # (which has the added special tokens)
    if is_adapter and os.path.exists(os.path.join(model_name, "tokenizer_config.json")):
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info(f"Loaded tokenizer from adapter checkpoint: {model_name}")

    # Add special tokens for visual primitives
    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")

    # CRITICAL: resize embeddings BEFORE prepare_model_for_kbit_training
    model.resize_token_embeddings(len(processor.tokenizer))
    logger.info(f"Resized embeddings to {len(processor.tokenizer)}")

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Prepare for 4-bit training
    model = prepare_model_for_kbit_training(model)

    if is_adapter:
        # Scenario 2: Load existing adapter
        logger.info(f"Loading existing adapter from {model_name}")
        model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
        logger.info(f"Adapter loaded. Resuming training from {model_name}")
    else:
        # Scenario 1: Add new LoRA adapter
        if target_modules is None:
            actual_modules = [n for n, _ in model.named_modules()]
            target_modules = [
                m for m in DEFAULT_LORA_TARGET_MODULES
                if any(m in n for n in actual_modules)
            ]
            logger.info(f"Auto-detected LoRA target modules: {target_modules}")

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model, processor


def load_reference_model(
    model_path: str,
    adapter_path: str | None = None,
) -> Qwen3VLForConditionalGeneration:
    """Load a reference model for DPO (4-bit, frozen, GPU resident).

    Args:
        model_path: Base model name or local path.
        adapter_path: Path to LoRA adapter to load (optional).
            If None and model_path itself is an adapter checkpoint,
            loads the adapter from model_path.

    Returns:
        Frozen reference model on GPU.
    """
    logger.info(f"Loading reference model from {model_path}")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Auto-detect adapter checkpoint
    is_adapter = _is_adapter_checkpoint(model_path)
    base_model_path = model_path
    if is_adapter:
        import json
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get(
            "base_model_name_or_path", BASE_MODEL_NAME
        )
        logger.info(
            f"Reference: detected adapter. Base: {base_model_path}"
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )

    # Determine which adapter to load and which tokenizer to use
    adapter_to_load = adapter_path
    if adapter_to_load is None and is_adapter:
        adapter_to_load = model_path

    # Load tokenizer from the same source as the adapter (for special tokens)
    tokenizer_source = adapter_to_load if adapter_to_load else base_model_path
    processor = AutoProcessor.from_pretrained(tokenizer_source, trust_remote_code=True)

    # CRITICAL: add special tokens and resize embeddings to match policy model
    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Reference: added {num_added} special tokens, resizing embeddings...")
    model.resize_token_embeddings(len(processor.tokenizer))

    if adapter_to_load is not None:
        model = PeftModel.from_pretrained(model, adapter_to_load)
        logger.info(f"Reference: loaded adapter: {adapter_to_load}")

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model
