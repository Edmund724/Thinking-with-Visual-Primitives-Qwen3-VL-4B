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

def _disable_gradient_checkpointing_on_frozen_modules(model: torch.nn.Module) -> None:
    """Turn off gradient checkpointing for modules whose parameters are all frozen.

    Qwen3-VL enables checkpointing on every submodule that exposes the flag,
    including the frozen vision blocks. Checkpointing frozen blocks only wastes
    memory and triggers PyTorch warnings (``None of the inputs have
    requires_grad=True``). This keeps checkpointing active on trainable layers
    (text decoder LoRA) and disables it elsewhere.
    """
    n_disabled = 0
    for module in model.modules():
        if not getattr(module, "gradient_checkpointing", False):
            continue
        has_trainable = any(p.requires_grad for p in module.parameters(recurse=True))
        if not has_trainable:
            module.gradient_checkpointing = False
            n_disabled += 1
    logger.debug(f"Disabled gradient checkpointing on {n_disabled} frozen module(s)")


# Modules whose full parameters must be trainable (not just LoRA) so that
# newly-added special-token embeddings (<|box|>, <|ref|>, etc.) are learned and
# saved with the adapter.  Without this, the embeddings/lm_head stay frozen at
# their random/base-model initialization and the model emits garbage non-Latin
# characters instead of the visual-primitive tokens.
_MODULES_TO_SAVE = ["model.language_model.embed_tokens", "lm_head"]

logger = logging.getLogger(__name__)


def _resolve_local_path(model_name: str) -> str:
    """Resolve relative paths to absolute to prevent HuggingFace Hub fallback.

    Strategy:
      - If path exists locally -> use absolute version.
      - If path contains '/' and starts with a common local dir prefix
        (outputs/, models/, data/, ./, ../, /) -> treat as local, error if missing.
      - Otherwise -> assume HuggingFace Hub ID, pass through.
    """
    if os.path.exists(model_name):
        return os.path.abspath(model_name)

    # Check if it looks like a local path (not a Hub ID like "org/model")
    local_prefixes = ("./", "../", "/", "outputs/", "models/", "data/", "checkpoints/")
    if any(model_name.startswith(p) for p in local_prefixes):
        resolved = os.path.abspath(model_name)
        raise FileNotFoundError(
            f"Local model path does not exist: {model_name} (resolved to {resolved}). "
            f"Please check the path or run the preceding pipeline stage first."
        )

    return model_name


def _is_adapter_checkpoint(path: str) -> bool:
    """Check if path contains a saved PEFT adapter."""
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_config.json"))
        or os.path.exists(os.path.join(path, "adapter_model.safetensors"))
    )


def _resolve_grpo_expert_path(path: str) -> str:
    """Resolve a GRPO stage output directory to its actual adapter checkpoint.

    GRPO saves the final adapter of each round under ``round_N/`` inside the
    stage output directory.  If *path* itself is already an adapter checkpoint
    (or a base model), return it unchanged.  Otherwise, look for the latest
    ``round_N`` subdirectory that contains an adapter and return that path.
    """
    if not os.path.isdir(path) or _is_adapter_checkpoint(path):
        return path

    rounds = []
    for name in os.listdir(path):
        if not name.startswith("round_"):
            continue
        try:
            round_num = int(name.split("_")[-1])
        except ValueError:
            continue
        round_path = os.path.join(path, name)
        if _is_adapter_checkpoint(round_path):
            rounds.append((round_num, round_path))

    if rounds:
        rounds.sort()
        return rounds[-1][1]
    return path


def _set_use_cache_deep(module: torch.nn.Module, use_cache: bool) -> None:
    """Recursively set ``use_cache`` on every config object in a module tree.

    Qwen3VL has deeply nested submodules; the ``@merge_with_config_defaults``
    decorator on ``Qwen3VLTextModel.forward`` reads ``self.config.use_cache``
    from the innermost ``Qwen3VLTextConfig``.  A top-level
    ``model.config.use_cache = ...`` does NOT propagate there, so we must reach
    every config object.
    """
    seen_configs: set[int] = set()

    def _walk(m: torch.nn.Module) -> None:
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg_id = id(cfg)
            if cfg_id not in seen_configs:
                seen_configs.add(cfg_id)
                cfg.use_cache = use_cache
        for child in m.children():
            _walk(child)

    _walk(module)
    logger.debug(f"Set use_cache={use_cache} on {len(seen_configs)} config object(s)")


def _get_use_cache_states(module: torch.nn.Module) -> dict[int, bool]:
    """Capture current ``use_cache`` values for every config object."""
    states: dict[int, bool] = {}

    def _walk(m: torch.nn.Module) -> None:
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            states[id(cfg)] = cfg.use_cache
        for child in m.children():
            _walk(child)

    _walk(module)
    return states


def _set_use_cache_states(module: torch.nn.Module, states: dict[int, bool]) -> None:
    """Restore ``use_cache`` values captured by ``_get_use_cache_states``."""

    def _walk(m: torch.nn.Module) -> None:
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg_id = id(cfg)
            if cfg_id in states:
                cfg.use_cache = states[cfg_id]
        for child in m.children():
            _walk(child)

    _walk(module)


def _patch_lm_head_dtype_cast(model):
    """Patch lm_head so fp32 activations are cast to the weight dtype.

    ``prepare_model_for_kbit_training`` casts layer norms to fp32 for training
    stability. During generation without autocast, hidden_states therefore stay
    fp32 all the way to ``lm_head``. Depending on the checkpoint / PEFT state,
    ``lm_head`` weights may be bfloat16 (or another dtype different from fp32),
    so the matrix multiply raises ``RuntimeError: expected mat1 and mat2 to
    have the same dtype``. This wrapper casts the input to the weight dtype
    only when they differ, fixing generation while leaving training (run under
    autocast) intact.
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return

    # Unwrap PEFT ModulesToSaveWrapper to reach the actual Linear module.
    inner = lm_head
    while hasattr(inner, "modules_to_save"):
        adapter = getattr(inner, "active_adapters", ["default"])[0]
        modules_to_save = inner.modules_to_save
        if isinstance(modules_to_save, dict):
            next_inner = modules_to_save.get(adapter)
        else:
            # nn.ModuleDict: use item accessor
            try:
                next_inner = modules_to_save[adapter]
            except (KeyError, IndexError, TypeError):
                next_inner = None
        if next_inner is None:
            return
        inner = next_inner

    if not isinstance(inner, torch.nn.Linear):
        return

    # Idempotent: do not stack wrappers if this model is re-patched.
    if getattr(inner.forward, "_tvp_dtype_cast_applied", False):
        return

    original_forward = inner.forward

    def forward_with_dtype_cast(input, *args, **kwargs):
        target_dtype = inner.weight.dtype
        if input.dtype != target_dtype:
            input = input.to(target_dtype)
        return original_forward(input, *args, **kwargs)

    forward_with_dtype_cast._tvp_dtype_cast_applied = True
    inner.forward = forward_with_dtype_cast


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
    unfreeze_vit_layers: int = 0,
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

    # Resolve relative paths to absolute to avoid HuggingFace Hub fallback
    model_name = _resolve_local_path(model_name)

    # GRPO stages save adapters under round_N/ inside the stage output dir.
    # If the user passes the stage output dir, resolve it to the actual adapter.
    model_name = _resolve_grpo_expert_path(model_name)

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
        base_model_path = _resolve_local_path(base_model_path)
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
    new_tokenizer_len = len(processor.tokenizer)
    logger.info(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")

    # Decoder-only models need left padding for batched generation.
    processor.tokenizer.padding_side = "left"

    # Explicitly align model config / generation_config with tokenizer token IDs
    # to suppress the auto-alignment warning from transformers.
    tokenizer = processor.tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # CRITICAL: resize embeddings BEFORE prepare_model_for_kbit_training
    # Only expand — never shrink. Model may have more embeddings than tokenizer tokens.
    current_embed_size = model.get_input_embeddings().num_embeddings
    if new_tokenizer_len > current_embed_size:
        model.resize_token_embeddings(new_tokenizer_len)
        logger.info(f"Resized embeddings: {current_embed_size} → {new_tokenizer_len}")
    else:
        logger.info(
            f"No resize needed: embedding ({current_embed_size}) covers tokenizer ({new_tokenizer_len})"
        )

    # (Step 4) Prepare for 4-bit training and enable gradient checkpointing in one
    # call. Passing gradient_checkpointing_kwargs ensures ``use_reentrant=False``
    # is forwarded to torch.utils.checkpoint, suppressing the PyTorch 2.11 warning.
    # We call this *before* adding PEFT so the checkpoint function is set on the
    # base modules and survives PEFT wrapping.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gradient_checkpointing else None,
    )
    if use_gradient_checkpointing:
        logger.info("Gradient checkpointing enabled (use_reentrant=False)")

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
            modules_to_save=_MODULES_TO_SAVE,
            ensure_weight_tying=True,
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"embed_tokens/lm_head set as modules_to_save for special-token learning")

    # After PEFT wrapping, ensure use_cache=False on ALL nested config objects.
    # Qwen3VL nesting: PeftModel → LoraModel → Qwen3VLForConditionalGeneration
    #   → Qwen3VLModel → Qwen3VLTextModel (the one with @merge_with_config_defaults
    #   decorator that reads self.config.use_cache during forward).
    _set_use_cache_deep(model, False)

    # Gradient checkpointing is only useful on trainable layers. Frozen modules
    # (e.g., the vision backbone) keep the flag from the base model but have no
    # trainable parameters, which causes spurious PyTorch warnings.
    _disable_gradient_checkpointing_on_frozen_modules(model)

    # Fix dtype mismatch between fp32 layer-norm outputs and lm_head weights
    # during generation (PEFT modules_to_save + prepare_model_for_kbit_training).
    _patch_lm_head_dtype_cast(model)

    # Optionally unfreeze the last N ViT blocks + merger.
    # Access the base model through the PeftModel wrapper if present.
    if unfreeze_vit_layers > 0:
        base = model.base_model.model if hasattr(model, "base_model") else model
        visual = getattr(base, "visual", None)
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
                    f"Unfroze last {unfreeze_vit_layers} ViT blocks "
                    f"(blocks {len(blocks)-unfreeze_vit_layers}-{len(blocks)-1}, "
                    f"{n_params:,} params)"
                )
            merger = getattr(visual, "merger", None)
            if merger is not None:
                for param in merger.parameters():
                    param.requires_grad = True
                n_params = sum(p.numel() for p in merger.parameters())
                logger.info(f"Unfroze ViT merger (vision→language projection, {n_params:,} params)")

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

    # Resolve relative paths to absolute to avoid HuggingFace Hub fallback
    model_path = _resolve_local_path(model_path)
    if adapter_path is not None:
        adapter_path = _resolve_local_path(adapter_path)

    # GRPO stages save adapters under round_N/ inside the stage output dir.
    model_path = _resolve_grpo_expert_path(model_path)

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
        base_model_path = _resolve_local_path(base_model_path)
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
    new_tokenizer_len = len(processor.tokenizer)
    logger.info(f"Reference: added {num_added} special tokens, tokenizer len={new_tokenizer_len}")

    # Decoder-only models need left padding for batched generation.
    processor.tokenizer.padding_side = "left"

    # Explicitly align model config / generation_config with tokenizer token IDs
    ref_tokenizer = processor.tokenizer
    model.config.pad_token_id = ref_tokenizer.pad_token_id
    model.config.bos_token_id = ref_tokenizer.bos_token_id
    model.config.eos_token_id = ref_tokenizer.eos_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = ref_tokenizer.pad_token_id
        model.generation_config.bos_token_id = ref_tokenizer.bos_token_id
        model.generation_config.eos_token_id = ref_tokenizer.eos_token_id

    current_embed_size = model.get_input_embeddings().num_embeddings
    if new_tokenizer_len > current_embed_size:
        model.resize_token_embeddings(new_tokenizer_len)
        logger.info(f"Reference: resized embeddings: {current_embed_size} → {new_tokenizer_len}")
    else:
        logger.info(f"Reference: no resize needed: embedding ({current_embed_size}) covers tokenizer ({new_tokenizer_len})")

    if adapter_to_load is not None:
        model = PeftModel.from_pretrained(model, adapter_to_load)
        logger.info(f"Reference: loaded adapter: {adapter_to_load}")

    # Fix dtype mismatch between fp32 layer-norm outputs and lm_head weights
    # during generation (same PEFT modules_to_save issue as policy).
    _patch_lm_head_dtype_cast(model)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model
