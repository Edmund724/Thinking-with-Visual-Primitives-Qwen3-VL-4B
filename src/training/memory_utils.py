"""GPU memory monitoring and optimization utilities."""

import gc
import logging

import torch
from transformers import TrainerCallback

from ..utils.constants import GPU_MEMORY_WARNING_GB, MAX_GPU_MEMORY_GB

logger = logging.getLogger(__name__)


def get_gpu_memory_gb() -> float:
    """Get current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def get_gpu_memory_reserved_gb() -> float:
    """Get current GPU memory reserved in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        # Finish any pending kernels before releasing blocks, then wait again
        # after empty_cache to avoid "CUDA driver error: device not ready" when
        # a new model is loaded immediately afterwards.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def log_memory_status(prefix: str = ""):
    """Log current GPU memory status."""
    allocated = get_gpu_memory_gb()
    reserved = get_gpu_memory_reserved_gb()
    msg = f"{prefix} GPU allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
    if allocated > GPU_MEMORY_WARNING_GB:
        logger.warning(msg)
    else:
        logger.info(msg)


def cast_ref_adapter_to_bf16(model: torch.nn.Module) -> int:
    """Cast the GRPO reference adapter (``ref``) to bfloat16 to save VRAM.

    TRL creates the ``ref`` adapter by copying the policy adapter before it
    casts trainable parameters to bfloat16, so the reference adapter is left in
    float32. The reference is only used for inference (computing log
    probabilities for the KL penalty), so bfloat16 is safe and roughly halves
    its GPU memory footprint.
    """
    peft_config = getattr(model, "peft_config", None)
    if peft_config is None or "ref" not in peft_config:
        return 0

    count = 0
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.data = param.data.to(torch.bfloat16)
            count += 1
    if count:
        logger.info(f"Cast {count} ref-adapter parameters to bfloat16")
    return count


def build_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    vit_lr: float = 1e-6,
) -> list[dict]:
    """Build per-layer parameter groups for mixed-precision training.

    Separates ViT blocks and merger from LLM parameters so they can use
    different learning rates.  ViT layers should be trained with very low
    LR (e.g. 1e-6) to avoid disrupting pretrained visual features.

    Returns a list of dicts suitable for ``torch.optim.AdamW(params=...)``.
    """
    vit_params: list[torch.nn.Parameter] = []
    merger_params: list[torch.nn.Parameter] = []
    llm_params: list[torch.nn.Parameter] = []

    def _is_vit_block(param: torch.nn.Parameter) -> bool:
        """Heuristic: a param belongs to a ViT block if its name contains
        'visual.blocks'."""
        return False  # determined during iteration

    # Access base model through PeftModel wrapper if present.
    base = model.base_model.model if hasattr(model, "base_model") else model
    visual = getattr(base, "visual", None)

    vit_block_params: set[int] = set()
    merger_param_ids: set[int] = set()
    if visual is not None:
        blocks = getattr(visual, "blocks", None)
        if blocks is not None:
            for block in blocks:
                for p in block.parameters():
                    if p.requires_grad:
                        vit_block_params.add(id(p))
        merger = getattr(visual, "merger", None)
        if merger is not None:
            for p in merger.parameters():
                if p.requires_grad:
                    merger_param_ids.add(id(p))

    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in vit_block_params:
            vit_params.append(p)
        elif pid in merger_param_ids:
            merger_params.append(p)
        else:
            llm_params.append(p)

    groups: list[dict] = []
    if llm_params:
        groups.append({"params": llm_params, "lr": base_lr})
        logger.info(
            "Param groups: LLM %s params @ lr=%.1e",
            f"{sum(p.numel() for p in llm_params):,}", base_lr,
        )
    if merger_params:
        merger_lr = vit_lr * 10.0
        groups.append({"params": merger_params, "lr": merger_lr})
        logger.info(
            "Param groups: ViT merger %s params @ lr=%.1e",
            f"{sum(p.numel() for p in merger_params):,}", merger_lr,
        )
    if vit_params:
        groups.append({"params": vit_params, "lr": vit_lr})
        logger.info(
            "Param groups: ViT blocks %s params @ lr=%.1e",
            f"{sum(p.numel() for p in vit_params):,}", vit_lr,
        )
    if not groups:
        # Fallback: all trainable params at base_lr
        all_params = [p for p in model.parameters() if p.requires_grad]
        groups.append({"params": all_params, "lr": base_lr})

    return groups


class GPUMemoryMonitor(TrainerCallback):
    """Hugging Face Trainer callback to monitor GPU memory."""

    def __init__(self, clear_threshold_gb: float = MAX_GPU_MEMORY_GB):
        self.clear_threshold_gb = clear_threshold_gb

    def on_step_end(self, args, state, control, **kwargs):
        allocated = get_gpu_memory_gb()
        if allocated > self.clear_threshold_gb:
            logger.warning(
                f"Step {state.global_step}: GPU {allocated:.2f}GB > "
                f"{self.clear_threshold_gb}GB, clearing cache..."
            )
            clear_memory()
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        log_memory_status(f"Epoch {state.epoch:.1f} end:")
        return control
