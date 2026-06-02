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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_memory_status(prefix: str = ""):
    """Log current GPU memory status."""
    allocated = get_gpu_memory_gb()
    reserved = get_gpu_memory_reserved_gb()
    msg = f"{prefix} GPU allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
    if allocated > GPU_MEMORY_WARNING_GB:
        logger.warning(msg)
    else:
        logger.info(msg)


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
