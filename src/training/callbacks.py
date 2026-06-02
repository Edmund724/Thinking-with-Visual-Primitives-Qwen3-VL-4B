"""Training callbacks for TVP project."""

import logging

import torch
from transformers import TrainerCallback

from .memory_utils import clear_memory, get_gpu_memory_gb, log_memory_status

logger = logging.getLogger(__name__)


class MemoryMonitorCallback(TrainerCallback):
    """Monitor GPU memory during training and clear if approaching limit."""

    def __init__(self, max_memory_gb: float = 22.0, warning_threshold_gb: float = 20.0):
        self.max_memory_gb = max_memory_gb
        self.warning_threshold_gb = warning_threshold_gb
        self.peak_memory = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        allocated = get_gpu_memory_gb()
        self.peak_memory = max(self.peak_memory, allocated)
        if allocated > self.max_memory_gb:
            logger.warning(
                f"Step {state.global_step}: GPU {allocated:.2f}GB > {self.max_memory_gb}GB, "
                "clearing cache..."
            )
            clear_memory()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        allocated = get_gpu_memory_gb()
        if allocated > self.warning_threshold_gb:
            logger.warning(
                f"Step {state.global_step} end: GPU {allocated:.2f}GB "
                f"(peak: {self.peak_memory:.2f}GB)"
            )
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        log_memory_status(f"Epoch {state.epoch:.1f} end (peak: {self.peak_memory:.2f}GB):")
        self.peak_memory = 0.0
        return control


class WandBLogPrimitiveMetricsCallback(TrainerCallback):
    """Log process-level primitive metrics to wandb during evaluation."""

    def __init__(self, eval_dataloader=None, processor=None):
        self.eval_dataloader = eval_dataloader
        self.processor = processor

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if self.eval_dataloader is None or model is None:
            return control
        # TODO: Implement primitive evaluation
        return control
