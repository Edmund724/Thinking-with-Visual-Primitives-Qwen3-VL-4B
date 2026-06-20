"""Training callbacks for TVP project."""

import logging

import torch
from transformers import TrainerCallback

from ..utils.conversation_builder import ConversationBuilder
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


class ValidationSubsetEarlyStoppingCallback(TrainerCallback):
    """Run a small validation subset every `eval_steps` and stop if no improvement.

    This is useful for flow-through runs where we want to verify the pipeline
    without training to convergence.
    """

    def __init__(
        self,
        model,
        processor,
        eval_data,
        reward_fn,
        eval_steps: int = 50,
        patience: int = 2,
        subset_size: int = 32,
    ):
        self.model = model
        self.processor = processor
        self.eval_data = eval_data[:subset_size]
        self.reward_fn = reward_fn
        self.eval_steps = eval_steps
        self.patience = patience
        self.best_score = -float("inf")
        self.bad_steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return control
        if not self.eval_data:
            return control

        from ..utils.batch_inference import batch_generate_completions

        self.model.eval()
        try:
            outputs, input_len = batch_generate_completions(
                model=self.model,
                processor=self.processor,
                samples=self.eval_data,
                num_generations=1,
                max_completion_length=args.max_completion_length,
                temperature=0.7,
            )

            rewards = []
            for j, sample in enumerate(self.eval_data):
                gt_text = ConversationBuilder.build_gt_text(
                    sample.get("reasoning", ""),
                    sample.get("answer", ""),
                )
                pred = self.processor.tokenizer.decode(
                    outputs[j][input_len:], skip_special_tokens=False
                )
                reward = self.reward_fn(
                    [pred],
                    inputs=[{"gt_text": gt_text, "task_type": sample.get("task_type", "box")}],
                )[0]
                rewards.append(reward)

            mean_reward = float(torch.tensor(rewards).mean()) if rewards else 0.0
            logger.info(
                f"Validation subset (n={len(self.eval_data)}) mean reward: {mean_reward:.4f} "
                f"@ step {state.global_step}"
            )

            if mean_reward > self.best_score:
                self.best_score = mean_reward
                self.bad_steps = 0
            else:
                self.bad_steps += 1

            if self.bad_steps >= self.patience:
                logger.info(
                    f"Early stopping triggered at step {state.global_step} "
                    f"(no improvement for {self.patience} evals)"
                )
                control.should_training_stop = True
        except Exception as e:
            logger.warning(f"Validation subset eval failed: {e}")
        finally:
            self.model.train()

        return control


def maybe_compile_model(model, enable: bool = True):
    """Wrap model with torch.compile if requested and supported.

    Compiling QLoRA models can fail on some PyTorch/Transformer versions, so
    failures are caught and logged instead of raised.
    """
    if not enable:
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        logger.info("torch.compile enabled on policy model.")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile failed, continuing without it: {e}")
        return model
