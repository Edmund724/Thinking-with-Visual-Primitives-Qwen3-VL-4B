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


class TensorBoardPrimitiveMetricsCallback(TrainerCallback):
    """Log process-level primitive metrics to TensorBoard during training.

    Every ``log_steps`` steps, samples a small batch of completions and
    computes primitive-level statistics (format compliance, coordinate
    validity, ref usage, reward decomposition).
    """

    def __init__(
        self,
        model=None,
        processor=None,
        eval_data=None,
        reward_fn=None,
        log_steps: int = 100,
        sample_size: int = 8,
    ):
        self.model = model
        self.processor = processor
        self.eval_data = eval_data[:sample_size] if eval_data else []
        self.reward_fn = reward_fn
        self.log_steps = log_steps
        self._writer = None

    def _get_writer(self, args):
        """Lazily create a TensorBoard SummaryWriter."""
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                import os
                log_dir = os.path.join(args.output_dir, "tb_primitive_logs")
                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard primitive metrics → {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed; primitive metrics disabled")
                return None
        return self._writer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_steps != 0:
            return control
        if not self.eval_data or self.model is None or self.processor is None:
            return control

        writer = self._get_writer(args)
        if writer is None:
            return control

        from ..utils.batch_inference import batch_generate_completions
        from ..models.visual_primitive_parser import PrimitiveParser
        from ..utils.reward.format_rm import format_reward

        self.model.eval()
        try:
            outputs, input_len = batch_generate_completions(
                model=self.model,
                processor=self.processor,
                samples=self.eval_data,
                num_generations=1,
                max_completion_length=256,
                temperature=0.7,
            )

            format_scores = []
            ref_usage = []
            coord_valid = []
            total_rewards = []

            for j, sample in enumerate(self.eval_data):
                pred = self.processor.tokenizer.decode(
                    outputs[j][input_len:], skip_special_tokens=False
                )

                # Format check
                fmt = format_reward(pred)
                format_scores.append(fmt.get("total_format_score", 0.0))

                # Ref usage
                refs = PrimitiveParser.extract_refs(pred)
                boxes = PrimitiveParser.extract_boxes(pred)
                ref_usage.append(1.0 if refs and boxes else 0.0)

                # Coordinate validity
                all_coords = [c for b in boxes for c in b]
                points = PrimitiveParser.extract_points(pred)
                all_coords.extend([c for p in points for c in p])
                if all_coords:
                    valid = all(0 <= c <= 999 for c in all_coords)
                    coord_valid.append(1.0 if valid else 0.0)
                else:
                    coord_valid.append(0.0)

                # Total reward if reward_fn available
                if self.reward_fn:
                    gt_text = ConversationBuilder.build_gt_text(
                        sample.get("reasoning", ""),
                        sample.get("answer", ""),
                    )
                    try:
                        reward = self.reward_fn(
                            [pred],
                            inputs=[{"gt_text": gt_text, "task_type": sample.get("task_type", "box")}],
                        )[0]
                        total_rewards.append(reward)
                    except Exception:
                        pass

            import numpy as np
            step = state.global_step
            writer.add_scalar("primitive/format_compliance_rate", float(np.mean(format_scores)), step)
            writer.add_scalar("primitive/ref_usage_rate", float(np.mean(ref_usage)), step)
            writer.add_scalar("primitive/coord_validity_rate", float(np.mean(coord_valid)), step)
            if total_rewards:
                writer.add_scalar("primitive/avg_total_reward", float(np.mean(total_rewards)), step)

            # Log one sample completion as text
            if self.eval_data:
                sample_pred = self.processor.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=False
                )
                writer.add_text("primitive/sample_completion", sample_pred[:500], step)

        except Exception as e:
            logger.warning(f"Primitive metrics logging failed: {e}")
        finally:
            self.model.train()

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._writer is not None:
            self._writer.close()
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
