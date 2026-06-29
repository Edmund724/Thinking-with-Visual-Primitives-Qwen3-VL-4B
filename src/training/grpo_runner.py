"""Shared GRPO round-loop for stage 4a (Box Expert) and stage 4b (Point Expert).

Extracts the ~140-line duplicated round-loop from both scripts into a single
function.  Stage scripts now supply only their task-specific reward factory
and thresholds; all training orchestration lives here.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

import torch
from trl import GRPOConfig, GRPOTrainer

from ..models.qwen_vl_loader import _set_use_cache_deep, load_qlora_model
from ..utils.constants import GPU_MEMORY_WARNING_GB
from .callbacks import CastRefAdapterCallback, TimeLoggingCallback, ValidationSubsetEarlyStoppingCallback
from .grpo_fixes import apply_grpo_fixes
from .memory_utils import GPUMemoryMonitor, clear_memory, log_memory_status


def _latest_checkpoint(round_dir: Path) -> Path | None:
    """Return the latest checkpoint-* directory inside a round dir, or None."""
    checkpoints = [p for p in round_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=_step)


def run_grpo_rounds(
    *,
    policy_model,
    processor,
    train_data: list[dict],
    output_dir: str,
    num_rounds: int,
    reward_fn_factory: Callable,  # (threshold: float) -> Callable
    thresholds: list[float],      # e.g. [0.3, 0.5, 0.7] or [20.0, 10.0, 5.0]
    quality_fn_factory: Callable,  # (tokenizer, task_type_default: str) -> Callable
    quality_task_type: str,        # "box" or "point"
    args,                          # argparse.Namespace
    logger: logging.Logger,
) -> None:
    """Run multi-round GRPO training with threshold progression.

    Parameters
    ----------
    policy_model:
        QLoRA model to train.  Modified in-place and reloaded between rounds.
    processor:
        HuggingFace processor for the model.
    train_data:
        List of data dicts (already difficulty-filtered).
    output_dir:
        Root output directory.  Each round gets a ``round_N/`` subdirectory.
    num_rounds:
        Number of GRPO rounds (typically 2–3).
    reward_fn_factory:
        Callable that takes a threshold and returns the task-specific reward
        function, e.g. ``make_box_reward_fn`` or ``make_point_reward_fn``.
    thresholds:
        Per-round threshold values.  Length must be ≥ *num_rounds*.
    quality_fn_factory:
        ``make_quality_reward_fn`` or ``make_quality_reward_api_fn``.
    quality_task_type:
        Passed as ``task_type_default`` to *quality_fn_factory*.
    args:
        Parsed argparse / YAML config namespace.  Must contain all the
        standard GRPO hyperparams (batch_size, learning_rate, beta, etc.).
    logger:
        Stage logger.
    """
    apply_grpo_fixes(GRPOTrainer)

    for round_idx in range(num_rounds):
        threshold = thresholds[round_idx] if round_idx < len(thresholds) else thresholds[-1]
        round_dir = Path(output_dir) / f"round_{round_idx + 1}"

        # ── Skip already-completed rounds ───────────────────────────
        round_adapter = round_dir / "adapter_model.safetensors"
        if round_adapter.exists():
            logger.info(
                f"Round {round_idx + 1}/{num_rounds} already done ({round_adapter}), skipping."
            )
            try:
                policy_model, processor = load_qlora_model(
                    model_name=str(round_dir),
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                )
            except Exception as e:
                logger.warning(f"Could not reload round {round_idx + 1}: {e}")
            continue

        round_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{'=' * 60}")
        logger.info(f"GRPO Round {round_idx + 1}/{num_rounds} (threshold: {threshold})")
        logger.info(f"{'=' * 60}")

        # ── Resume from latest checkpoint if interrupted ───────────
        resume_from = _latest_checkpoint(round_dir)
        if resume_from is not None:
            logger.info(
                f"Found checkpoint {resume_from.name} for round {round_idx + 1}, resuming."
            )
            # Keep the policy_model loaded from this round's starting point
            # (args.model_path for round 1, previous round's output otherwise).
            # Trainer.train(resume_from_checkpoint=...) will load the adapter
            # weights, optimizer state, and trainer state from the checkpoint.
            # Reloading the adapter here first causes double-loading and can
            # leave several extra GB in the CUDA allocator / memory pool.
            adapter_config_path = resume_from / "adapter_config.json"
            if not adapter_config_path.exists():
                logger.warning(
                    f"Checkpoint {resume_from} has no adapter_config.json; cannot resume."
                )
                resume_from = None
            else:
                log_memory_status(
                    f"Will resume from checkpoint {resume_from.name}:"
                )

        # ── Build reward function ──────────────────────────────────
        reward_fn = reward_fn_factory(threshold, tokenizer=processor.tokenizer, logger=logger)

        # ── Build GRPO config ──────────────────────────────────────
        grpo_config = GRPOConfig(
            output_dir=str(round_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="tensorboard",
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            num_generations=args.num_generations,
            generation_batch_size=getattr(args, "generation_batch_size", None) or args.num_generations,
            max_completion_length=args.max_completion_length,
            beta=args.beta,
            temperature=args.temperature,
            scale_rewards="group",
            num_iterations=2,
        )

        # ── Dataset ────────────────────────────────────────────────
        # Lazy import to avoid circular dependency with sft_trainer
        from ..data.datasets.grpo_dataset import GRPODataset

        dataset = GRPODataset(train_data)

        # ── Disable cache (incompatible with gradient checkpointing)
        _set_use_cache_deep(policy_model)

        # ── Memory monitor ─────────────────────────────────────────
        total_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        mem_threshold_gb = max(GPU_MEMORY_WARNING_GB, total_vram_gb * 0.85)

        # ── Quality RM ─────────────────────────────────────────────
        quality_fn = quality_fn_factory(
            tokenizer=processor.tokenizer,
            task_type_default=quality_task_type,
        )

        # ── Callbacks ──────────────────────────────────────────────
        callbacks = [
            GPUMemoryMonitor(clear_threshold_gb=mem_threshold_gb),
            TimeLoggingCallback(),
            CastRefAdapterCallback(),
        ]
        if args.early_stopping_subset_size > 0:
            callbacks.append(
                ValidationSubsetEarlyStoppingCallback(
                    model=policy_model,
                    processor=processor,
                    eval_data=train_data,
                    reward_fn=reward_fn,
                    eval_steps=args.early_stopping_eval_steps,
                    patience=args.early_stopping_patience,
                    subset_size=args.early_stopping_subset_size,
                )
            )

        # ── Trainer ────────────────────────────────────────────────
        trainer = GRPOTrainer(
            model=policy_model,
            reward_funcs=[reward_fn, quality_fn],
            args=grpo_config,
            train_dataset=dataset,
            processing_class=processor,
            callbacks=callbacks,
        )

        logger.info("Training GRPO...")
        trainer.train(
            resume_from_checkpoint=str(resume_from) if resume_from is not None else None
        )
        trainer.save_model(str(round_dir))
        processor.save_pretrained(str(round_dir))

        log_memory_status(f"Round {round_idx + 1} complete:")

        # ── Cleanup between rounds ─────────────────────────────────
        # Aggressively free GPU memory before loading the next round's model.
        # Moving the model to CPU first releases its CUDA allocations; then we
        # delete the trainer (which may hold references to the model) and force
        # a full garbage collection + cache clear. This prevents the common
        # failure mode where Round 2 OOMs because Round 1 left the allocator
        # pool/fragmentation populated.
        if policy_model is not None:
            try:
                policy_model.to("cpu")
            except Exception as e:
                logger.debug(f"Moving model to CPU before cleanup failed: {e}")
        del trainer
        del policy_model
        gc.collect()
        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

        # Reload for next round
        try:
            policy_model, processor = load_qlora_model(
                model_name=str(round_dir),
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
            log_memory_status("Reloaded model for next round:")
        except Exception as e:
            logger.warning(f"Could not reload: {e}, continuing")
