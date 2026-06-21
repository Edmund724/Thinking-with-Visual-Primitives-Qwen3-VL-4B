"""Difficulty grading for GRPO/RFT data selection.

Matches the paper's Specialized RL data selection (Sec 2.5.2):
a prompt is Easy/Normal/Hard based on how many of its rollouts are correct.
"""

from typing import List

import numpy as np

from .batch_inference import batch_generate_completions, generate_single_completion
from .conversation_builder import ConversationBuilder
from ..models.visual_primitive_parser import PrimitiveParser
from .reward.format_rm import format_reward, _NON_LATIN_SCRIPT_RE
from .reward.accuracy_rm import process_reward, compute_total_reward


def is_rollout_correct(
    pred_text: str,
    gt_text: str,
    task_type: str = "box",
    iou_threshold: float = 0.5,
    point_dist_threshold: float = 10.0,
    maze_grid: np.ndarray | None = None,
) -> bool:
    """Binary correctness used for paper-style difficulty grading.

    The paper (Sec 2.5.2) categorizes rollouts by the *number of correct
    rollouts*, not by a continuous reward threshold. A rollout is considered
    correct when its final answer is correct AND the output contains at least
    one valid visual primitive.

    This function checks **task correctness only** (IoU for boxes, distance
    for points, exact answer match for counting).  Format quality (think tags,
    non-Latin script, tag pairing) is NOT part of this check — it is handled
    separately by Format RM during GRPO training.  This keeps the difficulty
    grading signal focused on "can the model solve the task", which is the
    paper's intent.

    This is intentionally more lenient than ``format_reward`` on tag order:
    after SFT the model often emits coordinates with reversed or duplicated
    tags. As long as coordinates are parseable and the answer matches, the
    rollout counts as correct for difficulty selection. GRPO's reward will
    still penalize malformed tags.
    """

    # For box/point tasks with spatial GT, use IoU / distance matching
    # instead of exact answer string comparison.  The paper (Sec 2.5.2)
    # defines correctness by whether the output is *actually correct*,
    # and for localization tasks that means IoU ≥ threshold, not exact
    # coordinate string match.
    if task_type == "box":
        pred_boxes = PrimitiveParser.extract_boxes(pred_text) or PrimitiveParser.lenient_extract_boxes(pred_text)
        gt_boxes = PrimitiveParser.extract_boxes(gt_text)

        if gt_boxes:
            # Box localization: IoU-based correctness.
            if not pred_boxes:
                return False
            coords = [c for b in pred_boxes for c in b]
            if any(c < 0 or c > 999 for c in coords):
                return False
            _, num_match, _ = PrimitiveParser.match_boxes(
                pred_boxes, gt_boxes, iou_threshold
            )
            # Require at least one box matched at the given IoU threshold.
            if num_match == 0:
                return False
            return True
        else:
            # Counting or refusal (no GT boxes): fall back to answer match.
            pred_answer = PrimitiveParser.normalize_answer_text(
                PrimitiveParser.extract_answer(pred_text)
            )
            gt_answer = PrimitiveParser.normalize_answer_text(
                PrimitiveParser.extract_answer(gt_text)
            )
            if pred_answer is None or gt_answer is None or pred_answer != gt_answer:
                return False
            # Require parseable boxes only when the answer implies objects exist.
            if pred_boxes:
                coords = [c for b in pred_boxes for c in b]
                if any(c < 0 or c > 999 for c in coords):
                    return False
            return True

    if task_type in ("point", "path"):
        pred_points = PrimitiveParser.extract_points(pred_text)
        gt_points = PrimitiveParser.extract_points(gt_text)

        if gt_points:
            # Point/path: distance-based correctness.
            if not pred_points:
                return False
            coords = [c for p in pred_points for c in p]
            if any(c < 0 or c > 999 for c in coords):
                return False
            _, num_match, _ = PrimitiveParser.match_points(
                pred_points, gt_points, point_dist_threshold
            )
            if num_match == 0:
                return False
            return True
        else:
            pred_answer = PrimitiveParser.normalize_answer_text(
                PrimitiveParser.extract_answer(pred_text)
            )
            gt_answer = PrimitiveParser.normalize_answer_text(
                PrimitiveParser.extract_answer(gt_text)
            )
            if pred_answer is None or gt_answer is None or pred_answer != gt_answer:
                return False
            return True

    # Maze-specific: reuse process_reward for wall/answer checks.
    if task_type == "maze":
        proc = process_reward(
            pred_text, gt_text, task_type,
            iou_threshold, point_dist_threshold, maze_grid,
        )
        if not proc.get("answer_correct", False):
            return False
        return True

    # Fallback: exact answer match for other task types.
    pred_answer = PrimitiveParser.normalize_answer_text(PrimitiveParser.extract_answer(pred_text))
    gt_answer = PrimitiveParser.normalize_answer_text(PrimitiveParser.extract_answer(gt_text))
    return pred_answer is not None and gt_answer is not None and pred_answer == gt_answer


def filter_normal_level_data(
    model,
    processor,
    data: List[dict],
    num_generations: int = 4,
    max_completion_length: int = 384,
    task_type: str = "box",
    iou_threshold: float = 0.5,
    point_dist_threshold: float = 10.0,
    batch_size: int = 4,
    empty_cache_every: int = 50,
    reward_threshold: float = None,
    logger=None,
) -> List[dict]:
    """Pre-filter GRPO training data to keep only Normal-difficulty samples.

    For each prompt, generate `num_generations` on-policy rollouts and classify
    the sample as Easy/Normal/Hard based on the *number of correct rollouts*,
    matching the paper's Specialized RL data selection (Sec 2.5.2):

        * Easy:   all rollouts are correct.
        * Hard:   all rollouts are incorrect.
        * Normal: at least one correct and at least one incorrect rollout.

    When ``reward_threshold`` is set (recommended), a rollout is considered
    correct when ``total_reward >= reward_threshold``.  This uses the
    continuous reward signal (Format RM + Accuracy RM) as the correctness
    indicator — more robust than binary checks and better captures partial
    correctness, aligning with the paper's intent of identifying prompts
    where the model sometimes succeeds and sometimes fails.

    When ``reward_threshold`` is None (legacy mode), the binary
    ``is_rollout_correct`` function is used instead.

    Batched generation is used to saturate the GPU: we process ``batch_size``
    distinct prompts at once and repeat each prompt ``num_generations`` times,
    so a single ``model.generate`` call handles ``batch_size * num_generations``
    sequences. Gradient checkpointing is temporarily disabled and ``use_cache``
    is enabled during filtering because we are in inference mode.
    """
    import torch

    was_training = model.training
    model.eval()

    # Disable gradient checkpointing and enable KV-cache for fast inference.
    grad_ckpt_enabled = getattr(model, "is_gradient_checkpointing", False) or getattr(
        model, "gradient_checkpointing", False
    )
    if grad_ckpt_enabled:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    cache_configs = {}

    def _set_use_cache(module, value):
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cid = id(cfg)
            if cid not in cache_configs:
                cache_configs[cid] = cfg.use_cache
            cfg.use_cache = value
        for child in module.children():
            _set_use_cache(child, value)

    _set_use_cache(model, True)

    normal_data = []
    easy_count = 0
    hard_count = 0
    last_input_len = 0

    def _process_chunk(chunk):
        nonlocal easy_count, hard_count, normal_data
        try:
            outputs, input_len = batch_generate_completions(
                model=model,
                processor=processor,
                samples=chunk,
                num_generations=num_generations,
                max_completion_length=max_completion_length,
                temperature=0.7,
            )
        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(f"Batched generation failed: {e}; falling back to singles.")
            for sample in chunk:
                _process_single(sample)
            return
        nonlocal last_input_len
        last_input_len = input_len
        for j, sample in enumerate(chunk):
            gt_text = ConversationBuilder.build_gt_text(
                sample.get("reasoning", ""),
                sample.get("answer", ""),
            )
            rollouts = []
            start = j * num_generations
            for offset in range(num_generations):
                pred = processor.tokenizer.decode(
                    outputs[start + offset][input_len:], skip_special_tokens=False
                )
                try:
                    total = compute_total_reward(
                        pred_text=pred,
                        gt_text=gt_text,
                        task_type=task_type,
                        iou_threshold=iou_threshold,
                        point_dist_threshold=point_dist_threshold,
                        maze_grid=sample.get("maze_grid"),
                    )
                    rollouts.append({"pred_text": pred, "total_reward": total["total_reward"]})
                except Exception:
                    rollouts.append({"pred_text": pred, "total_reward": 0.0})

            correct_count = sum(
                1
                for r in rollouts
                if (
                    r["total_reward"] >= reward_threshold
                    if reward_threshold is not None
                    else is_rollout_correct(
                        pred_text=r["pred_text"],
                        gt_text=gt_text,
                        task_type=task_type,
                        iou_threshold=iou_threshold,
                        point_dist_threshold=point_dist_threshold,
                        maze_grid=sample.get("maze_grid"),
                    )
                )
            )
            if correct_count == num_generations:
                easy_count += 1
            elif correct_count == 0:
                hard_count += 1
            else:
                normal_data.append(sample)

    def _process_single(sample):
        nonlocal easy_count, hard_count, normal_data
        gt_text = ConversationBuilder.build_gt_text(
            sample.get("reasoning", ""),
            sample.get("answer", ""),
        )
        rollouts = []
        for _ in range(num_generations):
            try:
                outputs, input_len = generate_single_completion(
                    model=model,
                    processor=processor,
                    sample=sample,
                    max_completion_length=max_completion_length,
                    temperature=0.7,
                )
            except Exception:
                continue
            pred = processor.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=False
            )
            try:
                total = compute_total_reward(
                    pred_text=pred,
                    gt_text=gt_text,
                    task_type=task_type,
                    iou_threshold=iou_threshold,
                    point_dist_threshold=point_dist_threshold,
                    maze_grid=sample.get("maze_grid"),
                )
                rollouts.append({"pred_text": pred, "total_reward": total["total_reward"]})
            except Exception:
                rollouts.append({"pred_text": pred, "total_reward": 0.0})

        correct_count = sum(
            1
            for r in rollouts
            if (
                r["total_reward"] >= reward_threshold
                if reward_threshold is not None
                else is_rollout_correct(
                    pred_text=r["pred_text"],
                    gt_text=gt_text,
                    task_type=task_type,
                    iou_threshold=iou_threshold,
                    point_dist_threshold=point_dist_threshold,
                    maze_grid=sample.get("maze_grid"),
                )
            )
        )
        if correct_count == num_generations:
            easy_count += 1
        elif correct_count == 0:
            hard_count += 1
        else:
            normal_data.append(sample)

    for chunk_start in range(0, len(data), batch_size):
        chunk = data[chunk_start : chunk_start + batch_size]
        _process_chunk(chunk)

        processed = min(chunk_start + batch_size, len(data))
        if logger is not None and processed % 50 == 0:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(
                f"  {processed}/{len(data)}: "
                f"{len(normal_data)} normal kept, {easy_count} easy, {hard_count} hard | "
                f"input_len={last_input_len} peak_mem={peak_mb:.0f}MB"
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    if logger is not None:
        logger.info(
            f"Difficulty filtering done: {len(normal_data)} Normal kept, "
            f"{easy_count} Easy skipped, {hard_count} Hard skipped"
        )

    # Restore gradient checkpointing and KV-cache settings.
    if grad_ckpt_enabled:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    _set_use_cache(model, False)

    if was_training:
        model.train()
    else:
        model.eval()

    return normal_data
