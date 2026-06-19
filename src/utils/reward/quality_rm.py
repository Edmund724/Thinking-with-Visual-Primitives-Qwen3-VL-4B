"""Quality Reward Model (Quality RM) for visual primitives."""

import re
from typing import Callable, List

from ...models.visual_primitive_parser import PrimitiveParser
from ...training.grpo_utils import extract_completion_text


def _count_in_answer(answer_text: str | None) -> int | None:
    """Extract a count integer from answer text, if possible."""
    if not answer_text:
        return None
    # Look for the last integer in the answer
    nums = re.findall(r"\d+", answer_text)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


def _looks_like_reward_hacking(pred_text: str, gt_text: str) -> bool:
    """Detect common reward-hacking patterns in the generated text."""
    pred_lower = pred_text.lower()
    # Model explicitly claims to use ground truth or labels its own prediction as GT
    suspicious_phrases = [
        "ground truth",
        "ground-truth",
        "gt answer",
        "gt is",
        "true answer is",
    ]
    if any(p in pred_lower for p in suspicious_phrases):
        return True

    # Predicted final answer appears verbatim inside GT reasoning (copy-paste GT)
    pred_answer = PrimitiveParser.extract_answer(pred_text)
    if pred_answer and gt_text:
        # If pred answer is a long chunk of GT reasoning, likely copied
        if pred_answer in gt_text and len(pred_answer) > 20:
            return True

    return False


def _meaningful_references(text: str) -> bool:
    """Check that box references (<|ref|>...<|/ref|>) are not empty or pure numbers."""
    refs = re.findall(r"<\|ref\|>(.*?)<\|/ref\|>", text)
    if not refs:
        # No refs is okay for points or simple formats
        return True
    for ref in refs:
        ref_clean = ref.strip()
        if not ref_clean or ref_clean.isdigit():
            return False
    return True


def quality_reward_text(pred_text: str, gt_text: str, task_type: str = "box") -> float:
    """Quality Reward Model (Quality RM) — rule-based approximation.

    The paper uses an LLM-based Generative Reward Model (GRM) that evaluates
    redundancy, consistency, self-contradiction, meaningful entities, and reward
    hacking (Sec 2.5.2). Because running an LLM judge inside the GRPO loop is
    expensive on a single 24GB GPU, this implementation uses a rule-based
    approximation that targets the same failure modes.

    Scores from {0.0, 0.5, 1.0} matching the paper's discrete tiers:
        1.0  No quality issues detected.
        0.5  Minor issues (small redundancy, weak consistency).
        0.0  Serious issues (reward hacking, contradiction, missing backtracking).
    """
    from .format_rm import _NON_LATIN_SCRIPT_RE

    pred_reasoning = PrimitiveParser.extract_reasoning(pred_text) or ""
    pred_answer = PrimitiveParser.extract_answer(pred_text)

    issues = 0
    major_issue = False

    # Non-Latin script is a major quality issue.
    if _NON_LATIN_SCRIPT_RE.search(pred_text):
        major_issue = True

    # 1. Redundancy: duplicate boxes/points in the thinking trace.
    pred_boxes = PrimitiveParser.extract_boxes(pred_reasoning)
    pred_points = PrimitiveParser.extract_points(pred_reasoning)
    if PrimitiveParser.has_duplicate_coords(pred_boxes) or PrimitiveParser.has_duplicate_coords(pred_points):
        issues += 1

    # 2. Consistency (counting tasks): final answer count matches number of visual primitives.
    if task_type == "box":
        answer_count = _count_in_answer(pred_answer)
        if answer_count is not None and len(pred_boxes) > 0:
            # Allow small tolerance; severe mismatch is a major issue.
            if abs(answer_count - len(pred_boxes)) > max(1, len(pred_boxes) * 0.2):
                major_issue = True
        elif answer_count is not None and answer_count > 0 and len(pred_boxes) == 0:
            major_issue = True

    # 3. Contradiction checks.
    if task_type == "maze":
        # Maze: claims solvable but no path points, or vice versa.
        solvable_claim = False
        if pred_answer is not None:
            solvable_claim = "true" in pred_answer.lower() or "yes" in pred_answer.lower()
        has_path = len(pred_points) >= 2
        if solvable_claim and not has_path:
            major_issue = True
        if not solvable_claim and has_path:
            issues += 1
    elif task_type == "box":
        # Box: answer says "no X" / "False" / "0" but outputs boxes, or vice versa.
        answer_lower = (pred_answer or "").lower()
        negative_answer = any(
            tok in answer_lower for tok in ["false", "no ", "none", "0", "not found"]
        ) or (pred_answer == "0")
        if negative_answer and len(pred_boxes) > 0:
            major_issue = True
        if not negative_answer and len(pred_boxes) == 0:
            issues += 1

    # 4. Reward hacking: suspicious phrases or copied ground truth.
    if _looks_like_reward_hacking(pred_text, gt_text):
        major_issue = True

    # 5. Self-contradiction inside reasoning (e.g., "there is no dog" followed by a dog box).
    reasoning_lower = pred_reasoning.lower()
    negation_markers = [
        "no ", "not ", "none", "does not exist", "cannot see", "isn't", "aren't",
        "no visible", "no such", "doesn't appear",
    ]
    has_negation = any(m in reasoning_lower for m in negation_markers)
    if has_negation and (len(pred_boxes) > 0 or len(pred_points) > 0):
        major_issue = True

    # 6. Meaningful references: box refs are not empty or numeric codes.
    if not _meaningful_references(pred_text):
        issues += 1

    if major_issue:
        return 0.0
    if issues > 0:
        return 0.5
    return 1.0


def make_quality_reward_fn(tokenizer=None, task_type_default: str = "box") -> Callable:
    """Factory for a TRL-compatible Quality RM reward function."""

    def quality_reward(completions, prompts=None, **kwargs):
        inputs = kwargs.get("inputs", [])
        gt_texts = kwargs.get("gt_text", [])
        task_types = kwargs.get("task_type", [])
        completion_ids_list = kwargs.get("completion_ids", [])

        rewards = []
        for i, completion in enumerate(completions):
            if i < len(inputs):
                gt_text = inputs[i].get("gt_text", "")
                task_type = inputs[i].get("task_type", task_type_default)
            elif i < len(gt_texts):
                gt_text = gt_texts[i]
                task_type = task_types[i] if i < len(task_types) else task_type_default
            else:
                rewards.append(0.0)
                continue

            comp_id = completion_ids_list[i] if i < len(completion_ids_list) else None
            pred_text = extract_completion_text(
                completion, tokenizer=tokenizer, completion_id=comp_id
            )

            try:
                rewards.append(quality_reward_text(pred_text, gt_text, task_type))
            except Exception:
                rewards.append(0.0)
        return rewards

    return quality_reward
