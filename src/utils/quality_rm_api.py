"""LLM-as-Judge Quality Reward Model via OpenAI-compatible API.

Falls back to the rule-based quality_reward_text if the API is unavailable,
times out, or returns an unparseable response.

Subset sampling (QUALITY_RM_SAMPLE_RATIO) reduces API cost by routing a
random fraction of completions through the LLM judge; the rest use the fast
rule-based fallback.
"""

import os
import random
import re
import time
from typing import Any

from .reward.quality_rm import quality_reward_text


def _load_api_config() -> dict:
    """Load API config from environment.

    Returns empty dict if the required env vars are missing, which signals
    that API judging is disabled.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("QUALITY_RM_MODEL", "gpt-4o-mini")
    if not api_key:
        return {}

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": int(os.getenv("QUALITY_RM_TIMEOUT", "30")),
        "max_retries": int(os.getenv("QUALITY_RM_MAX_RETRIES", "2")),
        "sample_ratio": float(os.getenv("QUALITY_RM_SAMPLE_RATIO", "1.0")),
    }


def _build_judge_prompt(pred_text: str, gt_text: str, task_type: str = "box") -> str:
    """Build a chain-of-thought judge prompt (paper-style GRM evaluation).

    The judge is asked to reason about specific quality dimensions before
    outputting a structured score, which produces more reliable ratings than
    a bare score request.
    """
    return (
        "You are a quality judge for visual reasoning model outputs. "
        "Evaluate the model's response against the ground truth.\n\n"
        "Check these dimensions:\n"
        "1. Redundancy — duplicate or near-duplicate coordinates in the thinking trace?\n"
        "2. Consistency — does the answer count match the number of primitives (<|box|>, <|point|>)?\n"
        "3. Contradiction — wrong answer while claiming correctness?\n"
        "4. Reward hacking — copied ground truth text, boilerplate like \"the answer is\"?\n"
        "5. Self-contradiction — \"no objects\" or \"nothing found\" but outputs primitives?\n"
        "6. Meaningful references — are <|ref|> tags non-empty and refer to actual objects?\n\n"
        f"Task type: {task_type}\n\n"
        "=== Ground Truth ===\n"
        f"{gt_text}\n\n"
        "=== Model Prediction ===\n"
        f"{pred_text}\n\n"
        "Briefly identify any issues found (one line per issue, or \"none\" if clean). "
        "Then on the LAST line output exactly: Score: X.X\n"
        "Where X.X is 1.0 (no issues), 0.5 (minor issues), or 0.0 (serious issues)."
    )


def _parse_score(response: str) -> float:
    """Parse a discrete score from the judge response.

    Tries these strategies in order:
    1. ``Score: X.X`` pattern (from the chain-of-thought prompt)
    2. Bare number match (``1.0``, ``0.5``, ``0.0`` — legacy prompt format)
    """
    response = response.strip()

    # Strategy 1: "Score: X.X" pattern
    score_match = re.search(r"Score:\s*(1\.0|0\.5|0\.0)\b", response, re.IGNORECASE)
    if score_match:
        return float(score_match.group(1))

    # Strategy 2: Direct match (legacy format)
    if response in ("1.0", "1", "1.00"):
        return 1.0
    if response in ("0.5", ".5", "0.50"):
        return 0.5
    if response in ("0.0", "0", "0.00"):
        return 0.0

    # Strategy 3: First float-like occurrence
    match = re.search(r"\b(1\.0|0\.5|0\.0)\b", response)
    if match:
        return float(match.group(1))

    raise ValueError(f"Could not parse judge score from: {response!r}")


def quality_reward_api(
    pred_text: str,
    gt_text: str,
    task_type: str = "box",
    client: Any | None = None,
) -> float:
    """Call an OpenAI-compatible API to score prediction quality.

    Args:
        pred_text: Model-generated text.
        gt_text: Ground truth text.
        task_type: "box", "point", or "maze".
        client: Optional pre-built OpenAI client. If None, one is created lazily.

    Returns:
        Score in {0.0, 0.5, 1.0}. Falls back to rule-based scoring on any failure.
    """
    cfg = _load_api_config()
    if not cfg:
        return quality_reward_text(pred_text, gt_text, task_type)

    try:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

        prompt = _build_judge_prompt(pred_text, gt_text, task_type)
        retries = cfg["max_retries"]
        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,  # reasoning models need headroom for CoT
                    timeout=cfg["timeout"],
                )
                content = resp.choices[0].message.content
                # Fallback: reasoning models may put everything in reasoning_content
                if not content:
                    reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
                    if reasoning:
                        content = reasoning
                return _parse_score(content)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                break

        # All retries failed; fall back.
        import logging
        logging.getLogger(__name__).warning(
            f"Quality RM API failed after {retries} retries: {last_error}. "
            "Falling back to rule-based scoring."
        )
        return quality_reward_text(pred_text, gt_text, task_type)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Quality RM API unexpected error: {e}. Falling back to rule-based scoring."
        )
        return quality_reward_text(pred_text, gt_text, task_type)


def make_quality_reward_api_fn(tokenizer=None, task_type_default: str = "box"):
    """Factory for a TRL-compatible Quality RM using API judging.

    Subset sampling: controlled by ``QUALITY_RM_SAMPLE_RATIO`` env var
    (default 1.0).  Completions not sampled for API judging use the
    rule-based fallback directly, saving cost with minimal quality impact.

    Signature matches make_quality_reward_fn so it can be swapped in directly.
    """

    def quality_reward(completions, prompts=None, **kwargs):
        inputs = kwargs.get("inputs", [])
        gt_texts = kwargs.get("gt_text", [])
        task_types = kwargs.get("task_type", [])
        completion_ids_list = kwargs.get("completion_ids", [])

        # Pre-build client once per reward call to avoid repeated instantiation.
        cfg = _load_api_config()
        sample_ratio = cfg.get("sample_ratio", 1.0)
        client = None
        if cfg:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
            except Exception:
                client = None

        from ..training.grpo_utils import extract_completion_text

        rewards = []
        # Use a seeded random state per batch for reproducibility.
        rng = random.Random(42 + len(completions))

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

            # Subset sampling: only a fraction of completions go to the API.
            if rng.random() > sample_ratio:
                try:
                    rewards.append(quality_reward_text(pred_text, gt_text, task_type))
                except Exception:
                    rewards.append(0.0)
                continue

            try:
                rewards.append(quality_reward_api(pred_text, gt_text, task_type, client=client))
            except Exception:
                rewards.append(0.0)
        return rewards

    return quality_reward


# ---------------------------------------------------------------------------
# Spatial / VQA Accuracy RM (LLM-as-Judge for CLEVR complex questions)
# ---------------------------------------------------------------------------


def _build_spatial_judge_prompt(
    pred_text: str,
    gt_text: str,
    question_type: str = "multihop",
) -> str:
    """Build a judge prompt for spatial/VQA accuracy evaluation.

    The judge evaluates whether the model's reasoning chain is logically sound
    and the final answer is semantically consistent with the ground truth.
    """
    return (
        "You are an accuracy judge for visual spatial reasoning. "
        "Evaluate the model's response against the ground truth.\n\n"
        "Check these aspects:\n"
        "1. Reasoning quality — does each reasoning step have visual evidence (primitives like "
        "<|box|>, <|point|>) supporting it?\n"
        "2. Logical consistency — is the reasoning chain internally consistent?\n"
        "3. Answer correctness — does the final answer match the ground truth semantically "
        "(not just string match, but meaning)?\n"
        "4. Hallucination — does the model claim objects or relationships that don't exist?\n\n"
        f"Question type: {question_type}\n\n"
        "=== Ground Truth ===\n"
        f"{gt_text}\n\n"
        "=== Model Prediction ===\n"
        f"{pred_text}\n\n"
        "Briefly identify any issues found (one line per issue, or \"none\" if correct). "
        "Then on the LAST line output exactly: Score: X.X\n"
        "Where X.X is 1.0 (correct reasoning and answer), "
        "0.5 (partially correct reasoning or minor answer deviation), "
        "or 0.0 (wrong answer or fundamentally flawed reasoning)."
    )


def spatial_accuracy_rm_api(
    pred_text: str,
    gt_text: str,
    question_type: str = "multihop",
    client: "object | None" = None,
) -> float:
    """Call an OpenAI-compatible API to score spatial/VQA accuracy.

    Used for complex CLEVR questions (multihop, compare, spatial_existence,
    spatial_count) where rule-based accuracy RM is insufficient.

    Args:
        pred_text: Model-generated text.
        gt_text: Ground truth text.
        question_type: CLEVR question type string.
        client: Optional pre-built OpenAI client.

    Returns:
        Score in {0.0, 0.5, 1.0}. Falls back to rule-based scoring on failure.
    """
    cfg = _load_api_config()
    if not cfg:
        # Fallback: simple answer match
        from .reward.accuracy_rm import process_reward
        proc = process_reward(pred_text, gt_text, task_type="box")
        return 1.0 if proc.get("answer_correct", False) else 0.0

    try:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

        prompt = _build_spatial_judge_prompt(pred_text, gt_text, question_type)
        retries = cfg["max_retries"]
        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                    timeout=cfg["timeout"],
                )
                content = resp.choices[0].message.content
                if not content:
                    reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
                    if reasoning:
                        content = reasoning
                return _parse_score(content)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                break

        import logging
        logging.getLogger(__name__).warning(
            f"Spatial accuracy RM API failed: {last_error}. Falling back to rule-based."
        )
        from .reward.accuracy_rm import process_reward
        proc = process_reward(pred_text, gt_text, task_type="box")
        return 1.0 if proc.get("answer_correct", False) else 0.0
    except Exception:
        from .reward.accuracy_rm import process_reward
        proc = process_reward(pred_text, gt_text, task_type="box")
        return 1.0 if proc.get("answer_correct", False) else 0.0
