"""LLM-as-Judge Quality Reward Model via OpenAI-compatible API.

Falls back to the rule-based quality_reward_text if the API is unavailable,
times out, or returns an unparseable response.
"""

import os
import re
import time
from typing import Any, List

from .metrics import quality_reward_text


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
    }


def _build_judge_prompt(pred_text: str, gt_text: str, task_type: str = "box") -> str:
    """Build a strict, parseable prompt for the LLM judge."""
    return (
        "You are a quality judge for visual reasoning outputs. "
        "Score the model's response based on the ground truth.\n\n"
        "Evaluation criteria (paper-style Quality RM):\n"
        "- 1.0: No quality issues (correct, consistent, no redundancy, no contradictions).\n"
        "- 0.5: Minor issues (small redundancy, weak consistency, minor contradictions).\n"
        "- 0.0: Serious issues (reward hacking, self-contradiction, copied ground truth, "
        "wrong answer despite claiming correctness).\n\n"
        f"Task type: {task_type}\n\n"
        "--- Ground Truth ---\n"
        f"{gt_text}\n\n"
        "--- Model Prediction ---\n"
        f"{pred_text}\n\n"
        "Respond with ONLY one of: 1.0, 0.5, or 0.0. No explanation."
    )


def _parse_score(response: str) -> float:
    """Parse a discrete score from the judge response."""
    response = response.strip()
    # Direct match
    if response in ("1.0", "1", "1.00"):
        return 1.0
    if response in ("0.5", ".5", "0.50"):
        return 0.5
    if response in ("0.0", "0", "0.00"):
        return 0.0

    # Extract first occurrence of 1.0 / 0.5 / 0.0 in the text
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
                    max_tokens=10,
                    timeout=cfg["timeout"],
                )
                content = resp.choices[0].message.content
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

    Signature matches make_quality_reward_fn so it can be swapped in directly.
    """

    def quality_reward(completions, prompts=None, **kwargs):
        inputs = kwargs.get("inputs", [])
        gt_texts = kwargs.get("gt_text", [])
        task_types = kwargs.get("task_type", [])
        completion_ids_list = kwargs.get("completion_ids", [])

        # Pre-build client once per reward call to avoid repeated instantiation.
        cfg = _load_api_config()
        client = None
        if cfg:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
            except Exception:
                client = None

        from .metrics import extract_completion_text

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
                rewards.append(quality_reward_api(pred_text, gt_text, task_type, client=client))
            except Exception:
                rewards.append(0.0)
        return rewards

    return quality_reward
