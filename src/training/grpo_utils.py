"""Shared utilities for GRPO reward functions.

Handles the gap between TRL GRPOTrainer's internal representation and our
reward computation:
  - TRL 1.5+ wraps conversational completions as [{"role": "assistant", "content": ...}]
  - TRL decodes with skip_special_tokens=True, which strips <|box|>, <|/box|>, etc.
  - Our reward functions need the raw text WITH visual primitive tags intact.

Solution: re-decode from completion_ids with skip_special_tokens=False.
"""

from typing import Any, List


def extract_completion_text(
    completion: Any,
    tokenizer: Any = None,
    completion_id: List[int] | None = None,
) -> str:
    """Extract usable text from a TRL GRPOTrainer completion.

    Priority:
      1. Re-decode from completion_ids with skip_special_tokens=False
         (preserves <|box|>, <|point|>, etc.)
      2. Extract content from conversational dict format
      3. Return as-is if already a plain string

    Args:
        completion: One element from the completions list.
            May be a list of message dicts, a plain string, or a dict.
        tokenizer: Processor's tokenizer for re-decoding completion_ids.
        completion_id: Token IDs for this completion (from completion_ids kwarg).

    Returns:
        Decoded text with all special tokens preserved.
    """
    # Best path: re-decode from token IDs (preserves all special tokens)
    if completion_id is not None and tokenizer is not None:
        text = tokenizer.decode(completion_id, skip_special_tokens=False)
        # Strip EOS token that TRL includes at the end
        eos = getattr(tokenizer, "eos_token", None)
        if eos and text.endswith(eos):
            text = text[: -len(eos)]
        return text.strip()

    # Fallback: extract from conversational message list
    if isinstance(completion, list) and len(completion) > 0:
        msg = completion[0]
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
        return str(msg)

    # Already a string
    if isinstance(completion, str):
        return completion

    # Unknown format — best effort
    return str(completion)
