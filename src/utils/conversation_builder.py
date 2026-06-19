"""Unified chat-message builder for SFT, GRPO, inference, and pretrain.

All message-construction logic lives here so that datasets, eval scripts, and
training loops share one canonical implementation.  The builder is configured
via a mode string (``"sft"``, ``"grpo"``, ``"opd"``, ``"pretrain"``) which
sets the system message; a custom system message can also be supplied.
"""

from __future__ import annotations

from typing import Any

from .constants import THINK_CLOSE as _THINK_CLOSE, THINK_OPEN as _THINK_OPEN


# ── system messages ──────────────────────────────────────────────────────
_SYSTEM_MESSAGES: dict[str, str] = {
    "sft": (
        "You are a helpful visual reasoning assistant. "
        "Think step by step and use visual primitives when needed. "
        "Respond in English only; do not use characters from other languages."
    ),
    "grpo": (
        "You are a helpful visual reasoning assistant. "
        "Think step by step inside <think>...</think> tags, then provide your final answer. "
        "Use visual primitives (<|box|>, <|point|>) when needed to mark locations. "
        "Always close your reasoning with </think> before giving the answer. "
        "Respond in English only; do not use characters from other languages."
    ),
    "opd": "You are a helpful visual reasoning assistant. Think step by step.",
    "pretrain": (
        "You are a helpful visual reasoning assistant. "
        "Think step by step and use visual primitives when needed. "
        "Respond in English only."
    ),
}


class ConversationBuilder:
    """Build conversation messages for SFT, GRPO, inference, and pretrain.

    Parameters
    ----------
    mode:
        Preset mode: ``"sft"``, ``"grpo"``, ``"opd"``, or ``"pretrain"``.
    system_message:
        Override the preset system message.  When ``None`` the default for
        *mode* is used.

    Examples
    --------
    >>> cb = ConversationBuilder("grpo")
    >>> prompt = cb.build_prompt("Where is the cat?", image=img)
    >>> gt = cb.build_gt_text("The cat is at <|box|>[[100,200,300,400]]<|/box|>", "box found")
    """

    __slots__ = ("_system",)

    def __init__(self, mode: str = "sft", system_message: str | None = None) -> None:
        if system_message is not None:
            self._system = system_message
        elif mode in _SYSTEM_MESSAGES:
            self._system = _SYSTEM_MESSAGES[mode]
        else:
            raise ValueError(
                f"Unknown mode {mode!r}.  Expected one of {list(_SYSTEM_MESSAGES)}."
            )

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _build_user_content(
        prompt: str,
        image: Any = None,
    ) -> str | list[dict[str, Any]]:
        """Build user-content block — multimodal when *image* is provided."""
        if image is not None:
            return [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        return prompt

    # ── public API ──────────────────────────────────────────────────────

    @property
    def system_message(self) -> str:
        """The resolved system message string."""
        return self._system

    def build_prompt(
        self,
        prompt: str,
        image: Any = None,
    ) -> list[dict[str, Any]]:
        """Build ``[system, user]`` messages for GRPO / inference / OPD / eval.

        This is the simplest form — a system message followed by a single
        user turn.  It is used by **GRPODataset**, **batch_inference**,
        **OPDDataset**, eval scripts, and Stage 5 expert rollouts.
        """
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": self._build_user_content(prompt, image)},
        ]

    def build_sft(
        self,
        sample: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build ``(prompt_messages, full_messages)`` for SFT training.

        *prompt_messages* are used for loss masking (everything before the
        assistant turn is masked).  *full_messages* include the assistant
        turn with ``reasoning_content`` for Qwen3-Thinking.

        The raw reasoning text is stripped of any existing
        ``<think>`` / ``</think>`` wraps because the Qwen3 chat template
        already injects them around ``reasoning_content``.
        """
        user_content = self._build_user_content(
            sample["prompt"],
            sample.get("image"),
        )

        prompt_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user_content},
        ]

        # Strip <think> / </think> so the chat template doesn't double-wrap.
        reasoning: str = sample.get("reasoning", "")
        if reasoning.startswith(_THINK_OPEN):
            reasoning = reasoning[len(_THINK_OPEN):].lstrip("\n")
        if reasoning.endswith(_THINK_CLOSE):
            reasoning = reasoning[: -len(_THINK_CLOSE)].rstrip()

        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": str(sample.get("answer", "")),
                "reasoning_content": reasoning,
            },
        ]

        return prompt_messages, full_messages

    def build_pretrain(
        self,
        user_text: str,
        assistant_text: str,
    ) -> list[dict[str, Any]]:
        """Build a complete pretrain conversation ``[system, user, assistant]``.

        The *assistant_text* is placed directly into the assistant content
        field, typically with literal ``<think>...</think>`` tags.
        """
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

    @staticmethod
    def build_gt_text(reasoning: str, answer: str) -> str:
        """Build ground-truth text for GRPO reward computation.

        Format: ``reasoning + "\\n</think>\\n\\nThe answer is " + answer + "."``
        """
        return reasoning + "\n</think>\n\nThe answer is " + answer + "."
