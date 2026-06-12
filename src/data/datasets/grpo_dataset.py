"""GRPO Dataset for multimodal visual primitive training.

Each item returns:
    prompt: Qwen3 conversational messages with image embedded in content blocks
    gt_text: Ground truth text for reward computation
    task_type: "box", "point", or "maze"
    maze_grid: Optional maze grid for collision detection
"""

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .image_loader import load_image


def _build_gt_text(sample_or_reasoning, answer: str = "") -> str:
    """Build ground-truth text from a sample dict or (reasoning, answer) pair.

    Args:
        sample_or_reasoning: Either a dict with "reasoning" and "answer" keys,
            or a reasoning string (when used with the answer parameter).
        answer: The answer string (ignored if sample_or_reasoning is a dict).
    """
    if isinstance(sample_or_reasoning, dict):
        reasoning = sample_or_reasoning.get("reasoning", "")
        answer = sample_or_reasoning.get("answer", "")
    else:
        reasoning = sample_or_reasoning
    return reasoning + "\n" + "\x3c/think\x3e" + "\n\nThe answer is " + answer + "."


class GRPODataset(Dataset):
    """Dataset for GRPO training with multimodal inputs.

    TRL 1.5.1 GRPOTrainer expects VLM images to be embedded in the message
    content as ``{"type": "image", "image": <PIL>}`` blocks.  The standalone
    ``"image"`` key is NOT consumed by the trainer.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        image = load_image(sample.get("image"))

        system_content = (
            "You are a helpful visual reasoning assistant. "
            "Think step by step inside " + "\x3cthink\x3e" + "..." + "\x3c/think\x3e" + " tags, then provide your final answer. "
            "Use visual primitives (<|box|>, <|point|>) when needed to mark locations. "
            "Always close your reasoning with " + "\x3c/think\x3e" + " before giving the answer."
        )

        if image is not None:
            # Embed image in message content as multimodal blocks (TRL 1.5+ format)
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ]
        else:
            # Text-only (e.g., some maze samples without images)
            user_content = sample["prompt"]

        return {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "gt_text": _build_gt_text(
                sample["reasoning"],
                sample.get("answer", ""),
            ),
            "task_type": sample.get("task_type", "box"),
            "maze_grid": sample.get("maze_grid"),
        }
