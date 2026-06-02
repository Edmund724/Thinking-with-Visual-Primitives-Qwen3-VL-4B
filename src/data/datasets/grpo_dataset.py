"""GRPO Dataset for multimodal visual primitive training.

Each item returns:
    prompt: Qwen3 conversational messages (system + user with image)
    image: PIL Image
    gt_text: Ground truth text for reward computation
    task_type: "box", "point", or "maze"
    maze_grid: Optional maze grid for collision detection
"""

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class GRPODataset(Dataset):
    """Dataset for GRPO training with multimodal inputs.

    TRL's GRPOTrainer expects items with:
        - "prompt": conversational messages list
        - "image": PIL image (or "images": list of PIL images)
        - Additional metadata fields for reward function
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        return {
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful visual reasoning assistant. "
                        "Think step by step and use visual primitives when needed."
                    ),
                },
                {
                    "role": "user",
                    "content": sample["prompt"],
                },
            ],
            "image": sample["image"],
            "gt_text": (
                sample["reasoning"]
                + "\n</think>\n\nThe answer is "
                + sample.get("answer", "")
                + "."
            ),
            "task_type": sample.get("task_type", "box"),
            "maze_grid": sample.get("maze_grid"),
        }
