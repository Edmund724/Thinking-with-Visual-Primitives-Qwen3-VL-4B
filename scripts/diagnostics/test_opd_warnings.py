#!/usr/bin/env python3
"""Minimal end-to-end smoke test for OPD gradient-checkpointing warnings.

Loads the Stage 5 student, simulates one OPD step (generate + student forward
+ backward), and fails if any of the known gradient-checkpointing / use_cache
warnings are emitted.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

from PIL import Image

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn.functional as F

from src.models.qwen_vl_loader import load_qlora_model
from src.training.opd_trainer import OPDDataset, _opd_collate, _no_gradient_checkpointing
from torch.utils.data import DataLoader


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping GPU test.")
        return 0

    student_path = "outputs/stage5_rft_unified/final_model"
    if not os.path.isdir(student_path):
        print(f"Student checkpoint not found: {student_path}")
        return 1

    print("Loading student model...")
    student_model, processor = load_qlora_model(model_name=student_path)
    student_model.train()

    pad_token_id = processor.tokenizer.pad_token_id or 0
    eos_token_id = processor.tokenizer.eos_token_id

    dummy_image = Image.new("RGB", (128, 128), color=(73, 109, 137))
    sample = {
        "prompt": "What color is the square? Answer with one word.",
        "image": dummy_image,
        "task_type": "box",
    }

    dataset = OPDDataset(data=[sample], processor=processor)
    loader = DataLoader(dataset, batch_size=1, collate_fn=_opd_collate)
    batch = next(iter(loader))

    # Replicate _opd_single_batch input preparation
    prompt_inputs = {}
    for k, v in batch.items():
        if k in ("task_type", "prompt_text"):
            continue
        if k == "image_grid_thw":
            prompt_inputs[k] = v.to(student_model.device)
        else:
            prompt_inputs[k] = v[0].to(student_model.device)

    prompt_inputs["input_ids"] = prompt_inputs["input_ids"].unsqueeze(0)
    if "attention_mask" in prompt_inputs:
        prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"].unsqueeze(0)
    if "mm_token_type_ids" in prompt_inputs:
        prompt_inputs["mm_token_type_ids"] = prompt_inputs["mm_token_type_ids"].unsqueeze(0)

    image_kwargs = {
        k: v for k, v in prompt_inputs.items()
        if k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
    }
    prompt_ids = prompt_inputs["input_ids"]

    target_substrings = [
        "use_cache=True` is incompatible with gradient checkpointing",
        "use_reentrant parameter should be passed explicitly",
        "None of the inputs have requires_grad=True",
        "Caching is incompatible with gradient checkpointing",
    ]

    with tempfile.TemporaryDirectory():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            # 1. Generate (on-policy) - OPD disables gradient checkpointing for
            # generation, so mirror that here.
            with torch.no_grad(), _no_gradient_checkpointing(student_model):
                generated = student_model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=16,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    **image_kwargs,
                )
            full_ids = generated[0]

            # 2. Student forward + backward
            full_len = full_ids.shape[0]
            prompt_len = prompt_ids.shape[1]
            if "mm_token_type_ids" in image_kwargs and full_len > prompt_len:
                mm = image_kwargs["mm_token_type_ids"]
                pad_len = full_len - prompt_len
                mm_padding = torch.zeros(1, pad_len, dtype=mm.dtype, device=mm.device)
                image_kwargs["mm_token_type_ids"] = torch.cat([mm, mm_padding], dim=1)

            outputs = student_model(
                input_ids=full_ids.unsqueeze(0),
                labels=full_ids.unsqueeze(0),
                **image_kwargs,
            )
            # Dummy loss so we trigger the backward path through gradient checkpointing.
            loss = outputs.logits[:, :-1, :].float().mean()
            loss.backward()

    bad = []
    for w in caught:
        msg = str(w.message)
        for substr in target_substrings:
            if substr in msg:
                bad.append((w.category.__name__, msg))
                break

    if bad:
        print(f"\nFAIL: {len(bad)} target warning(s) emitted:")
        for cat, msg in bad:
            print(f"  [{cat}] {msg}")
        return 1

    print("\nPASS: no gradient-checkpointing / use_cache warnings emitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
