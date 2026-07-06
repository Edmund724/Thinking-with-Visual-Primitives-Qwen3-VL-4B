#!/usr/bin/env python3
"""Memory profile for a single OPD batch with long generation.

Loads the Stage 5 student + Stage 4a box expert, constructs one image batch,
and runs generate + student forward + expert forward + backward with
``max_new_tokens=512``.  Memory (allocated/reserved) is printed after each
phase so we can see where VRAM grows.
"""

import os
import sys
import warnings
from pathlib import Path

from PIL import Image

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from src.models.qwen_vl_loader import load_qlora_model
from src.training.opd_trainer import OPDDataset, _opd_collate
from torch.utils.data import DataLoader


def mem(prefix: str):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[MEM] {prefix}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0

    student_path = "outputs/stage5_rft_unified/final_model"
    expert_path = "outputs/stage4a_grpo_box"
    if not os.path.isdir(student_path) or not os.path.isdir(expert_path):
        print(f"Missing checkpoint: {student_path} or {expert_path}")
        return 1

    # Capture warnings so we can still verify no checkpointing/use_cache spam.
    target_substrings = [
        "use_cache=True` is incompatible with gradient checkpointing",
        "use_reentrant parameter should be passed explicitly",
        "None of the inputs have requires_grad=True",
        "Caching is incompatible with gradient checkpointing",
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        torch.cuda.empty_cache()
        mem("start")

        print("Loading student...")
        student_model, processor = load_qlora_model(model_name=student_path)
        student_model.train()
        mem("student loaded")

        print("Loading expert...")
        expert, _ = load_qlora_model(model_name=expert_path)
        for p in expert.parameters():
            p.requires_grad = False
        expert.eval()
        if getattr(expert, "is_gradient_checkpointing", False):
            expert.gradient_checkpointing_disable()
        expert.to(student_model.device)
        mem("expert loaded")

        pad_token_id = processor.tokenizer.pad_token_id or 0
        eos_token_id = processor.tokenizer.eos_token_id

        dummy_image = Image.new("RGB", (512, 512), color=(73, 109, 137))
        sample = {
            "prompt": "Describe what you see in the image in detail.",
            "image": dummy_image,
            "task_type": "box",
        }
        dataset = OPDDataset(data=[sample], processor=processor)
        loader = DataLoader(dataset, batch_size=1, collate_fn=_opd_collate)
        batch = next(iter(loader))

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

        print("Generating 512 tokens (student, on-policy)...")
        from src.training.opd_trainer import _no_gradient_checkpointing
        with torch.no_grad(), _no_gradient_checkpointing(student_model):
            generated = student_model.generate(
                input_ids=prompt_ids,
                max_new_tokens=512,
                temperature=1.0,
                do_sample=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **image_kwargs,
            )
        mem("after generate")

        full_ids = generated[0]
        full_len = full_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        print(f"  prompt_len={prompt_len}  generated_len={full_len - prompt_len}  full_len={full_len}")

        if "mm_token_type_ids" in image_kwargs and full_len > prompt_len:
            mm = image_kwargs["mm_token_type_ids"]
            pad_len = full_len - prompt_len
            mm_padding = torch.zeros(1, pad_len, dtype=mm.dtype, device=mm.device)
            image_kwargs["mm_token_type_ids"] = torch.cat([mm, mm_padding], dim=1)

        print("Student forward + backward...")
        student_outputs = student_model(
            input_ids=full_ids.unsqueeze(0),
            use_cache=False,
            **image_kwargs,
        )
        mem("after student forward")
        loss = student_outputs.logits[:, :-1, :].float().mean()
        loss.backward()
        mem("after student backward")

        print("Expert forward...")
        with torch.no_grad():
            expert_outputs = expert(
                input_ids=full_ids.unsqueeze(0),
                use_cache=False,
                **image_kwargs,
            )
        mem("after expert forward")
        _ = expert_outputs.logits[:, :-1, :]

    bad = []
    for w in caught:
        msg = str(w.message)
        for substr in target_substrings:
            if substr in msg:
                bad.append((w.category.__name__, msg))
                break
    if bad:
        print(f"\nFAIL: {len(bad)} target warning(s):")
        for cat, msg in bad:
            print(f"  [{cat}] {msg}")
        return 1

    print("\nPASS: memory profile completed without target warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
