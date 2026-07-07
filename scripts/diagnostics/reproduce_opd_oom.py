#!/usr/bin/env python3
"""Reproduce Stage 6 OPD VRAM blow-up in a minimal, instrumented loop.

This script mirrors train_opd_parallel but on a single batch, including the
optimizer.  It is designed to go red (OOM or >24 GB peak) when the student is
loaded with the default modules_to_save (embed_tokens + lm_head), and green
once those full parameters are frozen so only LoRA remains trainable.
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
from src.training.memory_utils import clear_memory
from src.training.opd_trainer import OPDDataset, _opd_collate, _freeze_opd_embeddings, _no_gradient_checkpointing
from torch.utils.data import DataLoader


def mem(prefix: str):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[MEM] {prefix}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def _make_optimizer(trainable, use_8bit: bool):
    if use_8bit:
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(trainable, lr=1e-6, weight_decay=0.0)
    return torch.optim.AdamW(trainable, lr=1e-6, weight_decay=0.0)


def run_batch(student_model, expert, processor, freeze_embeddings: bool, use_8bit: bool):
    """Run one OPD batch and return peak allocated GB."""
    if freeze_embeddings:
        frozen = _freeze_opd_embeddings(student_model)
        print(f"  frozen embed_tokens/lm_head params: {frozen:,}")

    trainable = [p for p in student_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable params: {n_trainable:,}")

    optimizer = _make_optimizer(trainable, use_8bit)
    # Force optimizer state allocation.
    for p in trainable:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad()
    mem("after optimizer init")

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
    if "mm_token_type_ids" in image_kwargs and full_len > prompt_len:
        mm = image_kwargs["mm_token_type_ids"]
        pad_len = full_len - prompt_len
        image_kwargs["mm_token_type_ids"] = torch.cat(
            [mm, torch.zeros(1, pad_len, dtype=mm.dtype, device=mm.device)], dim=1
        )

    student_model.train()
    student_outputs = student_model(
        input_ids=full_ids.unsqueeze(0),
        use_cache=False,
        **image_kwargs,
    )
    mem("after student forward")
    loss = student_outputs.logits[:, :-1, :].float().mean()
    loss.backward()
    mem("after student backward")

    with torch.no_grad():
        expert_outputs = expert(
            input_ids=full_ids.unsqueeze(0),
            use_cache=False,
            **image_kwargs,
        )
    mem("after expert forward")

    torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.3)
    optimizer.step()
    optimizer.zero_grad()
    mem("after optimizer step")

    peak = torch.cuda.max_memory_allocated() / 1e9
    return peak


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0

    student_path = "outputs/stage5_rft_unified/final_model"
    expert_path = "outputs/stage4a_grpo_box"
    if not os.path.isdir(student_path) or not os.path.isdir(expert_path):
        print(f"Missing checkpoint: {student_path} or {expert_path}")
        return 1

    target_substrings = [
        "use_cache=True` is incompatible with gradient checkpointing",
        "use_reentrant parameter should be passed explicitly",
        "None of the inputs have requires_grad=True",
        "Caching is incompatible with gradient checkpointing",
    ]

    # Default (current) path: full embed_tokens + lm_head trainable.
    print("\n=== Baseline: embed_tokens + lm_head trainable, fp32 AdamW ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem("start")

        student_model, processor = load_qlora_model(model_name=student_path)
        mem("student loaded")

        expert, _ = load_qlora_model(model_name=expert_path)
        for p in expert.parameters():
            p.requires_grad = False
        expert.eval()
        if getattr(expert, "is_gradient_checkpointing", False):
            expert.gradient_checkpointing_disable()
        expert.to(student_model.device)
        mem("expert loaded")

        peak = run_batch(student_model, expert, processor, freeze_embeddings=False, use_8bit=False)
        print(f"  peak allocated: {peak:.2f}GB")

    bad = []
    for w in caught:
        msg = str(w.message)
        for substr in target_substrings:
            if substr in msg:
                bad.append((w.category.__name__, msg))
                break
    if bad:
        print(f"  WARNINGS: {len(bad)}")
        for cat, msg in bad:
            print(f"    [{cat}] {msg}")

    del student_model, expert
    clear_memory()

    # Fix path: freeze embed_tokens + lm_head, use 8-bit AdamW.
    print("\n=== With fix: only LoRA trainable, 8-bit AdamW ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem("start")

        student_model, processor = load_qlora_model(model_name=student_path)
        mem("student loaded")

        expert, _ = load_qlora_model(model_name=expert_path)
        for p in expert.parameters():
            p.requires_grad = False
        expert.eval()
        if getattr(expert, "is_gradient_checkpointing", False):
            expert.gradient_checkpointing_disable()
        expert.to(student_model.device)
        mem("expert loaded")

        peak = run_batch(student_model, expert, processor, freeze_embeddings=True, use_8bit=True)
        print(f"  peak allocated: {peak:.2f}GB")

    bad = []
    for w in caught:
        msg = str(w.message)
        for substr in target_substrings:
            if substr in msg:
                bad.append((w.category.__name__, msg))
                break
    if bad:
        print(f"  WARNINGS: {len(bad)}")
        for cat, msg in bad:
            print(f"    [{cat}] {msg}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
