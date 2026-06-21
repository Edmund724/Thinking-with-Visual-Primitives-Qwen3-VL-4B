#!/usr/bin/env python3
"""Quick inference test for stage3a SFT Box Expert."""

import os
import sys
import json
import torch
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.models.qwen_vl_loader import load_qlora_model
from src.utils.conversation_builder import ConversationBuilder
from src.data.datasets.image_loader import load_image
from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
)

MODEL_PATH = "outputs/stage3a_sft_box"
COCO_IMAGE_DIR = "data/coco/train2017"
COCO_ANN_FILE = "data/coco/annotations/instances_train2017.json"

print("=" * 60)
print("Loading stage3a model...")
model, processor = load_qlora_model(
    model_name=MODEL_PATH,
    lora_r=256,
    lora_alpha=512,
)
model.eval()
print("Model loaded.\n")

# Generate test samples
print("Generating test samples...")
box_samples = generate_coco_box_samples(
    image_dir=COCO_IMAGE_DIR,
    ann_file=COCO_ANN_FILE,
    num_samples=4,
)
counting_samples = generate_coco_counting_samples(
    image_dir=COCO_IMAGE_DIR,
    ann_file=COCO_ANN_FILE,
    num_samples=4,
)
test_samples = box_samples[:4] + counting_samples[:4]
print(f"Got {len(test_samples)} samples.\n")

# Run inference
cb = ConversationBuilder("sft")

for i, sample in enumerate(test_samples):
    print("=" * 60)
    print(f"[Sample {i+1}] type={sample.get('task_type', '?')}")
    print(f"  Prompt: {sample['prompt']}")
    print(f"  Expected answer: {sample['answer']}")
    if sample.get('reasoning'):
        print(f"  Expected reasoning: {sample['reasoning'][:200]}...")

    # Load image
    try:
        image = load_image(sample.get("image"))
    except Exception as e:
        print(f"  SKIP: cannot load image: {e}")
        continue

    # Build messages
    messages = cb.build_prompt(sample["prompt"], image)

    # Apply chat template
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = processor(
        text=[prompt_text],
        images=[image] if image is not None else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    completion = processor.batch_decode(
        outputs[:, input_len:], skip_special_tokens=False
    )[0]

    print(f"  Completion ({len(completion)} chars):")
    print(f"  ---BEGIN---")
    print(completion)
    print(f"  ---END---")
    print()

print("=" * 60)
print("Done.")