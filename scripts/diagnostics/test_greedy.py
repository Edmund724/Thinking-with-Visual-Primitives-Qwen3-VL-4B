#!/usr/bin/env python3
"""Phase 4: Instrument — test greedy decoding (temperature=0) to isolate the issue."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

import torch
from src.models.qwen_vl_loader import load_qlora_model
from src.utils.conversation_builder import ConversationBuilder
from src.data.datasets.image_loader import load_image
from src.data.generators.coco_box_generator import generate_coco_box_samples

MODEL_PATH = "outputs/stage3a_sft_box"
COCO_IMAGE_DIR = "data/coco/train2017"
COCO_ANN_FILE = "data/coco/annotations/instances_train2017.json"

print("Loading model...")
model, processor = load_qlora_model(model_name=MODEL_PATH, lora_r=256, lora_alpha=512)
model.eval()

# Fixed test sample for reproducibility
sample = {
    "prompt": "How many persons are in this image? Use <|box|> to mark each one.",
    "image": "data/coco/train2017/000000000009.jpg",
    "task_type": "box",
}
print(f"Prompt: {sample['prompt']}")
print(f"Image: {sample['image']}")

image = load_image(sample["image"])
cb = ConversationBuilder("sft")
messages = cb.build_prompt(sample["prompt"], image)

# Test 1: Greedy decoding (temperature=0)
print("\n=== Test 1: Greedy (temperature=0) ===")
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,  # greedy
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
input_len = inputs["input_ids"].shape[1]
completion = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=False)[0]
print(f"Output:\n{completion}")

# Test 2: Check logits for special tokens on a simple prompt
print("\n=== Test 2: Special token logit analysis ===")
special_tokens = ["<|box|>", "<|/box|>", "<|ref|>", "<|/ref|>", "<|point|>", "<|/point|>"]
special_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]

# Get logits for the first generated token
with torch.inference_mode():
    out = model(**inputs)
    first_logits = out.logits[0, -1, :]  # logits for next token after prompt

# Show top-20 tokens
topk = torch.topk(first_logits, 20)
print("Top-20 predicted tokens:")
for i, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
    tok = processor.tokenizer.decode([idx])
    marker = " ★ SPECIAL" if idx in special_ids else ""
    print(f"  {i+1:2d}. id={idx:6d} logit={val:.2f} '{tok}'{marker}")

# Check where special tokens rank
print("\nSpecial token rankings:")
for tok, sid in zip(special_tokens, special_ids):
    logit_val = first_logits[sid].item()
    rank = (first_logits > logit_val).sum().item() + 1
    print(f"  '{tok}' (id={sid}): logit={logit_val:.2f}, rank={rank}/{len(first_logits)}")