#!/usr/bin/env python3
"""Phase 4: Compare greedy vs sampling on an image WITH persons."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

import torch
from src.models.qwen_vl_loader import load_qlora_model
from src.utils.conversation_builder import ConversationBuilder
from src.data.datasets.image_loader import load_image

MODEL_PATH = "outputs/stage3a_sft_box"

print("Loading model...")
model, processor = load_qlora_model(model_name=MODEL_PATH, lora_r=256, lora_alpha=512)
model.eval()

sample = {
    "prompt": "How many persons are in this image? Use <|box|> to mark each one.",
    "image": "data/coco/train2017/000000262145.jpg",
}
print(f"Prompt: {sample['prompt']}")
print(f"Image: {sample['image']}")

image = load_image(sample["image"])
cb = ConversationBuilder("sft")
messages = cb.build_prompt(sample["prompt"], image)
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Test 1: Greedy
print("\n=== Test 1: Greedy (temperature=0) ===")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
input_len = inputs["input_ids"].shape[1]
completion = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=False)[0]
print(completion)

# Test 2: Sampling temp=0.7
print("\n=== Test 2: Sampling (temperature=0.7) ===")
torch.manual_seed(42)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
completion = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=False)[0]
print(completion)

# Test 3: Sampling temp=0.3 (lower noise)
print("\n=== Test 3: Sampling (temperature=0.3) ===")
torch.manual_seed(42)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
completion = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=False)[0]
print(completion)

# Test 4: Check special token logits at the point where <|ref|> should appear
# The model should output "Grounding: <|ref|>persons<|/ref|><|box|>..."
# Let's check logits right after "Grounding: "
print("\n=== Test 4: Logit analysis at 'Grounding: ' position ===")
# Force the model to output the prefix up to "Grounding: "
prefix = "Intent Analysis: I need to count all persons in the image.\nGrounding: "
prefix_ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
print(f"Prefix tokens: {len(prefix_ids)}")

# Get the full input + prefix
full_ids = torch.cat([inputs["input_ids"], torch.tensor([prefix_ids], device=model.device)], dim=1)
with torch.inference_mode():
    out = model(input_ids=full_ids, pixel_values=inputs.get("pixel_values"), image_grid_thw=inputs.get("image_grid_thw"))
    next_logits = out.logits[0, -1, :]

special_tokens = ["<|ref|>", "<|/ref|>", "<|box|>", "<|/box|>"]
special_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]

print("Top-10 tokens after 'Grounding: ':")
topk = torch.topk(next_logits, 10)
for i, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
    tok = processor.tokenizer.decode([idx])
    print(f"  {i+1}. id={idx} logit={val:.2f} '{tok}'")

print("\nSpecial token logits at this position:")
for tok, sid in zip(special_tokens, special_ids):
    logit_val = next_logits[sid].item()
    rank = (next_logits > logit_val).sum().item() + 1
    print(f"  '{tok}': logit={logit_val:.2f}, rank={rank}/{len(next_logits)}")