# Inference Guide

## Basic Inference

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

model, processor = load_qlora_model("outputs/stage6_opd")

image = Image.open("example.jpg")
messages = [
    {"role": "system", "content": "You are a helpful visual reasoning assistant. Think step by step."},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "How many cats are in the image? Mark each with a box."},
    ]},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
)
response = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

Expected output:

```text
<think>
I can see two cats in the image. Let me mark them.
<|box|>[[120, 80, 340, 290]]<|/box|>
<|box|>[[410, 95, 620, 310]]<|/box|>
</think>

The answer is 2.
```

## Batch Inference on JSONL

```python
import json
from pathlib import Path
from src.models.qwen_vl_loader import load_qlora_model
from src.utils.metrics import process_reward
from PIL import Image

model, processor = load_qlora_model("outputs/stage6_opd")

results = []
with open("eval_data.jsonl") as f:
    for line in f:
        item = json.loads(line)
        image = Image.open(item["image_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": "You are a helpful visual reasoning assistant. Think step by step."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": item["question"]},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
        pred = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)

        reward = process_reward(pred, item["answer"], task_type=item.get("task_type", "box"))
        results.append({"pred": pred, "reward": reward, "gt": item["answer"]})

with open("eval_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

## Compare Models Across Stages

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

stages = {
    "Stage 2 (Pretrain)": "outputs/stage2_merged_base",
    "Stage 3a (Box SFT)": "outputs/stage3a_sft_box",
    "Stage 6 (OPD)":      "outputs/stage6_opd",
}

image = Image.open("test_image.jpg")
prompt = "Locate the dog in the image."

for name, path in stages.items():
    model, processor = load_qlora_model(path)
    messages = [
        {"role": "system", "content": "You are a helpful visual reasoning assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    print(f"=== {name} ===")
    print(processor.tokenizer.decode(outputs[0], skip_special_tokens=False))
    print()
    del model  # release VRAM
```

Expected behavior:
- **Pretrain**: correct format, less accurate boxes.
- **SFT Box**: structured thinking + precise boxes.
- **OPD**: combined box and point capabilities.
