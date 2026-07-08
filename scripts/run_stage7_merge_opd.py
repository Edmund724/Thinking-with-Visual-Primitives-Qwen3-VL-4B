#!/usr/bin/env python3
"""Merge Stage 6 OPD LoRA adapter into the merged base model.

After Stage 6 OPD, the final LoRA weights are merged back into the Stage 2
merged base model so that the final checkpoint is a standalone full bf16 model
(no adapter loading required at inference time).

Usage:
    python scripts/run_stage7_merge_opd.py \
        --base_model outputs/stage2_merged_base \
        --adapter_path outputs/stage6_opd \
        --output_dir outputs/stage7_opd_merged
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from pathlib import Path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.utils.config_utils import load_yaml_config
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/run_stage7_merge_opd.log")

_OPD_README_MD = """# TVP-OPD-Qwen3-VL-4B

## 模型概述

本模型是 [Thinking with Visual Primitives — Qwen3-VL-4B Reproduction](https://github.com/Edmund724/Thinking-with-Visual-Primitives-Qwen3-VL-4B) 项目在 **OPD 阶段结束并合并后的产物**，即最终的 **OPD（On-Policy Distillation）统一模型**。

该模型以 Stage 5 的 RFT Unified 模型为学生，分别以 Stage 4a 的 Box GRPO 专家和 Stage 4b 的 Point GRPO 专家为教师，通过反向 KL 散度（`D_KL(student || expert)`）将两种专家的能力蒸馏到单一统一模型中。最后通过 Stage 7 将 OPD LoRA adapter 合并回 Stage 2 预训练基座，得到可独立加载的完整模型。这是本项目训练流程的最后一个阶段。

> ⚠️ **声明**：本项目为论文的非官方复现，受限于模型规模和数据量，效果有限，主要用于验证训练流程跑通，不建议直接用于生产环境。
>
> 注意：本仓库发布的是 **合并后的完整 bf16 模型**（Stage 6 OPD 的 LoRA adapter 合并到 Stage 2 预训练基座后的产物），可直接用 `Qwen3VLForConditionalGeneration.from_pretrained(...)` 或项目提供的 `load_qlora_model(...)` 加载。

## 模型介绍

| 属性 | 说明 |
|------|------|
| 基座模型 | Qwen3-VL-4B-Thinking |
| 初始化权重 | Stage 5 RFT Unified 模型 |
| 模型类型 | 多模态理解 + 视觉定位（Visual Grounding） |
| 训练框架 | PyTorch |
| 训练技术 | QLoRA + On-Policy Distillation（reverse KL） |
| 能力范围 | Box + Point 统一 |
| 教师模型 | Stage 4a GRPO Box Expert + Stage 4b GRPO Point Expert |

本阶段对应论文中的 **On-Policy Distillation** 阶段，目标是在统一模型上逼近专家模型的输出分布。最终通过 Stage 7 将 OPD adapter 合并回基座，得到可独立加载的完整模型。

## 模型训练

### 训练流程

```
Stage 5: RFT Unified → Stage 6: OPD Distillation → Stage 7: Merge OPD LoRA（本模型）
```

### Stage 6 训练配置

| 配置项 | 数值 |
|--------|------|
| 学生模型 | Stage 5 RFT Unified |
| Box 教师 | Stage 4a GRPO Box Expert |
| Point 教师 | Stage 4b GRPO Point Expert |
| 数据 | 3K box + 2K point + 2K maze |
| 训练轮数 | 2 epochs |
| 学习率 | 1e-6 |
| batch size | 1 |
| max_new_tokens | 512 |
| temperature | 1.0 |
| LoRA r | 256 |
| LoRA alpha | 512 |
| 优化器 | 8-bit AdamW |
| embed_tokens / lm_head | 冻结 |

训练时 GPU 中只保留一个学生模型和一个教师模型，教师按阶段轮换加载，以控制显存占用。

### Stage 7 合并

```bash
python scripts/run_stage7_merge_opd.py \\
    --base_model outputs/stage2_merged_base \\
    --adapter_path outputs/stage6_opd \\
    --output_dir outputs/stage7_opd_merged
```

## 模型推理

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

model_path = "your-modelscope-repo/TVP-OPD-Qwen3-VL-4B"
model, processor = load_qlora_model(model_path)

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

outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
response = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

## 模型效果

本模型作为最终 OPD 统一模型，预期表现：

- 同时具备 box grounding 和 point/path tracing 能力
- 输出结构遵循 "意图分析 → 视觉 grounding → 总结" 三段式
- 相比 RFT Unified，在两类任务上的综合表现更接近专家模型

> 由于训练数据规模和模型规模较小，复杂场景下仍可能出现定位偏差、格式错误或多任务冲突。

## 相关链接

- 项目仓库：[https://github.com/Edmund724/Thinking-with-Visual-Primitives-Qwen3-VL-4B](https://github.com/Edmund724/Thinking-with-Visual-Primitives-Qwen3-VL-4B)
- 论文：*Thinking with Visual Primitives* (DeepSeek, 2026)
"""


def _resolve_adapter_path(adapter_path: str) -> str:
    """Resolve a stage output dir to its final adapter checkpoint.

    If adapter_path itself contains adapter_config.json, use it directly.
    Otherwise, look for the latest checkpoint-*/ directory that contains
    an adapter, which is the case for outputs/stage6_opd/.
    """
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        return adapter_path

    candidates = []
    for name in os.listdir(adapter_path):
        if not name.startswith("checkpoint-"):
            continue
        ckpt_path = os.path.join(adapter_path, name)
        if os.path.isdir(ckpt_path) and os.path.exists(
            os.path.join(ckpt_path, "adapter_config.json")
        ):
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            candidates.append((step, ckpt_path))

    if not candidates:
        raise FileNotFoundError(
            f"No adapter checkpoint found under {adapter_path}"
        )

    candidates.sort()
    return candidates[-1][1]


def main(args):
    logger.info("=" * 60)
    logger.info("Merging Stage 6 OPD LoRA into merged base model")
    logger.info("=" * 60)

    args.base_model = os.path.abspath(args.base_model)
    args.adapter_path = os.path.abspath(args.adapter_path)
    args.output_dir = os.path.abspath(args.output_dir)

    adapter_ckpt = _resolve_adapter_path(args.adapter_path)
    logger.info(f"Resolved adapter checkpoint: {adapter_ckpt}")

    logger.info(f"Loading base model: {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer from adapter (has special tokens) and align with base model
    processor = AutoProcessor.from_pretrained(
        adapter_ckpt,
        trust_remote_code=True,
    )

    current_embed_size = model.get_input_embeddings().num_embeddings
    new_tokenizer_len = len(processor.tokenizer)
    if new_tokenizer_len > current_embed_size:
        model.resize_token_embeddings(new_tokenizer_len)
        logger.info(f"Resized embeddings: {current_embed_size} → {new_tokenizer_len}")
    else:
        logger.info(
            f"No resize needed: embedding ({current_embed_size}) covers tokenizer ({new_tokenizer_len})"
        )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.bos_token_id = processor.tokenizer.bos_token_id
        model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

    logger.info(f"Loading adapter: {adapter_ckpt}")
    model = PeftModel.from_pretrained(model, adapter_ckpt)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(_OPD_README_MD)
    logger.info(f"Wrote README to {readme_path}")

    logger.info(f"Merge complete. Final merged model saved to {args.output_dir}")
    logger.info("Next: upload outputs/stage7_opd_merged as the final standalone model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Stage 6 OPD LoRA into merged base")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    yaml_cfg = load_yaml_config("configs/stage7_merge_opd.yaml")
    for key, value in yaml_cfg.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)

    main(args)
