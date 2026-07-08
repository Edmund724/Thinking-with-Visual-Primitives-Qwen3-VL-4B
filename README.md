English | **[中文](README_zh.md)**

# Thinking with Visual Primitives — Qwen3-VL-4B Reproduction

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ModelScope](https://img.shields.io/badge/ModelScope-Collection-cyan)](https://modelscope.cn/collections/EdmundYY/Thinking-with-Visual-Primitives-Qwen3-VL-4B)

> Reproducing the core idea of DeepSeek's **"Thinking with Visual Primitives"** on a single **RTX 5090D (24GB)** using **Qwen3-VL-4B-Thinking**.

Instead of only "seeing clearer", the model learns to **point while it reasons** — interleaving spatial markers (boxes and points) directly into the Chain-of-Thought as **visual primitives**, closing the *Reference Gap* in complex visual reasoning.

---

## Highlights

- **Single-GPU friendly**: 4-bit QLoRA + Gradient Checkpointing + Paged AdamW 8-bit; peak VRAM ~18.5GB in OPD.
- **Lightweight specialists + distillation**: separated Box/Point experts are frozen after SFT and distilled into one unified model via On-Policy Distillation (OPD).
- **Full 6-stage pipeline**: Pretrain → Merge → Box SFT → Point SFT → Box GRPO → Point GRPO → Unified RFT → OPD.
- **Inline visual primitives**: `<|box|>[[x1,y1,x2,y2]]<|/box|>` and `<|point|>[[x,y]]<|/point|>` embedded in `<think>`.
- **ModelScope checkpoints**: all intermediate and final adapters are published (see table below).

---

## Quick Start

### 1. Install

```bash
conda create -n tvp python=3.12 -y
conda activate tvp

pip install torch>=2.11.0 torchvision>=0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# Optional: Flash Attention 2 (falls back to eager if compilation fails)
pip install flash-attn --no-build-isolation
```

### 2. Download Base Model

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Thinking --local-dir models/Qwen3-VL-4B-Thinking
```

### 3. Inference

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
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
print(processor.tokenizer.decode(outputs[0], skip_special_tokens=False))
```

For batch inference and cross-stage comparisons, see [docs/INFERENCE.md](docs/INFERENCE.md).

---

## Model Checkpoints

All checkpoints are available on ModelScope (China mainland friendly):

| Stage | ModelScope name | Type |
|-------|-----------------|------|
| Pretrain (merged) | [TVP-Pretrain-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-Pretrain-Qwen3-VL-4B) | Full bf16 model |
| SFT Box expert | [TVP-SFT-Box-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-SFT-Box-Qwen3-VL-4B) | LoRA adapter |
| SFT Point expert | [TVP-SFT-Point-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-SFT-Point-Qwen3-VL-4B) | LoRA adapter |
| GRPO Box expert | [TVP-GRPO-Box-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-GRPO-Box-Qwen3-VL-4B) | LoRA adapter |
| GRPO Point expert | [TVP-GRPO-Point-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-GRPO-Point-Qwen3-VL-4B) | LoRA adapter |
| Unified RFT | [TVP-RFT-Unified-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-RFT-Unified-Qwen3-VL-4B) | LoRA adapter |
| OPD (final adapter) | [TVP-OPD-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-OPD-Qwen3-VL-4B) | LoRA adapter |
| OPD (merged) | [TVP-OPD-Qwen3-VL-4B](https://modelscope.cn/models/EdmundYY/TVP-OPD-Qwen3-VL-4B) | Full bf16 model |

LoRA adapters are loaded on top of their respective base models; `load_qlora_model(repo_name)` handles this automatically by reading `adapter_config.json`.

**Collection**: [Thinking-with-Visual-Primitives-Qwen3-VL-4B](https://modelscope.cn/collections/EdmundYY/Thinking-with-Visual-Primitives-Qwen3-VL-4B)

> These weights are for research / pipeline verification only, trained on small-scale data with a single 24GB GPU. They are not recommended for production.

---

## Training Pipeline

Run the full pipeline with:

```bash
bash scripts/run_pipeline.sh
```

| Stage | Task | Key inputs | Output | Verified wall time |
|-------|------|------------|--------|--------------------|
| 1 | Unified Visual Pretrain | COCO + CLEVR | `outputs/stage1_visual_pretrain/` | ~7.4h |
| 2 | Merge LoRA | Stage 1 adapter | `outputs/stage2_merged_base/` | ~1m |
| 3a | Box Expert SFT | Box / counting / CLEVR | `outputs/stage3a_sft_box/` | ~13.4h |
| 3b | Point Expert SFT | Point / maze / path | `outputs/stage3b_sft_point/` | ~16h |
| 4a | Box Expert GRPO | Box prompts | `outputs/stage4a_grpo_box/` | ~20.1h |
| 4b | Point Expert GRPO | Point/maze/path prompts | `outputs/stage4b_grpo_point/` | ~36.4h |
| 5 | Unified RFT | Expert rollouts | `outputs/stage5_rft_unified/` | ~2.7h |
| 6 | OPD | Student + experts | `outputs/stage6_opd/` | ~7h |
| 7 | Merge OPD LoRA | Stage 6 adapter + Stage 2 base | `outputs/stage7_opd_merged/` | ~1m |

See [docs/TRAINING.md](docs/TRAINING.md) for per-stage commands, configs, resume, and memory tips.

---

## Project Structure

```
tvp-4b-5090d/
├── configs/                 # YAML configs for every stage
├── docs/                    # Detailed guides
│   ├── TRAINING.md
│   ├── ARCHITECTURE.md
│   ├── INFERENCE.md
│   ├── OPTIMIZATION.md
│   └── KNOWN_ISSUES.md
├── scripts/                 # Stage entry scripts + pipeline
│   ├── run_stage1_visual_pretrain.py
│   ├── run_stage2_merge.py
│   ├── run_stage3a_sft_box.py
│   ├── run_stage3b_sft_point.py
│   ├── run_stage4a_grpo_box.py
│   ├── run_stage4b_grpo_point.py
│   ├── run_stage5_rft_unified.py
│   ├── run_stage6_opd.py
│   └── run_pipeline.sh
├── src/                     # Data, models, training, utils
├── tests/                   # Unit + integration tests
├── outputs/                 # Training artifacts (per stage)
├── data/                    # COCO + generated caches
└── models/                  # Base model (download manually)
```

For technical design details (visual primitives, reward functions, memory optimization, domain seams), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Tests

```bash
# Unit tests (most do not require GPU)
pytest tests/ -v --ignore=tests/test_grpo_reward_integration.py --ignore=tests/test_stage_integration.py

# Integration tests (require model + COCO data; skipped automatically if missing)
pytest tests/test_stage_integration.py -v

# All tests
pytest tests/ -v
```

---

## Acknowledgements

- **Paper**: *Thinking with Visual Primitives* (Lu et al., DeepSeek, 2026)
- **Reference PyTorch reproduction**: [vra/Thinking-with-Visual-Primitives-pytorch](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch) — training pipeline design, visual primitive format, and reward design heavily inspired this work.
- **Base model**: [Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking)

This is an independent reproduction with different technical choices: Qwen3-VL-4B, TRL + QLoRA, single RTX 5090D 24GB.

---

## Citation

```bibtex
@article{lu2026think,
  title={Thinking with Visual Primitives},
  author={Lu, Ruijie and Ma, Yiyang and Chen, Xiaokang and Luo, Lingxiao and Wu, Zhiyu and Pan, Zizheng and Liu, Xingchao and Lin, Yutong and Li, Hao and Liu, Wen and Hao, Zhewen and Gao, Xi and Nie, Shaoheng and Wei, Yixuan and Xie, Zhenda and Chen, Ting and Zeng, Gang},
  year={2026}
}

@article{bai2025qwen3vl,
  title={Qwen3-VL Technical Report},
  author={Bai, Shuai and Cai, Yuxuan and Chen, Ruizhe and others},
  journal={arXiv preprint arXiv:2511.21631},
  year={2025}
}

@misc{tvp4b5090d2026,
  title={TVP-4B-5090D: Thinking with Visual Primitives on Qwen3-VL-4B},
  author={Edmund724},
  howpublished={\url{https://github.com/Edmund724/Thinking-with-Visual-Primitives-Qwen3-VL-4B}},
  note={Single-GPU reproduction with QLoRA + TRL GRPO + RFT},
  year={2026}
}
```

---

## License

MIT
