**[English](README.md)** | 中文

# Thinking with Visual Primitives — Qwen3-VL-4B 复现

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ModelScope](https://img.shields.io/badge/ModelScope-Collection-cyan)](https://modelscope.cn/collections/EdmundYY/Thinking-with-Visual-Primitives-Qwen3-VL-4B)

> 在单张 **RTX 5090D (24GB)** 上复现 DeepSeek **「Thinking with Visual Primitives」** 的核心思想，基座模型为 **Qwen3-VL-4B-Thinking**。

模型不再只是"看清"，而是**一边推理一边指点**——把 spatial marker（box 与 point）直接交织进 Chain-of-Thought 作为**视觉原语**，从而弥合复杂视觉推理中的 *Reference Gap*。

---

## 核心特点

- **单卡可跑**：4-bit QLoRA + Gradient Checkpointing + Paged AdamW 8-bit；OPD 阶段峰值显存约 **18.5GB**。
- **轻量专家 + 蒸馏**：Box/Point 两个 Specialist 在 SFT 后冻结，通过 On-Policy Distillation（OPD）蒸馏为单个 Unified 模型。
- **完整 6 阶段流水线**：预训练 → 合并 → Box SFT → Point SFT → Box GRPO → Point GRPO → Unified RFT → OPD。
- **内联视觉原语**：`<|box|>[[x1,y1,x2,y2]]<|/box|>` 与 `<|point|>[[x,y]]<|/point|>` 直接嵌入 `<think>`。
- **ModelScope 权重**：所有中间阶段与最终模型均已发布（见下表）。

---

## 快速开始

### 1. 安装依赖

```bash
conda create -n tvp python=3.12 -y
conda activate tvp

pip install torch>=2.11.0 torchvision>=0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 可选：Flash Attention 2（编译失败会自动 fallback 到 eager）
pip install flash-attn --no-build-isolation
```

### 2. 下载基座模型

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Thinking --local-dir models/Qwen3-VL-4B-Thinking
```

### 3. 推理示例

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

批量推理与跨阶段对比见 [docs/INFERENCE.md](docs/INFERENCE.md)。

---

## 模型权重

各阶段 checkpoints 已上传至 ModelScope（中国大陆访问友好）：

| 阶段 | ModelScope 模型名 |
|------|------------------|
| Pretrain（合并后） | `TVP-Pretrain-Qwen3-VL-4B` |
| SFT Box 专家 | `TVP-SFT-Box-Qwen3-VL-4B` |
| SFT Point 专家 | `TVP-SFT-Point-Qwen3-VL-4B` |
| GRPO Box 专家 | `TVP-GRPO-Box-Qwen3-VL-4B` |
| GRPO Point 专家 | `TVP-GRPO-Point-Qwen3-VL-4B` |
| Unified RFT | `TVP-RFT-Unified-Qwen3-VL-4B` |
| OPD（最终模型） | `TVP-OPD-Qwen3-VL-4B` |

**合集地址**：[Thinking-with-Visual-Primitives-Qwen3-VL-4B](https://modelscope.cn/collections/EdmundYY/Thinking-with-Visual-Primitives-Qwen3-VL-4B)

> 上述权重仅供研究 / 流程验证，基于单卡 24GB 显存与小规模数据训练，不建议直接用于生产环境。

---

## 训练流程

一键运行完整流水线：

```bash
bash scripts/run_pipeline.sh
```

| 阶段 | 任务 | 主要输入 | 输出目录 | 已验证墙钟时间 |
|------|------|---------|---------|---------------|
| 1 | 统一视觉 Grounding 预训练 | COCO + CLEVR | `outputs/stage1_visual_pretrain/` | ~7.4h |
| 2 | Merge LoRA | Stage 1 adapter | `outputs/stage2_merged_base/` | ~1m |
| 3a | Box Expert SFT | Box / 计数 / CLEVR | `outputs/stage3a_sft_box/` | ~13.4h |
| 3b | Point Expert SFT | Point / 迷宫 / 路径 | `outputs/stage3b_sft_point/` | ~16h |
| 4a | Box Expert GRPO | Box prompts | `outputs/stage4a_grpo_box/` | ~20.1h |
| 4b | Point Expert GRPO | Point/迷宫/路径 prompts | `outputs/stage4b_grpo_point/` | ~36.4h |
| 5 | Unified RFT | 专家 rollout | `outputs/stage5_rft_unified/` | ~2.7h |
| 6 | OPD | Student + 专家 | `outputs/stage6_opd/` | ~7h |

每阶段的命令、配置、断点续训与显存提示见 [docs/TRAINING.md](docs/TRAINING.md)。

---

## 项目结构

```
tvp-4b-5090d/
├── configs/                 # 各阶段 YAML 配置
├── docs/                    # 详细文档
│   ├── TRAINING.md
│   ├── ARCHITECTURE.md
│   ├── INFERENCE.md
│   ├── OPTIMIZATION.md
│   └── KNOWN_ISSUES.md
├── scripts/                 # 阶段入口脚本与一键流水线
│   ├── run_stage1_visual_pretrain.py
│   ├── run_stage2_merge.py
│   ├── run_stage3a_sft_box.py
│   ├── run_stage3b_sft_point.py
│   ├── run_stage4a_grpo_box.py
│   ├── run_stage4b_grpo_point.py
│   ├── run_stage5_rft_unified.py
│   ├── run_stage6_opd.py
│   └── run_pipeline.sh
├── src/                     # 数据、模型、训练、工具
├── tests/                   # 单元测试与集成测试
├── outputs/                 # 按 stage 组织的训练产物
├── data/                    # COCO 与生成数据缓存
└── models/                  # 基座模型（需手动下载）
```

视觉原语、奖励函数、显存优化、Domain Seam 等技术设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 运行测试

```bash
# 单元测试（大部分不需要 GPU）
pytest tests/ -v --ignore=tests/test_grpo_reward_integration.py --ignore=tests/test_stage_integration.py

# 集成测试（需要模型 + COCO 数据；缺失时自动跳过）
pytest tests/test_stage_integration.py -v

# 全部测试
pytest tests/ -v
```

---

## 致谢与参考

- **论文**：*Thinking with Visual Primitives*（Lu et al., DeepSeek, 2026）
- **参考 PyTorch 复现**：[vra/Thinking-with-Visual-Primitives-pytorch](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch) —— 训练流水线、视觉原语格式、奖励函数设计从中获得大量启发。
- **基座模型**：[Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking)

本项目是对上述工作的独立复现与扩展，在基座模型（Qwen3-VL-4B）、训练框架（TRL + QLoRA）、硬件约束（单卡 RTX 5090D 24GB）等方面做了不同的技术选型。

---

## 引用

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
