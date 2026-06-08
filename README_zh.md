**[English](README.md)** | 中文

# Thinking with Visual Primitives — Qwen3-VL-4B Reproduction

> **单卡 RTX 5090D (24GB) 复现 DeepSeek「Thinking with Visual Primitives」核心思想。**

---

## 📄 论文背景

**Thinking with Visual Primitives** (DeepSeek, 2026) 提出了一种全新的推理框架：
> 模型不仅要"看清"(see clearer)，更要**"一边推理一边指点"（"point while it reasons"）**。通过在 Chain-of-Thought (CoT) 推理过程中**直接交织 spatial markers（空间标记，即 point 和 bounding box）**作为 **Visual Primitives（视觉原语）**，将空间标记提升为 **"最小思维单元"（minimal units of thought）**——将抽象的语言概念锚定到具体的物理坐标，从而弥合复杂视觉推理中的 **Reference Gap（引用鸿沟）**。

论文识别了多模态推理中的两个不同挑战：
- **Perception Gap（感知鸿沟）**：模型感知细节的能力（通过高分辨率裁剪、动态 patching 等技术解决）
- **Reference Gap（引用鸿沟）**：自然语言无法在复杂场景中精确指代物体的固有问题

与传统的"先语言推理、后输出坐标"或"用后验框验证"（post-hoc grounding）的路线不同，论文的核心主张是：**spatial markers 既不是推理的结果也不是辅助证据，而是推理本身的主体介质**——它们是认知轨迹（cognitive trajectory）的内在组成部分。

---

## 🎯 本项目定位

| 维度 | 论文 (DeepSeek) | 本复现 |
|------|----------------|--------|
| 基座模型 | DeepSeek-V4-Flash (284B MoE) | **Qwen3-VL-4B-Thinking** (4B Dense) |
| 训练方法 | 大规模 GRPO + 自研训练框架 | **QLoRA + TRL GRPO + RFT** |
| 视觉原语 | 自定义 tokens | `<|box|>` / `<|point|>` |
| 显存要求 | 多卡 A100/H100 | **单卡 RTX 5090D 24GB** |
| 数据规模 | 460K+ 迷宫 / 125K+ 路径 | 50K 迷宫 / 15K 路径 (可扩展) |

由于 24GB 显存无法容纳 284B MoE 的在线多 rollout 训练，本项目采用**轻量级 Separated Experts（Specialist）架构 + On-Policy Distillation (OPD)**，在保持核心思想不变的前提下，通过 **4-bit QLoRA (r=256) + Gradient Checkpointing + Paged AdamW 8-bit** 实现单卡可跑。

> **⚠️ Pretrain Limitation**: 原论文的 Pretrain 是 **trillion-scale 多模态预训练**，模型在海量 web 数据上建立"Visual Primitives 作为思维单元"的基础能力。由于算力限制，本项目的 Stage 1 仅做 **Format Pretraining（格式预训练）**——让模型学会输出 `<|box|>`、`<|point|>` 等特殊 token 的语法格式。后续 Stage 2 Visual Pretrain 在 COCO 图像上补偿视觉→坐标的 Grounding 能力。

---

## 🖥️ 硬件与软件要求

### 硬件
- **GPU**: NVIDIA RTX 5090D (24GB VRAM)
- **显存上限**: 建议预留 22GB 以内（留 2GB 给 CUDA 上下文与显存碎片）

### 软件（Blackwell 兼容）

| 包 | 最低版本 |
|----|---------|
| PyTorch | 2.6.0 |
| transformers | 4.49.0 |
| flash-attn | 2.7.0+ (自动 fallback 到 eager) |
| bitsandbytes | 0.45.0 |
| accelerate | 1.2.0 |
| peft | 0.14.0 |
| trl | 0.15.0 |

---

## ⚡ 快速开始

### 1. 安装依赖

```bash
# 创建环境（推荐）
conda create -n tvp python=3.12 -y
conda activate tvp

# 安装 PyTorch (CUDA 12.4+)
pip install torch>=2.6.0 torchvision>=0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install -r requirements.txt

# 可选：安装 Flash Attention 2 加速
# 如果编译失败，代码会自动 fallback 到 eager attention
pip install flash-attn --no-build-isolation
```

### 2. 下载基座模型

```bash
# 自动从 Hugging Face 下载（首次运行时会缓存）
# 或手动下载到本地
huggingface-cli download Qwen/Qwen3-VL-4B-Thinking --local-dir models/Qwen3-VL-4B-Thinking
```

### 3. 准备 COCO 数据集（可选，用于 Box 任务）

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/train2017.zip -P data/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco
unzip data/coco/train2017.zip -d data/coco
unzip data/coco/annotations_trainval2017.zip -d data/coco
```

> 迷宫与路径数据为**程序实时生成**，无需提前下载。

---

## 🚀 训练流程（Separated Experts + On-Policy Distillation）

```
Stage 1:  Text Pretrain          文本-only embedding 初始化               ~1.5h  ✅
Stage 2:  Visual Pretrain        COCO 图像 + box/point 视觉预训练        ~14h   ✅
Stage 2M: Merge LoRA             将视觉预训练 LoRA 合并入基座模型          ~18s   ✅
Stage 3a: Box Expert SFT         70% 通用 + 30% Box 专项 SFT             ~9.5h  ✅
Stage 3b: Point Expert SFT       70% 通用 + 30% Point+Maze 专项 SFT      ~8h    ✅
Stage 4a: Box Expert GRPO        Box 专家 GRPO (3 轮循环，默认)          ~6h    (预计)
Stage 4b: Point Expert GRPO      Point 专家 GRPO (3 轮循环，默认)        ~6h    (预计)
Stage 5:  Unified RFT            专家生成 rollout → Unified 学习         ~5h    (预计)
Stage 6:  OPD                    On-Policy Distillation (D_KL(student || expert))   ~7h    (预计)
                              ──────────────────────────────────────────────
                              Total:                                     ~80h
```

**核心设计**：
- **Separated Experts (Specialists)**：Box Specialist 和 Point Specialist 共享同一个 4-bit 基座模型但各带独立的 LoRA adapter
- **冻结 Specialist**：两个 Specialist 在 Stage 3 训好后不再更新，作为固定的 Teacher 模型
- **Expert 生成 Rollout**：Stage 5 RFT 中，Specialist 负责生成 rollout（generator），Unified 模型学习（learner）
- **难度分级**：Easy/Normal/Hard 三级，仅 Normal 级样本用于训练
- **On-Policy Distillation (OPD)**：用 D_KL(student || expert) 将两个 Specialist 的能力蒸馏到单个 Unified 模型
- **三步 Chain-of-Thought**：Intent Analysis（意图分析）→ Grounding → Summarization（总结）

### Stage 1: Text Pretrain（格式预训练）✅

**纯文本训练，无图像**。只训练 `embed_tokens` 层。25K 条程序化生成样本，3 epochs。

> **实测数据**: 25K 样本，3 epochs，18750 步 (batch_size=4, lr=2e-4)，Epoch 1/2/3 Avg Loss: 1.0399 / 0.9944 / 0.9899，耗时 **~1h25min**。

```bash
python scripts/run_stage1_pretrain.py \
    --model_path models/Qwen3-VL-4B-Thinking \
    --output_dir outputs/stage1_pretrain \
    --num_epochs 3
```

**输出**: `outputs/stage1_pretrain/pretrain_state_dict.pt`

---

### Stage 2: Visual Pretrain（视觉预训练）✅

**在 COCO 图像上训练**，建立"视觉特征 → 坐标"的真实映射。不是随机猜坐标。

> **实测数据**: 60000 样本 (50K box + 10K point)，1 epoch，batch_size=2, grad_accum=2 (有效 batch=4)，耗时 **~14h**。

```bash
python scripts/run_stage2_visual_pretrain.py \
    --model_path models/Qwen3-VL-4B-Thinking \
    --pretrain_embedding_path outputs/stage1_pretrain \
    --output_dir outputs/stage2_visual_pretrain \
    --num_box 50000 --num_point 10000 \
    --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 2
```

**输出**: `outputs/stage2_visual_pretrain/`

---

### Merge Stage 2 LoRA

**必须合并**！避免双层 LoRA 叠加。

> **实测耗时**: **~18s**。

```bash
python scripts/merge_stage2.py \
    --base_model models/Qwen3-VL-4B-Thinking \
    --adapter_path outputs/stage2_visual_pretrain \
    --output_dir outputs/stage2_merged_base
```

### Stage 3a: Box Expert SFT ✅

> **实测数据**: 40000 样本 (25K general + 15K box)，1 epoch，batch_size=1, grad_accum=8 (有效 batch=8)，lr=1e-4，5000 步，~7.07s/step，耗时 **~9h37min**。
>
> **注**：Stage 3a 未启用数据缓存（pickle cache），保持原始生成逻辑以确保耗时数据的准确性。从 Stage 3b 起，所有脚本均支持训练数据 pickle 缓存，首次运行后自动保存，后续运行直接加载，跳过耗时的数据生成步骤。

```bash
python scripts/run_stage3a_sft_box.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3a_sft_box \
    --num_box 15000 --num_epochs 1 --learning_rate 1e-4
```

### Stage 3b: Point Expert SFT ✅

> **实测数据**: 85000 样本 (25K general + 10K point + 50K maze)，1 epoch，batch_size=4, grad_accum=2 (有效 batch=8)，lr=1e-4，10625 步，耗时 **~8h** (含 resume)。
> 
> 训练中途曾因显存碎片化导致速度从 3s/it 降至 30s/it，通过 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` resume 后恢复正常。

```bash
# 正常从头跑
python scripts/run_stage3b_sft_point.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3b_sft_point \
    --num_point 10000 --num_maze 50000 \
    --num_epochs 1 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation_steps 2

# 若中途显存碎片化减速，resume 时加环境变量
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_stage3b_sft_point.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3b_sft_point \
    --num_point 10000 --num_maze 50000 \
    --num_epochs 1 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation_steps 2 \
    --resume_from_checkpoint outputs/stage3b_sft_point/checkpoint-5000
```

### Stage 4a: Box Expert GRPO（默认 3 轮循环）

> **注**：GRPO 采用多轮循环结构，每轮结束后模型会被 reload，轮间是天然断点。每轮本身应一次性跑完，不支持轮内 resume。若某轮中断，重跑时脚本会自动检测并跳过已完成的轮次，从中断处继续。

```bash
python scripts/run_stage4a_grpo_box.py \
    --model_path outputs/stage3a_sft_box \
    --output_dir outputs/stage4a_grpo_box \
    --num_samples 5000
```

### Stage 4b: Point Expert GRPO（默认 3 轮循环）

```bash
python scripts/run_stage4b_grpo_point.py \
    --model_path outputs/stage3b_sft_point \
    --output_dir outputs/stage4b_grpo_point \
    --num_point 2000 --num_maze 5000
```

### Stage 5: Unified RFT（专家生成 rollout）

```bash
python scripts/run_stage5_rft_unified.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage5_rft_unified
```

### Stage 6: OPD（On-Policy Distillation）

```bash
python scripts/run_stage6_opd.py \
    --student_path outputs/stage5_rft_unified/final_model \
    --output_dir outputs/stage6_opd
```

### 一键运行（推荐）

```bash
bash scripts/run_pipeline.sh
```

---

## 🔬 推理示例

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

预期输出包含：
```
<think>
I can see two cats in the image. Let me mark them.
<|box|>[[120, 80, 340, 290]]<|/box|>
<|box|>[[410, 95, 620, 310]]<|/box|>
</think>

The answer is 2.
```

### 批量推理

支持对 JSONL 文件批量推理，适用于评估或大规模处理：

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

        # 计算过程奖励（可选）
        reward = process_reward(pred, item["answer"], task_type=item.get("task_type", "box"))
        results.append({"pred": pred, "reward": reward, "gt": item["answer"]})

with open("eval_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

### 跨 Stage 模型对比评估

对比不同训练阶段的输出质量，验证各阶段效果演进：

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

# 加载各阶段模型
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
    del model  # 释放显存
```

> **预期行为**：Pretrain 阶段输出格式正确但框不准确；SFT Box 阶段结构化思考 + 精确框；OPD 阶段同时具备 Box 和 Point 能力。

---

## 🧠 关键技术设计

### Visual Primitive（Spatial Marker）格式

论文将 Visual Primitives 定义为 Chain-of-Thought 中的 inline tokens（内联标记）：

```
<|box|>[[x1, y1, x2, y2]]<|/box|>        # 单个 Bounding Box
<|box|>[[x1,y1,x2,y2],[x3,y3,x4,y4]]<|/box|>  # 多个 Box
<|point|>[[x, y]]<|/point|>              # 点坐标（迷宫路径、关键点）
```

坐标统一归一化到 `[0, 999]` 区间。

### 内存优化策略

| 技术 | 效果 |
|-----|------|
| 4-bit NF4 + Double Quantization | ~6GB / 模型实例 |
| Gradient Checkpointing | 显存换时间，降低激活值占用 |
| Paged AdamW 8-bit | 优化器状态压缩 |
| bf16 计算 | 速度 + 显存双赢 |

单卡 24GB 可以同时容纳 **Policy 模型 + Reference 模型**（TRL 的 GRPOTrainer 对 PEFT 模型通过禁用 adapter 复用相同基座权重，峰值显存约 14-18GB，其中 KV cache 为最大支出）。

### VRAM Guide 显存适配指南

不同显存 GPU 的推荐配置：

| GPU 显存 | batch_size | grad_accum | LoRA r | image_size | max_length | 备注 |
|---------|-----------|-----------|--------|-----------|-----------|------|
| **24GB** (5090D / 4090) | 2 | 2 | 256 | 448 | 2048 | 本项目默认配置 |
| **16GB** (4080 / 4070 Ti Super) | 1 | 4 | 128 | 384 | 1536 | 降低 LoRA rank 以节省显存 |
| **12GB** (4070 Ti / 3060 12G) | 1 | 8 | 64 | 336 | 1024 | 激进压缩，GRPO `num_generations=3` |
| **80GB** (A100 / H100) | 4 | 1 | 256 | 448 | 4096 | 可跑全参数或更大 batch |

> **提示**：
> - 12GB 显卡建议在运行前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少显存碎片。
> - OPD 阶段需同时加载 3 个模型（student + 2 experts），教师模型自动使用 4-bit 量化，峰值显存约为单模型 SFT 的 1.8 倍。
> - GRPO 阶段显存占用与 `num_generations` 正相关，12GB 下建议 `num_generations=3`，24GB 下可用 `num_generations=5`。

### 过程奖励函数 (Process Reward)

不同于仅看最终答案正确与否，我们设计了细粒度的过程奖励——受论文三个 reward heads（**Format**、**Quality**、**Accuracy**）启发：

- **Box 任务**: IoU 匹配、漏检、格式合法性
- **Point / Maze 任务**: L2 距离、撞墙检测 (Bresenham 采样)、回溯缺失检测
- **通用**: 标签配对合法性 (`syntax_valid`)

```python
from src.utils.metrics import process_reward

reward = process_reward(
    pred_text=pred,
    gt_text=gt,
    task_type="maze",
    iou_threshold=0.5,
    point_dist_threshold=10.0,
    maze_grid=grid,
)
# 返回: answer_correct, syntax_valid, box_avg_iou, point_avg_dist,
#       wall_collision_count, backtracking_missing, ...
```

---

## 📁 项目结构

```
tvp-4b-5090d/
├── configs/                          # YAML 训练配置
│   ├── stage2_visual_pretrain.yaml
│   ├── stage3a_sft_box.yaml
│   ├── stage3b_sft_point.yaml
│   ├── stage4a_grpo_box.yaml
│   ├── stage4b_grpo_point.yaml
│   ├── stage5_rft_unified.yaml
│   └── stage6_opd.yaml
├── src/
│   ├── models/
│   │   ├── qwen_vl_loader.py         # Qwen3VL + QLoRA 加载器
│   │   ├── pretrain_loader.py        # Pretrain 模型加载 + embedding 注入
│   │   └── visual_primitive_parser.py # 视觉原语解析器
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── sft_dataset.py        # SFT 数据集（assistant-only loss mask）
│   │   │   ├── grpo_dataset.py       # GRPO 数据集
│   │   │   └── image_loader.py       # Lazy image loading（防 OOM）
│   │   ├── generators/
│   │   │   ├── coco_box_generator.py # COCO → box/point 训练样本（3-step thinking）
│   │   │   ├── synthetic_maze.py     # 合成迷宫生成器（3-step thinking）
│   │   │   └── synthetic_path.py     # 合成路径生成器（暂未使用）
│   │   └── formatters/
│   │       └── primitive_formatter.py # 坐标标签格式化
│   ├── training/
│   │   ├── trainers/
│   │   │   └── sft_trainer.py        # SFT Trainer 封装
│   │   ├── pretrain_trainer.py       # Pretrain Trainer（仅训练 embedding）
│   │   ├── opd_trainer.py            # OPD On-Policy Distillation 训练器
│   │   ├── callbacks.py              # 训练回调（内存监控）
│   │   └── memory_utils.py           # GPU 显存工具
│   └── utils/
│       ├── constants.py              # 特殊 token / 超参常量
│       ├── metrics.py                # Format RM + Accuracy RM + 难度分级
│       └── logging_utils.py          # 日志初始化
├── scripts/                          # 阶段入口脚本
│   ├── generate_pretrain_data.py     # 预训练数据生成器
│   ├── run_stage1_pretrain.py        # Stage 1: Text Pretrain
│   ├── run_stage2_visual_pretrain.py # Stage 2: Visual Pretrain
│   ├── merge_stage2.py               # Stage 2 LoRA Merge
│   ├── run_stage3a_sft_box.py        # Stage 3a: Box Expert SFT
│   ├── run_stage3b_sft_point.py      # Stage 3b: Point Expert SFT
│   ├── run_stage4a_grpo_box.py       # Stage 4a: Box Expert GRPO
│   ├── run_stage4b_grpo_point.py     # Stage 4b: Point Expert GRPO
│   ├── run_stage5_rft_unified.py     # Stage 5: Unified RFT
│   ├── run_stage6_opd.py             # Stage 6: OPD
│   └── run_pipeline.sh               # Master Pipeline 一键运行
├── tests/
│   ├── test_primitive_parser.py      # 坐标解析单元测试
│   ├── test_metrics.py               # 奖励函数与几何工具测试
│   ├── test_pretrain_format.py       # 预训练格式测试
│   └── test_logging_utils.py         # 日志工具测试
├── outputs/                          # 训练产物（按 stage 组织）
│   ├── stage1_pretrain/              # embedding state_dict
│   ├── stage2_visual_pretrain/       # LoRA adapter + checkpoints
│   ├── stage2_merged_base/           # merge 后的完整模型
│   ├── stage3a_sft_box/              # Box Expert SFT adapter
│   ├── stage3b_sft_point/            # Point Expert SFT adapter
│   ├── stage4a_grpo_box/             # Box Expert GRPO adapter
│   ├── stage4b_grpo_point/           # Point Expert GRPO adapter
│   ├── stage5_rft_unified/           # Unified RFT adapter
│   └── stage6_opd/                   # On-Policy Distillation 蒸馏产物
├── logs/                             # 各 stage 训练日志
├── data/
│   ├── pretrain/pretrain_data.json   # 格式预训练数据
│   ├── coco/                         # COCO 数据集（需手动下载）
│   └── cache/maze/                   # 迷宫图片缓存
├── models/Qwen3-VL-4B-Thinking/     # 基座模型（需手动下载）
├── requirements.txt
├── README.md
└── README_zh.md
```

---

## 🧪 运行测试

```bash
pytest tests/ -v
```

---

## 🙏 致谢与参考

本项目的实现参考了以下工作：

- **论文**：*Thinking with Visual Primitives*（Lu et al., DeepSeek, 2026）—— 提出将 spatial markers（bounding box 和 point）作为多模态 Chain-of-Thought 推理中的"最小思维单元"（minimal units of thought），弥合复杂视觉推理中的 Reference Gap。本项目的核心思想即来源于此论文。
- **[vra/Thinking-with-Visual-Primitives-pytorch](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch)**（作者：Yunfeng Wang）：基于 PyTorch 的非官方复现，采用 Qwen2-VL-2B + LoRA 在单卡 12GB+ 上实现了完整的 Pretrain → SFT → OPD 训练流程。本项目的整体训练流水线设计（Separated Experts + On-Policy Distillation）、Visual Primitive 格式定义、过程奖励函数设计等均从中获得了大量启发和参考。
- **[ailuntx/Thinking-with-Visual-Primitives](https://github.com/ailuntx/Thinking-with-Visual-Primitives)**：原论文官方仓库被删除后的社区存档/镜像，保留了原始论文的技术报告、代码和说明，作为 DeepSeek 论文原始实现的替代参考来源。

> **声明**：本项目是对上述工作的独立复现与扩展，在基座模型（Qwen3-VL-4B）、训练框架（TRL + QLoRA）、硬件约束（单卡 RTX 5090D 24GB）等方面做了不同的技术选型。如有任何问题或建议，欢迎提出 Issue。

---

## 📚 引用

首先引用原始论文：

```bibtex
@article{lu2026think,
  title={Thinking with Visual Primitives},
  author={Lu, Ruijie and Ma, Yiyang and Chen, Xiaokang and Luo, Lingxiao and Wu, Zhiyu and Pan, Zizheng and Liu, Xingchao and Lin, Yutong and Li, Hao and Liu, Wen and Hao, Zhewen and Gao, Xi and Nie, Shaoheng and Wei, Yixuan and Xie, Zhenda and Chen, Ting and Zeng, Gang},
  year={2026}
}
```

以及参考的 PyTorch 复现：

```bibtex
@software{wang2026tvp_pytorch,
  title={Thinking with Visual Primitives --- PyTorch Implementation},
  author={Wang, Yunfeng},
  url={https://github.com/vra/Thinking-with-Visual-Primitives-pytorch},
  year={2026}
}
```

如果你使用了本代码，也请引用 Qwen3-VL：

```bibtex
@article{bai2025qwen3vl,
  title={Qwen3-VL Technical Report},
  author={Bai, Shuai and Cai, Yuxuan and Chen, Ruizhe and others},
  journal={arXiv preprint arXiv:2511.21631},
  year={2025}
}
```

以及本项目：

```bibtex
@misc{tvp4b5090d2026,
  title={TVP-4B-5090D: Thinking with Visual Primitives on Qwen3-VL-4B},
  author={Edmund724},
  howpublished={\url{https://github.com/Edmund724/Thinking-with-Visual-Primitives-Qwen3-VL-4B}},
  note={Single-GPU reproduction with QLoRA + TRL GRPO + RFT},
  year={2026}
}
```

---

## ⚠️ 已知限制

1. **GRPO 在线 rollout 开销**: 单卡 24GB 下 `num_generations=5` 已是极限，如需更多 rollout 需要梯度累积或 offload。
2. **Flash Attention 兼容性**: Blackwell (RTX 5090D) 对 flash-attn 2.7.0+ 支持仍在完善中，代码已内置 `eager` fallback。
3. **COCO 数据**: 首次下载约 18GB，训练时按需读取。
4. **本实现为复现**: 论文原始 pipeline 包含更多阶段和更大规模数据，本项目在单卡约束下做了精简。

---

## 📄 License

MIT

---

## 🤗 模型权重（训练完成后计划上传）

**⚠️ 当前状态：训练进行中，权重尚未上传。**

训练完成后，**完整模型权重**（基座 + LoRA 合并后的全量 bf16 参数）将上传至 **ModelScope**，开箱即用，无需额外加载基座模型。

计划上传地址：

```bash
# 未来可用（当前不可用）
modelscope download Edmund724/tvp-4b-5090d-qwen3-vl-4b --local_dir ./weights
```

上线后的使用方式：
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./weights",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("./weights", trust_remote_code=True)
```

> 预计完整权重约 8-9GB（bf16），支持直接推理与继续微调。上线后会在此处更新实际下载链接。
