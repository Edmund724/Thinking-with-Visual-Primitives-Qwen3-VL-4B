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
| 数据规模 | 460K+ 迷宫 / 125K+ 路径 | 50K 迷宫 / 10K 路径 (可扩展) |

由于 24GB 显存无法容纳 284B MoE 的在线多 rollout 训练，本项目采用**轻量级 Separated Experts（Specialist）架构 + On-Policy Distillation (OPD)**，在保持核心思想不变的前提下，通过 **4-bit QLoRA (r=256) + Gradient Checkpointing + Paged AdamW 8-bit** 实现单卡可跑。

> **⚠️ Pretrain Limitation**: 原论文的 Pretrain 是 **trillion-scale 多模态预训练**，模型在 4000 万+ 筛选后的网页 grounding 数据上建立"Visual Primitives 作为思维单元"的基础能力。由于算力限制，本项目采用**统一视觉 Grounding 预训练**作为直接入口——模型在 COCO + CLEVR 数据上通过 QLoRA 同时学习 special token embedding 和视觉→坐标映射，避免了先文本后视觉的分裂路径，更贴近论文的单阶段多模态预训练范式。

---

## 🖥️ 硬件与软件要求

### 硬件
- **GPU**: NVIDIA RTX 5090D (24GB VRAM)
- **显存上限**: 建议预留 22GB 以内（留 2GB 给 CUDA 上下文与显存碎片）

### 软件（Blackwell 兼容）

| 包 | 最低版本 |
|----|---------|
| PyTorch | 2.11.0 |
| transformers | 5.10.0 |
| flash-attn | 2.8.3 (自动 fallback 到 eager) |
| bitsandbytes | 0.49.0 |
| accelerate | 1.13.0 |
| peft | 0.19.0 |
| trl | 1.6.0 |

---

## ⚡ 快速开始

### 1. 安装依赖

```bash
# 创建环境（推荐）
conda create -n tvp python=3.12 -y
conda activate tvp

# 安装 PyTorch (CUDA 13.0+)
pip install torch>=2.11.0 torchvision>=0.26.0 --index-url https://download.pytorch.org/whl/cu130

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
Stage 1:  Unified Visual Pretrain  COCO + CLEVR 图像，box/point 视觉预训练  ~?    ✅
Stage 2:  Merge LoRA              将视觉预训练 LoRA 合并入基座模型          ~27s   ✅
Stage 3a: Box Expert SFT          格式 token 加权的 Box 专项 SFT           ~?    ✅
Stage 3b: Point Expert SFT        Point+Maze 专项 SFT                      ~?    ✅ (含 resume)
Stage 4a: Box Expert GRPO         Box 专家 GRPO (3 轮循环，默认)          ~6h    (预计)
Stage 4b: Point Expert GRPO       Point 专家 GRPO (3 轮循环，默认)        ~6h    (预计)
Stage 5:  Unified RFT             专家生成 rollout → Unified 学习         ~5h    (预计)
Stage 6:  OPD                     On-Policy Distillation (D_KL(student || expert))   ~7h    (预计)
                                ──────────────────────────────────────────────
                                Total（已实测部分）:                         ~52h
```

**核心设计**：
- **Separated Experts (Specialists)**：Box Specialist 和 Point Specialist 共享同一个 4-bit 基座模型但各带独立的 LoRA adapter
- **冻结 Specialist**：两个 Specialist 在 Stage 3 训好后不再更新，作为固定的 Teacher 模型
- **Expert 生成 Rollout**：Stage 5 RFT 中，Specialist 负责生成 rollout（generator），Unified 模型学习（learner）
- **难度分级**：Easy/Normal/Hard 三级，仅 Normal 级样本用于训练
- **On-Policy Distillation (OPD)**：用 D_KL(student || expert) 将两个 Specialist 的能力蒸馏到单个 Unified 模型
- **三步 Chain-of-Thought**：Intent Analysis（意图分析）→ Grounding → Summarization（总结）

### Stage 1: Unified Visual Grounding Pretrain（统一视觉 Grounding 预训练）✅

**在 COCO + CLEVR 图像上训练**，从零开始建立"视觉特征 → 坐标"的真实映射。Special token（`<|box|>`、`<|point|>`）从随机初始化开始，与 LoRA adapter 同时学习——不需要独立的文本格式预训练。

> **当前默认配置**：`num_box=30000`, `num_point=10000`, `num_clevr=5000`（共 45K 样本）, `num_epochs=2`, `batch_size=1`, `gradient_accumulation_steps=4`（有效 batch=4）, `max_seq_length=2048`, `lora_r=256`, `lora_alpha=512`, `learning_rate=2e-6`，启用 curriculum。
> - 输出：`outputs/stage1_visual_pretrain/`

```bash
python scripts/run_stage1_visual_pretrain.py --config configs/stage1_visual_pretrain.yaml
```

**输出**: `outputs/stage1_visual_pretrain/`

---

### Merge Stage 2 LoRA

**必须合并**！避免双层 LoRA 叠加。

Special token embedding 在 Stage 1 中与 LoRA adapter 一同学习，无需额外的 pretrain embedding 注入。

```bash
python scripts/merge_stage2.py \
    --base_model models/Qwen3-VL-4B-Thinking \
    --adapter_path outputs/stage1_visual_pretrain \
    --output_dir outputs/stage2_merged_base
```

**输出**: `outputs/stage2_merged_base/`（完整 bf16 模型，~8.8GB `model.safetensors`）

> **合并后是否需要验证再进入 Stage 3？**  
> 严格来说不需要——`merge_stage2.py` 是确定性的，如果合并损坏 Stage 3a 会直接加载失败。但建议跑一个 **5 分钟 smoke test**：加载 `outputs/stage2_merged_base`，用一张 COCO 图像提问，检查模型是否在 `<think>` 里输出了空间坐标。若输出正常，即可直接进入 Stage 3a。
>
> ```bash
> python scripts/smoke_test_stage2.py
> # 或指定图片 / 问题
> python scripts/smoke_test_stage2.py \
>     --image_path data/coco/train2017/000000000009.jpg \
>     --question "Locate the main object in the image. Mark it with a box."
> ```

> **合并后是否需要验证再进入 Stage 3？**  
> 严格来说不需要——`merge_stage2.py` 是确定性的，如果合并损坏 Stage 3a 会直接加载失败。但因为刚刚修复了数据格式和 reward，建议跑一个 **5 分钟 smoke test**：加载 `outputs/stage2_merged_base`，用一张 COCO 图像提问，检查模型是否在 `<think>` 里输出了空间坐标。若输出正常，即可直接进入 Stage 3a。
>
> ```bash
> python scripts/smoke_test_stage2.py
> # 或指定图片 / 问题
> python scripts/smoke_test_stage2.py \
>     --image_path data/coco/train2017/000000000009.jpg \
>     --question "Locate the main object in the image. Mark it with a box."
> ```

### Stage 3a: Box Expert SFT ✅

> **当前默认配置**：15K box 定位 + 10K 粗粒度计数 + 5K CLEVR 空间/VQA + 2K 负样本 box，并混入 general pretrain 数据。`num_epochs=2`, `max_seq_length=4096`, `batch_size=1`, `grad_accum=8`（有效 batch=8），`lr=1e-4`。
>
> **近期改进**：
> - SFT 目标会先经过 `clean_primitive_tags()` 清洗，修复生成数据中错序/重复的标签。
> - `WeightedSFTTrainer` 对视觉原语 token 和 `<think>` token 加权（`format_token_weight=5.0`），让格式语法学得更快。
> - 支持 `--resume_from_checkpoint outputs/stage3a_sft_box/checkpoint-XXX` 断点续训。
>
> **注**：Stage 3a 未启用数据缓存（pickle cache），保持原始生成逻辑以确保耗时数据的准确性。从 Stage 3b 起，所有脚本均支持训练数据 pickle 缓存，首次运行后自动保存，后续运行直接加载，跳过耗时的数据生成步骤。

```bash
# 从头训练
python scripts/run_stage3a_sft_box.py --config configs/stage3a_sft_box.yaml

# 断点续训
python scripts/run_stage3a_sft_box.py \
    --config configs/stage3a_sft_box.yaml \
    --resume_from_checkpoint outputs/stage3a_sft_box/checkpoint-500
```

### Stage 3b: Point Expert SFT ✅

> **实测数据**: 约 96K 样本 (25K general + 10K point + 50K maze + 10K path tracing + 1K 负样本 point)，1 epoch，batch_size=4, grad_accum=2 (有效 batch=8)，lr=1e-4，约 12K 步，~2.9s/step，耗时 **~9.7h** (含负样本预估，含 resume)。
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

> **注**：GRPO 采用多轮循环结构，每轮结束后模型会被 reload，轮间是天然断点。每轮本身应一次性跑完；若中途 OOM，脚本会自动查找最新的 `checkpoint-*` 并从中断处恢复。已完成的轮次在重跑时自动跳过。
>
> **显存提示**：所有 stage 脚本现已内置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，无需手动添加，可有效缓解长时间训练中的 CUDA 显存碎片化问题。

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
- **通用**: 标签配对合法性 (`syntax_valid`)、非拉丁文字惩罚、完成长度惩罚

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

### 配置管理（YAML + argparse）

所有 stage 脚本遵循三层默认值级联：

```
argparse default (None) → YAML config value → CLI override
```

- **YAML 配置**（`configs/*.yaml`）是超参数的**唯一真实来源**。
- **argparse 默认值**统一为 `None`——YAML 是必须的。如果 YAML 中缺少某个键，脚本会尽早报错。
- **CLI 参数**覆盖 YAML 和 argparse 默认值，例如 `--num_epochs 5`。
- `StageRunner`（在 `src/training/stage_runner.py` 中）提供共享 boilerplate：argparse 设置、YAML 加载（`apply_yaml_defaults`）、日志和 pickle 数据缓存辅助。

### 视觉原语 Domain Seam

`PrimitiveParser`（在 `src/models/visual_primitive_parser.py` 中）是**所有视觉原语操作的唯一公共 API**——解析、验证、格式化和几何计算。生产代码只从这里（或向后兼容的 `src/utils/metrics` shim）导入。底层模块（`text_parsing.py`、`geometry.py`、`primitive_formatter.py`）是内部实现细节。

```python
from src.models.visual_primitive_parser import PrimitiveParser

boxes = PrimitiveParser.extract_boxes(text)            # 解析
tags  = PrimitiveParser.format_box([(10,20,100,200)])  # 格式化
iou   = PrimitiveParser.box_iou(pred, gt)              # 几何计算
```

### 数据生成与质量控制

除原始 COCO box/point 和合成迷宫外，训练流水线现己新增多种数据生成器，以扩展模型的推理能力：

| 生成器 | 任务类型 | 描述 |
|--------|---------|------|
| `coco_box_generator.py` | Box / 计数 | COCO 边框 + 几何过滤（剔除超大/超小/退化/贴边框）+ 粗糙计数（3–30 个实例） |
| `clevr_spatial.py` | 空间 VQA | 2D 合成场景（球/立方体/圆柱体），支持计数、存在性、空间计数、属性查询四类问题 |
| `path_tracing.py` | Point | 缠绕的 Bézier 曲线，模型需追踪目标路径至终点；支持 uniform-color 模式迫使模型依赖曲率连续性 |
| `synthetic_maze.py` | Point / Maze | 随机迷宫生成 + BFS 路径求解 |

**Thinking-chain 验证器**（`thinking_verifier.py`）：所有生成器在生成后自动经过校验过滤，检查内容包括：
- Tag 配对（`<|box|>`/`<|/box|>`、`<|point|>`/`<|/point|>`）
- 坐标范围合法性（0–999）
- 引用有效性（thinking 步骤引用的 primitive 真实存在）
- Counting 答案与 primitive 数量一致性
- 迷宫自相矛盾检测

未通过任何一项检查的样本会在训练前被丢弃，确保 SFT 和 GRPO 的 cold-start 数据质量。

---

## 📁 项目结构

```
tvp-4b-5090d/
├── configs/                          # YAML 训练配置
│   ├── stage1_visual_pretrain.yaml
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
│   │   └── visual_primitive_parser.py # **Domain seam**：视觉原语统一接口（解析、格式化、几何计算）
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── sft_dataset.py        # SFT 数据集（assistant-only loss mask）
│   │   │   ├── grpo_dataset.py       # GRPO 数据集
│   │   │   └── image_loader.py       # Lazy image loading（防 OOM）
│   │   ├── generators/
│   │   │   ├── __init__.py            # 生成器注册表
│   │   │   ├── coco_box_generator.py # COCO → box/point/counting 训练样本（3-step thinking + 几何过滤）
│   │   │   ├── synthetic_maze.py     # 合成迷宫生成器（3-step thinking）
│   │   │   ├── clevr_spatial.py      # CLEVR 风格 2D 空间 / VQA 生成器
│   │   │   ├── path_tracing.py       # Bézier 曲线路径追踪生成器
│   │   │   └── synthetic_path.py     # 合成路径生成器
│   │   └── formatters/
│   │       └── primitive_formatter.py # 坐标标签格式化（内部模块）
│   ├── training/
│   │   ├── stage_runner.py           # **StageRunner**：共享 argparse+YAML+日志 boilerplate
│   │   ├── trainers/
│   │   │   └── sft_trainer.py        # SFT Trainer 封装（WeightedSFTTrainer）
│   │   ├── pretrain_trainer.py       # Pretrain Trainer（文本 + 视觉两阶段）
│   │   ├── opd_trainer.py            # OPD On-Policy Distillation 训练器
│   │   ├── grpo_fixes.py             # GRPOTrainer 多模态猴补丁
│   │   ├── grpo_utils.py             # GRPO 辅助工具（completion 文本提取）
│   │   ├── callbacks.py              # 训练回调（内存监控）
│   │   ├── memory_utils.py           # GPU 显存工具（build_param_groups）
│   │   └── config_utils.py           # 阶段脚本的 YAML 配置加载工具
│   └── utils/
│       ├── constants.py              # 特殊 token / 超参常量
│       ├── conversation_builder.py   # **ConversationBuilder**：统一消息构建（SFT/GRPO/OPD/pretrain）
│       ├── text_parsing.py           # 答案 / 推理 / box / point 解析（内部模块）
│       ├── geometry.py               # IoU、点距离、迷宫几何（内部模块）
│       ├── metrics.py                # 向后兼容 shim → text_parsing + geometry + reward/*
│       ├── thinking_verifier.py      # Thinking-chain 校验（tag 配对、坐标范围、引用检查）
│       ├── quality_rm_api.py         # LLM-as-Judge Quality RM（OpenAI 兼容 API）
│       ├── logging_utils.py          # 日志初始化
│       ├── difficulty.py             # Easy/Normal/Hard 难度分级
│       ├── batch_inference.py        # 批量生成辅助工具
│       └── reward/
│           ├── format_rm.py          # Format Reward Model
│           ├── quality_rm.py         # Quality Reward Model（规则版）
│           └── accuracy_rm.py        # Accuracy Reward Model（process_reward, compute_total_reward）
├── scripts/                          # 阶段入口脚本
│   ├── run_stage1_visual_pretrain.py  # Stage 1: 统一视觉 Grounding 预训练
│   ├── merge_stage2.py               # Stage 2: Merge LoRA
│   ├── run_stage3a_sft_box.py        # Stage 3a: Box Expert SFT
│   ├── run_stage3b_sft_point.py      # Stage 3b: Point Expert SFT
│   ├── run_stage4a_grpo_box.py       # Stage 4a: Box Expert GRPO
│   ├── run_stage4b_grpo_point.py     # Stage 4b: Point Expert GRPO
│   ├── run_stage5_rft_unified.py     # Stage 5: Unified RFT
│   ├── run_stage6_opd.py             # Stage 6: OPD
│   ├── eval_stage2_structure.py      # Stage 2 结构评估
│   ├── eval_stage3a_paradigm.py      # Stage 3a 范式检查
│   ├── smoke_test_stage2.py          # Stage 2 冒烟测试
│   ├── diagnose_stage2_resume_loss.py # Stage 2 loss 诊断
│   └── run_pipeline.sh               # Master Pipeline 一键运行
├── tests/
│   ├── test_primitive_parser.py      # PrimitiveParser 单元测试（32 个方法）
│   ├── test_primitive_formatter.py   # Box/point 格式化测试
│   ├── test_metrics.py               # 奖励函数与几何工具测试
│   ├── test_conversation_builder.py  # ConversationBuilder 单元测试（21 测试）
│   ├── test_quality_rm_api.py        # Quality RM API 集成测试（23 测试）
│   ├── test_pretrain_format.py       # 预训练格式测试
│   ├── test_weighted_sft_trainer.py  # WeightedSFTTrainer loss 测试
│   ├── test_grpo_fixes.py            # GRPO 猴补丁单元测试
│   ├── test_grpo_reward_integration.py # GRPO 奖励集成测试
│   ├── test_stage_integration.py     # **Stage 集成测试**（14 测试，覆盖全部 8 个 stage）
│   ├── test_stage3a_data_composition.py # Stage 3a 数据组成测试
│   ├── test_logging_utils.py         # 日志工具测试
│   └── test_filter_normal_level_data.py # 难度过滤器测试
├── outputs/                          # 训练产物（按 stage 组织）
│   ├── stage1_visual_pretrain/       # LoRA adapter + checkpoints
│   ├── stage2_merged_base/           # merge 后的完整模型
│   ├── stage3a_sft_box/              # Box Expert SFT adapter
│   ├── stage3b_sft_point/            # Point Expert SFT adapter
│   ├── stage4a_grpo_box/             # Box Expert GRPO adapter
│   ├── stage4b_grpo_point/           # Point Expert GRPO adapter
│   ├── stage5_rft_unified/           # Unified RFT adapter
│   └── stage6_opd/                   # On-Policy Distillation 蒸馏产物
├── logs/                             # 各 stage 训练日志
├── data/
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
# 单元测试（快速，大部分不需要 GPU）
pytest tests/ -v --ignore=tests/test_grpo_reward_integration.py --ignore=tests/test_stage_integration.py

# 集成测试（需要模型 + COCO 数据；缺失时自动跳过）
pytest tests/test_stage_integration.py -v

# 全部测试
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

## 🔧 进一步贴近原论文的优化方向

本复现优先保证**核心思想**（视觉原语作为推理单元）在单卡约束下可跑。以下是在不重建万亿级预训练的前提下，仍可逐步缩小与原文差距的具体方向：

### Stage 1 — 预训练
- **当前**: 统一视觉 Grounding 预训练，COCO + CLEVR 图像通过 QLoRA 训练。Special token 随机初始化，与视觉特征同时学习，不再有独立的文本格式预训练。
- **论文**: 在 4000 万+ 筛选后的网页 grounding 数据上进行万亿级多模态预训练。
- **可行优化**:
  1. 扩充视觉预训练数据来源（Flickr30k Entities、RefCOCO、SA-1B 样本等），10 万~100 万来自不同域的真实样本即可提升泛化能力。
  2. 若无法做网页抓取，可在公开检测/grounding 数据集上复现论文的两步过滤（语义审查 + 几何质量审查）。
  3. 解冻 ViT 最后几层（`--unfreeze_vit_layers 2-4`），让视觉特征更好地适配坐标预测任务。

### Stage 1 视觉预训练 — 进一步提升数据多样性
- **当前**: COCO + CLEVR 合成数据，通过 QLoRA 训练，ViT 冻结。
- **论文**: DeepSeek-ViT + 3×3 token 压缩 + CSA 4× KV-cache 压缩，端到端海量数据训练。
- **可行优化**:
  1. 扩充视觉预训练数据来源（SA-1B、合成几何图形、领域 grounding 数据集等）。
  2. 若显存允许，以极低学习率解冻部分 ViT 层（`--unfreeze_vit_layers 2-4`）。

### Stage 3 — Cold-Start SFT
- **当前**: COCO box/point/counting + 简化 CLEVR + 单算法矩形迷宫 + path tracing。
- **论文**: MLLM 基于 GQA 场景图生成 thinking chain，46 万迷宫覆盖 DFS/Prim/Kruskal 与矩形/圆形/六边形拓扑，12.5 万 path tracing。
- **可行优化**:
  1. **细粒度计数**: 接入 GQA 场景图，用 MLLM/API 生成属性约束问题与 thinking chain，再用 `thinking_verifier.py` 校验。
  2. **迷宫多样性**: 在现有 DFS 基础上增加 Prim、Kruskal 生成器，以及圆形、六边形拓扑。
  3. **空间/VQA**: 把 CLEVR 问题扩展为多跳推理，并加入负样本（忠实拒绝）。
  4. **MLLM 生成 thinking**: 在有标注的数据（GQA、COCO panoptic、SA-1B）上，用本地小 MLLM 或 API 合成“意图分析→Grounding→总结”三段式 thinking，替代手工模板。

### Stage 4 — 专项 RL
- **当前**: 规则化 Quality RM + 已改为按“正确 rollout 数量”分难度（与论文 Sec 2.5.2 对齐）。
- **论文**: LLM-based Generative Reward Model 做 Quality RM。
- **可行优化**:
  1. 用本地小模型（如 Qwen2.5-3B-Instruct 或蒸馏后的 critic）替代规则 QM，或在边界样本上调用 API。
  2. 规则 QM 作为快速预筛，LLM judge 负责难分样本的二次打分。

## ⚠️ 已知限制

1. **GRPO 在线 rollout 开销**: 单卡 24GB 下 `num_generations=5` 已是极限，如需更多 rollout 需要梯度累积或 offload。
2. **Flash Attention 兼容性**: Blackwell (RTX 5090D) 对 flash-attn 2.8.3 支持仍在完善中，代码已内置 `eager` fallback。
3. **COCO 数据**: 首次下载约 18GB，训练时按需读取。
4. **本实现为复现**: 论文原始 pipeline 包含更多阶段和更大规模数据，本项目在单卡约束下做了精简。
5. **vLLM 不支持**: vLLM 与 TRL GRPO + Qwen3-VL 不兼容，所有 GRPO 阶段均使用 HuggingFace 原生生成。
6. **样本量小、质量有限**: 默认配置为了快速跑通流程做了大幅裁剪（例如 Stage 1 仅 1 万条、Stage 2 视觉预训练仅 2 万张、GRPO 仅 2 轮 2 条 rollout）。**这些默认值无法保证最终精度或生产出可用权重，仅用于验证训练流程**。如需更好效果，请按硬件承受能力放大样本量和训练轮数。

---

## 📄 License

MIT
