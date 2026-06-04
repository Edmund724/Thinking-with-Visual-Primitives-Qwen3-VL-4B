# Thinking with Visual Primitives — Qwen3-VL-4B Reproduction

> **单卡 RTX 5090D (24GB) 复现 DeepSeek「Thinking with Visual Primitives」核心思想。**

---

## 📄 论文背景

**Thinking with Visual Primitives** (DeepSeek, 2026) 提出：
> 让多模态大模型在 Chain-of-Thought 推理过程中，把 **bounding box** 和 **point** 作为"最小思维单元"（Visual Primitives）直接插入到思考链中，从而将抽象的语言概念锚定到具体的物理坐标，解决复杂视觉推理中的 **Reference Gap**。

与传统"先语言推理、后输出坐标"或"用后验框验证"的路线不同，论文的核心主张是：**视觉标记不是推理的结果，也不是辅助证据，而是推理本身的主体介质**。

---

## 🎯 本项目定位

| 维度 | 论文 (DeepSeek) | 本复现 |
|------|----------------|--------|
| 基座模型 | DeepSeek-V4-Flash (284B MoE) | **Qwen3-VL-4B-Thinking** (4B Dense) |
| 训练方法 | 大规模 GRPO + 自研训练框架 | **QLoRA + TRL GRPO + RFT** |
| 视觉原语 | 自定义 tokens | `<|box|>` / `<|point|>` |
| 显存要求 | 多卡 A100/H100 | **单卡 RTX 5090D 24GB** |
| 数据规模 | 460K+ 迷宫 / 125K+ 路径 | 50K 迷宫 / 15K 路径 (可扩展) |

由于 24GB 显存无法容纳 284B MoE 的在线多 rollout 训练，本项目采用**轻量级 Separated Experts 架构 + OPD**，在保持核心思想不变的前提下，通过 **4-bit QLoRA (r=256) + Gradient Checkpointing + Paged AdamW 8-bit** 实现单卡可跑。

> **⚠️ Pretrain Limitation**: 原论文的 Pretrain 是 **trillion-scale 多模态预训练**，模型在海量 web 数据上建立"视觉原语作为思维单元"的基础能力。由于算力限制，本项目的 Stage 0 仅做"格式预训练"（Format Pretraining）——让模型学会输出 `<|box|>`、`<|point|>` 等特殊 token 的语法格式。后续 Stage 0.5 视觉 Pretrain 在 COCO 图像上补偿视觉→坐标映射能力。

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

## 🚀 训练流程（Separated Experts + OPD）

```
Stage 0:  Text Pretrain          文本-only embedding 初始化               ~2.5h
Stage 0.5:Visual Pretrain        COCO 图像 + box/point 视觉预训练        ~8-12h
Stage 0.5M:Merge LoRA            将视觉预训练 LoRA 合并入基座模型          ~5min
Stage 1a: Box Expert SFT         70% 通用 + 30% Box 专项 SFT             ~5-8h
Stage 1b: Point Expert SFT       70% 通用 + 30% Point+Maze 专项 SFT      ~8-12h
Stage 2a: Box Expert GRPO        Box 专家 GRPO (3 轮，难度分级)          ~6-10h
Stage 2b: Point Expert GRPO      Point 专家 GRPO (3 轮，难度分级)        ~8-12h
Stage 3:  Unified RFT            专家生成 rollout → Unified 学习         ~4-6h
Stage 4:  OPD                    反向 KL 蒸馏 (D_KL(student || expert))   ~3-5h
                              ──────────────────────────────────────────────
                              Total:                                     ~50-70h
```

**核心设计**：
- **分离专家**：Box Expert 和 Point Expert 共享同一个 4-bit 基座模型但各带独立的 LoRA adapter
- **专家固定**：两个专家在 Stage 2 训好后不再更新，作为固定的 Teacher
- **专家生成**：Stage 3 RFT 中，专家负责生成 rollout（generator），Unified 模型学习（learner）
- **难度分级**：Easy/Normal/Hard 三级，仅 Normal 级样本用于训练
- **反向 KL 蒸馏**：OPD 用 D_KL(student || expert) 让 Unified 模型学习专家的分布
- **三步 Thinking**：Intent Analysis → Grounding → Summarization

### Stage 0: Text Pretrain（格式预训练）✅

**纯文本训练，无图像**。只训练 `embed_tokens` 层。25K 条程序化生成样本，3 epochs。

```bash
python scripts/run_pretrain.py \
    --model_path models/Qwen3-VL-4B-Thinking \
    --output_dir outputs/stage0_pretrain \
    --num_epochs 3
```

**输出**: `outputs/stage0_pretrain/pretrain_state_dict.pt`

### Stage 0.5: Visual Pretrain（视觉预训练）🆕

**在 COCO 图像上训练**，建立"视觉特征 → 坐标"的真实映射。不是随机猜坐标。

```bash
python scripts/run_stage0_5_visual_pretrain.py \
    --model_path models/Qwen3-VL-4B-Thinking \
    --pretrain_embedding_path outputs/stage0_pretrain \
    --output_dir outputs/stage0_5_visual_pretrain \
    --num_box 50000 --num_point 10000
```

**输出**: `outputs/stage0_5_visual_pretrain/`

### Merge Stage 0.5 LoRA

**必须合并**！避免双层 LoRA 叠加。

```bash
python scripts/merge_stage0_5.py \
    --base_model models/Qwen3-VL-4B-Thinking \
    --adapter_path outputs/stage0_5_visual_pretrain \
    --output_dir outputs/stage0_5_merged_base
```

### Stage 1a: Box Expert SFT

```bash
python scripts/run_stage1a_sft_box.py \
    --model_path outputs/stage0_5_merged_base \
    --output_dir outputs/stage1a_sft_box
```

### Stage 1b: Point Expert SFT

```bash
python scripts/run_stage1b_sft_point.py \
    --model_path outputs/stage0_5_merged_base \
    --output_dir outputs/stage1b_sft_point
```

### Stage 2a: Box Expert GRPO

```bash
python scripts/run_stage2a_grpo_box.py \
    --model_path outputs/stage1a_sft_box \
    --output_dir outputs/stage2a_grpo_box
```

### Stage 2b: Point Expert GRPO

```bash
python scripts/run_stage2b_grpo_point.py \
    --model_path outputs/stage1b_sft_point \
    --output_dir outputs/stage2b_grpo_point
```

### Stage 3: Unified RFT（专家生成 rollout）

```bash
python scripts/run_stage3_rft_unified.py \
    --model_path outputs/stage0_5_merged_base \
    --output_dir outputs/stage3_rft_unified
```

### Stage 4: OPD（反向 KL 蒸馏）

```bash
python scripts/run_stage4_opd.py \
    --student_path outputs/stage3_rft_unified/final_model \
    --output_dir outputs/stage4_opd
```

### 一键运行（推荐）

```bash
bash scripts/run_iterdpo_pipeline.sh
```

---

## 🔬 推理示例

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

model, processor = load_qlora_model("outputs/stage4_opd")

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

---

## 🧠 关键技术设计

### Visual Primitive 格式

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

### 过程奖励函数 (Process Reward)

不同于仅看最终答案正确与否，我们设计了细粒度的过程奖励：

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
│   ├── stage0_pretrain.yaml
│   ├── stage0_5_visual_pretrain.yaml
│   ├── stage1a_sft_box.yaml
│   ├── stage1b_sft_point.yaml
│   ├── stage2a_grpo_box.yaml
│   ├── stage2b_grpo_point.yaml
│   ├── stage3_rft_unified.yaml
│   └── stage4_opd.yaml
├── src/
│   ├── models/
│   │   ├── qwen_vl_loader.py         # Qwen3VL + QLoRA 加载器
│   │   └── pretrain_loader.py        # Pretrain 模型加载 + embedding 注入
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
│   │   ├── opd_trainer.py            # OPD 反向 KL 蒸馏训练器
│   │   ├── callbacks.py              # 训练回调（内存监控）
│   │   └── memory_utils.py           # GPU 显存工具
│   └── utils/
│       ├── constants.py              # 特殊 token / 超参常量
│       ├── metrics.py                # Format RM + Accuracy RM + 难度分级
│       └── logging_utils.py          # 日志初始化
├── scripts/                          # 阶段入口脚本
│   ├── run_pretrain.py               # Stage 0: Text Pretrain
│   ├── run_stage0_5_visual_pretrain.py  # Stage 0.5: Visual Pretrain
│   ├── merge_stage0_5.py             # Stage 0.5 LoRA Merge
│   ├── run_stage1a_sft_box.py        # Stage 1a: Box Expert SFT
│   ├── run_stage1b_sft_point.py      # Stage 1b: Point Expert SFT
│   ├── run_stage2a_grpo_box.py       # Stage 2a: Box Expert GRPO
│   ├── run_stage2b_grpo_point.py     # Stage 2b: Point Expert GRPO
│   ├── run_stage3_rft_unified.py     # Stage 3: Unified RFT
│   ├── run_stage4_opd.py             # Stage 4: OPD
│   └── run_iterdpo_pipeline.sh       # Master Pipeline 一键运行
├── tests/
│   ├── test_primitive_parser.py      # 坐标解析单元测试
│   ├── test_metrics.py               # 奖励函数与几何工具测试
│   └── test_logging_utils.py         # 日志工具测试
├── requirements.txt
└── README.md
```

---

## 🧪 运行测试

```bash
pytest tests/ -v
```

---

## 📚 引用

如果你使用了本代码，请引用原始论文：

```bibtex
@article{deepseek2026thinking,
  title={Thinking with Visual Primitives},
  author={DeepSeek-AI},
  year={2026},
  note={Preprint. arXiv}
}
```

以及 Qwen3-VL：

```bibtex
@article{qwen3vl2025,
  title={Qwen3-VL: Advancing Multimodal Understanding and Agentic Ability},
  author={Qwen Team},
  journal={arXiv preprint arXiv:2504.01955},
  year={2025}
}
```

以及本项目：

```bibtex
@misc{tvp4b5090d2026,
  title={TVP-4B-5090D: Thinking with Visual Primitives on Qwen3-VL-4B},
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
