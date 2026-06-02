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

由于 24GB 显存无法容纳 284B MoE 的在线多 rollout 训练，本项目采用**轻量级三阶段 pipeline**，在保持核心思想不变的前提下，通过 **4-bit QLoRA + Gradient Checkpointing + Paged AdamW 8-bit** 实现单卡可跑。

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

## 🚀 训练流程（三阶段）

```
Stage 1: SFT Unified      混合 Box + Maze + Path 有监督微调    ~6h
Stage 2: GRPO             组相对策略优化，3 轮阈值收紧         ~10h
Stage 3: RFT              拒绝采样 + SFT 提纯                  ~2h
                        ─────────────────────────────────────────
                        Total:                                ~18h
```

### Stage 1: SFT Unified

在 COCO Grounding + 合成迷宫 + 合成路径数据上进行统一的有监督微调，让模型学会输出包含视觉原语的 thinking 格式。

```bash
python scripts/run_stage1_sft_unified.py \
    --config configs/stage1_sft_unified.yaml \
    --coco_image_dir data/coco/train2017 \
    --coco_ann_file data/coco/annotations/instances_train2017.json \
    --num_coco 40000 \
    --num_maze 50000 \
    --num_path 15000
```

**输出**: `outputs/stage1_sft_unified/`

### Stage 2: GRPO

使用 TRL 的 `GRPOTrainer` 进行在线强化学习。共 3 轮，每轮逐步收紧 Hard Negative 阈值，迫使模型输出更精确的坐标。

```bash
python scripts/run_stage2_grpo.py \
    --config configs/stage2_grpo.yaml \
    --model_path outputs/stage1_sft_unified \
    --coco_image_dir data/coco/train2017 \
    --coco_ann_file data/coco/annotations/instances_train2017.json \
    --num_rounds 3
```

| 轮次 | IoU 阈值 | 点距阈值 (px) | 目标 |
|-----|---------|--------------|------|
| R1  | 0.3     | 20.0         | 学会基本对齐 |
| R2  | 0.5     | 10.0         | 收紧精度要求 |
| R3  | 0.7     | 5.0          | 输出高质量坐标 |

**输出**: `outputs/stage2_grpo/round_1~3/`

### Stage 3: RFT (Rejection Sampling Fine-Tuning)

用 Stage 2 的最终模型对数据做 **5 次 rollout**，按 process reward 选出最优样本，再用 SFT 提纯。

```bash
python scripts/run_stage3_rft.py \
    --config configs/stage3_rft.yaml \
    --model_path outputs/stage2_grpo/round_3 \
    --coco_image_dir data/coco/train2017 \
    --coco_ann_file data/coco/annotations/instances_train2017.json \
    --accept_threshold 1.2
```

**输出**: `outputs/stage3_rft/final_model/`

---

## 🔬 推理示例

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

model, processor = load_qlora_model("outputs/stage3_rft/final_model")

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
<thinking>
I can see two cats in the image. Let me mark them.
<|box|>[[120, 80, 340, 290]]<|/box|>
<|box|>[[410, 95, 620, 310]]<|/box|>
</thinking>

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

单卡 24GB 可以同时容纳 **Policy 模型 + Reference 模型**（GRPO 阶段各约 6GB，加上激活值与缓存，峰值约 18GB）。

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
│   ├── stage1_sft_unified.yaml
│   ├── stage2_grpo.yaml
│   └── stage3_rft.yaml
├── src/
│   ├── models/
│   │   ├── qwen_vl_loader.py         # Qwen3VL + QLoRA 加载器
│   │   └── inference_engine.py       # 推理封装（adapter 切换）
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── sft_dataset.py        # SFT 数据集（assistant-only loss mask）
│   │   │   └── grpo_dataset.py       # GRPO 数据集
│   │   ├── generators/
│   │   │   ├── coco_box_generator.py # COCO 标注 → box 训练样本
│   │   │   ├── synthetic_maze.py     # 合成迷宫生成器
│   │   │   └── synthetic_path.py     # 合成路径生成器
│   │   └── formatters/
│   │       └── primitive_formatter.py # 坐标标签格式化
│   ├── training/
│   │   ├── trainers/
│   │   │   └── sft_trainer.py        # SFT Trainer 封装
│   │   ├── callbacks.py              # 训练回调（内存监控）
│   │   └── memory_utils.py           # GPU 显存工具
│   └── utils/
│       ├── constants.py              # 特殊 token / 超参常量
│       ├── metrics.py                # IoU / 撞墙 / 过程 reward
│       └── logging_utils.py          # 日志初始化
├── scripts/                          # 阶段入口脚本
│   ├── run_stage1_sft_unified.py
│   ├── run_stage2_grpo.py
│   ├── run_stage3_rft.py
│   └── run_full_pipeline.sh          # 全流水线（参考）
├── tests/
│   └── test_primitive_parser.py      # 坐标解析单元测试
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
  year={2026}
}
```

以及 Qwen3-VL：

```bibtex
@article{qwen3vl2025,
  title={Qwen3-VL: Advancing Multimodal Understanding and Agentic Ability},
  author={Qwen Team},
  journal={arXiv preprint},
  year={2025}
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
