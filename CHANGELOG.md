# Changelog

All notable changes to the GRPO training pipeline are documented in this file.

## [Unreleased]

### Changed

- **Environment dependency version bump (2026-06)**
  - `torch`: 2.6.0 → 2.11.0
  - `torchvision`: 0.21.0 → 0.26.0
  - `transformers`: 4.49.0 → 5.10.2 (major version bump)
  - `accelerate`: 1.2.0 → 1.13.0
  - `peft`: 0.14.0 → 0.19.1
  - `trl`: 0.15.0 → 1.6.0 (major version bump)
  - `bitsandbytes`: 0.45.0 → 0.49.2
  - `flash-attn`: 2.7.0 → 2.8.3
  - `datasets`: 3.0.0 → 4.8.5 (major version bump)
  - `pillow`: 11.0.0 → 12.2.0
  - `numpy`: 1.26.0 → 2.2.6 (major version bump)
  - `safetensors`: 0.5.0 → 0.7.0
  - `huggingface-hub`: 0.27.0 → 1.18.0 (major version bump)
  - CUDA install target: cu124 → cu130

- **stage4b max_completion_length 提升至 768** (`49be15f`)
  - 原因：maze GT 数据约 447 tokens，512 长度对早期训练 verbose 没有安全余量。
  - 文件：`scripts/run_stage4b_grpo_point.py`, `configs/stage4b_grpo_point.yaml`

- **stage4b batch_size 降为 3** (`8fde423`)
  - 在 max_completion_length=768 下平衡 5090D 显存。
  - 文件：`scripts/run_stage4b_grpo_point.py`, `configs/stage4b_grpo_point.yaml`

- **GRPO generation_batch_size 恢复为 num_generations** (`c6887cd`)
  - 从 `batch_size * num_generations` 改回 `num_generations`，每个 gradient step 重新生成 completion。
  - 原因：大 generation batch 在 TRL 1.6.0 + Qwen3-VL 下导致 image token / pixel_values / image_grid_thw 对齐错误。
  - 文件：`scripts/run_stage4a_grpo_box.py`, `scripts/run_stage4b_grpo_point.py`

### Removed

- **vLLM dependency removed** — vLLM was incompatible with TRL GRPO generation (EOS bug, weight sync issues). GRPO now uses HuggingFace native generation exclusively.
  - Removed `vllm` from `requirements.txt`
  - Removed all `--use_vllm`, `--vllm_gpu_memory_utilization`, `--vllm_max_model_length`, `--vllm_enable_sleep_mode` flags from stage 4a/4b scripts
  - Removed vLLM parameters from `GRPOConfig` in both scripts

### Fixed

- **Visual primitive tag format consistency (multi-box / multi-point bracket bug)**
  - `format_box` and `format_point` previously produced triple brackets for multiple coordinates, e.g. `<|box|>[[[x1,...],[x2,...]]]<|/box|>`. This confused the model and led to ~68% malformed tags in stage3a eval.
  - Fixed to always emit the consistent form: single `<|box|>[[x1,y1,x2,y2]]<|/box|>` and multi `<|box|>[[x1,y1,x2,y2],[x3,y3,x4,y4]]<|/box|>`.
  - Files: `src/data/formatters/primitive_formatter.py`, `scripts/generate_pretrain_data.py`

- **SFT final answer format and reasoning cleanup**
  - Removed the hard-coded `f"The answer is {answer}."` wrapper in `SFTDataset`; assistant content now uses the raw answer string, preserving `\boxed{...}` forms and reducing trailing-punctuation mismatch.
  - Removed the dangerous `reasoning.startswith("<")` / `reasoning.endswith("<")` cleanup that could strip visual primitive tags.
  - Files: `src/data/datasets/sft_dataset.py`

- **GRPO reward weaknesses exposed by stage3a eval**
  - `format_reward` now also rejects extra inner brackets like `[[[...]]]` inside a box/point tag, so malformed syntax is penalized during RL.
  - `compute_total_reward` for box tasks now gives a full exact-match reward for non-count answers (color / TrueFalse) instead of relying only on IoU.
  - Box GRPO length target raised from 120 to 240 tokens with a smaller max penalty, so valid multi-box / counting completions are no longer punished.
  - Files: `src/utils/metrics.py`, `scripts/run_stage4a_grpo_box.py`

### Changed

- **Unified grounding style across generators**
  - Coarse-grained counting, synthetic dense counting, and CLEVR counting/spatial-count questions now use a single visual primitive tag with all relevant boxes, matching the paper's batch-grounding protocol.
  - Previously some generators emitted one tag per box while others put multiple boxes in one tag, with inconsistent inner bracket formats.
  - Files: `src/data/generators/coco_box_generator.py`, `src/data/generators/clevr_spatial.py`

### Fixed

- **Compatibility warnings after major dependency upgrade (transformers 5.x / TRL 1.5.1 / PyTorch 2.11)**
  - Removed stale `BNB_CUDA_VERSION=130` from `~/.bashrc` — no longer needed because PyTorch, bitsandbytes, and flash-attn are all natively built for CUDA 13.0.
  - Eliminated `tokenizer has new PAD/BOS/EOS tokens` warning by syncing `model.config.pad_token_id`, `eos_token_id`, and `bos_token_id` with the tokenizer after `add_special_tokens()` in:
    - `src/models/qwen_vl_loader.py` (`load_qlora_model`, `load_reference_model`)
    - `src/models/pretrain_loader.py` (`load_pretrain_model`)
  - Eliminated `use_cache=True is incompatible with gradient checkpointing` warning by explicitly setting `use_cache=False` recursively on all nested config objects via `_set_use_cache_deep()`:
    - Root cause: `Qwen3VLTextModel.forward` has a `@merge_with_config_defaults` decorator that reads `self.config.use_cache` from the innermost `Qwen3VLTextConfig`. A top-level `model.config.use_cache = False` on PeftModel/ForConditionalGeneration does NOT reach this deep config.
    - Added `_set_use_cache_deep()` helper in `src/models/qwen_vl_loader.py` that recursively walks `nn.Module.children()` and sets `use_cache=False` on every config found.
    - Called in: `load_qlora_model`, `create_sft_trainer`, `run_stage4a_grpo_box.py`, `run_stage4b_grpo_point.py`
  - Verified: stage1–stage3 scripts run without errors in the upgraded environment.

- **GRPO multimodal field mismatch with Qwen3-VL**  
  Root cause: TRL's `_generate_and_score_completions` builds `mm_token_type_ids` from `processing_class` which **right-pads**, while TRL **left-pads** `prompt_ids`. This causes `attention_mask` and `mm_token_type_ids` to disagree on padded positions, leading to `RuntimeError: shape mismatch` in Qwen3-VL's `get_rope_index`.
  - Fix 1: In `_get_per_token_logps_and_entropies`, rebuild `mm_token_type_ids` / `token_type_ids` from the actual `input_ids` (which has correct left-padding).
  - Fix 2: In `_generate`, strip generated image/video pad tokens from `completion_ids` to prevent orphan image tokens (no matching `pixel_values` features) from causing `ValueError: Image features and image tokens do not match`.
  - File: `src/training/grpo_fixes.py`

- **GRPO image not passed to model (critical)**
  - Root cause: `GRPODataset` put images in a standalone `"image"` key, but TRL 1.5.1 GRPOTrainer expects images embedded in message content as `{"type": "image", "image": <PIL>}` blocks. Images were silently ignored → model generated without visual input → all rewards 0.
  - Fix: Updated `GRPODataset.__getitem__` to embed images in user message content using TRL's multimodal format.
  - File: `src/data/datasets/grpo_dataset.py`

- **GRPO monkey-patches still required under TRL 1.6.0**
  - A minimal verification run without `src/training/grpo_fixes.py` appeared to pass, but full-scale training later failed with `ValueError: Image features and image tokens do not match, tokens: 769, features: 768` at step 2542/5000.
  - Root cause: the model occasionally emits an extra image/video pad token in the completion, creating a mismatch between the number of image tokens and the pre-computed `pixel_values` / `image_grid_thw` features.
  - Fix: Restored `src/training/grpo_fixes.py`, `tests/test_grpo_fixes.py`, and the `apply_grpo_fixes(GRPOTrainer)` calls in `scripts/run_stage4a_grpo_box.py` and `scripts/run_stage4b_grpo_point.py`.
  - Note: the small-scale verification script was removed; the only reliable test is the full training run.

- **GRPO format_reward incompatible with Qwen3-VL chat template**
  - Root cause: Qwen3-VL-Thinking chat template prepends `<think>` to the prompt, so GRPO completions only contain `</think>` (not `<think>`). `format_reward` required both → always failed → 0.2 reward lost.
  - Fix: Updated `format_reward` to accept completions with only `</think>`.
  - File: `src/utils/metrics.py`

- **GRPO reward function: added length penalty to fix zero within-group variance**
  - Root cause: reward function (`compute_total_reward`) was completely insensitive to completion length. Model had no incentive to generate EOS, so all completions were clipped at `max_completion_length`. Within each group, rewards were nearly identical → `frac_reward_zero_std≈1` → GRPO Advantage≈0 → near-zero loss.
  - Fix: Added two length penalties in `compute_total_reward`:
    1. **Truncation penalty (-0.15)**: if completion length ≥ 95% of max limit, penalize (model failed to stop naturally).
    2. **General length penalty**: if completion exceeds 1.5× target length, apply linear penalty up to -0.1.
  - Files: `src/utils/metrics.py`, `scripts/run_stage4b_grpo_point.py`, `scripts/run_stage4a_grpo_box.py`

- **GRPO max_completion_length increased 512 → 768 → 1024**
  - Maze GT data reaches ~447 tokens; 512 left almost no safety margin for early-training verbosity.
  - 1024 gives sufficient breathing room.
  - Files: `scripts/run_stage4b_grpo_point.py`, `scripts/run_stage4a_grpo_box.py`

- **GRPO VRAM growth / repeated OOM kills during long runs**
  - Root cause 1: `apply_grpo_fixes()` was called inside the per-round loop, causing the monkey-patches to wrap themselves every round. The nested wrappers and accidental in-place mutation of `input_ids` could increase memory pressure and corrupt reused tensors.
  - Root cause 2: TRL GRPO with Qwen3-VL is known to fragment CUDA memory because generated completions vary in length (`max_completion_length` up to 1024). Fragmentation causes the allocator to reserve more and more memory over thousands of steps until the process is OOM-killed.
  - Fix:
    1. Made `apply_grpo_fixes()` idempotent and moved the call outside the round loop in `scripts/run_stage4a_grpo_box.py` and `scripts/run_stage4b_grpo_point.py`.
    2. In `_patch_get_per_token_logps_and_entropies`, clone `input_ids` before truncating orphan image/video pad tokens instead of mutating the caller's tensor in-place.
    3. Added explicit cleanup between rounds: `del trainer`, `del policy_model`, `gc.collect()`, `clear_memory()`.
    4. Added `GPUMemoryMonitor` callback to each `GRPOTrainer` so the cache is aggressively cleared when allocated memory exceeds the configured threshold.
    5. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at the top of both scripts to reduce fragmentation.
  - Files: `src/training/grpo_fixes.py`, `scripts/run_stage4a_grpo_box.py`, `scripts/run_stage4b_grpo_point.py`

- **Stage 5: Unified RFT VRAM and cleanup issues**
  - Root cause 1: `scripts/run_stage5_rft_unified.py` did not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, leaving it vulnerable to the same CUDA fragmentation that caused stage 4 OOMs during long runs with variable-length completions.
  - Root cause 2: Box Expert and Point Expert models remained in GPU memory during the Unified model SFT phase, wasting VRAM.
  - Fix:
    1. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at the top of `scripts/run_stage5_rft_unified.py`.
    2. After difficulty grading / rejection sampling, explicitly `del box_expert; del point_expert; gc.collect(); clear_memory()` before constructing the SFT trainer.
  - Files: `scripts/run_stage5_rft_unified.py`

- **Stage 6: OPD image not passed to model (critical)**
  - Root cause: `OPDDataset.__getitem__` only returned text `prompt_ids`; `pixel_values` / `image_grid_thw` were never computed or passed to `student_model.generate()`, `student_model()`, or `expert()`. The models therefore processed text-only prompts and ignored the input image, making the distillation target meaningless for visual tasks.
  - Fix:
    1. Updated `OPDDataset.__getitem__` to load the image and process prompt + image through the processor, returning `pixel_values` and `image_grid_thw` alongside `input_ids`.
    2. Added `_opd_collate` to correctly batch/concatenate `pixel_values` and stack `image_grid_thw`.
    3. Threaded image kwargs through `student_model.generate()`, `student_model()`, and `expert()` in `train_opd`.
  - Also fixed: `generate` temperature was hard-coded to `0.7` instead of using the configured `temperature` argument.
  - Also added: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and per-epoch `torch.cuda.empty_cache()` to reduce fragmentation from variable-length student completions; expert models are released after OPD training before saving the final student.
  - Files: `src/training/opd_trainer.py`, `scripts/run_stage6_opd.py`

- **GRPO 多模态猴补丁恢复与修正** (`79f034f`, `5576938`, `d84aaa2`, `b78f435`)
  - 曾误以为 TRL 1.6.0 原生处理多模态对齐，移除 `src/training/grpo_fixes.py`；实际长训练仍触发 `Image features and image tokens do not match`。
  - 恢复并调整猴补丁逻辑：仅在 shape 不匹配时从 `input_ids` 重建 `mm_token_type_ids`，并剥离 completion 中的 orphan image/video pad tokens。
  - 尝试 always-rebuild 后出现 features > tokens 的 shape mismatch，最终回滚到 `a5f4baf` 原始逻辑。
  - 文件：`src/training/grpo_fixes.py`, `tests/test_grpo_fixes.py`, `scripts/run_stage4a_grpo_box.py`, `scripts/run_stage4b_grpo_point.py`

- **解决 GRPO/SFT 输出中的非英文学符乱码** (`9ba2ce8`)
  - 在 system prompt 中明确要求英文输出。
  - `format_reward` 增加非拉丁文字惩罚（西里尔、阿拉伯、CJK、泰文、希腊等），每个字符扣 0.01，最多扣 0.2。
  - 文件：`src/data/datasets/grpo_dataset.py`, `src/data/datasets/sft_dataset.py`, `src/utils/metrics.py`

### Added

- **Round 内 checkpoint-* 断点续训**
  - 之前脚本只在整轮完成后跳过，round 内中途 OOM 会从头重跑。现在每轮开始前会自动查找 `round_N/checkpoint-*` 中 step 最大的目录：
    - 存在 checkpoint：从该 checkpoint 加载 policy model 和 processor，并把路径传给 `trainer.train(resume_from_checkpoint=...)` 恢复 optimizer / scheduler / rng / trainer_state。
    - 不存在 checkpoint：按原逻辑从上一轮（或 stage3 SFT）初始化。
  - 由于 checkpoint 里同时保存了 `default`（当前策略）和 `ref/`（参考策略）两个 PEFT adapter，`resume_from_checkpoint` 会把两者都恢复，GRPO 的 KL 参考点不会错位。
  - 文件：`scripts/run_stage4a_grpo_box.py`、`scripts/run_stage4b_grpo_point.py`

- `src/training/grpo_fixes.py` — Monkey-patch module for TRL GRPOTrainer multimodal alignment
  - Fix 1: Rebuild `mm_token_type_ids` from actual `input_ids` to fix padding direction mismatch.
  - Fix 2: Strip orphan image/video pad tokens from `completion_ids`.
  - Fix 3: Log first completion every 5 steps for monitoring.

- **COCO 几何过滤 (Geometric Filtering)**
  - 新增 `_filter_annotations_by_geometry`，过滤 mega box (>90% 图像面积)、tiny box (<0.01% 面积)、退化 box 和强贴边 box。
  - 在 `generate_coco_box_samples` 和 `generate_coco_point_samples` 中自动应用。
  - 文件：`src/data/generators/coco_box_generator.py`

- **Thinking-chain 验证器 (Cold-start 数据校验)**
  - 新增 `src/utils/thinking_verifier.py`：检查 tag 配对、坐标范围、引用有效性、counting 答案与 primitive 数量一致性、maze 自相矛盾。
  - 集成到 COCO box/point、合成 dense counting、maze 生成器中，生成后自动过滤不合格样本。
  - 文件：`src/utils/thinking_verifier.py`, `src/data/generators/coco_box_generator.py`, `src/data/generators/synthetic_maze.py`

- **Coarse-grained Counting 数据生成器**
  - 新增 `generate_coco_counting_samples`：从 COCO 选择 3–30 实例的类别，按论文 3-step thinking 协议生成 batch grounding + count answer。
  - 集成到 `scripts/run_stage3a_sft_box.py` 和 `scripts/run_stage4a_grpo_box.py`。
  - 文件：`src/data/generators/coco_box_generator.py`, `scripts/run_stage3a_sft_box.py`, `scripts/run_stage4a_grpo_box.py`

- **CLEVR-style Spatial / VQA 数据生成器**
  - 新增 `src/data/generators/clevr_spatial.py`：生成 2D 合成场景（球/立方体/圆柱体），支持 counting、spatial existence、spatial count、attribute query 四类问题。
  - 集成到 Stage 3a SFT、Stage 4a GRPO 和 Stage 5 RFT 的 prompt pool。
  - 文件：`src/data/generators/clevr_spatial.py`, `scripts/run_stage3a_sft_box.py`, `scripts/run_stage4a_grpo_box.py`, `scripts/run_stage5_rft_unified.py`

- **Path Tracing 数据生成器**
  - 新增 `src/data/generators/path_tracing.py`：生成缠绕的 Bézier 曲线，随机选择一条作为目标路径，输出 waypoint 序列作为 thinking，答案为终点标签。
  - 支持 uniform-style 模式（所有线同色），迫使模型依赖曲率连续性而非颜色。
  - 集成到 `scripts/run_stage3b_sft_point.py` 和 `scripts/run_stage4b_grpo_point.py`。
  - 文件：`src/data/generators/path_tracing.py`, `scripts/run_stage3b_sft_point.py`, `scripts/run_stage4b_grpo_point.py`

- **Stage 5 RFT Prompt Pool 扩展**
  - Rejection sampling 的 prompt pool 新增 coarse-grained counting 和 CLEVR spatial/VQA，与 box/point/maze 一起用于生成专家 rollout。
  - 文件：`scripts/run_stage5_rft_unified.py`

- **代码清理**
  - 删除 `src/data/generators/coco_box_generator.py` 中未使用的 `Path` import。


- **统一设置 CUDA 显存碎片缓解环境变量**
  - 在 `scripts/run_stage1_pretrain.py`、`run_stage2_visual_pretrain.py`、`run_stage3a_sft_box.py`、`run_stage3b_sft_point.py` 顶部统一设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  - 现在 stage1–stage6 全部脚本都内置该环境变量，无需每次手动在命令行添加。
  - 文件：`scripts/run_stage1_pretrain.py`, `scripts/run_stage2_visual_pretrain.py`, `scripts/run_stage3a_sft_box.py`, `scripts/run_stage3b_sft_point.py`
