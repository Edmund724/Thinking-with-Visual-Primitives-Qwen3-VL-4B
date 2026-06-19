# Changelog

All notable changes to the GRPO training pipeline are documented in this file.

## [Unreleased]

### Changed

- **YAML duplicate keys cleaned + argparse defaults unified to `None`**
  - Fixed duplicate `num_epochs` in `configs/stage2_visual_pretrain.yaml` (was `1` then `2`; removed the dead `1`).
  - Fixed duplicate `early_stopping_*` block in `configs/stage4a_grpo_box.yaml` (was `0/50/2` then `16/50/2`; removed the dead first set).
  - All 8 stage scripts: `add_arg(default=<concrete>)` → `default=None` (~120 arguments). YAML configs are now the sole default source; `action="store_true"` flags unchanged.
  - Fixed latent bug in `run_stage1_pretrain.py`: only 5 args were registered but `train()` accessed ~17 — now all registered; `configs/stage1_pretrain.yaml` expanded with visual-phase, ViT, and `max_seq_length` keys.
  - 5 standalone scripts (`merge_stage2`, `smoke_test_stage2`, `eval_stage2_structure`, `eval_stage3a_paradigm`, `diagnose_stage2_resume_loss`): all `default=<concrete>` → `default=None`.
  - Fixed `--config` CLI override in `StageRunner.parse_args()`: `self.args.config` was never read, so CLI `--config` was effectively dead. Now correctly synced and defaults to `None`.
  - `apply_yaml_defaults` correctly handles `None == None` comparison for the three-layer default cascade (argparse `None` → YAML value → CLI override).

- **PrimitiveParser upgraded to a true domain seam**
  - Extended `PrimitiveParser` from 7 methods to 32 methods, now covering all concerns:
    - **Parsing**: `extract_answer`, `extract_reasoning`, `split_generated_text`, `normalize_answer_text`, `lenient_extract_boxes`
    - **Formatting**: `format_box`, `format_point`, `clean_primitive_tags`, `normalize_coordinate`, `denormalize_coordinate`
    - **Geometry**: `box_iou`, `match_boxes`, `point_distance`, `match_points`, 5 maze scoring functions, `has_duplicate_coords`, `count_repeated_coordinates`, `check_backtracking_missing`
    - **Existing**: `extract_boxes`, `extract_points`, `validate_syntax`, `validate_coordinates`, `check_wall_collision`, `check_wall_collision_points`, `count_tags`, `has_backtracking_keywords`
  - Updated 11 production files to route through `PrimitiveParser` instead of directly importing `text_parsing.py` / `geometry.py` / `primitive_formatter.py`:
    - `src/utils/reward/accuracy_rm.py`, `quality_rm.py`, `difficulty.py`
    - `scripts/eval_stage2_structure.py`, `scripts/run_stage5_rft_unified.py`
    - 5 generator files (`coco_box_generator`, `clevr_spatial`, `path_tracing`, `synthetic_path`, `synthetic_maze`)
  - `src/utils/metrics.py` now re-exports `PrimitiveParser` alongside the legacy flat functions for backward compatibility.
  - Added 18 new test cases in `tests/test_primitive_parser.py`.

- **Upgraded Quality RM LLM Judge (API-based)**
  - Improved judge prompt in `src/utils/quality_rm_api.py`: chain-of-thought evaluation with 6 quality dimensions (redundancy, consistency, contradiction, reward hacking, self-contradiction, meaningful references) → structured `Score: X.X` output.
  - Score parser updated to handle both new `Score: X.X` format and legacy bare-number format.
  - **Subset sampling** via `QUALITY_RM_SAMPLE_RATIO` env var (default 0.3): only a random fraction of completions go through the API judge; the rest use the fast rule-based fallback. Reduces API cost by ~70%.
  - Increased API `max_tokens` from 10 → 150 to accommodate brief reasoning output.
  - `.env.example` updated with `QUALITY_RM_SAMPLE_RATIO` documentation.

- **Stage 1 now supports real images (visual grounding pretrain)**
  - New `train_pretrain_visual()` in `src/training/pretrain_trainer.py` — uses `SFTDataset` for image handling in a custom PyTorch loop.
  - Stage 1 CLI flags: `--visual_data_ratio`, `--visual_num_box`, `--visual_num_point`, `--visual_epochs`, `--visual_learning_rate`, `--visual_batch_size`.
  - When `--visual_data_ratio > 0`, COCO box/point samples are generated and trained after text pretrain.
  - Closer to the paper's "large-scale grounding pretraining" Stage 1.

- **ViT last-layer unfreezing (experimental)**
  - `load_pretrain_model()` and `load_qlora_model()` now accept `unfreeze_vit_layers: int = 0`.
  - When > 0, unfreezes `model.visual.blocks[-N:]` + `model.visual.merger`.
  - New `build_param_groups()` helper in `src/training/memory_utils.py` assigns per-group LRs (ViT blocks: 1e-6, merger: 1e-5, LLM: normal).
  - CLI flags: `--unfreeze_vit_layers`, `--vit_lr` added to stages 1 and 2.

- **Refactored `src/utils/metrics.py` into focused modules**
  - Split the 1500+ line file into:
    - `src/utils/text_parsing.py`: answer / reasoning / box / point parsing
    - `src/utils/geometry.py`: IoU, point distance, maze geometry
    - `src/utils/reward/format_rm.py`: Format RM
    - `src/utils/reward/quality_rm.py`: Quality RM
    - `src/utils/reward/accuracy_rm.py`: Accuracy RM (`process_reward`, `compute_total_reward`)
    - `src/utils/difficulty.py`: Easy/Normal/Hard difficulty grading
  - `src/utils/metrics.py` remains as a backward-compatible shim re-exporting the public API.
  - Updated internal imports in stage scripts, `visual_primitive_parser.py`, and `quality_rm_api.py` to use the new modules directly.
  - Fixed incorrect `extract_completion_text` import in `src/utils/quality_rm_api.py` (was imported from `.metrics`, now from `..training.grpo_utils`).
  - Updated `tests/test_filter_normal_level_data.py` patch targets to match the new module locations.

- **Introduced `ConversationBuilder` to unify message construction**
  - New `src/utils/conversation_builder.py` with mode-based system messages (`sft`, `grpo`, `opd`, `pretrain`) and composable methods: `build_prompt()`, `build_sft()`, `build_pretrain()`, `build_gt_text()`, `build_user_content()`.
  - Wired into: `sft_dataset.py`, `grpo_dataset.py`, `batch_inference.py`, `opd_trainer.py`, `generate_pretrain_data.py`, `eval_stage2_structure.py`, `eval_stage3a_paradigm.py`, `smoke_test_stage2.py`, `run_stage5_rft_unified.py`.
  - Eliminates ~110 lines of duplicated message-building across 9 files.

- **Introduced `StageRunner` to eliminate stage script boilerplate**
  - New `src/training/stage_runner.py` handles: `PYTORCH_CUDA_ALLOC_CONF`, `sys.path`, argparse + YAML defaults, logging setup, `torch.cuda.empty_cache()` banners, and `pickle` data-cache pattern (via `runner.cached_data()`).
  - All 8 stage scripts (`run_stage1..6*.py`) refactored to use `StageRunner` with callback-driven `train(runner)` functions.
  - Eliminates ~220 lines of duplicated boilerplate across stage scripts.

- **Added unified generator registry**
  - `src/data/generators/__init__.py` now exports a `GENERATORS` dict mapping task names to generator functions, and re-exports all public generator APIs.
  - Backward-compatible: direct imports from individual generator modules still work.

### Added

- **Stage integration tests** (`tests/test_stage_integration.py`)
  - 14 tests covering all 8 training stages: each test generates data with minimal sample counts and verifies data shape, task types, and (for Stage 1) runs an actual forward pass.
  - Stage 1: text pretrain generation + forward pass through 4-bit base model.
  - Stage 2: COCO box/point sample generation.
  - Stage 3a: box/counting/CLEVR sample generation.
  - Stage 3b: point/maze/path sample generation.
  - Stage 4a: GRPO Box data type mixture.
  - Stage 4b: GRPO Point data type mixture.
  - Stage 5: Unified RFT all-prompt-type generation (box/counting/CLEVR/point/maze/path).
  - Stage 6: OPD box/point/maze sample generation.
  - All tests use `pytest.mark.skipif` to gracefully skip when models or COCO data are not present.

- **Stage 1 lightweight format SFT**
  - `load_pretrain_model()` in `src/models/pretrain_loader.py` now unfreezes the last 2 decoder layers in addition to `embed_tokens` / `lm_head`.
  - This moves Stage 1 from pure embedding initialization to a lightweight format pretrain that learns the conditional pattern of emitting visual primitives inside `<think>` chains, better matching the paper's pretraining objective.

- **Stage 1/2 data format alignment with paper**
  - `scripts/generate_pretrain_data.py`: samples now include a system message and wrap the assistant reply in `<think>...</think>`, with a natural-language sentence plus the primitive tags.
  - `src/data/generators/coco_box_generator.py` and `coco_point_generator`: `use_thinking=False` (Stage 2 visual pretrain) now emits natural-language reasoning that introduces the primitive tags, instead of bare `<|ref|>...<|box|>...` strings.
  - `src/training/pretrain_trainer.py`: prompt masking now uses the last message as the assistant target, supporting the new 3-message format.

- **Weighted SFT loss for format tokens**
  - New `WeightedSFTTrainer` in `src/training/trainers/sft_trainer.py` applies per-token loss weights.
  - `SFTDataset` computes `loss_weight`: visual primitive tokens (`<|box|>`, `<|/box|>`, `<|point|>`, `<|/point|>`) and `<think>` / `</think>` are up-weighted (default `format_token_weight=5.0`).
  - Stage 3a exposes `--format_token_weight`.

- **SFT target data cleaning**
  - `clean_primitive_tags()` in `src/data/formatters/primitive_formatter.py` fixes reversed, duplicate, or bad-variant primitive tags before training.
  - Integrated into `scripts/run_stage3a_sft_box.py` for all box/point samples.

- **Stage 3a resume-from-checkpoint support**
  - Fixed `SFTDataset` attribute bug (`format_token_ids` → `_format_token_ids`) that broke resume.
  - README documents resume command.

- **Stricter non-Latin / format reward signals**
  - `format_reward` non-Latin penalty increased from max -0.2 to max -1.0.
  - `quality_reward_text` treats non-Latin script as a major issue (0 reward).
  - `is_rollout_correct` rejects any output containing non-Latin characters.
  - Added `primitive_format_compliance_reward` for paired/ordered tags and `box_count_answer_consistency_reward` for matching box count to numeric answer.

- **Lenient box parsing for difficulty grading**
  - `lenient_parse_boxes()` extracts `[[x1,y1,x2,y2]]` arrays even when tags are missing or wrong order.
  - `is_rollout_correct` uses normalized numeric/boolean answer matching.

- **Batched generation system prompt enforces English**
  - `src/utils/batch_inference.py` now adds "Respond in English only; do not use characters from other languages."

### Fixed

- **`merge_stage2.py` now preserves Stage 1 embeddings**
  - Adds special tokens, resizes embeddings, and injects `outputs/stage1_pretrain/pretrain_state_dict.pt` before loading/merging the Stage 2 LoRA adapter.
  - Without this, special-token embeddings in the merged base were randomly initialized.

- **`scripts/run_stage3a_sft_box.py` UnboundLocalError**
  - Removed premature `all_data.extend(negative_box_data)` referencing `all_data` before assignment.

- **`filter_normal_level_data` NameError**
  - Undefined variable `g` replaced with `num_generations`.

- **`format_reward` no_nested_tokens false positive**
  - No longer flags valid inner `[[...]]` brackets as nested tags.

- **GRPO `generation_batch_size` compatibility**
  - Set to `args.batch_size` so it is divisible by per-device train batch size in TRL 1.6.0.

### Changed

- **Stage 1/2 data scale increased**
  - Stage 1: `num_samples` 10K → 30K, `num_epochs` 2 → 3.
  - Stage 2: `num_box` 15K → 30K, `num_point` 5K → 10K, `num_epochs` 1 → 2.

- **Stage 3a config restored and strengthened**
  - `num_box` 8K → 15K, `num_counting` 5K → 10K, `num_clevr` 3K → 5K, `num_negative_box` 1K → 2K.
  - `max_seq_length` 2048 → 4096, `num_epochs` 1 → 2.
  - Fixed config keys so `num_epochs` and `batch_size` are actually applied.

- **Stage 4a early stopping disabled by default**
  - `early_stopping_subset_size: 0` in `configs/stage4a_grpo_box.yaml` to avoid premature stops on small/noisy validation subsets.
  - `max_completion_length` and `filter_max_completion_length` raised to 384.

### Documentation

- **README 与 requirements.txt 版本标注修正**
  - `flash-attn` 版本在 README.md、README_zh.md 和 requirements.txt 中明确标注为 `2.8.3`（实际安装版本为 `2.8.3.post1`）。
  - `wandb` 最低版本从 `>=0.19.0` 修正为 `>=0.27.0`（与实际安装版本对齐）。

### Added

- **API-based Quality RM (LLM-as-Judge)**
  - New `src/utils/quality_rm_api.py` with `quality_reward_api()` and `make_quality_reward_api_fn()`.
  - Reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `QUALITY_RM_MODEL` from `.env`.
  - Falls back to rule-based `quality_reward_text` if API is unavailable or fails.
  - Stage 4a/4b add `--use_quality_rm_api` flag and config key.
  - Added `python-dotenv` and `openai` to `requirements.txt`.

- **CLEVR question types extended**
  - Added existence, compare-integer, query-material, and 2-hop multi-hop questions.
  - Added `material` attribute with simple visual cues (metal highlight, matte border).
  - File: `src/data/generators/clevr_spatial.py`

- **Maze backtracking traps**
  - `add_backtracking_trap()` carves dead-end corridors off the solution path.
  - `generate_maze_dataset()` exposes `backtracking_trap_ratio`.
  - File: `src/data/generators/synthetic_maze.py`

- **COCO counting with attribute constraints**
  - `generate_coco_counting_samples()` supports `attribute_constraint_ratio`.
  - Adds color (dominant bbox color) and size (area ratio) constraints.
  - Stage 3a exposes `--counting_attribute_ratio`.

- **Offline CLEVR augmentation**
  - `generate_scene()` supports mild brightness/contrast jitter and random occlusion patches.
  - Enabled by default via `augment=True`.

- **Stage 1/2 curriculum**
  - `generate_dataset()` supports `curriculum` (sort by complexity).
  - Stage 1/2 scripts add `--curriculum` flag; configs enable it.

- **Repeat-token penalty in reward**
  - `repeat_token_penalty()` detects repeated n-grams and duplicate coordinates.
  - Integrated into `compute_total_reward()`.

- **Batched generation helper**
  - New `src/utils/batch_inference.py` with `batch_generate_completions()` and `generate_single_completion()`.
  - `filter_normal_level_data()` now uses the helper and falls back to singles on failure.

- **Early stopping + torch.compile support**
  - `ValidationSubsetEarlyStoppingCallback` evaluates a small subset every N steps.
  - `maybe_compile_model()` best-effort wraps model with `torch.compile`.
  - Stage 4a/4b add `--compile_model`, `--early_stopping_subset_size`, etc.

- **Stage 1 config file**
  - Added `configs/stage1_pretrain.yaml`; Stage 1 script now supports `--config`.

### Removed

- **ModelScope upload section** removed from `README.md` and `README_zh.md`.

### Changed

- **Stage 1/2/Merge actual run times updated in README**
  - Stage 1: 10K samples, 2 epochs, batch_size=4 → **~23min** (was ~57min with 25K/3epochs)
  - Stage 2: 15K box + 5K point, 1 epoch, curriculum → **~2h23min** (was ~9h36min with 60K/2epochs)
  - Merge Stage 2: **~27s** (was ~22s)
  - Updated in both `README.md` and `README_zh.md`.

- **README disclaimer**: Added explicit note that default configs use small sample sizes for fast run-through and do not guarantee high-quality final weights.

- **Stage 3 negative sample ratio raised** from 0.15 to 0.25.
- **All stage configs trimmed** for faster run-through while preserving pipeline shape:
  - Stage 1: 25K → 10K samples, 3 → 2 epochs.
  - Stage 2: 50K box / 10K point → 15K box / 5K point, 2 → 1 epoch.
  - Stage 3a: 15K box / 10K counting / 5K CLEVR → 8K / 5K / 3K.
  - Stage 3b: 50K maze / 10K point / 10K path → 10K / 5K / 5K.
  - Stage 4a/4b: `num_generations` 6 → 2, `num_rounds` 3 → 2, batch/GA tuned.
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

- **GRPO 难度筛选改为按“正确 rollout 数量”分级**
  - `src/utils/metrics.py` 新增 `is_rollout_correct`，以“答案正确 + 语法合法”作为 binary correct 判定。
  - `filter_normal_level_data` 和 `scripts/run_stage5_rft_unified.py` 的 `difficulty_grading` 不再使用 reward threshold，改为统计 correct rollout 数量来划分 Easy/Normal/Hard，对齐论文 Sec 2.5.2 / 2.5.3。
  - 移除 `scripts/run_stage4a_grpo_box.py` 的 `--filter_correct_threshold` 参数。

- **Quality RM 规则增强**
  - `src/utils/metrics.py` 的 `quality_reward_text` 新增 self-contradiction（“没有 X”但输出 box/point）、更细粒度的 reward-hacking 与一致性检查，作为论文 LLM-based GRM 的单卡近似。

- **stage4b max_completion_length 提升至 768** (`49be15f`)
  - 原因：maze GT 数据约 447 tokens，512 长度对早期训练 verbose 没有安全余量。
  - stage4a 保持 384（box 任务较短即可容纳）。
  - 文件：`scripts/run_stage4a_grpo_box.py`, `scripts/run_stage4b_grpo_point.py`, `configs/stage4a_grpo_box.yaml`, `configs/stage4b_grpo_point.yaml`

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

- **GRPO max_completion_length increased 512 → 768 (stage4b)**
  - Maze GT data reaches ~447 tokens; 512 left almost no safety margin for early-training verbosity.
  - stage4b 使用 768；stage4a 针对 box 任务保持 384。
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

- **`src/training/grpo_utils.py` — GRPO helper utilities**
  - 提供 `extract_completion_text` 等工具函数，统一从 TRL GRPO completion 中解码保留特殊 token 的文本，供 reward 函数和评估复用。
  - 已在 Stage 4a/4b GRPO 脚本中导入使用。

- **`src/utils/config_utils.py` — YAML 配置加载**
  - 为所有带 YAML 配置的阶段脚本（stage2、3a、3b、4a、4b、5、6）提供 `apply_yaml_defaults`，使 `configs/*.yaml` 成为默认超参数来源，CLI 参数仍可覆盖。

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

- **Stage 1 可选 COCO grounding 混合**
  - `scripts/generate_pretrain_data.py` 新增 `generate_coco_grounding_pretrain_samples`：从 COCO 标注中采样真实类别与坐标，生成文本-only 预训练样本。
  - `scripts/run_stage1_pretrain.py` 新增 `--coco_grounding_ratio`（默认 0），在格式预训练阶段即可引入真实 grounding 分布，向论文 Sec 2.3 靠近。

- **Cold-start 负样本增强（Faithful Refusal）**
  - CLEVR spatial generator 新增 `negative_ratio` 参数，生成“查询不存在颜色/形状组合”的负样本，答案为 `\boxed{False}` 且不输出 box。
  - COCO box/point generator 新增 `generate_coco_negative_box_samples` / `generate_coco_negative_point_samples`，询问图像中不存在的类别，训练模型忠实拒绝而非幻觉框/点。
  - Stage 3a/3b 脚本默认混入负样本，对齐论文 Sec 2.4.2 的 negative sample augmentation。
  - 文件：`src/data/generators/clevr_spatial.py`、`src/data/generators/coco_box_generator.py`、`scripts/run_stage3a_sft_box.py`、`scripts/run_stage3b_sft_point.py`

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
