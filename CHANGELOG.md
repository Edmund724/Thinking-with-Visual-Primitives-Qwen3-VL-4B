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

### Removed

- **vLLM dependency removed** — vLLM was incompatible with TRL GRPO generation (EOS bug, weight sync issues). GRPO now uses HuggingFace native generation exclusively.
  - Removed `vllm` from `requirements.txt`
  - Removed all `--use_vllm`, `--vllm_gpu_memory_utilization`, `--vllm_max_model_length`, `--vllm_enable_sleep_mode` flags from stage 4a/4b scripts
  - Removed vLLM parameters from `GRPOConfig` in both scripts

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
