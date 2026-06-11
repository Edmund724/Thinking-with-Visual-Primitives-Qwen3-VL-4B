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
  - `trl`: 0.15.0 → 1.5.1 (major version bump)
  - `bitsandbytes`: 0.45.0 → 0.49.2
  - `flash-attn`: 2.7.0 → 2.8.3
  - `datasets`: 3.0.0 → 4.8.5 (major version bump)
  - `pillow`: 11.0.0 → 12.2.0
  - `numpy`: 1.26.0 → 2.2.6 (major version bump)
  - `safetensors`: 0.5.0 → 0.7.0
  - `huggingface-hub`: 0.27.0 → 1.18.0 (major version bump)
  - CUDA install target: cu124 → cu130

### Added

- **vLLM integration for GRPO acceleration**
  - New dependency: `vllm>=0.22.0` (optional, for GRPO generation acceleration)
  - Stage 4a/4b GRPO scripts now support `--use_vllm`, `--vllm_gpu_memory_utilization`, `--vllm_max_model_length`, `--vllm_enable_sleep_mode` flags
  - vLLM uses `colocate` mode for single-GPU training
  - Updated README.md and README_zh.md with vLLM installation and usage instructions

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

### Changed

- **`scripts/run_stage4b_grpo_point.py`**
  - Added vLLM acceleration options: `--use_vllm`, `--vllm_gpu_memory_utilization`, `--vllm_max_model_length`, `--vllm_enable_sleep_mode`
  - Adjusted defaults for **24GB VRAM**:
    - `batch_size`: 1 → 2
    - `gradient_accumulation_steps`: 4 → 6 (keeps effective batch ~12)
    - `num_generations`: 5 → 3
    - `max_completion_length`: 1024 → 512
  - `gradient_checkpointing=False` (required for vLLM compatibility)

- **`scripts/run_stage4a_grpo_box.py`**
  - Same vLLM options and default parameter adjustments as stage4b
  - Changed `gradient_checkpointing=True` → `False` (vLLM incompatible)

### Added

- `src/training/grpo_fixes.py` — Monkey-patch module for TRL GRPOTrainer multimodal alignment
- **Safety protection for vLLM LoRA merge/unmerge** — Added `unmerge_adapter()` guard before `trainer.save_model()` in both stage4a/4b scripts to prevent saving potentially merged (polluted) weights if vLLM weight sync was interrupted.

### Fixed

- **GRPO reward function: added length penalty to fix zero within-group variance**
  - Root cause: reward function (`compute_total_reward`) was completely insensitive to completion length. Model had no incentive to generate EOS, so all completions were clipped at `max_completion_length=512`. Within each group of 3 completions, rewards were nearly identical → `frac_reward_zero_std≈1` → GRPO Advantage≈0 → near-zero loss.
  - Fix: Added two length penalties in `compute_total_reward`:
    1. **Truncation penalty (-0.4)**: if completion length ≥ 95% of max limit, penalize heavily (model failed to stop naturally).
    2. **General length penalty**: if completion exceeds 1.5× target length (point: ~80, box: ~150, maze: ~300 tokens), apply linear penalty up to -0.15.
  - Updated `make_point_reward_fn` and `make_box_reward_fn` to pass actual `completion_length` (from `completion_ids`) and `max_completion_length` to `compute_total_reward`.
  - Files: `src/utils/metrics.py`, `scripts/run_stage4b_grpo_point.py`, `scripts/run_stage4a_grpo_box.py`

- **GRPO reward function: fixed gradient explosion & KL collapse**
  - Problem: Initial `-0.4` truncation penalty was too aggressive. All completions were still clipped at 512, so every completion in a group got the same penalty → `frac_reward_zero_std` stayed ~0.9 → massive uniform negative advantage → grad_norm exploded to 7872, KL diverged to 104.
  - Fix 1: Reduced truncation penalty `-0.4 → -0.15`.
  - Fix 2: Added **not-truncated bonus `+0.2`** — gives the model a positive signal when it learns to stop naturally.
  - Fix 3: Reduced general length penalty cap `0.15 → 0.1`.
  - Fix 4: Hyperparameter changes to stabilize training:
    - `learning_rate`: `1e-6 → 5e-7`
    - `beta` (KL penalty coeff): `0.04 → 0.1`
    - `temperature`: `1.0 → 1.2`
  - Files: `src/utils/metrics.py`, `scripts/run_stage4b_grpo_point.py`, `scripts/run_stage4a_grpo_box.py`

- **GRPO max_completion_length increased 512 → 768**
  - Maze GT data reaches ~447 tokens; 512 left almost no safety margin for early-training verbosity. Model frequently hit the limit, got truncated, then punished — creating a vicious cycle.
  - 768 gives ~300 tokens of breathing room while length penalties still encourage conciseness.
  - Files: `scripts/run_stage4b_grpo_point.py`, `scripts/run_stage4a_grpo_box.py`

- **Training speed optimization: batch_size doubled + gradient checkpointing re-enabled**
  - `batch_size`: `2 → 4`
  - `gradient_accumulation_steps`: `6 → 3`
  - `gradient_checkpointing`: `False → True` (was disabled for vLLM compatibility)
  - Result: effective batch size stays 12, but total steps halved from 1750 to 875 per epoch. Estimated time per round drops from ~36h to ~18h.
  - Removed all vLLM parameters from GRPOConfig (vLLM is confirmed incompatible with TRL 1.5.1).
  - Installed `liger-kernel` for potential additional speedup (can enable `use_liger_kernel=True` in GRPOConfig if compatible with Qwen3-VL).
  - Files: `scripts/run_stage4b_grpo_point.py`, `scripts/run_stage4a_grpo_box.py`

- **Fixed vLLM 0.22.1 EOS bug causing 100% max-length completions**
  - Root cause: vLLM 0.22.1 + TRL 1.5.1 incompatibility — `SamplingParams` does not automatically pick up the model's EOS token, so vLLM ignores `<|im_end|>` and generates to `max_completion_length` every time. This killed within-group variance (all completions same length → same truncation penalty → `frac_reward_zero_std≈1`) and made GRPO fail to learn.
  - Fix: Added `generation_kwargs={"stop_token_ids": [processor.tokenizer.eos_token_id]}` to `GRPOConfig` in both stage4a/4b scripts. This explicitly passes the EOS token ID to vLLM's `SamplingParams`, forcing generation to stop at `<|im_end|>`.
  - Verified: Without vLLM (native generation), `clipped_ratio` drops to 0 and `frac_reward_zero_std` drops to ~0.3 within 50 steps, confirming vLLM was the culprit.
  - Also reverted `beta` from `0.1` back to `0.04` (the gradient explosion was caused by the vLLM EOS bug, not beta being too small).
  - Files: `scripts/run_stage4b_grpo_point.py`, `scripts/run_stage4a_grpo_box.py`

- Diagnostic scripts (removed after debugging):
  - `diagnose_tokenization.py`
  - `diagnose_tokenization2.py`
  - `test_grpo_fix.py`
