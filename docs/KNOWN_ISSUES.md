# Known Limitations & Troubleshooting

## 1. GRPO online rollout overhead

Single-GPU 24GB can support `num_generations=5` at most. More rollouts require gradient accumulation or offloading.

## 2. Flash Attention compatibility

Blackwell (RTX 5090D) support for flash-attn 2.8.3 is still maturing. The code automatically falls back to eager attention.

## 3. COCO data download

First download is ~18GB. Images are read on demand during training.

## 4. vLLM not supported

vLLM is incompatible with TRL GRPO + Qwen3-VL in this codebase. All GRPO stages use Hugging Face native generation.

## 5. Small sample sizes

Default configs are heavily downsampled to let the pipeline run quickly on a single GPU (e.g., Stage 1 ~10K–45K samples, GRPO 1 epoch, small rollouts). These defaults are for **flow validation**, not for producing production-quality weights. Scale samples and epochs according to your hardware for better results.

## 6. Garbled output after GRPO

If rollouts produce non-Latin garble near coordinates, the special token embeddings were likely frozen during SFT. Ensure you are using the latest `src/models/qwen_vl_loader.py` (`embed_tokens` / `lm_head` in `modules_to_save`), then retrain Stage 3a/3b (ideally from Stage 1 so the merged base also carries trained embeddings).

## 7. Dtype mismatch in GRPO generation

If you see `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`, `_patch_lm_head_dtype_cast()` in `src/models/qwen_vl_loader.py` already casts layer-norm outputs to the `lm_head` weight dtype automatically.

## 8. OOM between GRPO rounds

If a round is killed by OOM after previous rounds finish, it is usually due to PyTorch/BitsAndBytes allocator pools not returning memory immediately. `src/training/grpo_runner.py` moves the policy model to CPU and runs garbage collection + CUDA cache clear between rounds. If a round still fails, delete the interrupted `outputs/stage4a_grpo_box/round_N` (or `stage4b`) and rerun; completed rounds are skipped and the interrupted round resumes from its latest checkpoint.

## 9. Resume VRAM bloat (fixed)

Previously, resuming from `round_N/checkpoint-*` loaded the adapter twice. The current resume path loads the policy once and lets `Trainer.train(resume_from_checkpoint=...)` load the checkpoint. The reference adapter is also cast from fp32 to bf16 at training start, saving ~2.6GB.

## 10. Embedding / lm_head are duplicated in `modules_to_save`

Qwen3-VL-4B has `tie_word_embeddings=True`. PEFT's `ensure_weight_tying` does not detect this binding for Qwen3-VL's nested structure, so both `embed_tokens` and `lm_head` are saved as trainable modules. This is intentional: it makes special token embeddings trainable and fixes garbled output. The PEFT warning is harmless.

## 11. Windows shared GPU memory

If Task Manager shows "Shared GPU memory" constantly occupied while dedicated+shared total is far below the limit, this is typically a WDDM allocation issue, not a real OOM. The stage scripts set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. If it persists, try disabling Hardware-Accelerated GPU Scheduling (HAGS) and close other GPU processes.
