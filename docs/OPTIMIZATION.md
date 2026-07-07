# Optimization Directions

This reproduction prioritizes the core idea (visual primitives as reasoning units) within a single-GPU constraint. The directions below can shrink the gap to the original paper without rebuilding trillion-scale pretraining.

## Stage 1 — Pretrain

**Current**: unified visual grounding pretrain on COCO + CLEVR; special tokens learned jointly with LoRA.

**Paper**: trillion-scale multimodal pretrain on 40M+ filtered web grounding samples.

**Feasible improvements**:
1. Expand visual pretrain data (Flickr30k Entities, RefCOCO, SA-1B, etc.). 100K–1M diverse real samples improve generalization.
2. Reproduce the paper's two-step filtering (semantic + geometric quality review) on public detection/grounding datasets if web crawling is unavailable.
3. Unfreeze the last 2–4 ViT layers (`--unfreeze_vit_layers 2-4`) so visual features better adapt to coordinate prediction.

## Stage 3 — Cold-Start SFT

**Current**: COCO box/point/counting + simplified CLEVR + single-algorithm rectangular mazes + path tracing. `<|ref|>` token is supported end-to-end.

**Paper**: MLLM-generated thinking chains from GQA scene graphs; 460K mazes covering DFS/Prim/Kruskal and rectangular/circular/hexagonal topologies; 125K path tracing samples.

**Feasible improvements**:
1. **Fine-grained counting**: use GQA scene graphs to generate attribute-constrained questions and thinking chains, then verify with `thinking_verifier.py`.
2. **Maze diversity**: add Prim and Kruskal generators, plus circular and hexagonal topologies.
3. **Spatial / VQA**: extend CLEVR to multi-hop reasoning and add faithful-refusal negatives.
4. **MLLM-generated thinking**: on labeled data (GQA, COCO panoptic, SA-1B), use a small local MLLM or API to synthesize three-step thinking chains (Intent Analysis → Grounding → Summarization).

## Stage 4 — Task-Specific RL

**Current**: rule-based Quality RM + difficulty grading by correct rollout count. Path tracing uses the paper's 4-component Accuracy RM. Complex CLEVR questions can use an LLM API judge. LLM-based Generative RM is implemented in `src/utils/quality_rm_api.py` but disabled by default (`use_quality_rm_api: false`).

**Paper**: LLM-based Generative Reward Model as Quality RM.

**Feasible improvements**:
1. Replace the rule QM with a small local critic model, or call API only on boundary samples.
2. Use the rule QM as a fast pre-filter and LLM judge for hard cases.

### LLM-as-Judge call volume

Stages 4a/4b/5 default to `use_quality_rm_api: false` (zero LLM Quality-RM calls). If enabled with default ratios:

| Stage | Units | LLM calls ≈ | Tokens ≈ |
|-------|-------|-------------|----------|
| 4a Box GRPO | 4K prompts × 8 generations | ~32,000 | ~50M |
| 4b Point GRPO | 4K prompts × 6 generations | ~24,000 | ~40M |
| 5 Unified RFT | 17K prompts × 5 rollouts | ~85,000 | ~140M |

Token estimate: ~700–800 input tokens + up to 1,024 output tokens ≈ 1.5–1.8K tokens per call.

**Why default is off**: each judge call is a synchronous blocking `chat.completions.create` (30s timeout, 2 retries). GPU sits idle while waiting. At ~1s per call the extra wall-clock wait is roughly **9h (4a) / 7h (4b) / 24h (5)**. Keep it off for speed; enable only when fighting reward hacking.

Note: Stage 4a also sends ~303 complex CLEVR samples (×8 generations, ~0.4M tokens) to the spatial-VQA LLM judge whenever `OPENAI_API_KEY` is set, regardless of `use_quality_rm_api`.

## Stage 5 — RFT

**Current**: Expert rollout → difficulty grading → Normal + 5% Easy → SFT. Prompt pool includes path tracing; Quality RM is used for best-rollout selection (rule by default).

**Feasible improvements**:
- Increase prompt budget and `num_rollouts` beyond fast-mode defaults.
- Enable LLM-based Quality RM for best-rollout selection if API budget allows.

## Stage 6 — OPD

**Current**: parallel two-expert gradient accumulation (`train_opd_parallel()`); Box expert accumulates gradients on box data, then Point expert on point/maze/path data, followed by one `optimizer.step()`. Only one expert resides in VRAM at a time; distillation temperature is 1.2.

**Feasible improvements**:
- Experiment with distillation temperature and loss weighting between box/point experts.
- Use larger-capacity student or full fine-tuning if VRAM permits.

## Observability

TensorBoard primitive metrics are logged every N steps: format compliance rate, valid-coordinate rate, ref usage rate, average reward, etc. All stage configs set `report_to: tensorboard`.

```bash
tensorboard --logdir outputs/stageX_xxx/tb_primitive_logs
```
