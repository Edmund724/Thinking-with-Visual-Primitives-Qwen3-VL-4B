# Changelog

All notable changes to the GRPO training pipeline are documented in this file.

## [Unreleased]

### Added

- **全训练阶段自动断点续训 + 时间戳日志**
  - `src/training/stage_runner.py`：新增 `latest_checkpoint()` 方法，统一查找 `checkpoint-*` 目录中 step 最大的路径，供所有训练阶段复用。
  - `src/training/callbacks.py`：新增 `TimeLoggingCallback`，在每个 logging step 记录当前时间戳、step、loss/learning_rate/epoch 等指标和已运行时间；训练开始/结束时记录 Wall-clock 时间。
  - `scripts/run_stage1_visual_pretrain.py`、`scripts/run_stage3a_sft_box.py`、`scripts/run_stage3b_sft_point.py`、`scripts/run_stage5_rft_unified.py`：未指定 `--resume_from_checkpoint` 时自动从各自 `outputs/.../checkpoint-*` 中最新 checkpoint 恢复；恢复时直接从 checkpoint 加载模型，避免从上游阶段输出重新初始化；`create_sft_trainer()` 调用统一附加 `TimeLoggingCallback`。
  - `src/training/grpo_runner.py`：GRPO 每轮已有的 `checkpoint-*` 自动续训逻辑保持不变，额外添加 `TimeLoggingCallback`，让 Stage 4a/4b 的 per-step 日志也带时间戳。
  - `scripts/run_stage6_opd.py` + `src/training/opd_trainer.py`：Stage 6 自动检测最新 OPD checkpoint；`train_opd_parallel()` 新增 `resume_from_checkpoint` 支持，可恢复 optimizer/scheduler/全局 step/epoch，从中断的 epoch 边界继续并行蒸馏；`_load_opd_checkpoint()` 增强为可安全替换已存在的 `default` adapter。
  - `scripts/run_pipeline.sh`：所有训练阶段均检测是否存在 checkpoint，若存在则提示“从最新 checkpoint 继续”；GRPO 阶段会查找 `round_N/checkpoint-*` 中的最新 checkpoint。
  - 效果：Stage 1、3a、3b、4a、4b、5、6 中途中止后，再次运行同一命令即可自动续训；每次 logging 都能看到精确时间戳和已运行时间，便于分多次跑时估算剩余时间和核对进度。
  - 验证：单元测试 `test_logging_utils.py`、`test_config_utils.py` 通过；所有 stage 脚本 `python <script> --help` 可正常解析；`StageRunner.latest_checkpoint()`、`TimeLoggingCallback`、bash pipeline 语法均已手动验证。

- **StageRunner 自动记录进程终止/完成时间**
  - `src/training/stage_runner.py`：`run()` 内注册 `atexit` 与 `SIGTERM` 处理器；正常结束时记录“Stage completed normally”，被 `kill` / 超时 / 用户中断时记录“Process terminated/interrupted”或“Received SIGTERM ... exiting...”，并主动 flush 日志处理器。
  - 效果：今后各 stage 被手动终止或异常退出后，日志文件里会留下明确的终止时间戳，不再需要手动追加。

- **Stage 3b 实测耗时 ~16h**
  - 训练日志记录两段实际跑时：(1) 2026-06-29 13:50→06-30 01:26 约 11.6h（因进程中断），(2) 2026-06-30 10:42→15:02 约 4.4h（从 checkpoint-12000 resume 到完成）。已同步更新 `README.md`、`README_zh.md` 中训练流程总耗时。
  - 验证：通过 `timeout` 发送 SIGTERM/SIGINT 手动验证，日志中正确出现终止时间戳。

### Fixed

- **Stage 4a/4b GRPO resume 后显存占用比从头训练高数 GB 的问题**
  - `src/training/grpo_runner.py`：发现 checkpoint 后不再通过 `load_qlora_model()` 预先加载 policy adapter，而是保持当前轮次起始点加载的 policy model，让 `Trainer.train(resume_from_checkpoint=...)` 统一加载 adapter 权重、optimizer 状态与 trainer 状态。避免同一个 adapter 被加载两次，显著降低 resume 时的 CUDA 显存占用与碎片。
  - `src/training/memory_utils.py`：新增 `cast_ref_adapter_to_bf16()`，将 GRPO 的参考 adapter（`ref`）从 float32 转换为 bfloat16。TRL 在复制 policy adapter 创建 `ref` 之后才把可训练参数 cast 到 bf16，导致 `ref` 默认留在 fp32；参考模型仅用于推理（计算 KL 惩罚的 log prob），转 bf16 安全且可节省约一半 reference adapter 显存（约 2.6 GB）。
  - `src/training/callbacks.py`：新增 `CastRefAdapterCallback`，在训练开始、checkpoint 加载完成后执行 `cast_ref_adapter_to_bf16()`。
  - `src/training/grpo_runner.py`：GRPO 训练回调列表中追加 `CastRefAdapterCallback`。
  - 新增单元测试 `tests/test_grpo_memory.py`：验证 `cast_ref_adapter_to_bf16()` 只转换 `ref` 参数、`CastRefAdapterCallback` 正确触发、无 `ref` adapter 时安全 no-op。
  - 效果：Stage 4a/4b 在从 `round_N/checkpoint-*` 续训时，显存占用与从头训练更接近，降低因 VRAM 不足导致 BitsAndBytes 把 optimizer 状态换入系统内存而大幅降速的概率。
  - 验证：`tests/test_grpo_memory.py`、`tests/test_grpo_fixes.py` 通过。

### Changed

- **Stage 4a/4b GRPO 默认数据量减半，加速流程复现**
  - `configs/stage4a_grpo_box.yaml`：`num_samples` 4000 → 2000，`num_counting` 2000 → 1000，`num_clevr` 2000 → 1000（总计 4K 样本）。
  - `configs/stage4b_grpo_point.yaml`：`num_point` 2000 → 1000，`num_maze` 4000 → 2000，`num_path` 2000 → 1000（总计 4K 样本）。
  - 原因：默认配置定位为“快速跑通 pipeline”，8K 样本对单卡验证来说训练时间过长；减半后单 epoch step 数减少约 50%，更便于迭代调试。需要追论文精度时再把样本量加回来。
  - `README.md` 和 `README_zh.md` 已同步更新 Stage 4a/4b 的默认数据量说明和示例命令。

- **Stage 4a/4b GRPO 进一步降低默认显存占用（修复 resume OOM）**
  - `configs/stage4a_grpo_box.yaml`：`generation_batch_size` 16 → 8，`num_epochs` 2 → 1。`batch_size=2`、`gradient_accumulation_steps=3`、`num_generations=8`，每步有效 completions 仍为 48，但单次 generate 调用从 16 条降到 8 条，峰值 VRAM 更低。
  - `configs/stage4b_grpo_point.yaml`：`batch_size` 3 → 2，`gradient_accumulation_steps` 4 → 3，`num_epochs` 2 → 1。`generation_batch_size=6`、`num_generations=6`，每步有效 completions 36，并修复了之前 `grad_accum=4` 导致 `clip_ratio` 恒为 0 的问题（`4 % (1*2) == 0` → `3 % (1*2) == 1`）。
  - `README.md` 和 `README_zh.md` 已同步更新 Stage 4a/4b 显存与配置说明。

- **Stage 4a/4b GRPO 默认策略改为“单轮 + 无早停 + 固定阈值”**
  - `configs/stage4a_grpo_box.yaml`：`num_rounds` 2 → 1，`num_epochs` 1 → 2，`early_stopping_subset_size` 16 → 0，`filter_iou_threshold` 0.3 → 0.5。
  - `configs/stage4b_grpo_point.yaml`：`num_rounds` 2 → 1，`num_epochs` 1 → 2，`early_stopping_subset_size` 16 → 0，`filter_point_dist_threshold` 20.0 → 10.0。
  - `scripts/run_stage4a_grpo_box.py`：多轮 IoU 阈值 `[0.3, 0.5, 0.7]` → 单轮固定 `[0.5]`。
  - `scripts/run_stage4b_grpo_point.py`：多轮距离阈值 `[20.0, 10.0, 5.0]` → 单轮固定 `[10.0]`。
  - 原因：原配置使用训练集前 16 条样本做早停，默认 2 轮且阈值逐步收紧，这是为了快速跑通 pipeline 的简化；改为更贴近论文思路的单轮训练到收敛，避免 tiny-subset 波动导致 Round 1 过早结束。difficulty filter 仍默认跳过（`skip_difficulty_filter: true`），以节省单卡前置 rollout 时间。
  - `README.md` 和 `README_zh.md` 已同步更新 Stage 4a/4b 说明。

- **Stage 4a GRPO 显存再优化：降低单步 rollout batch，保持 group size 与 clip 条件**
  - `configs/stage4a_grpo_box.yaml`：`batch_size` 3 → 2，`generation_batch_size` 24 → 16；`gradient_accumulation_steps` 保持 3，`num_generations` 保持 8，`max_completion_length` 保持 384。
  - 效果：每步同时生成的 completion 数从 24 降到 16，峰值 VRAM 显著下降；有效 completions per optimizer step 从 72 降到 48（2 × 3 × 8）。
  - 兼容性：`generation_batch_size=16` 可被 `num_generations=8` 整除；`batch_size=2` 使得 `steps_per_generation=8`，`gradient_accumulation_steps=3` 与 `num_iterations=2` 的余数 `3 % 16 != 0`，GRPO clip_ratio 仍不会恒为 0。
  - `README.md` 和 `README_zh.md` 已同步更新 Stage 4a 显存提示。

- **Stage 4a GRPO 超参数调整：激活 clip 并扩大 group size**
  - `configs/stage4a_grpo_box.yaml`：`gradient_accumulation_steps` 4 → 3，`num_generations` 6 → 8，`generation_batch_size` 6 → 24（有效 completions per optimizer step 保持 72 不变）。
  - `src/training/grpo_runner.py`：`GRPOConfig` 新增 `num_iterations=2`。
  - 原因：TRL 1.6.0 在 `gradient_accumulation_steps % (steps_per_generation * num_iterations) == 0` 时会把 `old_per_token_logps` 设为当前 policy 的 detach 版本，导致 `clip_ratio` 始终为 0；调整参数使余数非零，从而让 GRPO clip 真正生效，同时通过扩大 group size 提升组内 reward 方差。

- **Stage 3a 显存优化：降低单步 batch 与序列长度**
  - `configs/stage3a_sft_box.yaml`：`batch_size` 4 → 2，`gradient_accumulation_steps` 3 → 6（有效 batch size 保持 12 不变），`max_seq_length` 4096 → 2048。
  - 原因：Stage 3a 在 RTX 5090D 24 GB 上出现 OOM；降低单步激活内存占用，同时通过梯度累积保持相同的有效 batch。
  - `README.md` 和 `README_zh.md` 已同步更新 Stage 3a 配置说明与显存提示。

- **Stage 3b 缓存升级：手动 pickle → runner.cached_data + 参数 hash**
  - `scripts/run_stage3b_sft_point.py`：将手动 pickle 缓存替换为 `runner.cached_data()`，缓存 key 覆盖 `num_point`、`num_maze`、`num_path`、`num_negative_point`、`coco_image_dir`、`coco_ann_file`、`general_data_path`。新增 `--regenerate_data` CLI 参数。
  - 效果：缓存 key 随参数自动变化，避免旧缓存误用。

- **Stage 4a/4b 缓存 key 升级：固定文件名 → 参数 hash**
  - `scripts/run_stage4a_grpo_box.py`、`scripts/run_stage4b_grpo_point.py`：原始数据缓存和过滤后数据缓存均改用参数 hash 文件名，覆盖数据生成参数和过滤参数。新增 `--regenerate_data` CLI 参数。
  - 效果：修改参数自动生成新缓存；`--regenerate_data` 同时清除 raw 和 filtered 两个缓存。

- **Stage 6 缓存 key 升级：固定文件名 → 参数 hash**
  - `scripts/run_stage6_opd.py`：缓存文件名改用参数 hash（覆盖 `num_box`、`num_point`、`num_maze`、`coco_image_dir`、`coco_ann_file`）。新增 `--regenerate_data` CLI 参数。
  - 效果：修改参数自动生成新缓存。

- **Stage 1 shuffle seed 固定**
  - `scripts/run_stage1_visual_pretrain.py`：在 `random.shuffle(all_data)` 前添加 `random.seed(42)`，确保 resume 时数据顺序与首次运行一致，消除 resume 的最后一个不确定因素。

- **Stage 3a 新增训练数据 pickle 缓存**
  - `scripts/run_stage3a_sft_box.py`：将 box/counting/CLEVR/negative/general 数据生成 + 清洗步骤包装为 `runner.cached_data()` 调用。缓存 key 覆盖所有数据相关参数（`num_box`、`num_counting`、`num_clevr`、`num_negative_box`、`counting_attribute_ratio`、`clevr_negative_ratio`、`coco_image_dir`、`coco_ann_file`、`general_data_path`）。新增 `--regenerate_data` CLI 参数。
  - 效果：Stage 3a resume 或重复运行时直接加载缓存，跳过 ~5.7 万样本的生成/清洗耗时。

- **Stage 5 新增数据缓存（prompts + 过滤后训练数据）**
  - `scripts/run_stage5_rft_unified.py`：prompts 缓存从固定文件名改为参数 hash（覆盖 `num_box_prompts`、`num_counting_prompts`、`num_clevr_prompts`、`num_point_prompts`、`num_maze_prompts`、`num_path_prompts`、`coco_image_dir`、`coco_ann_file`）。
  - 新增 `filtered_data_cache_<hash>.pkl` 缓存，包装专家模型加载 + 生成 + 难度分级。缓存 key 包含了专家模型路径（`box_expert_path`、`point_expert_path`），更换专家自动触发重新生成。
  - 新增 `--regenerate_data` CLI 参数，同时清除 prompts 和 filtered_data 两个缓存。
  - 效果：Stage 5 resume 或重复运行时，若缓存存在，跳过专家模型加载和推理生成，直接加载过滤后数据进入 SFT。

- **Stage 1 新增训练数据 pickle 缓存**
  - `scripts/run_stage1_visual_pretrain.py`：将 COCO box/point + CLEVR 数据生成步骤包装为 `runner.cached_data()` 调用，缓存文件保存为 `outputs/stage1_visual_pretrain/train_data_cache_<hash>.pkl`。
  - 缓存 key 基于 `num_box`、`num_point`、`num_clevr`、`coco_image_dir`、`coco_ann_file` 的 MD5 前缀，修改任一参数会自动生成新缓存，避免旧缓存失效导致的数据不一致。
  - 新增 `--regenerate_data` CLI 参数，可强制删除已有缓存并重新生成数据。
  - 效果：Stage 1 resume 或重复运行时可直接加载缓存，跳过约 4.5 万样本的生成/验证耗时。
  - `README.md` 和 `README_zh.md` 已同步更新 Stage 1 说明与全 pipeline 缓存说明。

- **Stage 1 训练加速：提升 per-device batch size，减少梯度累积步数**
  - `configs/stage1_visual_pretrain.yaml`：`batch_size` 1 → 4，`gradient_accumulation_steps` 4 → 1（有效 batch size 保持 4，消除全部梯度累积开销）。

### Fixed

- **修复 Stage 4 GRPO 多轮连续运行导致第二轮 OOM**
  - `src/training/grpo_runner.py`：每轮结束后新增“先 `policy_model.to('cpu')` 再删除”的清理步骤，随后触发完整 `gc.collect()`、`torch.cuda.empty_cache()` / `synchronize()` / `ipc_collect()`，并在重新加载下一轮模型后记录显存状态。
  - `src/training/memory_utils.py`：`clear_memory()` 增加 `torch.cuda.ipc_collect()`，进一步回收 CUDA IPC 共享内存碎片。
  - 原因：单进程内连续跑多轮 GRPO 时，Round 1 训练结束后 PyTorch/BitsAndBytes 的显存池和碎片不会立即归还给系统；Round 2 再加载一份 base + adapter 时，峰值显存/内存叠加，容易在 24 GB 显存 + 19 GB 内存环境下把资源顶满。该修复让每轮之间尽可能回到干净的显存状态。
  - 验证：`python -m py_compile src/training/grpo_runner.py src/training/memory_utils.py` 通过；`python -c "from src.training.grpo_runner import run_grpo_rounds"` 导入正常。

- **修复 GRPO 生成阶段 `lm_head` dtype 不匹配错误**
  - `src/models/qwen_vl_loader.py`：新增 `_patch_lm_head_dtype_cast()`，在 `load_qlora_model()` 和 `load_reference_model()` 完成后，为 `lm_head` 注入一个轻量前向包装：当输入 activation 的 dtype 与 `lm_head` 权重 dtype 不一致时（例如 `prepare_model_for_kbit_training` 把 layer norm 输出保留为 fp32，而 `lm_head` 权重为 bfloat16 等其他 dtype），自动将输入 cast 到权重 dtype。
  - 修复报错：`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`。
  - 该包装对训练无影响：训练在 autocast 下运行时输入与权重 dtype 通常一致，不会触发 cast；生成阶段无 autocast 时会自动修正。
  - 新增回归测试 `tests/test_qwen_vl_loader.py`，覆盖 PEFT-wrapped `lm_head`、plain `lm_head`、bf16 权重 + fp32 输入等场景。
  - 验证：`tests/test_qwen_vl_loader.py` 通过；`load_qlora_model()` 加载 base model 与 adapter checkpoint 后均能正确注入 patch。

- **Stage 3a/3b SFT 配置对齐：`format_token_weight=40`, `num_epochs=3`**
  - `configs/stage3a_sft_box.yaml`：`format_token_weight` 从默认值上调至 40.0（ref token 梯度信号更强），`num_epochs` 1 → 3（embedding 获得更多更新机会）。
  - `configs/stage3b_sft_point.yaml`：同步应用 `format_token_weight: 40.0`、`num_epochs: 3`、`max_grad_norm: 1.0`，与 stage3a 保持一致。
  - `scripts/run_stage3b_sft_point.py`：新增 `--format_token_weight` 和 `--max_grad_norm` CLI 参数；新增 `clean_primitive_tags()` 数据清洗步骤（与 stage3a 对齐，防止训练数据包含损坏语法）；`create_sft_trainer()` 调用传入上述两个参数。

### Fixed

- **Stage 3/4 输出乱码（非拉丁字符 / 特殊标签损坏）根本原因修复**
  - 根因：新增的视觉原语特殊 token（`<|box|>`、`<|/box|>`、`<|point|>`、`<|/point|>`、`<|ref|>`、`<|/ref|>`）虽然通过 `add_special_tokens()` 加入 tokenizer，但 `embed_tokens` / `lm_head` 始终被 LoRA 冻结，导致这些 token 的 embedding 停留在随机 / 预训练初始值。模型在生成时无法稳定输出这些特殊 token，转而生成 CJK / 泰文等乱码字符。
  - 修复：`src/models/qwen_vl_loader.py` 在创建新 LoRA adapter 时把 `model.language_model.embed_tokens` 和 `lm_head` 加入 `modules_to_save`（并开启 `ensure_weight_tying=True`），使特殊 token 的 embedding 可以被训练并随 adapter 一起保存/加载。训练参数从 ~528M 增至 ~917M，适配器体积会相应增大。
  - 修复：同步修正 `src/utils/conversation_builder.py` 中 GRPO system message 缺失的 `<think>` / `</think>` 尖括号（原消息被错误渲染为 "inside  thinking... response tags"）。
  - 配置调整：由于 embedding 已可训练，`configs/stage3a_sft_box.yaml` 和 `configs/stage3b_sft_point.yaml` 的 `format_token_weight` 从 **40.0 降到 10.0**。40.0 原是为补偿冻结 embedding 的权宜之计；现在 10.0 已足够给 GRPO 提供 ~80-90% 格式合规的热启动，同时避免过度挤压内容/坐标/推理 token。
  - 影响：需要重新训练 Stage 1 → Stage 2 → Stage 3a/3b（最稳妥）；如想节省时间，至少重新跑 Stage 3a/3b，Stage 4a 的 policy model 会加载到训练好的 embedding。
  - 验证：50-sample 小规模 Stage 3a 训练后，模型可稳定输出 `<|box|>[[x1,y1,x2,y2]]<|/box|>`，不再出现 `𬒈` 等非拉丁字符；全部 157 个单元测试通过。

- **Stage 4 GRPO `generation_batch_size` 与 `num_generations` 不兼容导致 ValueError**
  - `src/training/grpo_runner.py`：`generation_batch_size` 从硬绑定 `args.batch_size` 改为独立配置参数 `args.generation_batch_size`，默认回退到 `args.num_generations`。解决 `batch_size=2` 不能被 `num_generations=6` 整除的报错。
  - `configs/stage4a_grpo_box.yaml` 和 `configs/stage4b_grpo_point.yaml`：新增 `generation_batch_size: 6`。
  - `scripts/run_stage4a_grpo_box.py` 和 `scripts/run_stage4b_grpo_point.py`：新增 `--generation_batch_size` CLI 参数。

- **Stage 4a `make_box_reward_fn` 缺少 `logger` 参数导致 TypeError**
  - `scripts/run_stage4a_grpo_box.py`：`make_box_reward_fn` 签名添加 `logger=None`，与 `grpo_runner.py` 的 `reward_fn_factory(threshold, tokenizer=..., logger=...)` 调用保持一致。闭包内改用 `_log` 局部变量，优先使用传入的 logger，回退到全局 logger。

### Changed

- **Stage 4 GRPO 难度筛选默认跳过，数据量加倍补偿**
  - `configs/stage4a_grpo_box.yaml` 和 `configs/stage4b_grpo_point.yaml`：`skip_difficulty_filter: true`（默认关闭），数据量各参数翻倍（stage4a: 4000/2000/2000；stage4b: 2000/4000/2000）。
  - 原因：单卡环境下难度筛选步骤需要额外 on-policy rollout 推理，容易引发 OOM。Hard 样本在 GRPO 训练中 reward 方差 ≈ 0、梯度 ≈ 0，不会损害训练，只是浪费算力。用更大数据量补偿“有效信号密度”的下降。
  - 多卡环境可通过 YAML 中 `skip_difficulty_filter: false` 重新启用论文原始流程。
  - `README.md` 和 `README_zh.md` 已同步添加说明。

- **Stage 4a GRPO filter 参数优化**
  - `configs/stage4a_grpo_box.yaml`: `num_generations` 2 → 6，与论文常用的 4~8 次 rollout 对齐，显著放宽 normal 难度判定窗口；`filter_batch_size` 4 → 6，利用显存余量加速 filter；`filter_max_completion_length` 384 → 512，减少截断导致的格式错误判定。
  - 新增 `filter_iou_threshold` 独立参数（默认 0.3），将难度过滤阶段的 IoU 阈值与训练 round 阈值解耦，可单独调优而不影响奖励函数。

- **Stage 4b GRPO filter 参数同步优化**
  - `configs/stage4b_grpo_point.yaml`: `num_generations` 2 → 6，`filter_batch_size` 4 → 6，与 stage 4a 保持一致。
  - 新增 `filter_max_completion_length: 512` 和 `filter_point_dist_threshold: 20.0` 独立参数，将 filter 阶段的距离阈值和生成长度与训练参数解耦，与 stage 4a 的参数化模式对齐。

- **难度筛选新增 `filter_reward_threshold` 备用模式**
  - `filter_normal_level_data` 新增 `reward_threshold` 参数（默认 `null`，向后兼容）。
  - 启用后以 `total_reward >= threshold` 作为 rollout 正确性判定，取代二值 `is_rollout_correct` 检查。
  - **默认不启用**：论文 Sec 2.5.2 的“correct response”应仅基于任务正确性（IoU/distance/count），不含格式质量；当前 `is_rollout_correct` 已严格对齐这一原则，无需 reward 混合判定。
  - `reward_threshold` 仅保留作为可调旋钮，供探索性实验使用。

### Fixed

- **`is_rollout_correct` 移除格式门控，只判断任务正确性（严格对齐论文 Sec 2.5.2）**
  - 此前 `is_rollout_correct` 包含 think tag 和非拉丁字符检查，导致即使模型任务答案正确（IoU=0.9），也因格式小疵被判为 hard。这与论文“correct response”的定义（任务答案是否正确）不符。格式质量应由 GRPO 训练时的 Format RM 负责，不应污染难度筛选。
  - 修复后：`is_rollout_correct` 仅检查 IoU/distance/count 匹配 + 可解析性，不再检查 think tags 和非拉丁字符。预期 normal 比例将显著提升。

- **`is_rollout_correct` 对 box/point 任务改用 IoU/距离匹配（修复 normal 比例过低的根本原因）**
  - 此前 `is_rollout_correct` 对所有任务类型使用精确字符串比较判定答案正确与否，导致 localization 任务中即使模型输出的 box IoU 很高（如 0.8），也因坐标数值不完全相同而被判为 hard，`iou_threshold` 参数在 box 任务中实际未生效。
  - 修复后：box 任务使用 `match_boxes(iou_threshold)` 判定，point/path 任务使用 `match_points(point_dist_threshold)` 判定，只有 counting 任务（GT 无 box）才回退到精确答案匹配。预期 normal 样本比例将大幅提升。
  - 新增 8 个单元测试覆盖 IoU/distance 匹配、counting fallback、格式验证等场景。

- **`process_reward.answer_correct` 对 localization 任务改用空间匹配**
  - 此前 `process_reward` 的 `answer_correct` 对所有任务均使用字符串精确匹配，导致 box localization 训练中即使模型输出 IoU 极高，只要坐标字符串不完全相同，`answer_r` 就为 0，削弱了 GRPO 的奖励信号。
  - 修复后：box 任务若 GT 含 boxes，`answer_correct = (IoU match > 0) AND (答案字符串匹配)`，同时保留计数正确性检查；point 任务若 GT 含 points，改用距离匹配。counting 任务（GT 无 boxes）逻辑不变。

- **Path tracing continuity penalty 参数传错 bug**
  - `src/utils/reward/accuracy_rm.py` 中 `path_continuity_penalty` 的第二个参数错误地传入了 `pred_end`（应为 `gt_end`），导致 continuity penalty 始终为 0，path tracing 的奖励信号不完整。现已修复为传入 `gt_end`。

### Added

- **Stage 1 & 2 timing updated to latest wall-clock time** — Stage 1: **~7.4h wall-clock** (2026-06-23 13:34:45 → 20:57:45; 45K samples, 2 epochs, data cache hit). Stage 2: **~24s** (2026-06-23 21:25:15 → 21:25:24). Results: 22,500 steps, loss 6.88→2.34 (−66%), grad norm 14.48→1.25, stable convergence. Updated README.md and README_zh.md pipeline tables and Stage 1 sections.

- **Stage 3a timing recorded** — Stage 3a: ~12.1h GPU time. Results: 14,250 steps (2 epochs), loss 2.87→1.62 (−44%), average 1.65, grad norm 6.20→0.44, stable convergence. 57,000 samples (15K box + 10K counting + 5K CLEVR + 2K negative + 25K general). Updated README.md and README_zh.md pipeline tables, total time (~52h→~57h), and Stage 3a sections.

- **Stage 4a completed — actual runtime recorded** — Stage 4a Box Expert GRPO completed in **~20.1h wall-clock** across 3 resume segments: (1) 2026-06-27 16:53→2026-06-28 00:03 ~7h 10m (checkpoint-3800), (2) 2026-06-28 06:53→08:26 ~1h 33m (checkpoint-4000), (3) 2026-06-28 08:58→20:22 ~11h 24m (final). 4,000 steps (1 epoch, 4K samples: 2K box + 1K counting + 1K CLEVR). Output: `outputs/stage4a_grpo_box/`. Updated README.md and README_zh.md pipeline tables (Stage 4a ~6h est. → ~20.1h measured, total ~58h→~72h) and added Stage 4a results sections.

### Removed

- **Dead code cleanup (round 2)** — removed ~470 lines of unused code and stale artifacts:
  - `src/training/pretrain_trainer.py` (356 lines) — zero references after Stage 1 unified refactor
  - `inject_pretrained_embeddings()` and `save_pretrain_state()` in `src/models/pretrain_loader.py` (~110 lines) — code paths no longer reachable
  - `pretrain_embedding_path` and `old_vocab_size` parameters from `load_qlora_model()` in `src/models/qwen_vl_loader.py` — all callers passed `None`
  - Stale log files: `stage1_pretrain.log`, `stage2_visual_pretrain.log`, `merge_stage2.log`, `verify_grpo_fixes.log`, `stage1_sft_unified.log` (empty), `test_a/b/c.log` (empty)
  - Stale smoke-test cache directories: `data/cache/clevr_smoke/`, `clevr_smoke2/`, `clevr_spatial/`, `maze_smoke/`, `path_tracing/`, and empty `tmp_trainer/`
- **Dead code cleanup** — removed ~352 lines of unused code:
  - `src/data/generators/synthetic_path.py` (169 lines) — superseded by `path_tracing.py`
  - `TensorBoardPrimitiveMetricsCallback` in `src/training/callbacks.py` (133 lines) — defined but never wired into any trainer
  - `GENERATORS` dict and `__all__` in `src/data/generators/__init__.py` (25 lines) — unused registry
  - 8 unused constants in `src/utils/constants.py` (`DEFAULT_MAX_SEQ_LENGTH`, `DEFAULT_IMAGE_SIZE`, `HN_BOX_IOU_THRESHOLDS`, `HN_POINT_DIST_THRESHOLD_PX`, `DEFAULT_DPO_BETA`, `MAZE_UNSOLVABLE_RATIO`, `ANSWER_TRUE`, `ANSWER_FALSE`)
  - `PrimitiveParser.count_tags` and `PrimitiveParser.has_backtracking_keywords` (15 lines) — only used in tests, never in production
  - Duplicate `import sys` in `scripts/run_stage3b_sft_point.py`, `scripts/run_stage5_rft_unified.py`, `scripts/run_stage6_opd.py`
  - Stale documentation references to nonexistent `src/utils/metrics.py`

### Fixed

- **Stage 1 point samples all filtered out** — `verify_thinking_chain` in `thinking_verifier.py` had a count-consistency check that applied to `"point"` task type, but point answers are coordinate strings (e.g. `"(500, 300)"`), not counts. `_parse_int` would extract the last integer (300) and compare it against the number of primitives (1), rejecting every point sample. Changed `task_type in ("box", "point")` to `task_type == "box"` on line 105.
- **LLM Judge API returned empty content for reasoning models** — `quality_rm_api.py` used `max_tokens=150` which was insufficient for reasoning models (e.g. `step-3.7-flash`) that produce verbose chain-of-thought in `reasoning_content` before the final answer. Increased `max_tokens` to 1024 and added a fallback: when `content` is empty, try to parse the score from `reasoning_content`. Applied to both `quality_reward_api` and `spatial_accuracy_rm_api`.

### Added

- **Path Tracing 4-component Accuracy RM** — Stage 4b GRPO now uses the full paper reward decomposition:
  - `path_forward_accuracy` (0.30): each predicted waypoint → GT Bézier curve distance.
  - `path_reverse_accuracy` (0.25): each GT point → predicted polyline distance, catching skipped segments.
  - `path_endpoint_accuracy` (0.20): start/end coordinate distance decay.
  - `path_continuity_penalty` (−0.1): penalises jumping from a partial trace to a guessed endpoint.
  - `path_answer_correctness` (0.25): endpoint label binary match.
  - New functions in `src/utils/geometry.py` (`point_to_segment_distance`, `point_to_polyline_distance`, `path_forward_accuracy`, `path_reverse_accuracy`, `path_endpoint_accuracy`, `path_continuity_penalty`) and proxy methods in `PrimitiveParser`.
  - `task_type` changed from `"point"` to `"path"` in `path_tracing.py`; GT data now stores dense `gt_curve` of 30 Bézier waypoints.
  - Stage 4b reward function routes `"path"` tasks to the 4-component RM; `gt_curve` passed through `GRPODataset`.
  - Stage 3b/5 scripts updated: no override of path `task_type`, Stage 5 prompt pool expanded with path tracing data.

### Changed

- **`<|ref|>` token full-pipeline integration** — 6 special tokens were already defined in `constants.py` but never used by data generators. Now every box/point primitive in COCO and CLEVR data is preceded by a `<|ref|>target_name<|/ref|>` tag, matching the paper's reference-before-grounding format.
  - `format_ref(name)` added to `primitive_formatter.py` and `PrimitiveParser.format_ref()`.
  - `extract_refs(text)` and `validate_ref_box_pairing(text)` added to `text_parsing.py` and `PrimitiveParser`.
  - `REF_PATTERN` regex added to `constants.py`.
  - COCO generators: coarse-grained counting → batch ref (`<|ref|>dogs<|/ref|><|box|>[[...],[...]]<|/box|>`); fine-grained/localization → individual ref (one ref+box per category). Negative samples → no ref.
  - CLEVR `clevr_spatial.py`: all 8 question types updated with individual ref tags. Each sample now carries a `question_type` field (`counting`/`existence`/`spatial_existence`/`spatial_count`/`attribute_query`/`query_material`/`compare`/`multihop`).
  - `format_reward()` in `format_rm.py` adds a ref-box pairing check (deduct 0.1 for unpaired boxes, 0.05 for bad ref content).
  - `SFTDataset` now applies `format_token_weight` to `<|ref|>` / `<|/ref|>` tokens.

- **Spatial/VQA Accuracy RM (LLM API)** — for complex CLEVR questions, accuracy is now scored by an LLM judge rather than simple answer matching.
  - `_build_spatial_judge_prompt()` and `spatial_accuracy_rm_api()` added to `quality_rm_api.py`.
  - `compute_total_reward()` reroutes CLEVR `multihop`/`compare`/`spatial_existence`/`spatial_count` tasks to the API judge; simpler question types stay rule-based.
  - Stage 4a GRPO reward function passes `question_type` through to `compute_total_reward()`.
  - No API key required — gracefully falls back to rule-based answer matching when API is unavailable.

- **TensorBoard primitive metrics callback** — replaced the empty `WandBLogPrimitiveMetricsCallback` with `TensorBoardPrimitiveMetricsCallback`.
  - Every N steps logs: `primitive/format_compliance_rate`, `coord_validity_rate`, `ref_usage_rate`, `avg_total_reward`, and a sample completion text.
  - All stage config YAMLs: `report_to: none` → `report_to: tensorboard`.
  - `grpo_runner.py` and `sft_trainer.py`: default `report_to` changed from `"none"` to `"tensorboard"`.

- **OPD gradient-accumulation parallel distillation** — replaced the sequential Box→Point OPD with gradient accumulation that approximates the paper's parallel KL-sum formula.
  - New `train_opd_parallel()` in `opd_trainer.py`: processes box batches → accumulates gradients → swaps to point expert → accumulates → single `optimizer.step()`.
  - Only one expert in GPU memory at a time; gradient direction is the sum of both experts' signals.
  - `run_stage6_opd.py` updated to use `train_opd_parallel()`; both experts loaded at start and released together after training.
  - `DEFAULT_DISTILL_TEMPERATURE` raised from 1.0 → 1.2 (paper range 1.0~1.5).

### Changed

- **Merged Stage 1+2 into Unified Visual Grounding Pretrain**
  - Removed text-only format pretrain — special token embeddings now start from random init and are learned alongside visual features during visual pretrain (closer to the paper's single-stage multimodal pretraining paradigm).
  - New `scripts/run_stage1_visual_pretrain.py` replaces `run_stage1_pretrain.py` + `run_stage2_visual_pretrain.py`. Trains on COCO box/point + CLEVR spatial data with QLoRA (r=256). No separate pretrain embedding injection needed.
  - New `scripts/run_stage2_merge.py` replaces the old merge script. Simplified: drops `--pretrain_embedding_path` and `inject_pretrained_embeddings()` call.
  - Deleted `configs/stage1_pretrain.yaml` and `configs/stage2_visual_pretrain.yaml`; added `configs/stage1_visual_pretrain.yaml` with CLEVR data generation (`num_clevr: 5000`).
  - Updated `configs/stage3a_sft_box.yaml` and `configs/stage3b_sft_point.yaml`: `model_path` now points to `outputs/stage2_merged_base`.
  - Updated `scripts/run_pipeline.sh`: stages renumbered from 8 → 6 (removed old Stage 1, merged Stage 2 into new Stage 1).
  - Updated `README.md` and `README_zh.md`: renumbered all stage documentation, updated pipeline diagram, project structure tree, and "Closing the Gap" optimization guidance.
  - Old `scripts/run_stage1_pretrain.py` and `scripts/run_stage2_visual_pretrain.py` kept as reference but removed from main pipeline documentation.

- **YAML duplicate keys cleaned + argparse defaults unified to `None`**
  - Fixed duplicate `num_epochs` in `configs/stage2_visual_pretrain.yaml` (was `1` then `2`; removed the dead `1`).
  - Fixed duplicate `early_stopping_*` block in `configs/stage4a_grpo_box.yaml` (was `0/50/2` then `16/50/2`; removed the dead first set).
  - All 8 stage scripts: `add_arg(default=<concrete>)` → `default=None` (~120 arguments). YAML configs are now the sole default source; `action="store_true"` flags unchanged.
  - Fixed latent bug in `run_stage1_pretrain.py`: only 5 args were registered but `train()` accessed ~17 — now all registered; `configs/stage1_pretrain.yaml` expanded with visual-phase, ViT, and `max_seq_length` keys.
  - 5 standalone scripts (`run_stage2_merge`, `smoke_test_stage2`, `eval_stage2_structure`, `eval_stage3a_paradigm`, `diagnose_stage2_resume_loss`): all `default=<concrete>` → `default=None`.
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

- **`run_stage2_merge.py` now preserves Stage 1 embeddings**
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

- **README 增加 `embed_tokens` / `lm_head` 独立训练说明**
  - 在 README.md 和 README_zh.md 的 Stage 3a 重要提示中补充：Qwen3-VL-4B 虽配置 `tie_word_embeddings=True`，但 PEFT 的 `ensure_weight_tying` 对其嵌套结构检测不到 tied weights，因此回退到把两层都放入 `modules_to_save` 独立训练；该设计解决了特殊 token 乱码问题，代价是这两层可训练参数约翻倍，且 PEFT 会报无害 warning。

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
