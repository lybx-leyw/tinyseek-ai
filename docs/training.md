# 训练流程与实践指南

快速概览

- 训练入口位于 `train_agent/*.py`（例如 `sft_train.py`, `lora_train.py`, `pre_train.py`, `train_gate.py`），它们读取 `model_config.json` 并调用 `train_agent/trainers/*` 中的训练器。
- 训练主要损失：CE(fc1) + CE(fc2) + alpha * Lexp（Lexp 来自 MoE 路由，作为负载均衡正则项）。

混合精度与优化

- 当 `device=='cuda'` 时，训练器会使用 `torch.amp`（`autocast` + `GradScaler`）。
- 优化器：`Adam`；学习率策略支持指数衰减或基于 `warmup_index` 的线性 warmup。
- 使用 `accumulation_steps` 支持小显存训练；建议使用梯度裁剪 `max_norm=1.0`。

训练数据与 Dataset

- 数据通常为 JSONL 格式，训练器会把对话拼接为 `User：...` 和 `Assistant：...<|im_end|>` 的形式。
- `tools/vocab.Vocab` 用于生成/加载词表并做 encode/decode。
- 数据遮掩：通过 `tools/mask.mask_from_id_to_id` 按标记区间屏蔽 labels（用于 LoRA/SFT 训练）。

检查点与日志

- 中间检查点示例：`out\TinySeek_Pre{index}_{epoch}.pkl`, `out\TinySeek_Lora{index}_{epoch}.pkl` 等。
- 最终：`out\TinySeek_Pre_final.pkl`, `out\TinySeek_Lora_final.pkl`, `out\TinySeek_SFT_final.pkl`。
- 日志写入 `log.txt`，并可用 `tools/plot.draw_plt` 绘制训练曲线。

建议实践

- 在新实验前备份/清理 `out/` 与 `logs/` 以避免混淆旧 checkpoint。
- 小显存场景优先使用 LoRA/Prefix 或减小模型深度与 batch，并开启 `accumulation_steps`。


训练入口
- Wrapper（示例）：`train_agent/sft_train.py`, `train_agent/lora_train.py`, `train_agent/train_gate.py`。
- 每个 wrapper 加载 `model_config.json`（通过 `tools.ConfigManager`），并调用 `train_agent/trainers/*` 中具体训练器函数。

训练器通用约定
- 多进程支持：在 wrapper 中调用 `torch.multiprocessing.freeze_support()`。
- 参数传递：wrapper 将常用参数（`json_data_path, vocab_trg_path, max_len, batch_size, num_workers, accumulation_steps, init_lr` 等）传入训练器。
- 检查点命名与保存点：统一使用 `out\` 前缀并在训练显著节点保存带 index 的 name，最终保存 `TinySeek_Pre_final.pkl`。

调试与本地快速验证
- 使用小数据子集：通过 wrapper 中的 `n_of_samples` / `n_of_samplings` 参数限制数据规模。
- 本地单卡调试：将 `config['device']='cpu'` 或 `cuda` 并在 wrapper 中把 `batch_size` 设置为很小值以快速检测错误。

混合精度与梯度累积
- 在 GPU (cuda) 时，训练器会启用 `torch.amp.GradScaler` 并在前向/反向使用 `autocast('cuda', dtype=torch.float16)`。
- 梯度累积用于模拟大 batch：损失按 `accumulation_steps` 除后反向传播，达到步数时更新优化器并清零梯度。

常用训练参数建议
- 小型实验（验证逻辑）：`n_layer=1-2, batch_size=8-16, accumulation_steps=2, init_lr=5e-4`
- 中型训练：`n_layer=6, batch_size=16-32, accumulation_steps=2-4, init_lr=5e-4`
- MoE 专家数量：`other_experts` 与 `shared_experts` 依显存调整，先试小值如 `other_experts=16, shared_experts=2`。

日志与监控
- 训练脚本会把关键信息写入 `log.txt` 与 `logs/` 下的特定日志文件。请在开始训练前清理或重命名历史日志以便区分实验。
