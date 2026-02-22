# SFT（Supervised Fine-Tuning）说明

快速概述
- 入口：`train_agent/sft_train.py`（wrapper）→ `train_agent/trainers/sft_training.sft_train`。
- 目标：在预训练模型基础上，用指令/对话风格数据进行监督微调，使模型在生成质量和角色对话上更合适。

关键实现细节（摘录）
- 加载策略：
  - 若 `last_index < 0`，尝试加载 `pretraining_model_path`（默认 `out\\TinySeek_Pre_final.pkl`）。
  - 否则加载已有 SFT checkpoint，例如 `out\\TinySeek_SFT{last_index}_1.pkl`。
- 数据：输入 JSONL 中 `conversations` 列表被拼接为 `User：...` / `Assistant：...<|im_end|>` 格式，随后用 `tools/vocab.Vocab.encode` 转为固定长度 ids。
- 掩码：使用 `tools.mask_from_id_to_id` 将 `[MASK]` 区间（wrapper 默认 id 9 到 10）在 label 中遮蔽。

训练参数摘要
- 优化器：`Adam`。
- 初始 lr：`init_lr`（默认 `5e-4`）。
- Mixed-precision：如果 `config['device']=='cuda'`，启用 `torch.amp.GradScaler` 与 `autocast(dtype=torch.float16)`。
- 梯度累积：使用 `accumulation_steps`（wrapper 常见值 2-8）。
- 学习率调度：支持两种方式：指数衰减（scheduler True，默认每步乘 `decay`）或基于 `warmup_index` 的线性 warmup（当 scheduler=False 时）。
- 梯度裁剪：`clip_grad_norm_`，`max_norm=1.0`。

检查点与输出
- 训练期间会按 `out\\TinySeek_SFT{number}_{epoch}.pkl` 保存，最终保存为 `out\\TinySeek_SFT_final.pkl`。

建议/注意事项
- SFT 的数据构造对 prompt 格式敏感，请保持与 `train_agent/trainers/sft_training` 中拼接方式一致。
- 若基于 LoRA/Prefix 微调得到的模型用于 SFT，请注意参数冻结与加载顺序。
