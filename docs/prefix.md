# Prefix 微调说明

快速概述
- 入口：`train_agent/prefix_train.py`（wrapper）→ `train_agent/trainers/symprefix_tuning.prefix_train`。
- 目标：通过插入 prefix 模块（可学习的前缀向量）来对模型进行参数高效微调。

关键实现细节（摘录）
- 应用 prefix：
  - 在加载预训练模型后会调用 `model.prefix_model.apply_prefix(model, scale)` 来在模块中插入 `prefix` 参数。
  - 训练前会冻结大部分参数，仅对 `module.prefix.prefix_1` 与 `module.prefix.prefix_2` 这类参数解冻并训练。
- 数据与掩码：与 SFT/LoRA 一致，使用 `PrefixDataset` 并屏蔽 mask 区间。

训练参数摘要
- 初始 lr：`init_lr`（默认 `5e-4`）。
- prefix scale：`scale`（示例 0.02），与 LoRA 的 scale 类似，用于控制 prefix 激活的影响强度。
- 检查点：`out\\TinySeek_Prefix{number}_{epoch}.pkl`；最终 `out\\TinySeek_Prefix_final.pkl`。

注意事项
- Prefix 与 LoRA 都是参数高效微调方式，但实现细节不同；Prefix 直接插入可学习的前缀向量并在前向传播中使用，应确保解冻参数正确。
- 若同时使用 LoRA/Prefix，请在训练脚本中明确哪部分参数需要解冻并记录对应 checkpoint 名称。
