# LoRA 微调说明

快速概述
- 入口：`train_agent/lora_train.py`（wrapper）→ `train_agent/trainers/lora_tuning.lora_train`。
- 目标：在冻结大部分基模型参数的前提下，仅训练低秩适配层（LoRA），以节省显存与加速微调。

关键实现细节（摘录）
- LoRA 应用：
  - 在加载预训练模型后会调用 `model.lora_model.apply_lora(model, rank, scale)` 将 LoRA 层插入模型（或在加载已有 LoRA checkpoint 后依然保留 LoRA 层）。
  - 在训练前，会把除 `lora` 或 `prefix` 命名模块外的参数全部冻结，仅保留包含 `lora` 的模块参数 `requires_grad=True`。
- 数据与掩码行为与 SFT 类似（使用 `LoraDataset`，mask id 同样为 9/10）。

训练参数摘要
- 初始 lr：`init_lr`（默认 `5e-4`）。
- LoRA rank：`rank`（wrapper 示例 `rank=16` 或 64），scale 常用 0.02。应用 `apply_lora` 时需指定。
- 训练器保存最终权重为 `out\\TinySeek_Lora_final.pkl`，中间 checkpoint 命名 `out\\TinySeek_Lora{number}_{epoch}.pkl`。
- 训练时仍使用两路 cross-entropy + `alpha * Lexp` 的损失项，与其它训练器一致。

注意事项
- LoRA 权重通常仅包含低秩矩阵（部分 state_dict），在加载与合并模型时需保留对应 rank/scale 元信息以便 `apply_lora` 兼容。
- 在 `gated_TinySeek.load_part_model` 中会尝试兼容 LoRA 权重的加载流程（见 `model/tiny_seek.py`）。
