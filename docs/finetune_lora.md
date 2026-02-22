# LoRA 微调与兼容性

概要
- 本项目支持 LoRA 微调（低秩适配），以在保持基模型参数冻结的情况下高效训练任务特定权重。
- 相关实现文件：`model/lora_model.py`（包含 `apply_lora` 工具函数）。

加载与应用
- 在 `gated_TinySeek.load_part_model` 中，仓库尝试直接 `load_state_dict`；若失败会调用 `apply_lora` 来适配 LoRA 格式的权重。
- 若你有 LoRA 权重，推荐先确认权重对应的 rank/scale 等参数，然后通过 `apply_lora(model, rank, scale, prefix)` 应用权重。

保存策略
- LoRA 专用权重可能以部分 state dict 形式保存（只包含低秩矩阵），请在保存时记录对应的 rank 与 scale，并在 README/`docs/finetune_lora.md` 中说明兼容性。

微调示例（快速）

```powershell
# 运行 LoRA wrapper
python train_agent/lora_train.py
```

注意事项
- 在合并多个微调模型到 `gated_TinySeek` 时，需确保 tokenEmbedding 权重的一致性。`gated_TinySeek.load_part_model` 会尝试将六个模型的 tokenEmbedding 权重取均值并复制到 gate 的 tokenEmbedding 上。
