# 门控与专家路由（Gating）

概览
- 门控模块用于在样本级或 token 级选择合适的专家以降低整体计算成本并提升模型容量。

关键实现
- `model/tiny_seek.gate`：小型模型，输出维度 `6`（代表最多 6 个专家选择）。它通过 `get_last_non_pad_output` 取得序列最后有效 token 的表示并预测专家分配。
- `model/tiny_seek.gated_TinySeek`：包含 6 个 `TinySeek` 模型作为专家与一个 `gate` 模型。推理过程中：
  - 计算 gate 的 softmax，选最大概率专家（通过阈值 Mask 保留 top），并对样本进行分组，由对应专家分别计算输出，然后按 gate 权重合并。

训练门控的注意事项
- `cacu_gate_loss`（拼写为 `cacu`）通过对每个专家在当前 batch 上计算其生成损失并选择最小损失对应的专家 index，作为 gate 的交叉熵目标。
- 该函数使用 `torch.no_grad()` 在每个专家上计算 per-sample loss，然后 `argmin` 选出最佳专家作为监督信号，最后计算 `F.cross_entropy(gate_output, best_model_indices)`。

合并专家模型与 tokenEmbedding
- `gated_TinySeek.load_part_model` 在加载多个训练好的子模型时会尝试直接 `load_state_dict`，若失败则调用 LoRA 兼容路径；若所有模型含有 `tokenEmbedding`，函数会把六个专家的 `tokenEmbedding` 权重平均后复制到 gate 的 `tokenEmbedding`，保证输入编码一致性。

可扩展性建议
- 若需更多专家或动态专家数：调整 `gated_TinySeek` 的硬编码 6 的部分并同步更新 `gate` 输出维度、训练器的 gate loss 计算与 `docs/experiments.md` 的记录表。
