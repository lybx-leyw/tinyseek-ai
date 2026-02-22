# 预训练说明

目的
- 提供模型在大规模无监督/自监督语料上的预训练流程样例，生成可复现的检查点以便后续 LoRA / SFT / Gate 训练。

入口与交互
- 主要 wrapper：`train_agent/pre_train.py`（支持“顺序训练”与“随机训练”两种模式，运行时交互选择）。

关键超参与约定（摘录）
- 来自 `model_config.json`（示例）：
  - `vocab_size=5000, device=cuda, n_layer=6, n_head=8, d_model=512`
  - MoE 相关：`other_experts=64, shared_experts=4, keep=8`
- 训练示例（wrapper 默认值）：
  - `max_len=512, batch_size=32 (顺序) / 16 (随机), accumulation_steps=2/4, init_lr=5e-4`

训练器实现要点（`train_agent/trainers/pre_training.py`）
- 数据读取：逐行读取 JSONL，每行 `data['text']` 通过 `<|im_end|>` 切分为若干对话单元并编码为固定 `max_len` 的 ids。
- 数据聚合：每 300 行合并一次并调用 `train_epoch`，通过 `number//n_of_samples` 控制 warmup 与 keep 阶段。
- 损失与优化：两路交叉熵之和 + `alpha * Lexp`（MoE 返回的正则项）；使用 `Adam`，并在 GPU 时使用 `torch.amp` 混合精度。
- 学习率策略：支持指数衰减（scheduler=True）或基于 index 的 warmup。

检查点策略
- 保存规则：`torch.save(model.state_dict(), f"out\\TinySeek_Pre{number}_{epoch}.pkl")`，训练结束时保存为 `out\\TinySeek_Pre_final.pkl`。
- 建议：在长期训练中定期把关键 checkpoint 复制到外部存储并在 `docs/experiments.md` 中记录对应元信息（超参、数据快照、硬件）。

数据与词表
- 词表生成与加载由 `tools/vocab.Vocab` 控制：会在缺失时从语料生成 `vocab.json`，并在词表前部插入若干固定特殊 token（见 `tools/vocab.py`）。
- Tokenizer：正则分词（`tools/tokenizer.Tokenizer.tokenize`）；请保证预处理脚本与该 tokenizer 行为一致。

运行示例

```powershell
# 交互式运行并选择顺序或随机训练
python train_agent/pre_train.py

# 非交互自动化（可直接调用 wrapper 中的 pre_train 函数）
python run.py  # 然后按 2 选择随机训练
```

性能与调优建议
- 小批量多步尝试：先用 `batch_size=8-16`、`n_layer=1-2` 的小模型验证训练逻辑与数据处理。
- MoE 调优：增大 `keep` 会降低稀疏性但提升单 token 表达；`other_experts`/`shared_experts` 增加会显著增加显存占用。
