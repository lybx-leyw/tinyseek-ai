# 实验与实践成果模板

建议在每次重要实验中记录以下信息（可放入 `docs/experiments.md` 或单独表格）：

- 名称 / 简要描述
- 日期
- 配置摘要（`model_config.json` 关键字段）
- 数据集路径与版本
- 硬件（GPU 型号 / 内存）
- 超参数（batch, lr, accumulation_steps, epochs 等）
- checkpoint 路径
- 关键指标（loss, ppx, gate accuracy）及日志片段
- 备注（问题、改动、后续计划）

示例条目

- Name: gated_lora_experiment_2024-11
- Config: n_layer=12, d_model=768, other_experts=48, keep=2
- Data: dataset/minimind_dataset/sft_512.jsonl
- HW: 1x A100 40GB
- Hyperparams: batch=16, lr=5e-5, accumulation_steps=4
- Checkpoint: out/TinySeek_gate_0_1_200.pkl
- Metrics: final ppx1=12.3, ppx2=9.7, gate_acc=0.82

记录格式化有助于后续比较与自动化分析。

请在此记录每次重要实验的可复现元数据与关键结果。建议使用 Markdown 表格或 YAML front-matter 记录便于脚本解析。

示例表格：

| Name | Date | Config (摘要) | Data | HW | Hyperparams | Checkpoint | Metrics / Notes |
|---|---:|---|---|---|---|---|---|
| pretrain-small | 2026-02-01 | d_model=512,n_layer=6,other_experts=64 | dataset/minimind_dataset/pretrain_hq.jsonl (50k samples) | 1xA100 40GB | batch=32,lr=5e-4,acc=2 | out/TinySeek_Pre_final.pkl | PPL: -- ; 生成示例见 docs/examples.md |

建议字段说明：
- Name：简短描述，例如 `pretrain-small`。
- Config：给出关键模型字段或 `model_config.json` 的引用。
- Data：标明数据路径和样本量。
- HW：GPU 型号与显存、CPU/内存 等。
- Hyperparams：列出 lr、batch、accumulation、amp、keep、experts 等。
- Checkpoint：指向 `out/` 下的文件名及日期。
- Metrics / Notes：可包含 PPL、生成示例链接、训练失败/调参要点。

如何共享结果
- 将关键 checkpoint 与生成样例上传 Releases 或挂载到 CDN，然后在表中写下外部链接。
- 如需自动化，可写脚本解析本文件并生成实验比较图（建议放到 `docs/figures/`）。
