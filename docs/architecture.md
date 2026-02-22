# TinySeek 架构说明

概览

TinySeek 的核心由下列模块组成：

- `TokenEmbedding`（token 编码）
- 多层 `ModelBlock`：每层包含 `MLA` 注意力 + `MoE` 前馈（稀疏专家）
- 双路输出头：`fc1`（主预测）与 `fc2`（候选预测）
- `gated_TinySeek`：封装多个 TinySeek 专家与一个小型 gate 网络，用于样本级专家分配

关键实现文件

- MLA 注意力：[modules/mla.py](modules/mla.py#L1)
- MoE（专家路由）：[modules/moe.py](modules/moe.py#L1)
- 模型与 gate：[model/tiny_seek.py](model/tiny_seek.py#L1)
- LoRA / Prefix 插入工具：`model/lora_model.py`, `model/prefix_model.py`

要点

- MoE 在 logits 上注入噪声并做 top-k 选择，返回 (output, routing_score)，routing_score 在训练中作为正则项。
- 双路输出用于在训练与生成中分离主预测与备选预测，生成策略在 `tools/vocab.generate` 中实现。
- `gated_TinySeek` 支持合并多位专家并冻结大部分参数，仅保留 LoRA/Prefix 层作为可训练子集（训练或微调时常用）。

阅读建议

从 `model/tiny_seek.py` 开始，然后查看 `modules/mla.py` 与 `modules/moe.py`，最后阅读训练器实现以理解损失与 checkpoint 策略。

概览
- TinySeek 是一个基于 Transformer 的轻量级 MoE（Mixture-of-Experts）模型，主要目标是在保持推理效率的同时利用专家路由提升表达能力。
- 主要组件：`TokenEmbedding`、多层 `ModelBlock`（每层包含 MLA 注意力与 `MoE` 层）、两个输出头 (`fc1`/`fc2`)、可选的 `gate` 模块与 `gated_TinySeek` 封装多个专家模型。

数据流（高层）
- 输入文本 → `tools/vocab.Vocab.encode`（使用 `tools/tokenizer.Tokenizer`）→ token ids
- token ids → `TokenEmbedding` → 若干 `ModelBlock`（注意力 + MoE）→ 得到隐表示
- 隐表示 → `fc1`/`fc2` 输出两路 logits（用于下一 token 的预测与采样策略）

MoE 与 Gate 的协同
- `modules/moe.MoE`：基于输入计算 gating logits（含噪声与 top-k 保留），按专家分配 token 并聚合专家输出；返回 `(output, score)`，其中 `score` 用作训练正则化或统计。
- `model.tiny_seek.gate`：以少量层预测样本级的专家索引（通过 `get_last_non_pad_output` 取得序列最后非填充位置的表示）。
- `gated_TinySeek`：封装 6 个 `TinySeek` 专家模型与一个 `gate` 模型，用 softmax+阈值选择专家后在推理时对样本路由并加权输出。

设计理由与注意点
- 两路输出 (`c_t`,`n_t`) 源于对生成策略的实验：将两路概率分开以便更灵活的组合与重复惩罚策略（见 `tools/vocab.generate`）。
- MoE 使用 `keep_topk` 保证稀疏路由，减少计算；`shared_experts` 用于对所有样本共享的额外容量。
- 修改模型结构时请注意：
  - 保持 `MoE.forward` 返回 `(output, score)`。
  - 若调整 embedding 或输出头尺寸，需同步更改 `train_agent/trainers/*` 中对 logits 的处理代码。

架构特点（TinySeek 作为“小型 DeepSeek”）

- 轻量化设计：在保持 DeepSeek 核心思想（MoE + gate + 双路预测）的前提下，缩减层数与隐藏维度以支持更快的实验迭代与更低的资源需求，适合在单卡或小规模集群上开展研究。
- 复合层单元（ModelBlock）：每层集成 MLA 注意力、残差 + LayerNorm 以及基于 MoE 的 FFN（`modules/moe.MoE`），其中 MoE 会返回额外的路由得分 `Lexp`，由训练器累加为额外正则项。
- 双路预测头与采样策略：使用 `fc1` / `fc2` 产生两路 logits（`c_t`,`n_t`），训练与生成时分别用于主预测与候选/替代采样，并在 `tools/vocab.generate` 中结合重复惩罚与置信阈值控制生成长度与质量。
- 稀疏专家路由细节：MoE 在计算 gating logits 时注入噪声、应用 `keep_topk` 并通过 `softmax` 归一化；对每个专家按选择的 token 批量调用对应 FFN 并按 gate 权重累加输出；shared experts 在所有 token 上统一应用以提供共享容量。
- 样本级门控（gated_TinySeek）：通过一个轻量 gate 网络预测每个样本应使用的专家（默认 6 个专家），推理时将样本分组送入对应专家并按 gate 权重合并输出，`gated_TinySeek.load_part_model` 支持专家权重合并与 LoRA 兼容加载。
- Mask 与填充策略：`get_mask` 生成因果下三角矩阵并与 pad mask 结合，保证自回归时序信息与 padding 屏蔽的正确性。
- 嵌入一致性策略：合并多个专家模型时，会对专家的 `tokenEmbedding` 权重取平均并复制到 gate 的 embedding，保持输入编码一致，便于多模型集成与门控训练。
- 可插拔微调机制：代码原生支持 LoRA（`model/lora_model.apply_lora`）与 Prefix（`model/prefix_model.apply_prefix`）两种参数高效微调方式；训练器会根据模块命名自动解冻对应参数以简化微调流程。

示例实现位置
- 层复合单元：`model/tiny_seek.ModelBlock`
- MoE 关键实现：`modules/moe.MoE`
- Gate 与合并策略：`model/tiny_seek.gate` 与 `model/tiny_seek.gated_TinySeek`

定位关键文件
- `model/tiny_seek.py` — 模型与门控封装
- `modules/moe.py` — MoE 实现细节
- `modules/mla.py` — 自定义注意力实现
- `tools/tokenizer.py` 与 `tools/vocab.py` — 分词与词表构建
