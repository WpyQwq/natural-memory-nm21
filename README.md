# Natural Memory NM2.1

在**已经训练好的 Qwen3.5-4B** 上做「架构手术」，插入一个可读、可写、可持续更新的记忆模块，让模型在不把历史对话反复塞回上下文的前提下持续积累和使用信息。

本仓库是 [natural-memory](https://github.com/WpyQwq/natural-memory)（V2 时代工程）的**代码分叉**，承载 **NM2 / NM2.1** 阶段的全部源码、评测证据与报告。分叉动机、逐轴结果与踩坑记录见 **[README_FORK.md](README_FORK.md)**（本仓库最详细的一篇文档）。

---

## 一句话结论

> 在**完全相同的存储预算**下（128 维 = 每条记录 512 字节、2,037,774 参数），
> V2-128 v6 路由器在参数、地址字节、延迟、吞吐**全部不劣**的前提下，
> 把 Top-1 从 **41.12% → 94.62%**、未知拒答率从 **0.00% → 100.00%**、hop 正确率从 **73.23% → 100.00%**，
> 逐轴判定 **22/22 通过**。

而这次提升**不是新架构的胜利**：XL-128 / XL-512 准确率最高（Top-1 95.66% / 95.78%），但真实地慢 26–52%、参数 3.3–3.9 倍，并在召回与多跳证据完整度上略逊于同数据的 V2。收益来自**数据与标签的修正**，不是路由器容量。这一条负面结论同样写进了仓库，没有被藏起来。

---

## 本仓库回答的问题

| 轴 | 结果 |
|---|---|
| 检索排序 | Top-1 41.12% → 94.62%；Recall@3 46.73% → 96.48%；MRR 47.69% → 97.12% |
| 未知拒答 | 0.00% → **100.00%**（已知问题被误拒率 0.00%） |
| 多跳证据 | hop 73.23% → 100.00%；Top-3 全中 43.43% → 98.64% |
| 写入存活 | 写 20 条存活 12/20 → 20/20 |
| 零字面重叠改写 | 43.75% → 68.75% |
| 跨架构可移植 | 14 / 14 架构施加记忆手术后**逐位相同**（读写关闭时与原生模型 logits 完全一致） |
| 通用能力回归 | 54 任务上原版与动态记忆版均为 96.30%，差值 0，门禁通过 |

**没有通过的**（诚实记录）：成本轴（单查询延迟、QPS、每条记录地址字节共 22 项未通过）、部分平均分数余量项、XL-final 的多跳证据全中。

---

## 三个被测量推翻的「显然优化」

这是本仓库最有价值的部分——三条都是**先有直觉、后被实测否决**，代码与结论一并保留以便复核：

1. **`logits_to_keep=1`**：`_encode_model_key` 每次前向都算出 `[B, L, 248320]` 的 logits 并全部丢弃（3.05 GB 中间张量 + 约 8.5 TFLOP）。看起来显然该裁掉，实测反而**慢 55%**（283 → 183 texts/s）——`slice(-1, None)` 产生非连续视图，让 4-bit matmul 掉进慢路径。
2. **加大路由器容量**：XL-512 相对同数据 V2-512 只换来 Top-1 **+0.73pp**、MRR +0.12pp，代价是 1.4 倍延迟与 1.67 倍参数，召回与多跳还略降。
3. **一个 46.8 小时的性能缺陷**：`prepare_memory_router_dataset.py` 对每条 episode 重建候选池，引入 mega 验证集后单次列表复制约 2.1 秒。修复后 mega 家族 **3568×** 加速，数据集重建从约 47 小时降到 **114.9 秒**，且输出**逐字节相同**（A/B 同 seed，train/eval sha 一致）。

---

## 仓库结构

```
.
├─ README_FORK.md                    # ★ 分叉说明：逐轴结果、缺陷根因、常用命令（最详细）
├─ ENGINEERING.md                    # 原工程说明（Dynamic Memory Lab 时代，与 natural-memory 同源）
├─ NM2_1_FINAL.md                    # NM2.1 最终报告
├─ NM2_VS_NM2_1.md                   # NM2 → NM2.1 逐轴对比
├─ V6_FINAL_REPORT.md                # v6 路由器训练与判定报告
├─ WRITE_PATH_FIX.md                 # 写入路径缺陷的根因与修复（含真实调用栈）
├─ ABSTENTION_BREAKTHROUGH.md        # 未知拒答从 0% 到 100% 的机制
├─ Natural_Memory_v2_Paper.md        # 技术论文
├─ Natural_Memory_v2_对外技术总结.md    # 对外技术总结
│
├─ memory_os_v2.py / qwen_integration.py / model.py / router_xl.py   # 记忆核心与模型手术
├─ train_*.py                        # 路由器 / 检索器 / 原生记忆 / 写入策略训练器
├─ eval_*.py / benchmark_*.py        # 多轴评分卡与基线对比
├─ audit_*.py / verify_*.py          # 独立数据审计与验证
├─ probe_*.py / diagnose_*.py        # 诊断探针（剂量曲线、可分离性、可移植性）
├─ stream_feature_bank.py            # 流式多线程特征编码（212 万条文本 → mmap 特征库）
│
├─ qwen3_5_4b_natural_memory_v2/     # 模型手术配方（config / memory_config / memory_merge）
├─ qwen3_5_4b_natural_memory_v2_1/   # 同上（NM2.1 终态）
├─ data/                             # 可复现的评测语料（zero_overlap / realistic_v2 / locomo 等）
│
├─ router_*.json / nm2_1_*.json ...  # ★ 全部评测证据（评分卡、判定、审计结果）
└─ tests/                            # 单元测试
```

`*.json` / `*.jsonl` 是**评测证据本体**，不是缓存——每张报告里的数字都能在对应 JSON 里逐条查到。

---

## 未包含在仓库中的内容（及原因）

| 内容 | 体积 | 原因 |
|---|---:|---|
| `qwen3_5_4b_natural_memory_v2{,_1}/model.safetensors-*` | 2 × 9.0 GB | 合并后的模型权重；可用 `merge_memory_weights.py` 从基座 + adapter 重建 |
| `checkpoints/`（router 最终权重与中间快照） | 607 MB | 二进制训练产物；训练脚本与超参已完整保留 |
| `data/router_training_v{3,5,6}/`、`router_replay_v7/` | 4 × 1.4 GB | 冻结训练集，由 `prepare_memory_router_dataset.py` 约 115 秒重建（逐字节可复现） |
| mmap 特征库 `features.f16.npy` | 10.86 GB | 由 `stream_feature_bank.py` 重新编码生成 |
| 用户记忆快照 `*_runtime.pt` / `persistent_memory.pt` | — | **含真实对话内容，按设计绝不入库** |
| Qwen3.5-4B 基座权重 | — | 第三方模型，请自行获取并遵守其许可证 |

---

## 复现

```powershell
# 环境：Python 3.12 + CUDA
pip install -r requirements.txt

# 单元测试
python -m unittest discover -s tests

# 重建数据集（含拒答 / 多跳标签修正，约 2 分钟）
python -m V2_dpskw.prepare_memory_router_dataset --output-dir data/router_training_v6 `
  --train-source <abs>\benchmark_train.jsonl ... --candidate-count 32 --conflict-aware --seed 20260909

# 独立审计：类别完整性 + 严格泄漏检查
python -m V2_dpskw.audit_router_dataset `
  --train-file data/router_training_v6/train.jsonl --eval-file data/router_training_v6/eval.jsonl `
  --output router_dataset_audit_v6.json

# 流式多线程特征编码（212 万条唯一文本 → mmap 特征库）
python -m V2_dpskw.stream_feature_bank `
  --train-file data/router_training_v6/train.jsonl --eval-file data/router_training_v6/eval.jsonl `
  --model-path qwen3_5_4b_natural_memory_v2 --output-dir <nvme>/nm_router_v6/feature_cache `
  --tokenizer-threads 8 --max-batch 192 --token-budget 12288 --gpu-memory-gb 10

# 训练 + 评分
pwsh -File .\run_router_v6.ps1 -Configs v2_512_v6,xl512_v6
python -m V2_dpskw.eval_router_v5 --feature-cache <nvme>/nm_router_v6/feature_cache `
  --train-file data/router_training_v6/train.jsonl --eval-file data/router_training_v6/eval.jsonl `
  --run "XL-512 v6=checkpoints\router_v6_xl512\router_best.pt" `
  --output router_scorecard_v6.json --markdown router_scorecard_v6.md
```

> 原文档中的 `H:\Memory\...`、`W:\Flash\model` 等路径是作者本机路径，复现时请替换为你自己的路径。

---

## 第三方数据归属

`data/net_locomo/` 来自公开研究数据集 **LoCoMo**（*Evaluating Very Long-Term Conversational Memory of LLM Agents*），用于跨语料验证。该目录内的图片签名 URL、对话内容均为上游数据集自带，**不属于本项目**，请遵守上游数据集许可证。其余 `data/` 内容由本仓库脚本生成。

---

## 已知局限

- 未在真实业务分布上做长时间压力测试；合成数据指标**不等于**生产承诺。
- 生产级多用户隔离、加密、并发写入、版本迁移与合规删除尚未实现。
- 需要外部持久化介质保存用户 checkpoint —— 这是信息存在的必要条件，不是实现缺陷。
- 内存治理（事实抽取、去重、时间衰减、审计导出）仍是后续工作。

详见 [README_FORK.md](README_FORK.md) 与 [NM2_1_FINAL.md](NM2_1_FINAL.md)。

---

## 免责声明

本项目为研究性质代码，不保证在所有任务上提升。使用第三方模型权重或数据集时，请遵守对应的模型许可证、数据许可证与隐私要求。
