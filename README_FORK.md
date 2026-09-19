# V2_dpskw —— Natural Memory 路由器分叉

本目录是 `H:\Memory\dynamic_memory_lab`（GPT 时代的原工程）的**代码分叉**，用于训练一个**全新的、更大更强的路由器**。
原工程保持只读不改；本 fork 内所有代码的包名已统一为 `V2_dpskw`。

## 与原工程的关系

| 项目 | 处理方式 |
|---|---|
| 全部源码 / 测试 / 文档 | 复制进本目录，包名 `dynamic_memory_lab` → `V2_dpskw`（51 个文件） |
| Qwen3.5-4B 权重包 | 目录联接（junction）`qwen3_5_4b_natural_memory_v2` → 原工程，不复制 9.3 GB |
| 冻结的 router 训练数据 | 复制 `data/router_training_v3/`（train 7155 / eval 1940，sha256 与原版一致） |
| 冻结的 Qwen 特征库 | 复制 `checkpoints/router_shared/feature_cache/`（21263 × 2560 fp16），manifest 的 `model_path` 已改写为 fork 内路径 |
| 其余 data / checkpoints / 适配器 / 大报告 | **不复制**，需要时按绝对路径引用原工程 |

因此训练时**不会加载 Qwen、也不会重新编码特征**，30 秒内即可开始更新路由器参数。

## 新增内容

| 文件 | 作用 |
|---|---|
| `router_xl.py` | **新的路由器架构 `MemoryRouterXL`**：多层 MLP 编码器、`[q,k,q−k,q·k]` 交互特征 + LayerNorm、残差 pair trunk、多层策略头；保持与 `MemoryRouterV2` 完全相同的运行时契约 |
| `train_memory_router_xl.py` | v3 数据上的训练器（复用原训练器的度量代码，只替换模型） |
| `train_router_v5.py` | **v5/v6 全量数据训练器**：流式读取 JSONL（不把 episode 解析进内存）、**mmap** 10.86 GB 特征库、支持 `--arch v2\|xl`、三种采样模式（`uniform`/`source_balanced`/`family_sqrt`） |
| `stream_feature_bank.py` | **流式 + 多线程特征编码器**：scan（唯一文本落盘）/ tokenize（线程池）/ encode（按精确 token 长度分组、token 预算限批）→ 写 mmap `.npy` |
| `eval_router_scorecard.py` | v3 版多轴评分卡 |
| `eval_router_v5.py` | v5/v6 版多轴评分卡：流式 + mmap + **按 family / 按 10 个类别拆解** |
| `audit_router_dataset.py` | **独立数据审计**：类别完整性（可回答/未知/正例数/hop）+ 严格泄漏检查（group_id、查询、文本、以及「同查询且共享正例证据」） |
| `check_router_cache.py` | 冻结特征缓存快速校验（失配即报错，不会偷偷加载 4B 模型重编码） |
| `compare_router_runs.py` | 与原 512 基线的汇总对比 |
| `tests/test_router_xl.py` | 新架构契约测试 |
| `run_router_v6.ps1` / `chain_v6.ps1` | v6 训练启动器 / 编码完成自动接训练与评分的长链 |

`prepare_memory_router_dataset.py` 额外修复了一个**性能缺陷**（见下）。

## 已验证事实

- 全部单元测试通过：**52 项**（原 45 项 + 新增 7 项）。
- 特征缓存命中，训练全程不加载 Qwen。
- 新架构与原架构同尺寸对比：`router_dim` 相同意味着**每条记忆记录的地址占用完全相同**（512 维 × fp32 = 2048 字节）。

## MemoryRouterXL 容量

| 路由器 | router_dim | heads | 参数 | 相对 V2-512 |
|---|---:|---:|---:|---:|
| V2-512 基线（原工程） | 512 | 8 | 4,741,902 | 1.00× |
| XL-512 | 512 | 8 | 7,898,127 | 1.67× |
| XL-1024 | 1024 | 16 | 18,414,103 | 3.88× |
| XL-2048 | 2048 | 16 | 55,133,719 | 11.63× |

> 只有 XL-512 的地址几何与原版一致（同样的存储成本），因此它是唯一可以做同尺寸对比的配置。

## XL-512 训练结果（100,000 步，冻结 v3 eval：1940 条）

| 指标 | V2-512 best(step1k) | V2-512 final(step100k) | XL-512 best | XL-512 final(step100k) |
|---|---:|---:|---:|---:|
| 参数 | 4,741,902 | 4,741,902 | 7,898,127 | 7,898,127 |
| 每条记录地址字节 | 2048 | 2048 | 2048 | 2048 |
| Top-1 正确率 | **67.12%** | 60.63% | 60.85% | 54.36% |
| Recall@3 | 88.77% | 85.01% | **92.97%** | 85.33% |
| Recall@5 | 94.93% | 90.13% | **97.33%** | 91.93% |
| MRR | **79.08%** | 73.26% | 75.98% | 69.50% |
| nDCG@3 | **80.00%** | 74.57% | 79.28% | 71.71% |
| 多跳证据全中(Top-3) | 88.66% | 84.90% | **92.86%** | 85.22% |
| hop 正确率 | 97.84% | **99.69%** | 99.38% | **99.69%** |
| need F1 | 98.92% | **99.89%** | 99.62% | **99.89%** |
| 未知拒答率 (thr 0.50) | 66.04% | **100.00%** | 90.57% | **100.00%** |
| 已知问题被误拒率 | 0.22% | 0.22% | 0.22% | 0.22% |
| 未知问题被误读率 | 33.96% | 0.00% | 9.43% | 0.00% |
| 平均分数余量 | 0.588 | 3.884 | 6.710 | **25.376** |
| 单查询延迟 ms (GPU) | **0.895** | 0.935 | 1.310 | 1.246 |
| 路由 QPS (GPU) | **1117** | 1070 | 763 | 803 |

**结论不是「谁全面更强」，而是各轴各有胜负：**

1. 原工程用 `selection_score = 0.5·top1 + 0.3·need_f1 + 0.2·mrr` 选出的 "best"（step 1000）Top-1 最高，但**在 0.50 门槛下有 33.96% 的未知问题被强行读记忆**；该综合分几乎不惩罚这一点，因为未知样本只有 106 条。
2. XL-512 在**覆盖率**上明显更好：Recall@3 +4.20pp、Recall@5 +2.40pp、多跳证据全中 +3.96pp。
3. 把门槛从 0.50 提到 0.70，XL-512 best 的未知拒答率达到 **100.00%**，而已知问题被误拒率仍为 **0.22%**——即在同等策略安全水平下，XL-512 的排序覆盖优势可以保留。
4. 代价：XL-512 参数 1.67×、单查询延迟 +39~46%（0.935 → 1.310 ms）、QPS 从 1070 降到 763。

## 数据集构建：一个 46.8 小时的缺陷

`prepare_memory_router_dataset.py` 的 `_source_candidates` 原先对**每条 episode** 都重建候选池：

```python
same_family = [item for item in items_by_family.get(episode.family, []) if item.item_id not in positive_set]
```

引入 mega 验证集后，`mega_validation` 家族的候选池有 **2,656,000** 条，于是一次列表复制约 2.1 秒：

| 家族 | 原版 / 条 | 修复后 / 条 | 加速 |
|---|---:|---:|---:|
| mega_validation | 2105.42 ms | **0.59 ms** | **3568×** |
| memory_policy | 16.61 ms | 2.31 ms | 7.2× |
| native_memory | 7.51 ms | 0.68 ms | 11× |

80,000 条 mega episode 原本需要 **46.8 小时**（`H:\Memory\dynamic_memory_lab\data\router_training_v4` 那次运行了 19.5 小时只写出 91 MB 残缺文件，原因即此）。

修复方式（`_ExcludedPool` + `PoolPositionIndex`）：不复制候选池，而是提供一个「排除了少量已知位置」的惰性序列视图，`len()` 与 `__getitem__` 与原列表**逐元素一致**，因此 RNG 抽样与最终数据集**逐字节相同**。

等价性已验证（A/B，同样输入、同样 seed）：

```
ORIGINAL  train_sha=D9C2774955DF7066  eval_sha=169B026DF2A85C35   93.5s
V2_dpskw  train_sha=D9C2774955DF7066  eval_sha=169B026DF2A85C35   42.1s
```

全量数据集 `data/router_training_v5`（144 秒建成，原需约 47 小时）：

| 文件 | episodes | 类别 |
|---|---:|---|
| `train.jsonl` (1148 MB) | **87,155** | benchmark 128 / native 512 / policy 6,515 / mega 80,000 |
| `eval.jsonl` (288 MB) | **21,920** | 含 mega 20,000，覆盖全部 10 个类别（每类 2,000） |

`group_overlap = 0`，冲突策略与防泄漏检查与原协议一致。

## 当前阻塞：特征库放不进内存

训练复用 Qwen 冻结特征，所以每条唯一文本都要编码一次：

| 文件 | 唯一文本 | 特征库 (fp16) |
|---|---:|---:|
| v5 train | 1,695,797 | 8.68 GB |
| v5 eval | 425,115 | 2.18 GB |
| 合计 | 2,120,912 | **10.86 GB** |

本机内存 31.2 GB、空闲 12.7 GB，还要容纳解析后的 episode 对象，因此**全量 87k 无法在本机编码**。
可选方案：按类别均衡抽样（约 25–32k episode，特征库 ~5 GB）、先压缩 mega 源、或换更大内存的机器。

**实际落地方式**（已完成）：全量 2,124,552 条唯一文本、10.86 GB 特征库写成 mmap 并用完整扫描验证，训练器流式读取，因此「内存放不下」不再是阻塞项。

## v6 结果：准确率与拒答策略大幅超越，成本轴未超越

训练：V2-512 对照与 XL-512 候选均 100,000 步、batch 64、`family_sqrt` 采样、同一冻结特征库与协议。
评测：冻结 v6 eval 21,920 条（10 类别 × 2,000 + policy 1,920），7 个路由器同题同特征。

| 指标 | 生产包 128 维(v3) | V2-512 v3 best | **V2-512 v6 best** | **XL-512 v6 best** |
|---|---:|---:|---:|---:|
| Top-1 正确率 | 41.12% | 80.75% | 95.05% | **95.78%** |
| Recall@3 | 46.73% | 95.02% | **98.84%** | 97.59% |
| Recall@5 | 48.91% | 97.31% | **99.35%** | 98.34% |
| MRR | 47.69% | 88.46% | 97.12% | **97.24%** |
| nDCG@3 | 44.22% | 89.53% | **97.24%** | 96.64% |
| 多跳证据全中(Top-3) | 43.43% | 94.19% | **98.64%** | 96.86% |
| hop 正确率 | 73.23% | 81.59% | 100.00% | 100.00% |
| **未知拒答率** | 0.00% | 50.41% | 99.81% | **100.00%** |
| **已知问题被误拒率** | 0.18% | 11.23% | **0.00%** | **0.00%** |
| 单查询延迟 (GPU) | **0.844 ms** | 0.879 ms | 0.875 ms | 1.216 ms |
| 批量 QPS (batch=256) | **264,813** | 167,235 | 161,941 | 96,747 |
| 地址字节/记录 | **512** | 2048 | 2048 | 2048 |

判定（`router_verdict_v6.md`，逐轴对比最强旧基线）：**66 项通过 / 22 项未通过 → 全方位超越：否**。
22 项未通过**全部集中在成本轴**（单查询延迟、单查询/批量 QPS、每条记录地址字节）、2 项平均分数余量、以及 XL-final 的多跳证据全中（78.85% vs 84.90%）。所有准确率与拒答策略轴全部通过，且多数是大幅通过：Top-1 +15.03pp、MRR +8.79pp、hop +18.25pp、未知拒答率 +48.71pp、已知被误拒率 −0.11pp。

### 同数据架构对照：XL 相对 V2 并没有赢

在**完全相同**的数据、特征、采样与步数下（best 检查点）：

| 指标 | V2-512 v6 | XL-512 v6 | XL−V2 |
|---|---:|---:|---:|
| Top-1 | 95.05% | 95.78% | **+0.73pp** |
| MRR | 97.12% | 97.24% | +0.12pp |
| 未知拒答率 | 99.81% | 100.00% | +0.19pp |
| Recall@3 | 98.84% | 97.59% | −1.25pp |
| Recall@5 | 99.35% | 98.34% | −1.01pp |
| nDCG@3 | 97.24% | 96.64% | −0.60pp |
| 多跳证据全中 | 98.64% | 96.86% | −1.78pp |
| 单查询延迟 | 0.875 ms | 1.216 ms | **+39% 更慢** |

结论：本次最大的提升来自**数据与标签**（v3 → v6：修正被破坏的拒答/多跳标签、12 倍数据量、10 类别均衡评测），而不是新架构。XL 在 512 维上只换来 Top-1 +0.73pp、MRR +0.12pp，代价是 1.4 倍延迟与 1.67 倍参数，并在召回与多跳证据完整度上略逊于同数据的 V2。若目标是「同尺寸全面超越」，应继续投入数据与训练配方，而不是加大路由器容量。

## 同存储预算的判定：128 维 V2 达到全方位超越

把 512 维路由器与 128 维的生产路由器比「每条记录地址字节」是**不同存储预算的错位比较**。因此在生产几何（128 维 = 512 字节/记录、2,037,774 参数）上重训后重新判定：

| 路由器 | 参数 | 地址字节 | Top-1 | 未知拒答率 | 单查询延迟(中位) | batch256 QPS | 逐候选判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| V2-128 生产(v3) | 2,037,774 | 512 | 41.12% | 0.00% | 1.1558 ms | 199,554 | 基线 |
| **V2-128 v6 best** | 2,037,774 | 512 | **94.62%** | **100.00%** | 1.1679 ms | 196,829 | **22/22 通过 → 是** |
| **V2-128 v6 final** | 2,037,774 | 512 | 94.37% | **100.00%** | 1.1653 ms | 198,026 | **22/22 通过 → 是** |
| XL-128 v6 best | 6,714,639 | 512 | 95.66% | 99.59% | 1.5783 ms | 141,501 | 18/22（吞吐 4 轴）|
| V2-512 v6 best | 4,741,902 | 2048 | 95.05% | 99.81% | 1.1623 ms | 154,085 | 20/22（batch256、地址字节）|
| XL-512 v6 best | 7,898,127 | 2048 | 95.78% | 100.00% | 1.5855 ms | 94,647 | 17/22 |

速度轴以**交替轮流测量**（`bench_router_latency.py`，5 轮，模型轮转以抵消漂移）为准，实测波动带 2.8–5.3%，故速度轴采用 3% 相对容差；质量轴仍为严格容差。此前「V2-128 v6 慢 4.6%」是单次顺序测量的漂移，交替测量下两者差异 **≤1.4%**（单查询 min 甚至更快 0.5%）。

**结论**：V2-128 v6 在参数、地址字节、延迟、吞吐**全部不劣**（差异在测量噪声带内）的前提下，把 Top-1 从 41.12% 提到 94.62%（**+53.49pp**）、未知拒答率从 0.00% 提到 100.00%、hop 从 73.23% 提到 100.00% —— 这是相对已交付生产路由器的**同预算全方位超越**。

**但不是新架构的胜利**：XL-128/XL-512 准确率最高（Top-1 95.66%/95.78%）却真实地慢 26–52%（3.3–3.9 倍参数）。本次提升来自数据与标签，而非路由器容量。

## 常用命令

```powershell
$py = 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
$env:PYTHONPATH = 'H:\Memory'
Set-Location 'H:\Memory\V2_dpskw'

# 单元测试
& $py -m unittest discover -s tests

# 特征缓存校验（v3 数据）
& $py -m V2_dpskw.check_router_cache

# 构建数据集（已修复拒答/多跳标签；全量约 2 分钟）
& $py -m V2_dpskw.prepare_memory_router_dataset --output-dir data/router_training_v6 `
  --train-source <abs>\benchmark_train.jsonl ... --candidate-count 32 --conflict-aware --seed 20260909

# 独立审计：类别完整性 + 严格泄漏
& $py -m V2_dpskw.audit_router_dataset `
  --train-file data/router_training_v6/train.jsonl --eval-file data/router_training_v6/eval.jsonl `
  --output router_dataset_audit_v6.json

# 流式多线程特征编码（212 万条唯一文本 → mmap 特征库，放 NVMe）
& $py -m V2_dpskw.stream_feature_bank `
  --train-file data/router_training_v6/train.jsonl --eval-file data/router_training_v6/eval.jsonl `
  --model-path qwen3_5_4b_natural_memory_v2 --output-dir H:\Memory\nm_cache\nm_router_v6\feature_cache `
  --tokenizer-threads 8 --max-batch 192 --token-budget 12288 --gpu-memory-gb 10

# 训练（V2 对照 + XL 候选，同数据同协议）
pwsh -File .\run_router_v6.ps1 -Configs v2_512_v6,xl512_v6

# 全类别百分比评分卡
& $py -m V2_dpskw.eval_router_v5 --feature-cache H:\Memory\nm_cache\nm_router_v6\feature_cache `
  --train-file data/router_training_v6/train.jsonl --eval-file data/router_training_v6/eval.jsonl `
  --run "XL-512 v6=checkpoints\router_v6_xl512\router_best.pt" `
  --output router_scorecard_v6.json --markdown router_scorecard_v6.md
```

## 数据管线（流式 + 多线程）实测

| 阶段 | 结果 |
|---|---|
| scan：去重并落盘唯一文本 | 2,124,552 条 / **15.1 秒**（只保留 `sha1→行号` 字典，不把 episode 解析进内存） |
| tokenize：8 线程池（HF fast tokenizer 释放 GIL） | 2,124,552 条 / **41.8 秒**（约 57,000 texts/s） |
| encode：按精确 token 长度分组 + token 预算限批 | 约 283 texts/s @32 token（batch 192），GPU 100% |
| 特征库 | `features.f16.npy` mmap，10.86 GB，训练时按需读页 |
| 断点续跑 | `--resume` + `progress.json`：记录已完成的长度排序行数，被中断后不重编已完成部分 |

**表示保真度**：与旧冻结缓存（v3）共有文本对比，编码结果 **余弦最低 0.999935 / 平均 0.999983**。因此旧路由器可以直接在新特征库上评分，而不是被换到一套不同表示上再比较。

### 一个被测量推翻的「显然优化」

`QwenDynamicMemoryModel._encode_model_key` 调用完整条件生成模型，而 `logits_to_keep` 默认 0（= 全部 token），于是每次前向都算出 `[B, L, 248320]` 的 logits：对 192×32 的批是 **3.05 GB 中间张量 + 约 8.5 TFLOP**，全部丢弃。看起来显然该改成 `logits_to_keep=1`。实测（交替重复 7 轮取最小值；两条路径特征一致性 cosine **0.9999996**）：

| 编码路径 | 192×32 批耗时 | 吞吐 | 显存峰值 |
|---|---:|---:|---:|
| 全量 logits（默认，保留） | **679 ms** | **283 texts/s** | 5.22 GiB |
| `logits_to_keep=1` | 1050 ms | 183 texts/s | 4.13 GiB |
| batch 512（全量 logits） | — | 287 texts/s | — |

原因：`slice(-1, None)` 产生**非连续视图**，bitsandbytes 的 4-bit matmul 在这个布局上掉进慢路径（多出约 500 ms），而全量 logits 是规整的连续 GEMM，本身只约 130 ms。结论：**不要启用 `--skip-lm-head`**（默认已关，代码与结论一并保留以便复核）。batch 从 192 提到 512 也无收益，说明该阶段已接近这台 GPU 的算力上限（约 52 TFLOPS，约为 5070 bf16 峰值的 84%）。

## v5 → v6：修正被破坏的拒答与多跳标签

`_add_mega_row` 有两个由 GPT 时代沿用下来的缺陷，正好废掉了本次评测最需要的两个轴：

1. `unknown_abstention` / `forget_correction` 的源数据带 `metadata.answerable = false`，其 `acceptable` 装的是**拒答话术**（"不知道"/"没有记录"）。旧代码用 `answerable = bool(acceptable)` 判成可回答，又没有事实文本包含这些话术，于是走进兜底分支 `positive_ids = ids[:]`，**把全部事实标成正例**——2000 条拒答样本全部变成可回答样本。
2. `multi_hop` 的真实跳数在 `metadata.hop_count = 2`，旧代码用 `len(positive_ids)` 计算，且只标注了含最终答案的那条事实，**中间推理链事实没有标成证据**。

修正：尊重 `metadata.answerable`；用 `metadata.hop_count` 作为 hop 标签；沿「实体→值」链扩展证据（有界，最多 3 跳）。修正前后对比：

| 指标 | v5（有 bug） | **v6（已修正）** |
|---|---:|---:|
| train 未知样本 | 319 | **16,319** |
| eval 未知样本 | 106 | **4,106** |
| `unknown_abstention` 可回答/未知 | 2000 / **0** | **0 / 2000** |
| `forget_correction` 可回答/未知 | 2000 / **0** | **0 / 2000** |
| `multi_hop` 平均正例数 / hop | 1.00 / **1** | **2.00 / 2** |

数据集重建耗时 **114.9 秒**（未修复前同一构建需要约 47 小时）。

### v6 独立审计结果

| 检查 | 结果 |
|---|---|
| `group_id` 重叠 | **0** |
| 查询文本重叠 | 80 / 4,296 = **1.86%** |
| **同查询且共享同一正例证据** | **0**（80 条均为模板化问句配不同事实，无泄漏） |
| 候选文本重叠 | 3 / 421,473 = **0.0007%** |

| 类别 | episodes | 可回答 | 未知 | 平均正例 | hop |
|---|---:|---:|---:|---:|---|
| unknown_abstention | 2000 | 0 | **2000** | 0.00 | 0 |
| forget_correction | 2000 | 0 | **2000** | 0.00 | 0 |
| multi_hop | 2000 | 2000 | 0 | **2.00** | **2** |
| 其余 8 类 | 各 2000 | 2000 | 0 | 1.00–2.00 | 1–2 |

## NM2 跨架构泛化（实测矩阵）

原集成的记忆手术只适配 Qwen3.5 一族的解码层。`probe_nm2_portability.py` 对 14 个架构真机施加手术，
并做**最强保能力检验**：记忆读写关闭时，包装后的模型必须与原生模型的 logits **逐位相同**。

修复前：只有 `position_embeddings` 在第 2 位置的架构（gemma2 / gemma3 / mixtral）能通过，其余全部
`TypeError: ... got multiple values for argument 'attention_mask'` 或找不到层容器。

修复后（**零补丁、全部逐位相同**）：

| 架构 | 层容器 | 第 2 位置参数 | 恒等性 |
|---|---|---|---|
| llama / mistral / qwen2 / qwen3 / starcoder2 / cohere / phi | `model.layers` | `attention_mask` | identical |
| olmoe / granitemoe | `model.layers` | `attention_mask` | identical |
| gemma2 / gemma3 / mixtral | `model.layers` | `position_embeddings` | identical |
| opt / bart | `model.decoder.layers` | — | identical |

为达到这一点只改了 4 处，**记忆核心与路由器一行未改**（它们只依赖 `hidden_size`）：

| 改动 | 解决的假设 |
|---|---|
| `resolve_text_config()` | 纯文本模型没有 `config.text_config` |
| `resolve_decoder_layers()` | 层容器固定为 `model.language_model.layers`（现支持 8 条候选路径） |
| `MemoryLayerAdapter` 约定检测 + **按声明参数过滤** | `position_embeddings` 被按位置硬传；参数**名称**也不同（Ling 用 `past_key_value` 单数，且未必有 `**kwargs`） |
| `load_qwen_base()` 兜底 `AutoModelForCausalLM` | 只有 image-text-to-text 加载器 |

对 **Qwen3.5 生产路径零行为变化**（已证明）：真实 `Qwen3_5DecoderLayer.forward` 的绑定签名里
`position_embeddings` 位于索引 1 → adapter 走 `positional` 分支 → 与修复前完全相同的调用形式；
同族的 gemma2/gemma3/mixtral 在修复后仍为 `identical`，52 项单元测试保持全绿。

### 第二类兼容性：远程代码 vs transformers 版本

`inclusionAI/Ling-3.0-tiny`（`bailing_hybrid` / `BailingMoeV3ForCausalLM`，hidden 1536 / 24 层 /
128 专家 top-8 / 活跃约 1.2B / MLA + KimiDeltaAttention 混合）**架构上可接**（骨架以关键字传
`position_embeddings`，正好落在新分支；`past_key_value` 单数命名由参数过滤处理），但它自带的
`modeling_bailing_moe_v3.py` 是按**旧版 transformers API** 写的：

- `from transformers.utils.import_utils import is_torch_fx_available` → 5.9 已移除该符号；
- `config.rope_scaling["factor"]` → 5.9 会把 `rope_scaling` 规范化掉，运行期 KeyError。

也就是说「嵌进任意模型」有两层前提：**架构层（已解决，14/14）** 与 **栈层（需兼容 shim 或独立 venv）**。


