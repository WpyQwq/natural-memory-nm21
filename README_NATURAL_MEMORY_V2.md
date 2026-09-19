# Natural Memory v2

Natural Memory v2 是接在本地 Qwen3.5-4B 上的一层分层、可寻址、可纠错记忆系统。它的目标不是把一百万个 slot 当作一张巨大的 KV Cache，而是把历史中最适合长期保存的部分压缩成有地址的记忆记录；当前对话仍由 Qwen 的热 KV 负责连续理解。

## 先看结论

当前版本已经完成以下闭环：

```text
普通用户输入
    -> 自动写入策略
    -> 置信度 / 重要性 / 来源审计
    -> 紧凑地址投影
    -> 分页存储
    -> LSH 粗索引
    -> 候选页重排
    -> 候选记录精排
    -> Top-K 证据注入 Qwen
    -> 版本冲突 / 隔离 / 撤回 / 重启恢复
```

最终运行包位于：

`W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2`

它不需要额外的 `memory_state.pt` 才能恢复已嵌入的记忆；记忆快照、路由器权重、V2 页面元数据和上下文片段都写在该包的第三个 memory safetensors 切片中。当前包的 manifest 指向实际使用的 runtime memory shard；旧 runtime shard 不会被自动覆盖，以避免 Windows 进程仍持有文件句柄时破坏现有包。当前交付配置固定使用 embedded weight-shard 模式：启动时完整载入进程内存，路由命中的热点记录才进入有界 VRAM cache，不使用 SQLite 或磁盘分页。

## 核心设计

### KV 与 Memory Slot 的分工

```text
最近 32K token（最多可配置到 128K） -> GPU 热 KV：保留精确顺序和局部连贯性
长期个人事实 / 项目决策 / 纠错版本     -> Memory Slot：保存压缩、可寻址的证据
当前问题                             -> 稀疏路由：只读取少量相关记录
```

Memory Slot 不模拟完整 KV。它只接管 KV 中最昂贵、最适合长期保存、最容易重复利用的部分。这里的目标是比长 KV 获得更高压缩率，而不是把每条对话都压成一句极短摘要：短事实和有价值的对话碎片会保留原始 token 序列，同时另存一个较小的语义地址用于路由。这样仍不能承诺逐 token 无损复现数百万 token 的原始上下文，但能保留足够细节供模型复用。

### 路由路径

任何查询都必须经过有界路径：

```text
查询 hidden state
      |
      v
128 维紧凑地址
      |
      v
LSH 粗索引（精确桶 + Hamming-1/2 探针 + 热页）
      |
      v
候选页（不是全部页面）
      |
      v
页级与记录级精确重排
      |
      v
最多 top_k_pages / top_k_records
```

当前 token 不会对 1M slot 做全量注意力。页内才会做小规模记录评分；默认是最多 4 个页、8 条记录。粗索引结果还会记录在诊断 trace 中，便于检查“是否因为候选页不足而漏召回”。

### 记录结构

每条 `MemoryRecordV2` 包含：

- 原始短文本和可选 token 序列；独立碎片不会因为共享一个热槽而被合并；
- 128 维紧凑地址与摘要地址；
- 可选的冻结 Qwen 语义检索键，用于重启后的更新/删除判定，不参与全量注意力；
- `entity / attribute / value` 冲突键；
- 时间戳、版本号、来源、证据；
- 置信度、重要性、访问次数；
- `active / superseded / retracted / quarantined` 状态；
- `supersedes` 和 `related_ids`，用于版本追踪和多跳检索。

重复文本是幂等写入；同一实体和属性的新值会生成新版本并将旧值标记为 `superseded`。自动自然语言路径对“更新”使用比“读取”更严格的门槛：只有高置信语义匹配或明确词面证据才会退役旧碎片；无法确认同一属性时，宁可保留两条独立记录。不可信写入进入 quarantine，不参与正常读取，只有显式批准后才会变成 active。撤回不会抹掉审计记录，而是把记录置为 `retracted`。

### 多跳读取

记录可以带 `related_ids`。第一跳找到一个项目、人物或事件后，路由器会沿关联记录继续查找，直到：

- 达到 `memory_max_hops`；
- 达到 Top-K；
- 没有新关联页；
- 没有新证据。

每次读取都返回 `hop_trace` 和 `stop_reason`，不是只返回一段无法解释的文本。

### 写入安全边界

自动写入由原有自然语言记忆策略决定；V2 另外检查置信度和重要性。默认写入阈值为 `0.50`，读取阈值为 `0.65`。读取阈值是特意偏保守的：在实际 Qwen 测试中，未知的“我的血型是什么”最初会受到短中文问句的语义相似度干扰；提高阈值后该问题被路由层拒绝，不再把无关事实放进上下文前缀。

这不是“绝不出错”的证明，而是一个可检查的安全策略：

```text
低置信度写入 -> quarantine
旧值被纠正   -> 新版本 active，旧版本 superseded
用户撤回     -> retracted，读取隔离
证据不足     -> router_abstained / below_read_threshold
```

## 训练

### 通用路由器训练

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m V2_dpskw.train_memory_router_v2
```

训练目标包括：

1. 在 hard negatives 中选对目标记录；
2. 判断当前问题是否需要记忆；
3. 预测是否需要继续多跳；
4. 让无记忆问题学会 abstain。

### Qwen hidden-state 路由器训练

最终包使用的是 Qwen3.5 hidden state 上训练的路由器：

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m V2_dpskw.train_qwen_router_v2 `
  --output-dir W:\Flash\model\V2_dpskw\checkpoints\natural_memory_v2_qwen_router_entities
```

当前训练数据是本地生成的实体—属性—值事实及 hard negatives，不是公共榜单数据集。因此训练结果可以证明工程链路有效，但不能直接等同于公开 benchmark 的泛化能力。后续正式训练应加入真实对话脱敏集、改写问句、时间冲突、跨语言表达、未知事实和长文档事件链。

### 构建嵌入包

在支持硬链接的 NTFS 目录中，可以从 v1 包构建新的完整目录：

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  V2_dpskw\build_natural_memory_v2_package.py `
  --base-package W:\Flash\model\V2_dpskw\qwen3_5_4b_memory_merged_v13 `
  --output-dir W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2_new `
  --router-checkpoint W:\Flash\model\V2_dpskw\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt
```

W: 当前是 exFAT，不能创建硬链接。构建器现在默认拒绝复制多 GB 的冻结分片，必须明确加 `--allow-copy-base` 才允许复制；这样可以避免一次构建意外耗尽磁盘空间。当前交付包使用已存在的 v2 包原地更新，未重复复制两片 Qwen 主权重。

## 运行流式聊天

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m V2_dpskw.stream_chat_qwen_memory `
  --model-path W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2 `
  --max-new-tokens 128
```

当前生产默认不启用 SQLite/磁盘分页。若做独立的容量研究，旧版仍保留可选的分层页库参数，但它不属于本次默认运行路径：

```powershell
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m V2_dpskw.stream_chat_qwen_memory `
  --model-path W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2 `
  --tiered-memory-path W:\Flash\model\V2_dpskw\memory_pages.sqlite `
  --memory-resident-pages 64 `
  --kv-offload
```

启动后发送普通自然语言即可触发自动判断；不需要 `/remember`。常用控制命令仍保留：

- `/save`：把当前持久记忆写入 memory safetensors 切片；
- `/reset`：清空持久记忆并保存；
- `/quit`：退出。

模型重启时不会收到历史聊天记录。它只从嵌入式 memory shard 恢复记录、路由器和审计元数据。

## 单用户本地生产工作流

当前优先完成的 2/3/4/5 已经集中到一个入口：

```powershell
Set-Location W:\Flash\model
$py = "C:\Users\Administrator\miniconda3\envs\LLM\python.exe"

# 2. 规范化真实对话导出，并按 group_id 防止 train/eval 泄漏
& $py -m V2_dpskw.natural_memory_app build-dataset `
  --source W:\path\to\redacted_conversations.jsonl `
  --eval-source W:\path\to\redacted_eval.jsonl

# 2. 训练候选自动写入策略；不会覆盖当前生产适配器
& $py -m V2_dpskw.natural_memory_app train-policy `
  --model-path W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2 `
  --base-adapter W:\Flash\model\V2_dpskw\qwen_memory_adapter_natural_auto_v13 `
  --steps 240

# 4. 低显存连续运行与长上下文压缩压力测试
& $py -m V2_dpskw.natural_memory_app stress `
  --model-path W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2

# 5. 启动本地 API；默认只监听 localhost
& $py -m V2_dpskw.natural_memory_app serve `
  --model-path W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2 `
  --port 8765
```

默认训练数据目录是 `data/production_memory`，包含 `train.jsonl`、`eval.jsonl` 和
`manifest.json`。没有提供真实脱敏对话时，构建器会使用工程内 bootstrap 数据；这只能验证链路，不能冒充真实业务泛化结果。

管理接口：

```text
GET    /health
GET    /v1/memory?status=active&query=...
GET    /v1/memory/{record_id}
GET    /v1/memory/export
GET    /v1/memory/audit
POST   /v1/memory/{record_id}       # 版本化编辑
DELETE /v1/memory/{record_id}       # 可审计撤回
POST   /v1/memory/reset             # 清空并持久化
POST   /v1/chat                     # {message,max_new_tokens,stream}
POST   /v1/memory                    # 管理员/测试用显式写入
```

`POST /v1/chat` 默认由模型自己的自动策略决定是否写入；`stream: true` 返回 SSE token 流。服务以单模型锁串行化请求，避免同一用户的 memory state 被并发写坏。默认自动持久化会回写 embedded memory safetensors；测试时可加 `--no-auto-persist`。

## 记忆管理与审计

编辑不是覆盖原记录，而是生成 `version + 1` 的 successor，并把旧记录标记为
`superseded`；删除同样不物理抹除，而是标记为 `retracted`。`GET /v1/memory/audit`
会检查页容量、页指针、冲突索引和多跳关联是否存在悬空引用。所有管理接口只暴露 JSON-safe
元数据，不返回路由向量和模型内部 tensor。

## 当前受控验证结果

- 单元测试：20/20 通过；
- bootstrap 数据规范化：训练 1485 条，验证 363 条；
- 候选自动策略 smoke train：4bit、batch 2、24 steps，验证集 accuracy 96.88%、recall 90.91%、FPR 0；这不是最终生产成绩；
- 10 轮压力测试：写入 40 条，召回 39/40（97.5%），2 次长上下文压缩，审计 healthy，0 errors；
- 压力测试峰值约 3.2 GB allocated VRAM，使用 256 条/131072 token 的自适应热点缓存上限，并保留 2048 MB 显存安全余量；
- 压力测试不会写回模型包，生产服务只有在启用自动持久化时才会写回。
- v7 自然语言检索器 + 碎片保留策略烟测：teacher 19/20（95%），Natural Memory 20/20（100%），student/teacher = 1.0526，paired parity = 19/19；多跳、冲突更新、删除、干扰项、长上下文和未知拒答全部通过。
- 默认嵌入 v7 包的同一烟测：teacher 19/20（95%），Natural Memory 20/20（100%），parity gate 通过；报告为 `mega_memory_vs_full_kv_smoke20_embedded_v7.json`。

## 重启验证

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m V2_dpskw.test_natural_memory_v2_restart `
  --model-path W:\Flash\model\V2_dpskw\qwen3_5_4b_natural_memory_v2 `
  --report W:\Flash\model\V2_dpskw\natural_memory_v2_restart_test.json
```

测试会：

1. 清空选定包的持久记忆；
2. 通过普通用户句子自动写入一条控制事实；
3. 写入嵌入式 safetensors memory shard；
4. 释放第一个 Qwen 模型；
5. 重新加载模型，只输入一个新问题；
6. 检查 V2 router decision、内部 prefix 和生成答案；
7. 默认清理测试事实，避免污染工作包。

当前实际结果：自动写入成功；重启后路由器找到 `page_00000001` 和目标记录；内部前缀长度 36；生成结果精确返回 `NM-V2-RESTART`；清理后页面与记录均为 0。

## 评测

### 单元测试

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m unittest discover -s V2_dpskw\tests -v
```

当前结果：20/20 通过，覆盖路由张量形状、压缩地址、页粗索引、版本冲突、quarantine、批准、撤回、多跳、导出恢复、语义检索键导出恢复、KV 预算、页容量上限、批量上下文分块、记忆编辑/撤回/审计，以及分层后端重启、冷页卸载和隔离区恢复。

### 超大自然语言记忆验证集

`data\mega_validation\memory_validation_100k.jsonl` 已完整生成 100,000 条用例，固定分成 10 类、每类 10,000 条：单事实、32/128 干扰项、冲突更新、多跳、未知拒答、随机位置、长文本、改写问句和删除纠错。当前文件 SHA-256 为 `dafefba852ef1539fba3391b276aa264759395e9677d54fdd5b4fce49ada05b8`。

20 条分层 smoke 用例用于本机快速质量门槛；全量 100,000 条会显著增加推理时间，建议在长时间窗口执行，并持续观察 RTX 5070 显存，不要与训练任务并行：

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m V2_dpskw.natural_memory_app benchmark-kv `
  --model-path qwen3_5_4b_natural_memory_v2 `
  --adapter qwen_memory_adapter_natural_production_candidate_v7 `
  --data data\mega_validation\memory_validation_100k.jsonl `
  --output mega_memory_vs_full_kv_100k_v7.json `
  --limit 100000 --max-new-tokens 96
```

在全量端到端生成完成前，100,000 条数据是“完整验证集”，不是已经完成的 100,000 条生成成绩；当前可复现的质量结论以 smoke 报告和单元测试为准。

### V2 存储与路由评测

报告：`W:\Flash\model\V2_dpskw\natural_memory_v2_benchmark.json`

| 指标 | 实测结果 |
|---|---:|
| 合成记录 | 20,000 |
| 页面数 | 626 |
| 粗候选页平均数 | 178.13 |
| 粗候选页占比 | 28.45% |
| 记录 Recall@K | 100% |
| 页面 Recall@K | 100% |
| 多跳成功 | 100%，2 hops |
| 导出/恢复后召回 | 100% |
| 冲突版本/纠错/隔离/批准/撤回/幂等 | 全部通过 |
| 地址空间容量（32K 页 × 32） | 1,048,576 条记录 |
| 通用路由器 route accuracy | 99.32% |
| 通用 need-memory precision/recall/specificity | 100% / 100% / 100% |
| 通用 hop accuracy | 86.25% |

上述是控制变量下的合成存储评测，证明的是分页、索引和状态机，不是 1M 条真实用户记忆已经完成验证。

### 分层后端百万级压力测试

报告：`W:\Flash\model\V2_dpskw\tiered_memory_v2_1m_benchmark.json`

该测试实际写入 1,000,000 条轻量记录、31,250 页和 64 个常驻页。它是旧的 durable page store 容量实验，不是当前默认方案；当前默认方案要求所有记忆随第三个 safetensors 切片加载进进程内存，再用有界 VRAM cache 加速热点记录。无论哪种方案，这个数字都不等于百万条完整自然语言长文本在 Qwen 上的端到端生成质量。

| 指标 | 实测结果 |
|---|---:|
| 实际写入记录 | 1,000,000 |
| 实际页面 | 31,250 |
| 重启后记录总数 | 1,000,000 |
| 重启后常驻记录 | 160 |
| 冷页 | 31,186 |
| 目标记录重启召回 | 通过 |

### Qwen 路由器验证

报告：`W:\Flash\model\V2_dpskw\checkpoints\natural_memory_v2_qwen_router_entities\qwen_router_v2_training.json`

- 512 条生成事实；409 条训练，103 条 held-out；
- route accuracy：91.26%；
- need-memory precision/recall/specificity：100% / 100% / 100%；
- hop accuracy：36.70%。

hop controller 目前明显弱于候选记录路由，因此运行时不会把它当成唯一正确性来源；关联记录和显式 `related_ids` 仍由存储层约束，后续训练应重点补多跳样本。

### 与原版 Qwen3.5-4B 的综合回归

报告：`W:\Flash\model\V2_dpskw\natural_memory_v2_full_benchmark.json`

同一份 120 个固定用例、同一 Qwen3.5-4B 主干、同一 4-bit NF4 加载和贪心解码：

| 指标 | 原版 Qwen3.5-4B | Natural Memory v2 |
|---|---:|---:|
| 总分 | 0.85833 | 0.85833 |
| 总分变化 | - | 0 |
| 通用 / 数学 / 推理 / 语言 / 知识 / 逻辑 / 上下文分类 | 基线 | 各分类 delta = 0 |
| 自动写入 precision | - | 100% |
| 自动写入 recall | - | 100% |
| 自动写入 specificity | - | 100% |
| 无历史重启恢复 | - | 通过 |
| 清理后不再召回 | - | 通过 |

综合评测的硬件是 RTX 5070 11.94 GiB，完整模型使用 4-bit NF4。显存峰值字段来自 PyTorch allocator，在当前 Transformers/bitsandbytes 组合下可能高于物理显存读数，不能把该字段当成独立硬件测量；最终是否能运行，应以实际 GPU OOM 和 `nvidia-smi` 为准。

### 长上下文边界

报告：`W:\Flash\model\V2_dpskw\long_context_v1_stress.json`

现有直接 KV 路径在约 8K token 可以运行，16K 和 32K 会 OOM。现在 V2 已接入 Transformers 的 CPU-backed `DynamicCache(offloading=True)`，并在 Qwen3.5 混合线性/全注意力结构上完成真实生成验证；线性注意力的微小循环状态留在执行设备，只有昂贵的 full-attention KV 迁移到 CPU。

模型还提供自动热窗口压缩：当输入超过 `kv_budget_tokens`，旧前缀会按 `context_chunk_tokens` 分块写入 V2 context records，当前生成只保留最近热窗口；读取仍然走“粗索引 -> 候选页 -> 精排 -> Top-K”，不是让当前 token 对 1M slot 做注意力。真实自动路径测试将 371 token 压缩为 32 token，并写入 22 条可追溯上下文记录，随后成功生成。

这仍然不能宣称当前包支持 200M—300M 原始 token 上下文：CPU KV offload 不是 NVMe 分页注意力，自动压缩目前也没有完成百万 token 的 Qwen 质量训练。正确的工程目标是热 KV 保留当前窗口，超出窗口的内容经过事件切分、摘要、实体关系和可逆原文页写入 V2；查询时只加载少量相关页，再由 Qwen 做最终回答。

KV offload 与自动压缩验证：

```powershell
Set-Location W:\Flash\model
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe `
  -m V2_dpskw.benchmark_kv_offload
```

## 关键源文件

| 文件 | 作用 |
|---|---|
| `memory_os_v2.py` | V2 路由器、分页存储、LSH 粗索引、精排、多跳、版本和 KV 预算 |
| `qwen_integration.py` | 将 V2 接入 Qwen，负责内部 prefix、嵌入式 safetensors 读写和重启恢复 |
| `stream_chat_qwen_memory.py` | 不依赖历史上下文的流式聊天与自动写入 |
| `train_memory_router_v2.py` | 通用路由器 hard-negative 训练 |
| `train_qwen_router_v2.py` | Qwen hidden-state 路由器训练 |
| `tiered_memory_store_v2.py` | RAM/磁盘分层、二进制记录和可恢复 page store |
| `benchmark_tiered_memory_v2.py` | 1M 级 durable page store 压力测试 |
| `benchmark_memory_v2.py` | 独立存储/路由/完整性评测 |
| `benchmark_natural_memory_v1.py` | 兼容 v1/v2 的 Qwen 综合回归评测 |
| `test_natural_memory_v2_restart.py` | 真实模型释放、重载、无历史召回测试 |
| `tests/test_memory_os_v2.py` | V2 核心单元测试 |

## 当前限制与下一阶段

已经实现的是可运行的 V2 内核和百万级 durable 存储后端；模型质量和长上下文仍有明确边界：

1. 真实 Qwen 路由器只用 512 条本地合成事实训练，需继续做跨实体、改写、冲突、时间和未知事实泛化；
2. 多跳控制器的 held-out accuracy 只有 36.70%，需要专门的多跳 curriculum；
3. 当前交付包使用 embedded weight-shard 模式，所有 V2 记录随第三个 safetensors 切片加载进 RAM；热点记录优先进入 VRAM，最多 256 条/131,072 token，并动态保留 2 GiB 显存安全余量；显存不足时自动留在 RAM，避免把模型推理和 KV 顶满；
4. `memory_max_pages=32768` 代表 32K 页 × 32 条记录的地址上限；1M 轻量记录已经完成存储压力测试，但不是 1M 条完整长文本的 Qwen 端到端质量验证；
5. 还需要加入更强的压缩摘要、原文页、事件时间线、事实置信度校准和用户级撤销日志；
6. 还需要长序列课程训练、NVMe 级分页和更大规模端到端评测，才能验证 128K 热 KV 与百万级历史的实际吞吐和质量。

这些限制是设计边界，不是用一个“无限上下文”数字掩盖的未验证假设。
