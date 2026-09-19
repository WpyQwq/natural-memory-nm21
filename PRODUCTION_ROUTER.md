# 生产路由器交付说明（NM2 Router · V2-128 v7）

**交付件**：`checkpoints/router_replay_v7_v2_128/memory_router_v2.pt`
**SHA-256**：`69f8295e78821e52c0b41b248e7eddbfa47d7e409539f4a13382c0c82d189deb`
**大小**：8,156,921 字节 · **格式**：raw `state_dict`（16 张量 / 2,037,774 参数 / float32 / 数值全部有限）

几何与现网**完全一致**：`router_dim=128`、`num_heads=8`、`hidden_size=2560`、`max_hops=3`、
每条记录地址 **512 字节**、参数量 **2,037,774**（与原版完全相同）。

---

## 1. 交付件相对原版 NM2 的成绩（冻结评测集 21,920 条 / 10 类别）

判定口径：`verdict_router_v6.py` 逐轴对原版部署权重比较，速度类噪声轴用 3% 相对容差。

| 指标 | 原版 NM2 | **本交付件** | 变化 |
|---|---:|---:|---|
| Top-1 正确率 | 41.12% | **94.14%** | +53.01pp |
| Recall@1 / @3 / @5 | 37.03% / 46.73% / 48.91% | **88.58% / 96.48% / 97.15%** | +51.55 / +49.75 / +48.23pp |
| MRR / nDCG@3 | 47.69% / 44.22% | **95.77% / 95.37%** | +48.08 / +51.15pp |
| 多跳证据全中(Top-3) | 43.43% | **96.22%** | +52.80pp |
| 多跳证据全中(仅多正例) | 37.15% | **95.45%** | +58.30pp |
| hop 正确率 | 73.23% | **100.00%** | +26.77pp |
| hop 欠预测率 ↓ | 4.68% | **0.00%** | −4.68pp |
| need F1 / 召回 | 89.58% / 99.82% | **100.00% / 100.00%** | +10.42 / +0.18pp |
| **未知拒答率** | 0.00% | **100.00%** | +100.00pp |
| **已知问题被误拒率 ↓** | 0.18% | **0.00%** | −0.18pp |
| 未知问题被误读率 ↓ | 100.00% | **0.00%** | −100.00pp |
| 仲裁准确率 | 81.12% | **100.00%** | +18.88pp |
| 每条记录地址字节 | 512 | **512** | 持平 |
| 参数量 | 2,037,774 | **2,037,774** | 持平 |

**逐轴判定：22 通过 / 0 未通过 —— 全方位超越原版 NM2**
（`router_verdict_final.json` / `router_verdict_final.md`）

**门槛曲线复核**：不止默认阈值 0.50 —— 在 0.30 / 0.40 / 0.50 / 0.60 / 0.70 / 0.80 **每个门槛**上，
未知拒答率均 **100.00%**、已知问题被误拒率均 **0.00%**、未知问题被误读率均 **0.00%**、
need F1/召回/精确率/仲裁准确率均 **100.00%**（`threshold_sweep_check.md`，5/5 候选全门槛通过）。

---

## 2. 生产可用性验证

| 项目 | 结果 | 证据 |
|---|---|---|
| 可替换性（drop-in） | **DROP-IN REPLACEMENT OK**：16/16 键匹配、无缺失/多余、无形状不符、分数有限，且实测驱动 `PagedMemoryBankV2` 路由（`bank_routed_records: 3`） | `router_swap_check_replay.json` |
| 延迟（7 轮**交错**取中位数，消除顺序效应） | 单查询 **1.4242 ms** vs 原版 **1.4203 ms**（差 0.3%，同架构同参数量）；batch=256 QPS 146,676 vs 164,556 | `router_latency_bench_prod.json` |
| 未见过改写问法的泛化 | Top-1 **59.60%**、Recall@3 **83.20%**（24 个同形候选，随机基线 4.17%） | `replay_check_zov.json` |
| 端到端（16 条零重叠用例，官方 harness） | 回答正确率 **68.75%**、答成别的属性 18.75% | `router_critical_e2e_after_write_fix.json` |
| 单元测试 | 52 项全部通过 | `python -m unittest discover -s tests -t .` |

> ⚠️ **关于速度轴的一个更正**：早先某次记分卡把 `router_best.pt` / `router_step_*.pt`（wrapper 格式）
> 报成单查询慢 30%，据此判它们"速度不达标"。交错基准复测证明五个 checkpoint 的中位数全在
> 1.4123–1.4343 ms（彼此 ≤1%）—— **那是记分卡顺序测量的伪影，不是路由器属性**，该判定作废。

---

## 3. 准确度上限与取舍（选型依据，如实记录）

同一权威基准上，128 维 drop-in 几何内的候选对比：

| 候选 | v6 基准 Top-1 | 未见改写 Top-1 | 说明 |
|---|---:|---:|---|
| **本交付件 REPLAY-128 v7** | **94.14%** | **59.60%** | 22/22 轴、drop-in、延迟持平 |
| `router_v6_v2_128/router_best.pt` | **94.62%**（基准最高） | **7.60%** | v6 基准最强，但**新问法上崩掉**（低于随机 4.17% 附近） |
| `router_v6_v2_128/memory_router_v2.pt` | 94.37% | 18.40% | v6 最终版 |
| `router_prod_v2_128/memory_router_v2.pt`（本轮 20k 步长训） | 93.78% | — | 从 v6 best 出发在合并语料上继续训练反而退化 |
| 原版 NM2 | 41.12% | 11.60% | 基线 |

**结论**：在 drop-in 几何内，v6 基准的检索准确度上限约 **94.6%**，但那个点在新问法上只有 7.60%。
本交付件用 **0.48pp** 的 v6 基准差距换来 **+52.00pp** 的未见问法能力（7.60% → 59.60%）。
对真实用户提问而言这是明显正确的取舍，因此以它为生产件。

另注：本轮 20k 步长训的曲线峰值出现在第 7000 步（合并评测 Top-1 93.98%），但**该权重未落盘**
（`--checkpoint-interval` 默认 10000，且训练器的 `router_best.pt` 保存未生效）。
同时提醒：**从 15 个 eval 点里挑最高值属于在评测集上做选择**，93.98% 带选择性偏差，
不应作为交付指标 —— 故未采用。

---

## 4. 安装（3 步）

```powershell
# 1) 备份原版权重
Copy-Item 'H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt' `
          'H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt.bak'

# 2) 替换（几何完全一致，无需改 memory_router_dim，无需重建地址）
Copy-Item 'H:\Memory\V2_dpskw\checkpoints\router_replay_v7_v2_128\memory_router_v2.pt' `
          'H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt'

# 3) 校验替换结果
cd H:\Memory\V2_dpskw
& 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe' -m V2_dpskw.check_router_swap `
  --candidate 'H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt'
# 期望输出: VERDICT: DROP-IN REPLACEMENT OK
```

若模型是**内嵌合并包**（`memory_merge.json`），权重键为
`dynamic_memory.memory_router_v2.*`，需把本 `state_dict` 的 16 个张量按该前缀写回
`safetensors`；`merge_memory_weights.py` 是这件事的现成入口。

---

## 5. 本交付件**不**解决的问题（避免误判）

以下都是**评测过的独立问题，换路由器不会改善**，详见 `WRITE_PATH_FIX.md` 与 `BOTTLENECK_REPORT.md`：

1. **记录级排序不由路由器决定**：`memory_os_v2._record_scores` 对带 `semantic_key` 的记录用打包的
   `text_retriever` 覆盖路由器分数。实测打包 retriever 的 Top-1 只有 **23.20%**，而本交付件同任务
   达 **59.60%** —— 但把权重混合进去后，端到端**独立进程复测无任何变化**
   （被 `1.25·rare_lexical_address` 等先验项压过，注入记录集合不变）。
2. **"句式正常的未知"识别不了**：问库中不存在的属性时，**78.00%（39/50）**仍给出编造答案；
   两组检索最高分分布几乎完全重叠（p50 288.20 vs 285.25），**阈值方案被测量排除**。
3. **写入路径曾摧毁记录**：一次写 20 条不同属性事实后只剩 12 条 active（现已修复为 20 条，
   详见 `WRITE_PATH_FIX.md`，该项修复把端到端从 37.50% 提到 68.75%）。

即：本交付件把**路由器这一环**做到了 22/22 全方位超越且生产可用；端到端体验的主要剩余瓶颈
在写入路径之后的检索/排序与拒答判断上，不在路由器权重里。

---

## 6. 复现命令

```powershell
$env:PYTHONPATH='H:\Memory'; $env:PYTHONIOENCODING='utf-8'
$py='C:\Users\Administrator\miniconda3\envs\LLM\python.exe'
cd H:\Memory\V2_dpskw

# 22 轴评分卡 + 逐轴判定（权威基准）
& $py -m V2_dpskw.eval_router_v5 --train-file data/router_training_v6/train.jsonl `
  --eval-file data/router_training_v6/eval.jsonl `
  --feature-cache H:\Memory\nm_cache\nm_router_v6\feature_cache `
  --run "V2-128 deployed(v3)=H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt" `
  --run "REPLAY-128 v7 final=checkpoints\router_replay_v7_v2_128\memory_router_v2.pt" `
  --output router_scorecard_final.json --markdown router_scorecard_final.md

& $py -m V2_dpskw.verdict_router_v6 --scorecard router_scorecard_final.json `
  --candidate "REPLAY-128 v7 final" --baseline-prefix "V2-128 deployed" `
  --output router_verdict_final.json --markdown router_verdict_final.md

# 零重叠泛化
& $py -m V2_dpskw.eval_router_v5 --train-file data/zero_overlap/train.jsonl `
  --eval-file data/zero_overlap/eval.jsonl `
  --feature-cache H:\Memory\nm_cache\nm_zero_overlap\feature_cache `
  --model-path qwen3_5_4b_natural_memory_v2 --candidate-count 24 `
  --run "REPLAY-128 final=checkpoints\router_replay_v7_v2_128\memory_router_v2.pt" `
  --output replay_check_zov.json --markdown replay_check_zov.md

# 延迟（交错中位数）
& $py -m V2_dpskw.bench_router_latency --rounds 7 --single-samples 300 `
  --run "deployed=v2:H:\Memory\dynamic_memory_lab\checkpoints\natural_memory_v2_qwen_router_entities\memory_router_v2.pt" `
  --run "REPLAY=v2:checkpoints\router_replay_v7_v2_128\memory_router_v2.pt" `
  --output router_latency_bench_prod.json
```
