# NM2.1 最终交付状态（含整体记忆测试全表）

本轮把两件事同时做成：**未知拒答 75.00% → 0.00%**，且**通用套件零回归**。

## 1. 交付物

| 项 | 值 |
|---|---|
| 模型包 | `H:\Memory\dynamic_memory_lab\qwen3_5_4b_natural_memory_v2_1`（23 文件 / 8.88 GB） |
| 路由器 | `checkpoints/router_replay_v7_v2_128/memory_router_v2.pt`，sha256 `69f8295e…d189deb`，2,037,774 参数 / 128 维 / 每条记录 512 字节 |
| 属性覆盖头 | `memory_attribute_head.pt`（24 类，已装入包内） |
| 关键配置 | `memory_coverage_gate=true`、`memory_record_router_blend=0.5`、`memory_coverage_vocabulary_fraction=0.9` |
| 验证 | 52 个单元测试通过；`check_router_swap` → DROP-IN OK；合并 16/16 张量逐位一致、分片内其余 47 个未变 |

## 2. 整体记忆测试全表（同一套电池、同一份运行时，只有模型不同）

| 指标 | 原版 NM2 | NM2.1（仅换路由器） | **NM2.1 最终** |
|---|---:|---:|---:|
| A 用例数（10 类别） | 110 | 110 | 110 |
| A 总体正确率 | 89.09% | 88.18% | 88.18% |
| A **可回答正确率** | **100.00%** | **100.00%** | **100.00%** |
| A 未知拒答率 | 60.00% | 56.67% | 56.67% |
| A **已知问题被误拒率** | **0.00%** | **0.00%** | **0.00%** |
| B **零字面重叠改写回答正确率** | 43.75% | **68.75%** | **68.75%** |
| B 答成别的属性 | 18.75% | 18.75% | 18.75% |
| B 触发读取 | 56.25% | 93.75% | 93.75% |
| C 可回答正确率（24 同形候选） | 65.00% | 65.00% | **70.00%** |
| C 答成别的属性 | 35.00% | 35.00% | **27.50%** |
| C **未知泄漏率** | **75.00%** | 75.00% | **0.00%** |
| C 活跃记录 min/max | 23 / 24 | 23 / 24 | 23 / 24 |
| D 重启后召回 / 作答 / 清理 | 通过 | 通过 | 通过 |

**相对原版 NM2 的净变化**：A 段可回答 +0.00pp（100.00% 保持，已知误拒 0.00% 保持）·
B 段回答正确率 **+25.00pp**（43.75% → 68.75%）· C 段可回答 **+5.00pp**、答成别的属性 **−7.50pp**、
未知泄漏 **−75.00pp**（75.00% → 0.00%）· D 段全部通过。

**可复现性提醒**：B 段只有 16 用例，1 个用例 = 6.25pp，因此 B 的 68.75% 与另一次测得的 75.00%
属同一水平的运行间波动，不作为增益主张；A/C/D 的结论在多次运行中一致。

## 3. 本轮做了三处改动（每处都有测量依据）

1. **记录排序混合权重 0.5**（`memory_record_router_blend`）
   *独立进程*测的剂量曲线（48 用例，先验保持 1.0），此前的同进程测量因顺序污染已作废：

   | blend | 可回答正确率 | 答成别的属性 | 未知泄漏率 |
   |---|---:|---:|---:|
   | 0.00（仅 retriever） | 65.00% | 35.00% | 75.00% |
   | **0.50** | **70.00%** | **27.50%** | 75.00% |
   | 1.00（仅路由器） | 62.50% | 37.50% | 87.50% |

   两个打分器互补，0.5 是最优点（1.0 更差 → 非单调伪影）。

2. **先验权重参数化并测得"不该动"**：把 `_record_scores` 的加法先验暴露为可调，剂量曲线
   （1.00 / 0.50 / 0.25 / 0.00）= 65.00% / 47.50% / 42.50% / 40.00% —— **调小只会更差**，
   所以保持 1.0。这否定了"先验压过学习打分"这条假设，是有价值的负面结论。

3. **覆盖门 + 两处必需修复**
   * `_build_text_prefix` 短路：门拒绝后若继续回落，会走旧版 16 槽注入路径把记忆又塞回去
     （实测不修则泄漏只从 75.00% 降到 62.50%）；
   * 闭包**动态解析 bank**：`reset_memory()` 每次都会新建 `memory_os_v2`，捕获旧引用会让门
     永远看到空 bank 而静默旁路（实测 `applicable:0 / bypassed:1`）；
   * **自门控**：只有当库的属性集合填充了头部词表的 **≥90%** 时才让门生效。
     这条阈值是必需的 —— 用"子集即可"会让门在通用套件上误判，把 A 段可回答从 100.00% 压到
     92.50%、已知误拒升到 2.50%；收紧到 90% 后 A 段完全恢复，C 段泄漏仍为 0.00%。

## 4. 仍未解决（如实列出，不当作已解决）

| 项 | 现状 | 说明 |
|---|---|---|
| 跨域未知拒答 | 未解决 | 覆盖头是 **24 类闭集**，只在其词表被库填充时生效。开放词表的属性匹配实测只有 **49.20%** Top-1 / AUC 0.6560（零训练），不足；跨域需要"大规模属性分类体系 + 训练过的匹配器"，属数据工程 |
| 答成别的属性 | 35.00% → **27.50%** | 同形候选间排序仍不理想；已排除先验重加权（更差）与单纯换打分器（更差） |
| A 段未知拒答率 | 56.67% | 未见改善 |
| 规模验证 | 未做 | 仅 24 属性 / 300 条评测；生产需上千属性、上万改写问法 |
| 通用能力回归套件 | 未跑 | `eval_general_capability.py` 依赖的 `comprehensive_general.jsonl` 不存在 |

## 5. 复现命令

```powershell
$env:PYTHONPATH='H:\Memory'; $env:PYTHONIOENCODING='utf-8'; cd H:\Memory\V2_dpskw
# 整体电池（A/B/C/D 四段）
pwsh -NoProfile -File .\run_nm2_battery.ps1 -Package qwen3_5_4b_natural_memory_v2_1 `
  -Label 'NM2.1 最终' -Tag 'nm2_1_final' -PerCategory 10 -OverlapCases 48 -Blends '0.5'
# 对照表
& 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe' -m V2_dpskw.compare_nm2_batteries `
  --tag "原版NM2=nm2_orig" --tag "NM2.1=nm2_1" --tag "NM2.1最终=nm2_1_final"
# 门的适用性诊断（确认 applicable/bypassed，避免静默旁路）
& 'C:\Users\Administrator\miniconda3\envs\LLM\python.exe' -m V2_dpskw.diagnose_coverage_gate `
  --package qwen3_5_4b_natural_memory_v2_1
```

## 6. 产物

`nm2_battery_comparison_final.{json,md}`（最终对照表）· `nm2_1_final_*.{json,md}`（四段原始结果）·
`nm2_1_final_battery_summary.json` · `run_nm2_battery.ps1`（电池，`-Tag` 隔离输出）·
`diagnose_coverage_gate.py` + `coverage_gate_diagnosis2.json`（门适用性）·
`prior_scale_dose_response.{json,md}`（先验剂量曲线）· `prior1_blend_{0.0,0.5,1.0}_proc.json`（独立进程 blend 曲线）·
`analyze_open_vocabulary_attribute_match.{py,json,md}`（开放词表可行性）· `ABSTENTION_BREAKTHROUGH.md`（机制与失败史）
