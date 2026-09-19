# 写入路径修复：20 条不同属性事实不再互相销毁

## 症状

一次写入 20 条**不同属性**的事实后，库里有 20 条记录但只剩 **12 条 active**，**8 条在查询前就已被
retract**；16 个关键零重叠用例中有 **8 个的目标记录已不存在**，因此端到端正确率存在 **50% 的硬上限**
—— 任何排序器都救不回一条已被删除的记录。

## 根因（栈追踪确证，非推断）

16 次 retract 的调用栈完全一致：

```
eval_end_to_end_memory.py:106 write_fact
 -> qwen_integration.py:3807 forward
 -> qwen_integration.py:2195 _write_text_memory      <-- retract 调用点
 -> memory_os_v2.py:2578 MemoryOSV2.retract_record
 -> memory_os_v2.py:1959 PagedMemoryBankV2.retract   (status = retracted)
```

`status_transition_summary == {"active->retracted": 16}`，无 `superseded`、无 `quarantined`，
20 次 `bank.write` 全部返回 `inserted` —— 写入路径的**唯一**销毁机制就是 2195 行的 retract。

授权条件 `confirmed_update` 在 20 次写入中为真 **5 次**，且**全部只走"学习重排器
`learned_best >= 0.95`"这一条**（`qwen_integration.py:2140-2142`）：

| 写入 | 事实 | learned_best | 结果 |
|---|---|---:|---|
| 2 | 出生城市 | 0.998959 | confirmed_update=True |
| 3 | 办公城市 | 0.992638 | confirmed_update=True |
| 9 | 工位楼层 | 0.964987 | confirmed_update=True |
| 12 | 手机尾号 | 0.998516 | confirmed_update=True |
| 15 | 办公楼层 | 0.997487 | confirmed_update=True |

随后 2187-2197 的循环退掉 8 条：3 条走 `record.slot_index == slot`，5 条走
`score >= 0.95 and shared >= 2`；2196 行的 `break`（仅当 `score < 0.98` 才停）让写 #12 一次
级联退掉 3 条、写 #15 退掉 2 条。被退掉的正是写 1/2/3/4/9/10/11/14 —— 16 条关键事实中的 8 条。

三条辅助事实：

* `exact_slots.numel() > 0`（token 完全相同）**0/20 次**都没命中 —— 精确重复路径与此无关；
* `shared >= 2` 毫无区分力：19 个候选的 `shared` 全落在 {2,3,5,6,7}（每条事实都含"我的/是"模板词元）；
* 词面兜底分支（`>= 0.30`）在重排器就绪时是**死代码**（`learned_best is not None`），
  且更宽松：真跑起来 17/20 次会通过。

**根因**：retire 一条已存在记录的授权**完全来自学习打分加一个固定 0.95 阈值**，没有任何
"新文本与在位记录共享 `(entity, attribute)`"的结构性校验。而打包进来的 `text_retriever` 在真实
e2e 事实集上对**不相关**属性过度自信：190 个不相关对中 **10 个 ≥ 0.95**，最高 **0.9985**
（如"常住城市 CITY-A1B2C3" vs "出生城市 CITY-D4E5F6" = 0.9985）。

> 说明：这条结论修正了我早先的一次错误判断。我曾用自己合成的 `VAL-A%07d` 事实集测得
> "0/190 对达 0.95"并据此宣布"重排器过度自信假设被推翻"——那是**数据集不同**造成的：
> 同一测量在 e2e 协议真正写入的事实集上得到 10/190。因此调阈值不可行：假阳性高达 0.9989，
> 没有任何标量切点能分开"不相关属性"与"真更新"。

## 修复（已应用，两处）

1. `qwen_integration.py` `_write_text_memory`：`confirmed_update` 的两条**打分类**分支改为
   **结构优先** —— 仅当候选文本的 `entity::attribute` 已经在库的 `active_by_conflict` 账本中
   存在（即确实是同一条事实的更新）才允许由打分授权；`exact_slots`（token 完全相同）这条
   结构安全的分支保持不变，以保留"重复写入幂等"的既有语义。
2. 2187-2197 的 retract 循环增加守卫：`if record.conflict_key() != candidate_conflict_key: continue`，
   循环只能退掉**描述同一属性**的记录，不再因"共享热槽"或"词面重叠高"而销毁无关事实。

语义不受损：真·同属性更新本来就已经由 `PagedMemoryBankV2.write` 通过 `active_by_conflict`
做版本化（旧版本保留为 `superseded`，`version+1`），修复只是移除了那条**额外的、无监督的销毁路径**。

## 四级验证（全部通过）

| 级 | 检验 | 修复前 | 修复后 |
|---|---|---|---|
| ① | `probe_write_retraction_trace.py`：`status_transition_summary` | `{"active->retracted": 16}` | **`{}`** |
| ① | 同上：`retraction_calls` / `status_counts` | 16 次 / `{"retracted": 8, "active": 12}` | **`[]` / `{"active": 20}`** |
| ② | 正对照：同属性改值 | —— | 旧值 `superseded` + 新值 `active`，其余 19 属性不受影响，0 retract |
| ③ | 重排器缺席对照（更宽松的词面分支） | 会踩死代码分支 | 20 条全 active、0 retract |
| ④ | `probe_record_selection.py --gate-scan`：`target_record_active` | 8 / 16 | **16 / 16** |
| ④ | 同上：`target_record_retracted_or_absent` | 8 | **0** |
| ④ | 同上：`active_records_min/max` | 12 / 12 | **20 / 20** |
| ④ | 同上：probe 口径回答正确率 | 25.00%（4/16） | **43.75%**（7/16） |

`verify_write_path_fix.py` 的三阶段与 8 项检查全部通过（`write_path_fix_verification.json`）。

## 端到端效果（官方 harness，16 个零重叠用例）

| 路由器 | 修复前 | 修复后 | 答成别的属性 |
|---|---:|---:|---|
| deployed（原版 NM2） | 25.00% | **50.00%**（+25.00pp） | 18.75% → 12.50% |
| V2-128-v6 | 37.50% | **68.75%**（+31.25pp） | 25.00% → 18.75% |
| REPLAY-128 | 37.50% | **68.75%**（+31.25pp） | 25.00% → 18.75% |

`router_critical_e2e_after_write_fix.json` / `.md`。

## 注意事项与仍未解决的部分

* **路由器评分卡不受此修复影响**：评分卡直接读冻结特征库、不走运行时的写入/读取路径，
  所以 `router_scorecard_final.md` 的 22 轴结论无需重跑，也不因本次改动而改变。
* 单元测试 52 项仍全部通过（`python -m unittest discover -s tests -t .`）。
* **仍未解决**（本次未触碰）：这 16 个用例里 **93.75% 的读取仍走旧版 16 槽路径**
  （`legacy_prefix_used_pct: 93.75`）而非 V2 库记录，地址命中 0/16；记录级排序仍由打包的
  `text_retriever` 决定而非路由器。这两项是端到端正确率的下一个瓶颈。
* 220 用例的 `eval_end_to_end_memory.py` 全量套件本次未重跑。
