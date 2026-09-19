# 已移出本 fork 的内容（归档到 E 盘）

本文件记录 **2026-09-12** 从 `H:\Memory\V2_dpskw` 移出、归档到
`E:\归档\03_代码与项目\V2_dpskw-备份\` 的内容，便于日后查找或恢复。
归档规范见 `E:\归档\AGENTS.md`；操作已记入 `E:\归档\00_索引\操作日志.jsonl`。

## 归档包

| 包 | 大小 | 未压缩 | 内容 |
| --- | ---: | ---: | --- |
| `V2_dpskw_20260912_085921.7z` | 7.09 GB | 13.14 GB / 411 文件 | 整个 fork 的冻结备份：全部代码、`data/` 数据集、`checkpoints/` 最终权重与日志、全部评测证据 JSON/MD |
| `V2_dpskw_中间与废弃checkpoint_20260912_132740.7z` | 6.48 GB | 7.90 GB / 116 文件 | 训练中间快照与废弃运行（清单见下） |

两个包都通过 `7z t` 完整性测试，并与源文件**逐文件比对字节数一致（116/116）后才删除源文件**。

## 第二个包的具体内容（已从 H: 删除）

| 来源 | 文件数 | 大小 | 说明 |
| --- | ---: | ---: | --- |
| `checkpoints/router_v6_v2_128/router_step_*.pt` | 10 | 0.23 GB | 训练中间快照（每 10000 步一个） |
| `checkpoints/router_v6_v2_512/router_step_*.pt` | 10 | 0.53 GB | 同上 |
| `checkpoints/router_v6_xl128/router_step_*.pt` | 10 | 0.76 GB | 同上 |
| `checkpoints/router_v6_xl512/router_step_*.pt` | 10 | 0.92 GB | 同上 |
| `checkpoints/router_replay_v7_v2_128/router_step_*.pt` | 1 | 0.02 GB | 与最终版权重相同 |
| `checkpoints/router_zov_v2_128/router_step_*.pt` | 1 | 0.02 GB | 与最终版权重相同 |
| `checkpoints/router_xl_1024/` | 9 | 1.11 GB | 1024 维实验（已放弃）整目录 |
| `checkpoints/router_xl_512/` | 28 | 2.09 GB | v3 旧数据运行（已被 v6 版取代）整目录 |
| `checkpoints/router_xl_smoke/` | 8 | 0.97 GB | 冒烟测试整目录 |
| `checkpoints/_smoke_stream/`、`_smoke_v2/`、`_smoke_v2b/`、`_smoke_xl_ckpt/`、`_smoke_feature_bank/`、`router_shared/` | 29 | 0.70 GB | 冒烟/临时运行整目录 |

释放 **7.36 GB**；fork 由 12.24 GB / 534 文件降至 **4.89 GB / 418 文件**。
上表 9 个整目录归档后已变为空目录，一并删除。

## 保留在 fork 里的（未归档，仍是证据与可运行产物）

| 目录 | 保留内容 |
| --- | --- |
| `checkpoints/router_v6_v2_128/` | `memory_router_v2.pt`（最终，drop-in 部署件）、`router_best.pt`、`router_arch.json`、`metrics.jsonl`、`router_v5_training.json`、`v6_final_resume_wrapper.pt`、`ranking_scrambled_probe.pt` |
| `checkpoints/router_v6_v2_512/` | 最终 `memory_router_v2.pt` + `router_best.pt` + 日志 |
| `checkpoints/router_v6_xl128/` | 最终 `memory_router_xl.pt` + `router_best.pt` + 日志 |
| `checkpoints/router_v6_xl512/` | 最终 `memory_router_xl.pt` + `router_best.pt` + 日志 |
| `checkpoints/router_replay_v7_v2_128/` | `memory_router_v2.pt`（**推荐交付物**：22/22 轴超越 + 零重叠 +41.20pp） |
| `checkpoints/router_zov_v2_128/` | 朴素微调对照件（用于证明灾难性遗忘） |

## 恢复方法

```powershell
# 恢复某个中间快照目录（示例）
& 'C:\Program Files\7-Zip\7z.exe' x 'E:\归档\03_代码与项目\V2_dpskw-备份\V2_dpskw_中间与废弃checkpoint_20260912_132740.7z' -o'H:\Memory\V2_dpskw\_restored' 'checkpoints/router_xl_512/*'

# 恢复整个 fork 快照（注意：会覆盖同名文件，先解到空目录）
& 'C:\Program Files\7-Zip\7z.exe' x 'E:\归档\03_代码与项目\V2_dpskw-备份\V2_dpskw_20260912_085921.7z' -o'E:\_v2_restore'
```

⚠️ 整包备份里的 `V2_dpskw\qwen3_5_4b_natural_memory_v2` 是个**空目录占位**（原为指向
`H:\Memory\dynamic_memory_lab\qwen3_5_4b_natural_memory_v2` 的 junction，用 `-snl` 打包时未跟随，
以免把整个 4B 模型塞进包里）。解压后需要重建：

```powershell
cmd /c mklink /J "H:\Memory\V2_dpskw\qwen3_5_4b_natural_memory_v2" "H:\Memory\dynamic_memory_lab\qwen3_5_4b_natural_memory_v2"
```
