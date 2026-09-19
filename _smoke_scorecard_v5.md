| metric | V2-128 deployed |
|---|---|
| 参数 | 2,037,774 |
| router_dim | 128 |
| 每条记录地址字节 | 512 |
| **Top-1 正确率** | 37.27% |
| Recall@1 | 37.27% |
| Recall@3 | 57.45% |
| Recall@5 | 74.06% |
| MRR | 49.84% |
| nDCG@3 | 46.10% |
| 多跳证据全中(Top-3) | 51.60% |
| hop 正确率 | 73.23% |
| hop 欠预测率 | 12.99% |
| need F1 | 99.68% |
| need 召回 | 99.85% |
| **未知拒答率** | 0.00% |
| **已知问题被误拒率** | 0.15% |
| 未知问题被误读率 | 100.00% |
| 仲裁准确率 | 99.37% |
| 已知问题 need 概率均值 | 98.04% |
| 未知问题 need 概率均值 | 99.99% |
| 平均分数余量 | -10.4388 |
| 单查询延迟 ms(GPU) | - |
| 路由 QPS(GPU) | - |

### family = `benchmark_qa`

| metric | V2-128 deployed |
|---|---|
| episodes | 64 |
| Top-1 正确率 | 0.00% |
| Recall@3 | 0.00% |
| MRR | 0.00% |
| hop 正确率 | 0.00% |
| 未知拒答率 | 0.00% |
| 已知问题被误拒率 | 0.00% |
| 未知问题被误读率 | 100.00% |
| 平均分数余量 | 0.0000 |

### family = `mega_validation`

| metric | V2-128 deployed |
|---|---|
| episodes | 20000 |
| Top-1 正确率 | 39.42% |
| Recall@3 | 60.39% |
| MRR | 51.89% |
| hop 正确率 | 74.95% |
| 未知拒答率 | 0.00% |
| 已知问题被误拒率 | 0.00% |
| 未知问题被误读率 | 0.00% |
| 平均分数余量 | -9.5746 |

### family = `memory_policy`

| metric | V2-128 deployed |
|---|---|
| episodes | 1728 |
| Top-1 正确率 | 14.41% |
| Recall@3 | 26.36% |
| MRR | 28.14% |
| hop 正确率 | 57.41% |
| 未知拒答率 | 0.00% |
| 已知问题被误拒率 | 1.87% |
| 未知问题被误读率 | 100.00% |
| 平均分数余量 | -19.3981 |

### family = `native_memory`

| metric | V2-128 deployed |
|---|---|
| episodes | 128 |
| Top-1 正确率 | 0.00% |
| Recall@3 | 4.67% |
| MRR | 12.99% |
| hop 正确率 | 53.91% |
| 未知拒答率 | 0.00% |
| 已知问题被误拒率 | 0.00% |
| 未知问题被误读率 | 100.00% |
| 平均分数余量 | -29.0393 |

### 读取/拒答门槛曲线

| run | threshold | need F1 | need 召回 | 未知拒答率 | 已知问题被误拒率 | 未知被误读率 | tp/tn/fp/fn |
|---|---|---|---|---|---|---|---|
| V2-128 deployed | 0.30 | 99.76% | 100.00% | 0.00% | 0.00% | 100.00% | 21814/0/106/0 |
| V2-128 deployed | 0.40 | 99.76% | 100.00% | 0.00% | 0.00% | 100.00% | 21814/0/106/0 |
| V2-128 deployed | 0.50 | 99.68% | 99.85% | 0.00% | 0.15% | 100.00% | 21782/0/106/32 |
| V2-128 deployed | 0.60 | 99.45% | 99.39% | 0.00% | 0.61% | 100.00% | 21682/0/106/132 |
| V2-128 deployed | 0.70 | 99.37% | 99.23% | 0.00% | 0.77% | 100.00% | 21645/0/106/169 |
| V2-128 deployed | 0.80 | 99.22% | 98.92% | 0.00% | 1.08% | 100.00% | 21579/0/106/235 |

### 按 mega 类别拆解（每类 2,000 条）

| 类别 | episodes | metric | V2-128 deployed |
|---|---|---|---|
| (未分类) | - | Top-1 正确率 | - |
| (未分类) | - | Recall@3 | - |
| (未分类) | - | MRR | - |
| (未分类) | - | 多跳证据全中 | - |
| (未分类) | - | hop 正确率 | - |
| (未分类) | - | 未知拒答率 | - |
| (未分类) | - | 已知问题被误拒率 | - |
| (未分类) | - | 未知问题被误读率 | - |
| conflict_update | 2000 | Top-1 正确率 | 82.85% |
| conflict_update | 2000 | Recall@3 | 96.80% |
| conflict_update | 2000 | MRR | 90.01% |
| conflict_update | 2000 | 多跳证据全中 | 96.80% |
| conflict_update | 2000 | hop 正确率 | 49.55% |
| conflict_update | 2000 | 未知拒答率 | 0.00% |
| conflict_update | 2000 | 已知问题被误拒率 | 0.00% |
| conflict_update | 2000 | 未知问题被误读率 | 0.00% |
| distractor_128 | 2000 | Top-1 正确率 | 24.55% |
| distractor_128 | 2000 | Recall@3 | 37.50% |
| distractor_128 | 2000 | MRR | 36.13% |
| distractor_128 | 2000 | 多跳证据全中 | 37.50% |
| distractor_128 | 2000 | hop 正确率 | 100.00% |
| distractor_128 | 2000 | 未知拒答率 | 0.00% |
| distractor_128 | 2000 | 已知问题被误拒率 | 0.00% |
| distractor_128 | 2000 | 未知问题被误读率 | 0.00% |
| distractor_32 | 2000 | Top-1 正确率 | 23.80% |
| distractor_32 | 2000 | Recall@3 | 35.75% |
| distractor_32 | 2000 | MRR | 35.34% |
| distractor_32 | 2000 | 多跳证据全中 | 35.75% |
| distractor_32 | 2000 | hop 正确率 | 100.00% |
| distractor_32 | 2000 | 未知拒答率 | 0.00% |
| distractor_32 | 2000 | 已知问题被误拒率 | 0.00% |
| distractor_32 | 2000 | 未知问题被误读率 | 0.00% |
| forget_correction | 2000 | Top-1 正确率 | 41.60% |
| forget_correction | 2000 | Recall@3 | 107.45% |
| forget_correction | 2000 | MRR | 57.71% |
| forget_correction | 2000 | 多跳证据全中 | 43.60% |
| forget_correction | 2000 | hop 正确率 | 0.00% |
| forget_correction | 2000 | 未知拒答率 | 0.00% |
| forget_correction | 2000 | 已知问题被误拒率 | 0.00% |
| forget_correction | 2000 | 未知问题被误读率 | 0.00% |
| long_context | 2000 | Top-1 正确率 | 23.45% |
| long_context | 2000 | Recall@3 | 28.65% |
| long_context | 2000 | MRR | 29.51% |
| long_context | 2000 | 多跳证据全中 | 28.65% |
| long_context | 2000 | hop 正确率 | 100.00% |
| long_context | 2000 | 未知拒答率 | 0.00% |
| long_context | 2000 | 已知问题被误拒率 | 0.00% |
| long_context | 2000 | 未知问题被误读率 | 0.00% |
| multi_hop | 2000 | Top-1 正确率 | 71.50% |
| multi_hop | 2000 | Recall@3 | 98.05% |
| multi_hop | 2000 | MRR | 84.46% |
| multi_hop | 2000 | 多跳证据全中 | 98.05% |
| multi_hop | 2000 | hop 正确率 | 0.00% |
| multi_hop | 2000 | 未知拒答率 | 0.00% |
| multi_hop | 2000 | 已知问题被误拒率 | 0.00% |
| multi_hop | 2000 | 未知问题被误读率 | 0.00% |
| paraphrase | 2000 | Top-1 正确率 | 42.55% |
| paraphrase | 2000 | Recall@3 | 60.50% |
| paraphrase | 2000 | MRR | 55.07% |
| paraphrase | 2000 | 多跳证据全中 | 60.50% |
| paraphrase | 2000 | hop 正确率 | 100.00% |
| paraphrase | 2000 | 未知拒答率 | 0.00% |
| paraphrase | 2000 | 已知问题被误拒率 | 0.00% |
| paraphrase | 2000 | 未知问题被误读率 | 0.00% |
| random_position | 2000 | Top-1 正确率 | 27.55% |
| random_position | 2000 | Recall@3 | 47.40% |
| random_position | 2000 | MRR | 45.13% |
| random_position | 2000 | 多跳证据全中 | 47.40% |
| random_position | 2000 | hop 正确率 | 100.00% |
| random_position | 2000 | 未知拒答率 | 0.00% |
| random_position | 2000 | 已知问题被误拒率 | 0.00% |
| random_position | 2000 | 未知问题被误读率 | 0.00% |
| single_fact | 2000 | Top-1 正确率 | 35.85% |
| single_fact | 2000 | Recall@3 | 53.15% |
| single_fact | 2000 | MRR | 50.41% |
| single_fact | 2000 | 多跳证据全中 | 53.15% |
| single_fact | 2000 | hop 正确率 | 100.00% |
| single_fact | 2000 | 未知拒答率 | 0.00% |
| single_fact | 2000 | 已知问题被误拒率 | 0.00% |
| single_fact | 2000 | 未知问题被误读率 | 0.00% |
| unknown | 1920 | Top-1 正确率 | 13.56% |
| unknown | 1920 | Recall@3 | 25.08% |
| unknown | 1920 | MRR | 27.25% |
| unknown | 1920 | 多跳证据全中 | 25.08% |
| unknown | 1920 | hop 正确率 | 55.26% |
| unknown | 1920 | 未知拒答率 | 0.00% |
| unknown | 1920 | 已知问题被误拒率 | 1.76% |
| unknown | 1920 | 未知问题被误读率 | 100.00% |
| unknown_abstention | 2000 | Top-1 正确率 | 20.55% |
| unknown_abstention | 2000 | Recall@3 | 38.65% |
| unknown_abstention | 2000 | MRR | 35.13% |
| unknown_abstention | 2000 | 多跳证据全中 | 38.65% |
| unknown_abstention | 2000 | hop 正确率 | 100.00% |
| unknown_abstention | 2000 | 未知拒答率 | 0.00% |
| unknown_abstention | 2000 | 已知问题被误拒率 | 0.00% |
| unknown_abstention | 2000 | 未知问题被误读率 | 0.00% |
