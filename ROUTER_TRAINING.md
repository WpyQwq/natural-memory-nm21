# Natural Memory v2：正式路由器训练

这一阶段只训练 `MemoryRouterV2`，不更新 Qwen 主干。训练目标是让路由器在真实自然语言查询下学会：

- 从候选记忆中找出一个或多个证据；
- 遇到多跳问题时保留多个支持项；
- 碰到未知事实时拒绝读取；
- 区分同属性冲突、旧值、新值和无关噪声；
- 在个人事实、对话记忆、代码/文档和通用 QA 之间泛化。

训练分为两个文件级阶段：

```text
公开/本地数据
       ↓
prepare_memory_router_dataset.py
       ↓ 先写出并 hash
train.jsonl + eval.jsonl
       ↓
train_memory_router_large.py
       ↓ 一次性生成 Qwen 冻结特征
CPU feature bank + router training
```

## 当前环境

已检查 `Conda LLM`：

```text
Python:    C:\Users\Administrator\miniconda3\envs\LLM\python.exe
PyTorch:   2.9.0+cu128
CUDA:      available
GPU:       NVIDIA GeForce RTX 5070
Transformers: 5.9.0
Datasets:  4.8.3
```

默认使用 `qwen3_5_4b_natural_memory_v2`，Qwen 只在特征预计算阶段工作；训练更新的只有一个小型 `MemoryRouterV2`。默认 GPU 上限是 9 GiB，为模型、系统和 KV 留出安全空间。

## 1. 生成并冻结数据

从仓库已有的训练/评估文件生成正式路由 episode：

```powershell
Set-Location H:\Memory
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe -m V2_dpskw.prepare_memory_router_dataset `
  --output-dir data/router_training `
  --candidate-count 32 `
  --seed 20260907
```

输出：

```text
H:\Memory\V2_dpskw\data\router_training\train.jsonl
H:\Memory\V2_dpskw\data\router_training\eval.jsonl
H:\Memory\V2_dpskw\data\router_training\manifest.json
H:\Memory\V2_dpskw\data\router_training\eval.sha256
```

`eval.jsonl` 会在训练开始前生成，训练器启动时重新计算 SHA-256；如果被修改，可以通过 `--expected-eval-sha256` 让训练直接停止。

当前默认本地混合源包括：

| 领域 | 来源 | 用途 |
|---|---|---|
| 个人事实 | `benchmark_train/eval.jsonl` | 主题改写、实体区分、短事实召回 |
| 原生记忆 | `native_memory/train/eval.jsonl` | 写入—替换—读取链路 |
| 记忆策略 | `production_memory/train/eval.jsonl` | 写入、遗忘、临时信息和噪声 |
| 困难记忆策略 | `production_memory_hard_v2/train/eval.jsonl` | 冲突、引用噪声、旧值、新值 |
| 长上下文压力 | `mega_validation/smoke.jsonl` | 多跳、随机位置、未知拒答等 smoke eval |

## 2. 引入公开数据

脚本内置了可审计的公开数据配方：

- [HotpotQA](https://huggingface.co/datasets/hotpot_qa)：多跳证据选择；
- [CodeSearchNet](https://huggingface.co/datasets/code_search_net)：自然语言到代码/文档检索；
- [FEVER](https://huggingface.co/datasets/fever)：支持证据、冲突和无证据样本。

网络可用时执行：

```powershell
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe -m V2_dpskw.prepare_memory_router_dataset `
  --output-dir V2_dpskw/data/router_training_public `
  --include-public `
  --hf-max-rows 20000 `
  --candidate-count 32 `
  --seed 20260907
```

如果 Hugging Face 下载失败，`manifest.json` 会写入失败原因，失败源贡献 0 行；不能把失败的在线源计入实验结果。也可以手动指定来源：

```text
--hf-source DATASET_ID|SPLIT|CONFIG(optional)|SPLIT_KIND(optional)
```

例如：

```powershell
--hf-source hotpot_qa|train|distractor|train
--hf-source hotpot_qa|validation|distractor|eval
```

真实用户对话必须先获得同意并去除姓名、地址、密钥、账号等 PII，再通过 `--train-source`/`--eval-source` 加入。脚本不会把用户对话偷偷上传。

## 3. 监督训练

第一次运行会加载 4B 模型，给去重后的 query/candidate 文本生成冻结语义特征，保存到 CPU feature bank；之后重新训练会复用它：

```powershell
Set-Location H:\Memory
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe -m V2_dpskw.train_memory_router_large `
  --train-file V2_dpskw/data/router_training/train.jsonl `
  --eval-file V2_dpskw/data/router_training/eval.jsonl `
  --model-path V2_dpskw/qwen3_5_4b_natural_memory_v2 `
  --output-dir V2_dpskw/checkpoints/natural_memory_v2_router_large `
  --feature-cache-dir V2_dpskw/checkpoints/natural_memory_v2_router_large/feature_cache `
  --gpu-memory-gb 9 `
  --encode-batch-size 1 `
  --steps 10000 `
  --batch-size 64 `
  --eval-interval 500 `
  --overwrite-metrics
```

控制台和 `metrics.jsonl` 都会记录：

- 每一个 optimizer step 的 `loss`、四个子损失、学习率和梯度范数；
- step 0 初始 eval；
- step 500、1000、1500……的完整 eval；
- 最终 `memory_router_v2.pt`，可直接交给现有的 `build_natural_memory_v2_package.py`。

训练不会输出每 token 的 Qwen 生成，因而不会把推理生成路径混入路由器质量指标。

如果中途停止，使用保存的 checkpoint 恢复；`--steps` 是恢复后的最终全局 step，不是额外步数：

```powershell
& C:\Users\Administrator\miniconda3\envs\LLM\python.exe -m V2_dpskw.train_memory_router_large `
  --train-file V2_dpskw/data/router_training/train.jsonl `
  --eval-file V2_dpskw/data/router_training/eval.jsonl `
  --model-path V2_dpskw/qwen3_5_4b_natural_memory_v2 `
  --output-dir V2_dpskw/checkpoints/natural_memory_v2_router_large `
  --feature-cache-dir V2_dpskw/checkpoints/natural_memory_v2_router_large/feature_cache `
  --resume V2_dpskw/checkpoints/natural_memory_v2_router_large/router_step_00005000.pt `
  --steps 10000 `
  --eval-interval 500
```

恢复时不要使用 `--overwrite-metrics`，这样历史 loss/eval 会继续追加到同一个 `metrics.jsonl`。

## 4. 评估门槛

训练期间重点看 `eval` 行：

| 指标 | 含义 |
|---|---|
| `route_top1` | 有证据时，Top-1 是否命中支持记忆 |
| `route_recall_at3` | 多跳/多事实时，Top-3 是否包含支持记忆 |
| `route_mrr` | 支持记忆在排序中的平均倒数排名 |
| `need_recall` | 需要记忆时是否愿意读取 |
| `need_specificity` | 不需要/未知时是否能拒绝读取 |
| `need_f1` | 读取和拒绝的综合平衡 |
| `hop_accuracy` | 多跳控制预测是否正确 |

路由器指标不能代替端到端回答率。训练完成后必须重新运行现有的 dirty-corpus、strong-RAG 和 Qwen3.5 4B 对照；最终门槛仍然是端到端正确率、未知拒答、延迟和显存共同达标。

## 5. 重要边界

当前公开数据适配器是通用 schema 适配器，不会把所有数据集都自动变成完美标注。每个公开源的实际行数、错误信息和哈希都在 `manifest.json`。如果某个源的支持证据字段无法解析，它会少贡献 episode，而不是制造伪标签。

路由器训练也不会直接解决“召回正确但 Qwen 没有把证据融合进答案”的全部问题；那是后续证据融合/生成控制实验，必须单独报告。
