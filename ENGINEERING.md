# Dynamic Memory Lab

一个在现成 Qwen3.5-4B 语言模型上进行“架构手术”的实验工程。

> Natural Memory v2 已完成分页地址路由、粗索引→候选页→精排→Top-K、多跳、冲突版本、quarantine、撤回、嵌入式重启恢复和完整回归评测。请先阅读 [Natural Memory v2 工程说明](README_NATURAL_MEMORY_V2.md) 与 [技术论文](Natural_Memory_v2_Paper.md)。

本项目不重新训练整个语言模型，而是在 Transformer 的若干层之间插入一个可读、可写、可持续更新的动态记忆模块，研究下面这个问题：

> 能不能让一个已经训练好的 LLM，在不把所有历史对话反复塞回上下文的情况下，持续积累和使用信息？

答案是：工程中已经实现了一个生产导向的自然语言记忆核心，并完成了多事实写入、冲突更新、无历史跨重启读取、二次重启读取、未知事实拒答和 reset token 验收。它仍需要在真实业务数据上继续做压力测试，合成数据指标不能直接等同于所有场景的生产承诺。

## 结论先说

当前模型已经具备：

- 在一次进程运行期间维护一组独立于 token 上下文的 memory state；
- 对 memory 做 attention 读取；
- 根据当前输入生成写入内容，并更新 memory；
- 将 memory 读取结果注入 Qwen3.5-4B 的中间层；
- 只训练记忆模块和少量门控参数，而冻结原始 Qwen 主干；
- 对比原版 Qwen3.5-4B 与动态记忆版的 loss、困惑度、准确率、生成质量、速度和显存。

当前版本仍有边界：

- 还没有证明对任意自然语言、多事实和长时间运行都可靠；
- 还没有完成生产级多用户隔离、隐私策略和容量管理；
- 自由生成在合成留出集上仍会偶尔答错值，所以不能把实验指标当作生产承诺。

因此，当前版本更准确的定位是：

> 一个包含可部署自然语言记忆核心、持久化 checkpoint 和验收基准的 LLM 原生动态记忆工程；生产接入前仍需补齐业务侧鉴权、加密、并发和运维策略。

## 项目结构

    V2_dpskw/
    ├─ qwen_integration.py       # 动态记忆模块和 Qwen 模型适配器
    ├─ train_qwen_memory.py      # 记忆模块训练脚本
    ├─ chat_qwen_memory.py       # 多轮对话实验脚本
    ├─ benchmark_qwen.py         # 原版/动态版跑分脚本
    ├─ verify_persistent_memory.py # 重启后仅用 memory_state 验证回答
    ├─ make_native_memory_data.py # 生成写入/遗忘/未知/冲突数据
    ├─ train_native_memory.py    # 训练原生写入、遗忘和读出
    ├─ evaluate_native_memory.py # 留出集控制器与自由生成评估
    ├─ verify_native_checkpoint.py # checkpoint 重启和 reset token 验证
    ├─ train_natural_retriever.py # 训练自然语言记忆查询检索器
    ├─ train_auto_memory_policy.py # 训练自动记忆重要性策略
    ├─ build_production_memory_dataset.py # 真实对话导入、规范化和防泄漏切分
    ├─ train_production_memory_policy.py # 从规范化对话训练策略候选
    ├─ stress_test_natural_memory.py # 低显存连续运行/长上下文压力测试
    ├─ natural_memory_service.py # localhost 管理与聊天 API
    ├─ natural_memory_app.py    # 统一 chat/serve/train/stress 入口
    ├─ benchmark_natural_language_memory.py # 多事实/更新/重启/拒答验收
    ├─ stream_chat_qwen_memory.py # 流式聊天与随时重启测试入口
    ├─ make_benchmark_data.py    # 生成简单的记忆型 benchmark 数据
    ├─ requirements.txt          # Python 依赖
    ├─ memory.pt                 # 训练后的记忆模块参数，若已生成
    ├─ memory_config.json        # 记忆模块配置，若已生成
    ├─ surgery.pt                # 层手术相关参数，若已生成
    └─ README.md

## 运行环境

推荐使用已有的 Conda 环境 LLM。

    conda activate LLM
    cd W:\Flash\model\V2_dpskw
    pip install -r requirements.txt

如果模型路径不在默认位置，可以通过参数指定。当前工程主要面向本地 Hugging Face 格式的 Qwen3.5-4B 模型。

## 总体架构

原版 Qwen 的推理过程大致是：

    输入 token
       │
       ▼
    Embedding
       │
       ▼
    Transformer Layer 0
       │
       ▼
    Transformer Layer 1
       │
      ...
       │
       ▼
    Transformer Layer N
       │
       ▼
    LM Head
       │
       ▼
    下一个 token

动态记忆版会在若干个 Transformer 层上插入 Memory Adapter：

    输入 token
       │
       ▼
    Qwen Transformer Layer
       │
       ├──────────────► Memory Read
       │                    ▲
       │                    │
       │              Runtime Memory State
       │              M = [B, S, D]
       │                    │
       │                    ▼
       ├──────────────► Memory Delta
       │
       ▼
    Memory Layer Adapter
       │
       ▼
    后续 Qwen Transformer Layers
       │
       ├──────────────► Memory Write
       │                    │
       │                    ▼
       │              M_new = Update(M, input)
       │
       ▼
    LM Head

其中：

- B 是 batch size；
- S 是 memory slot 数量，默认 16；
- D 是 memory embedding 维度，默认 512；
- M 不是 token 序列，而是模型外部维护的一组连续向量；
- memory_state 可以在不同调用之间传递，因此它能够脱离上一轮的文本上下文。

## 核心概念：参数、上下文和运行时记忆

理解本项目最重要的是区分三种东西。

### 1. 模型参数

模型参数是 Qwen 的权重以及动态记忆模块的权重，例如：

    Wq, Wk, Wv, Wread, Wwrite

它们决定模型“如何读写记忆”，通常在训练阶段更新，在推理阶段保持不变。

### 2. 当前上下文

当前上下文是这次请求中送入模型的 token，例如：

    用户：我叫小明。
    助手：你好，小明。

上下文是临时的。上下文窗口结束以后，模型本身不会自动保存这些 token。

### 3. 运行时记忆状态

运行时记忆状态是：

    M = [batch_size, memory_slots, memory_dim]

默认情况下：

    M = [B, 16, 512]

它是模型运行时的一块连续状态。只要下一次调用仍然传入同一个 memory_state，模型就能继续使用之前写入的内容。

注意：

> memory.pt 保存的是“记忆模块的训练参数”，不是某个用户的聊天记忆。

用户记忆应该单独保存为某种 runtime state，例如：

    user_001_memory.pt
    user_002_memory.pt

或者保存到数据库、对象存储、向量数据库中。

## Memory Read：模型如何读取记忆

设某一层产生的隐藏状态为：

    h ∈ R^D_hidden

记忆矩阵为：

    M = [m₁, m₂, ..., mₛ] ∈ R^(S×D_memory)

首先把当前隐藏状态投影成 query：

    q = W_q h

把每个 memory slot 投影成 key 和 value：

    k_i = W_k m_i
    v_i = W_v m_i

然后计算当前输入与每个 slot 的匹配程度：

    score_i = q · k_i / sqrt(D_memory)

经过 softmax 得到读取权重：

    α_i = softmax(score_i)

最后将各个 slot 的 value 加权求和：

    r = Σ_i α_i v_i

r 就是当前输入从动态记忆中检索出来的内容。

为了避免记忆模块一开始就破坏 Qwen，代码还使用了一个 read gate：

    g = sigmoid(W_gate h)

最终的记忆增量大致为：

    Δh = read_scale × g × W_read(r)

然后再注入当前层：

    h_new = h + Δh

这和给 Transformer 增加一个小型、可训练的外部知识通道类似。

## Memory Write：模型如何写入记忆

读取解决的是“从记忆里找什么”，写入解决的是“把当前输入存什么”。

当前实现默认使用输入序列最后一个 token 的隐藏状态作为摘要：

    s = h_last

也支持固定 token 偏移位置作为写入摘要。

接着通过写入投影生成候选内容：

    p = W_write(s)

再根据当前输入生成写入地址和写入强度：

    a = softmax(W_addr(s))
    z = sigmoid(W_strength(s))

其中：

- a_i 表示第 i 个 slot 被写入的比例；
- z 表示本次写入总体有多强；
- p 是候选写入向量。

对于每个 slot，更新形式近似为：

    m_i_new = (1 - z × a_i) × m_i
              + (z × a_i) × p

这是一种可微分的软写入。它不会使用不可导的“直接选中某个 slot”操作，所以可以通过反向传播学习：

- 什么输入值得写入；
- 应该写入哪些 slot；
- 写入幅度应该多大；
- 如何从隐藏状态中压缩信息。

代码还支持 broadcast_write，让候选内容广播写入所有 slot。这个模式更适合做架构实验，但可能降低 slot 的分工能力。

## Qwen 接入方式

qwen_integration.py 会加载原版 Qwen，并替换指定层为带记忆能力的适配层。

默认会选择若干中间层；也可以通过 layer_indices 手动指定层号。工程中同时保留了一个轻量的自定义线性注意力记忆层，便于做对照实验。

动态模型默认冻结 Qwen 主干：

    Qwen 原始参数：冻结
    Memory Read/Write：训练
    层融合 gate：训练

这样做的好处是：

- 显存和训练成本更低；
- 不容易破坏原模型能力；
- 更容易判断提升来自记忆机制还是来自主干重新学习；
- 适合在单卡环境中快速迭代架构。

## 三种层融合模式

动态记忆读出后，需要决定如何注入 Qwen 的隐藏状态。当前支持三种模式。

### residual

    h_new = h + memory_delta

这是默认模式。它保留原始隐藏状态，并把记忆当作额外残差通道。

特点：

- 最稳定；
- 对原模型干扰小；
- 适合第一版训练和 benchmark。

### blend

    h_new = h_residual + (1 - α) × original_token_mixer + α × memory_delta

其中 α 是可训练的融合系数。

特点：

- 模型可以学习记忆通道应该占多大比例；
- 适合研究“原始表示”和“记忆表示”的权衡；
- 如果初始化或训练不稳定，可能导致原模型信息被过早削弱。

### replace

    h_new = memory_delta

完全使用记忆分支输出。

特点：

- 适合验证记忆分支的独立表达能力；
- 风险最高；
- 通常不建议作为默认生产方案。

## direct_logit_scale：直接影响输出概率

除了修改 Transformer 中间层，代码还支持把最后一次 memory readout 经过投影后直接加到 logits：

    logits_new = logits_qwen + scale × projection(memory_readout)

这个选项可以直接研究：

- memory 是否能记住某些目标答案；
- memory 是否能把目标 token 的概率推高；
- 记忆模块对最终预测的直接影响。

但它也更容易过拟合简单 benchmark，因此应该同时观察泛化测试和正常生成质量。

## raw token pointer：精确 token 记忆实验

项目还提供一个有意“开后门”的架构消融：memory pass 不仅写入连续向量，还可以保存某个 token 在 Qwen 输出投影矩阵中的行；查询生成的第一个 token 会读取这行向量。

这个实验用于回答一个非常具体的问题：

> 如果记忆里已经存在目标 token，当前 Qwen 接口能不能把它可靠地送进自由生成？

示例配置：

    --write-token-offset 4 --broadcast-write \
    --raw-token-write --raw-logit-scale 30

注意：`write-token-offset` 按“从有效序列末尾数起”计算，`1` 是最后一个 token。当前 benchmark 的答案位于 `H / 。 / <|im_end|> / 换行` 中的倒数第 4 个位置，所以使用 `4`。`raw-token-write` 是 pointer ablation，不是通用自然语言记忆方案；它直接保存 token id 对应的输出投影行，不能把它的 100% 结果等同于普通 learned memory 的能力。

raw pointer 只作用于第一个生成 token，后续 token 回到 Qwen 原本的生成分布，避免把同一个答案 token 重复写满整段输出。

## 训练原理

当前训练脚本采用“两阶段记忆训练”。

### 阶段一：写入阶段

输入一段 memory text：

    Memory: user=alice; favorite_color=blue

此阶段关闭 memory read，打开 memory write：

    outputs = model(
        memory_text,
        memory_state=memory_state,
        read_memory=False,
        update_memory=True,
        return_memory=True,
    )
    memory_state = outputs.memory_state

目标是让模型把关键信息写入 memory state。

### 阶段二：查询阶段

再输入查询：

    Question: What is alice's favorite color?

此阶段使用刚刚更新后的 memory state，并开启 memory read：

    outputs = model(
        query_text,
        memory_state=memory_state,
        read_memory=True,
        update_memory=False,
        labels=labels,
    )

通过语言模型 loss 训练记忆模块，让查询阶段能够根据 memory state 输出正确答案。

训练时通常只更新：

    Memory Read parameters
    Memory Write parameters
    Memory Layer Adapter parameters
    Blend/Gate parameters

而 Qwen 主干保持冻结。

## 数据格式

训练和 benchmark 数据使用 JSONL。当前脚本要求每行包含 `memory` 和 `query` 两个消息列表；最后一个 assistant 消息是监督目标：

    {"memory":[{"role":"user","content":"记住对象 A 的代号是 H。"},{"role":"assistant","content":"好的，已记住。"}],"query":[{"role":"user","content":"对象 A 的代号是什么？"},{"role":"assistant","content":"H"}]}

额外的 `id`、`subject`、`attribute`、`answer` 字段只用于 benchmark 统计。训练和评估数据应避免把答案直接重复到 query 的 user 内容中。

数据设计时要注意：

1. 写入文本中出现的信息，应该在查询文本中尽量不重复；
2. 如果查询中直接包含答案，模型可能只是在复制上下文，而不是读取 memory；
3. 训练集和评估集中的实体、属性、表述方式应尽量分离；
4. 要加入冲突样本，测试新记忆是否能覆盖旧记忆；
5. 要加入多条事实，测试不同 slot 是否会互相污染；
6. 要加入无关信息，测试模型能否避免把所有内容都写进去。

make_benchmark_data.py 可以生成一个简单的单字符映射任务，用来快速检查“写入—读取—回答”链路是否工作。它适合做冒烟测试，不足以证明模型拥有通用长期记忆。

## 常用命令

以下命令均在工程目录执行。

### 训练动态记忆模块

    conda activate LLM
    cd W:\Flash\model
    python -m V2_dpskw.train_qwen_memory --model-path "W:\Flash\model" --data V2_dpskw\data\benchmark_train.jsonl --output-dir V2_dpskw\qwen_memory_adapter --steps 100 --batch-size 1 --lr 1e-4 --max-length 128

如果要复现实验中的精确 token pointer：

    python -m V2_dpskw.train_qwen_memory --model-path "W:\Flash\model" --data V2_dpskw\data\benchmark_train.jsonl --output-dir V2_dpskw\qwen_memory_adapter_pointer --steps 100 --surgery-mode blend --blend-init 0.1 --write-token-offset 4 --broadcast-write --raw-token-write --raw-logit-scale 30

如果实际模型目录不同，请替换 --model-path。

### 训练原生记忆控制器

原生模式会训练：摘要池化、写入决策、slot 地址、候选值读出，以及依赖旧记忆的遗忘门。训练数据中显式包含闲聊噪声、未知查询和冲突覆盖样本：

    python -m V2_dpskw.make_native_memory_data --output-dir V2_dpskw\data\native_memory --train-count 512 --eval-count 128 --seed 20260904
    python -m V2_dpskw.train_native_memory --model-path "W:\Flash\model" --data V2_dpskw\data\native_memory\train.jsonl --output-dir V2_dpskw\qwen_memory_adapter_native_v3 --steps 2500 --lr 8e-5 --max-length 192 --save-every 250 --direct-logit-scale 12 --write-loss-weight 0.5 --forget-loss-weight 0.75 --forget-positive-weight 6 --value-loss-weight 1

评估：

    python -m V2_dpskw.evaluate_native_memory --model-path "W:\Flash\model" --adapter-dir V2_dpskw\qwen_memory_adapter_native_v3 --data V2_dpskw\data\native_memory\eval.jsonl --max-length 192 --max-new-tokens 8

### 生成 benchmark 数据

    python -m V2_dpskw.make_benchmark_data --output-dir V2_dpskw\data

### 对比原版和动态版

    python -m V2_dpskw.benchmark_qwen --model-path "W:\Flash\model" --data V2_dpskw\data\benchmark_eval.jsonl --adapter V2_dpskw\qwen_memory_adapter_pointer --output V2_dpskw\benchmark_qwen_pointer_eval.json --max-length 128 --max-new-tokens 4 --repeats 1 --warmup 0

benchmark 通常会报告：

- validation loss；
- perplexity；
- token accuracy；
- first target token accuracy；
- exact sequence accuracy；
- generation quality；
- tokens per second；
- 峰值显存。

最终报告必须同时关注效果和代价。一个模型如果只是在极小任务上准确率更高，却明显降低通用生成能力或推理速度，不能简单视为架构成功。

### 本机已验证结果

在 `data/benchmark_train.jsonl` 的 128 条训练样本和 `data/benchmark_eval.jsonl` 的 64 条 held-out 随机映射上，答案字符不是由实体名称推导出来的。当前已保存的结果文件是：

| 版本 | PPL | token accuracy | 首目标 token | 自由生成 prefix | 速度 |
|---|---:|---:|---:|---:|---:|
| 原版 Qwen3.5-4B | 210.20 | 42.71% | 0% | 0% | 19.45 tok/s |
| 普通 learned blend | 4.50 | 66.67% | 0% | 0% | 12.99 tok/s |
| blend + raw token pointer | 1.11 | 96.88% | 100% | 100% | 14.56 tok/s |
| native learned memory v3 | 1.24 | 91.67% | 75% | 75% | 13.33 tok/s |

对应文件分别是 `benchmark_qwen_native_v3_eval.json`、`benchmark_qwen_pointer_eval.json` 中的 baseline/dynamic 记录，以及 `benchmark_qwen_full_eval.json` 中保存的普通 blend 结果。native v3 行来自同一套 64 条 benchmark；pointer 行是精确 token 消融实验，不能替代通用 learned memory 的结论。不同运行的速度会受显存缓存和系统状态影响。

原生控制器 v3 在 128 条完全不同用户编号的留出集上得到：写入准确率 100%，遗忘准确率 99.58%，replacement 遗忘准确率 96.30%，查询 token 准确率 92.12%，自由生成事实召回 81.31%，未知查询安全拒答 100%。这些结果来自 `qwen_memory_adapter_native_v3/native_eval_report.json`；其中自由生成仍有少量错误值，不能宣称已经达到可靠生产级记忆。

### 多轮对话实验

    python -m V2_dpskw.chat_qwen_memory --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter

要让用户记忆跨进程保存，指定一个用户专属文件：

    python -m V2_dpskw.chat_qwen_memory --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter --memory-state V2_dpskw\data\users\alice.pt

首次运行可以输入：

    /remember 我叫小明，喜欢蓝色

退出后再次运行同一条命令，直接询问个人事实即可；启动时只会加载 `alice.pt`，不会自动加载上一轮聊天文本。`/reset` 会把该用户的 state 重置为零并保存，`/save` 可以手动保存。

需要区分“记忆持久化”和“记忆能力”：`qwen_memory_adapter_natural_auto_v3` 加上持久化 checkpoint 已验证自然语言写入、冲突覆盖、两次无历史重启读取、未知事实拒答和 reset；真实业务上线仍需要按业务数据继续扩充评测。

不启用 `--persistent-memory` 时，当前 chat 脚本中的 memory 默认是进程内状态；关闭脚本后，这块状态会消失。

使用原生模式并把用户记忆写进适配器 checkpoint：

    python -m V2_dpskw.chat_qwen_memory --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter_native_v3 --persistent-memory --persist-in-adapter

原生 adapter 默认使用 `<|fim_prefix|>` 作为 reset token；也可以通过 `--reset-token` 或 `--reset-token-id` 指定其他 tokenizer token。向模型发送该 token 会在模型内部清零 memory，不需要外部清理函数参与推理。

原生 chat 的普通用户消息会先经过自动记忆策略头判断是否值得长期保存；写入阶段不读取旧记忆，避免把回忆内容或问题句误写回去。随后生成阶段只读，不把模型自己的回答再次写回记忆。`/remember` 仍可用于强制写入。

## 当前保存机制的边界

当前代码已经提供真正的 runtime state 持久化接口：

    model.save_runtime_memory("user_memories/alice.pt")
    model.load_runtime_memory("user_memories/alice.pt")

保存文件包含：

- `memory_state`：连续动态记忆张量；
- `raw_memory`：如果启用了 token pointer，则保存对应的辅助状态；
- hidden size、memory shape、层配置等兼容性信息。

它不包含模型权重，也不包含历史聊天文本。因此重启后的调用可以只传入加载后的 state 和新的 query。

如果希望把“记忆模块参数 + 当前用户 memory state”放到同一个紧凑适配器包中：

    model.save_persistent_memory_checkpoint("V2_dpskw/qwen_memory_adapter_native_v3_persistent")

该包中的 `persistent_memory.pt` 会在 `load_memory_adapter()` 时自动加载，新的模型实例不需要再显式传入 `memory_state`。验证命令：

    python -m V2_dpskw.verify_native_checkpoint --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter_native_v3 --data V2_dpskw\data\native_memory\eval.jsonl --output-adapter V2_dpskw\qwen_memory_adapter_native_v3_persistent

可以用下面的脚本自动验证完整流程：

    python -m V2_dpskw.verify_persistent_memory --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter_pointer --data V2_dpskw\data\benchmark_eval.jsonl --memory-state V2_dpskw\data\user_memory_demo.pt

该脚本会执行：写入 memory → 保存 state → 销毁模型 → 重新加载模型 → 只用 state 生成回答。

项目会保存动态记忆模块的训练参数，例如：

    memory.pt
    memory_config.json
    surgery.pt

这些文件描述的是“模型如何使用记忆”；`persistent_memory.pt` 则是打包进适配器的“某个用户已经记住了什么”。

可以把它们类比成：

    memory.pt        = 记忆系统的大脑结构和读写规则
    user_memory.pt   = 某个用户实际写入的内容

当前实现的 runtime state 通常通过以下方式流动：

    outputs = model(
        input_ids=input_ids,
        memory_state=memory_state,
        return_memory=True,
    )
    memory_state = outputs.memory_state

只要把 memory_state 保存下来，之后重新加载并传回模型，就可以恢复对应记忆。

## 能不能实现跨对话、无上下文、原生记忆？

需要先把这个问题拆开。

### 现在能做到的版本：跨调用、无历史文本

可以。

同一个进程里：

    第 1 次调用：输入事实，更新 memory_state
    第 2 次调用：只输入问题，传入 memory_state
    第 3 次调用：继续传入更新后的 memory_state

第 2 次调用不必把第 1 次调用的完整聊天记录重新放进 prompt。模型可以从 memory state 中读取信息。

### 加一个持久化层后：跨程序、跨会话

也可以实现，但它不是模型单独完成的，而是：

    用户 ID
       │
       ▼
    加载该用户的 runtime memory state
       │
       ▼
    调用动态记忆模型
       │
       ▼
    保存更新后的 runtime memory state

最小实现可以是：

    from pathlib import Path
    import torch

    def load_user_memory(user_id, model):
        path = Path("user_memories") / f"{user_id}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu")
        return model.initial_memory(batch_size=1)

    def save_user_memory(user_id, memory_state):
        path = Path("user_memories") / f"{user_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(memory_state.detach().cpu(), path)

实际生产环境还需要：

- 用户隔离；
- 加密；
- 并发写入保护；
- 版本迁移；
- memory state 校验；
- 删除和导出接口；
- 过期时间或记忆衰减；
- 防止恶意 prompt 写入长期记忆。

### 严格意义上的“完全无上下文”不可能凭空发生

如果“无上下文”指：

    不提供历史文本
    不提供 memory state
    不提供数据库
    不提供任何外部信息

但又希望模型知道上一场对话发生了什么，那么这在信息上是不可能的。

信息必须存在某个地方：

    上下文窗口
    模型参数
    外部数据库
    向量索引
    神经 memory state

本项目选择的是“神经 memory state”这条路线。它可以让历史信息不以原始文本形式出现，但不能让信息在没有任何载体的情况下存在。

### 为什么仍然需要生产化治理

当前自然语言记忆核心已经不只是连续状态：它有精确文本槽、冻结 Qwen 检索键、训练过的查询检索器、写入门控、冲突替换、最老槽淘汰、checkpoint 校验和 reset。但上线前仍要针对真实业务补充：

- 用户身份隔离、加密、并发写入和版本迁移；
- 真实语言中的事实抽取、撤回/删除语义和多事实长时间压力测试；
- 记忆容量策略、审计日志、导出与合规删除；
- 不同语言、不同 tokenizer、不同 batch 和服务重启方式的回归测试。

更准确的说法是：

> 模型已经拥有内部的“是否写入、如何检索、何时拒绝读取、如何覆盖和淘汰”的自然语言记忆路径；用户 checkpoint 仍必须存在于某种持久化介质中，这是信息保存的必要条件，而不是外部代码替模型执行读取。

## 为什么不直接每轮修改 Qwen 主权重

每轮对话都直接更新主模型权重，理论上可以把信息写进参数，但会产生明显问题：

- 很容易灾难性遗忘；
- 不同用户之间会互相污染；
- 每次写入都需要保存或更新大模型权重；
- 难以撤销某条记忆；
- 难以处理隐私和权限；
- 推理延迟和存储成本都很高。

外部 runtime memory 的好处是：

    主模型参数 = 稳定的通用能力
    用户 memory = 可修改、可删除、可隔离的个体状态

这是更接近实际产品需求的拆分。

## 当前自然语言记忆核心

这是当前推荐的生产导向路径，不再把个人事实压缩成一个容易丢失多 token 值的连续向量：

    用户输入
       │
       ├─ Automatic memory policy：判断普通消息是否值得长期记忆
       ├─ Native write controller：提供写入表示和槽位地址
       ├─ Frozen Qwen key encoder：生成稳定检索键
       ├─ Learned retriever：查询与记忆槽匹配
       └─ Internal text bank：保存合法 memory prefix、原始 fact key、年龄和有效位
              │
              ▼
       相关记忆才被模型内部拼成 prefix
              │
              ▼
       原版 Qwen 生成回答

具体保证：

- 每个槽保存可直接参与 Qwen 对话的内部 memory prefix，并单独保存原始 fact key，因此“红富士苹果”“蓝鲸-47”等多 token 值不会被单个向量强行压缩；
- 普通消息由训练过的自动记忆策略决定是否写入；写入阶段关闭读取，生成阶段只读，避免把模型自己的回答或已召回事实再次写回记忆；
- 同一属性的更新由训练过的语义检索器确认，不同事实优先使用空槽，容量满时按最老槽淘汰；
- 读取由模型内部的 learned retriever 触发。无关问题低于阈值时，不注入记忆 prefix，降低个人事实幻觉；
- `persistent_memory.pt` 直接包含连续状态、文本槽、检索键、年龄和计数器。重启时只加载这个用户 checkpoint，不需要聊天历史，也不需要外部“记忆读取器”代码；
- `<|fim_prefix|>`（或显式配置的 reset token）由模型包装器在生成入口内识别并清空全部记忆槽。

训练检索器：

    conda activate LLM
    cd W:\Flash\model
    python -m V2_dpskw.train_natural_retriever --model-path "W:\Flash\model" --base-adapter V2_dpskw\qwen_memory_adapter_native_v3 --output-adapter V2_dpskw\qwen_memory_adapter_natural_controller_v13 --steps 3000 --pair-count 3200 --batch-size 32 --lr 2e-4

训练自动记忆策略头：

    python -m V2_dpskw.train_auto_memory_policy --model-path "W:\Flash\model" --base-adapter V2_dpskw\qwen_memory_adapter_natural_controller_v13 --output-adapter V2_dpskw\qwen_memory_adapter_natural_auto_v13 --steps 2600 --example-count 1600 --batch-size 32 --lr 2e-4 --threshold 0.35 --text-memory-threshold 0.30

启动自然语言记忆聊天：

    python -m V2_dpskw.chat_qwen_memory --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter_natural_auto_v13 --persistent-memory --persist-in-adapter

如果要测试流式输出和“随时重启”，使用：

    python -m V2_dpskw.stream_chat_qwen_memory --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter_natural_auto_v13 --memory-state V2_dpskw\data\users\stream_user_memory.pt

该脚本每轮只发送当前用户消息，不发送历史；写入或清空操作会在生成前原子保存。生成过程中按 `Ctrl+C` 退出后，再次执行同一命令即可从最近一次保存的 memory state 继续。普通消息会自动保存高价值个人事实，`/remember <事实>` 用于强制写入，`/reset` 清空全部记忆。

运行完整验收：

    python -m V2_dpskw.benchmark_natural_language_memory --model-path "W:\Flash\model" --adapter V2_dpskw\qwen_memory_adapter_natural_auto_v13 --output-adapter V2_dpskw\qwen_memory_adapter_natural_production_auto_v13 --report V2_dpskw\benchmark_natural_language_memory_auto_v13_final.json --max-new-tokens 48 --text-memory-threshold 0.30

当前验收报告为 `benchmark_natural_language_memory_auto_v13_final.json`：三条事实写入后保留两个独立有效槽，工作地点从 R7 更新为 K9；两条已知事实在两次模型重启后均能回答；未知血型不注入 memory prefix 并拒答；reset token 后连续状态和文本槽都清零。报告中的 `production_gate_pass` 为 `true`。

自动记忆策略的训练报告为 `qwen_memory_adapter_natural_auto_v13/auto_policy_training.json`：2600 步、每类 1600 条样本，评估准确率 99.22%，负样本特异度 100%，误写率 0%。这些是合成数据结果，仍需用真实用户语言继续扩充压力测试。

### 将记忆模块和当前用户状态合并进 safetensors

如果需要把 v13 的记忆模块参数、检索器、自动写入策略和某个用户的当前状态写进模型权重，可以生成一个新的合并目录。下面的命令是兼容的合并方式；推荐在新包上使用额外第三分片模式。

```powershell
python -m V2_dpskw.merge_memory_weights `
    --base-model "W:\Flash\model" `
    --adapter "W:\Flash\model\V2_dpskw\qwen_memory_adapter_natural_auto_v13" `
    --memory-state "W:\Flash\model\V2_dpskw\data\users\stream_user_memory.pt" `
    --output "W:\Flash\model\V2_dpskw\qwen3_5_4b_memory_merged_v13"
```

额外第三分片模式使用同样的参数，并增加 `--add-shard`；目标文件名应为 `model.safetensors-00003-of-00003.safetensors`：

```powershell
python -m V2_dpskw.merge_memory_weights `
    --base-model "<base-model>" `
    --adapter "<adapter>" `
    --memory-state "<memory-state>" `
    --output "<merged-package>" `
    --add-shard "<merged-package>\\model.safetensors-00003-of-00003.safetensors"
```

兼容旧合并方式生成的目录会把 `dynamic_memory.*` 张量放入第二个 safetensors 分片；原始 Qwen 两个分片不会被覆盖。自定义动态记忆模型可以直接从该目录加载：

```powershell
python -m V2_dpskw.stream_chat_qwen_memory `
    --model-path "W:\Flash\model\V2_dpskw\qwen3_5_4b_memory_merged_v13" `
    --memory-state "W:\Flash\model\V2_dpskw\data\users\merged_runtime.pt"
```

上面的旧模式仍可使用 `merged_runtime.pt` 保存后续变化；真正执行动态记忆仍需要本项目的模型架构代码。

当前推荐的第三分片启动方式是：

```powershell
python -m V2_dpskw.stream_chat_qwen_memory `
    --model-path "<merged-package>"
```

不传 `--adapter` 和 `--memory-state` 时，模型会从 `model.safetensors-00003-of-00003.safetensors` 读取内置 memory state；之后的自动写入、`/save` 和 `/reset` 也只原子重写第三分片，不再创建或依赖 `merged_runtime.pt`。旧的“把动态张量追加到第二分片”方式仍可通过不使用 `--add-shard` 保留，但每次保存会重写较大的第二分片。

检索器 v13 额外加入了短事实、无标点问法、未见过的值、登记/称呼表达、编程工具表达和“你是谁”硬负样本；训练与线上都使用 plain tokens，并按长度分组避免 Qwen 补零造成表示漂移。训练报告的标准留出、短事实留出和 raw sigmoid 严格阈值召回均为 100%；当前 Wpy 真实身份/拒读压力测试为 25/25。后两项是更接近线上行为的指标，仍应随真实用户分布持续回归。

通用能力回归报告为 `comprehensive_benchmark_natural_auto_v13_final.json`：54 个任务上原版 Qwen3.5-4B 与动态记忆版均为 96.30%，整体差值 0，回归门禁通过。

需要明确：模型内部已经拥有“何时读、读什么、如何拒绝无关记忆”的路径，但跨机器或跨服务保存用户 checkpoint 仍然需要持久化介质；这是信息存在的物理要求，不等于推理时依赖固定外部读取代码。

## 当前架构的局限

当前版本主要用于研究“记忆模块能否工作”，还不是最终架构。

### 记忆写入过于粗粒度

默认使用最后一个 token 的 hidden state 作为摘要。复杂文本可能包含多条事实，仅靠最后一个 token 很容易丢失信息。

后续可以加入：

- sentence-level summarizer；
- entity/value extractor；
- 多 token pooling；
- 特殊 memory token；
- 独立的写入路由器。

### slot 语义还不稳定

固定数量的连续 slot 不一定自动形成清晰分工。后续可以研究：

- slot type；
- key-value memory；
- 稀疏路由；
- memory usage regularization；
- slot 专家化；
- 多层级 memory。

### teacher forcing 与自由生成存在差距

在训练中，模型通常知道正确答案的标签；但在真正聊天时，需要连续生成多个 token。应该分别测试：

- 单 token 读取；
- 多 token 事实回答；
- 多轮连续记忆；
- 错误回答后的恢复；
- 新记忆覆盖旧记忆；
- memory 容量接近上限时的退化。

### 还缺少完整的记忆治理

生产级系统至少需要三层：

    模型内部 Memory
       │
       ├─ 记忆写入策略
       ├─ 记忆读取策略
       └─ 连续向量状态

    记忆管理器
       │
       ├─ 事实抽取
       ├─ 去重
       ├─ 冲突处理
       ├─ 时间衰减
       └─ 重要性评分

    持久化服务
       │
       ├─ 用户隔离
       ├─ 加密
       ├─ 版本管理
       └─ 删除/导出

## 评测时应该回答的关键问题

不要只问“准确率有没有变高”，还要问：

1. 模型是否真的使用了 memory，而不是从 query 中猜答案？
2. 新记忆能否覆盖旧记忆？
3. 无关内容是否会污染 memory？
4. memory 容量增加后，效果是否持续提升？
5. 程序重启后能否恢复？
6. 不同用户之间是否完全隔离？
7. memory read/write 是否降低原模型的通用能力？
8. 记忆状态是否可以解释、导出和删除？
9. 推理速度和显存成本是多少？
10. 长时间运行后是否出现状态漂移或数值爆炸？

## 下一步建议

如果目标是开发一个真正有新意的架构，建议按下面顺序推进：

1. 扩展到多事实、多属性和长序列连续写入；
2. 增加时间衰减、记忆容量压力和可解释 slot 诊断；
3. 做多用户隔离、并发读写和异常恢复测试；
4. 再尝试更激进的结构，例如 fast weights、可写 KV cache、分层记忆和稀疏路由。

## 免责声明

本项目是研究和实验性质的代码。它不保证训练出的 adapter 在所有任务上提升，也不保证当前连续 memory state 能可靠保存所有自然语言事实。

如果使用第三方模型权重或数据集，请遵守对应的模型许可证、数据许可证和隐私要求。
