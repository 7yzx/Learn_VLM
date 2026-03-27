# ChartQA强化学习Agent - VeRL框架详细说明

## 目录

1. [项目概述](#项目概述)
2. [VeRL框架架构](#verl框架架构)
3. [RL Agent的作用](#rl-agent的作用)
4. [训练流程详解](#训练流程详解)
5. [关键组件说明](#关键组件说明)
6. [两阶段Rollout机制](#两阶段rollout机制)
7. [奖励系统](#奖励系统)
8. [优势估计算法](#优势估计算法)
9. [配置说明](#配置说明)

---

## 项目概述

这是一个基于VeRL（Veritable Reinforcement Learning）框架构建的多模态强化学习Agent，用于解决ChartQA（图表问答）任务。该项目的核心创新点是"Think with Images"范式，允许模型使用视觉工具进行局部注意力处理。

### 核心特性

1. **多模态输入处理**：支持图像和文本的联合处理
2. **视觉工具使用**：模型可以调用`focus_on_*`系列函数进行图像聚焦、遮罩、绘制等操作
3. **两阶段Rollout**：首次rollout生成工具调用，执行工具后进行第二次rollout生成最终答案
4. **GRPO/DAPO训练**：使用GRPO（Group Relative Policy Optimization）和DAPO（Dual-clip PPO）算法
5. **分布式训练**：基于Ray + FSDP + vLLM的高效分布式训练框架

---

## VeRL框架架构

VeRL是一个基于Ray的强化学习训练框架，专门设计用于大规模语言模型的RLHF训练。其架构分为以下几个层次：

### 1. 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                   Driver Process                         │
│              (RayPPOTrainer)                         │
│
│  - 协调训练流程
│  - 优势计算
│  - 日志和检查点管理
└────────────┬────────────────────────────────────────────┘
             │ Ray RPC
             ↓
┌─────────────────────────────────────────────────────────────┐
│              Resource Pool Manager                      │
│
│  - GPU资源池管理
│  - Worker Group调度
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────┴────────┬────────┬────────┐
    ↓                 ↓        ↓        ↓
┌─────────┐    ┌─────────┐ ┌─────────┐ ┌─────────┐
│  Actor  │    │  Critic │ │   Ref   │ │ Rollout │
│  Worker │    │  Worker  │ │  Policy │ │ Worker  │
└────┬────┘    └────┬────┘ └────┬────┘ └────┬────┘
     ↓               ↓             ↓           ↓
┌──────────┐   ┌──────────┐  ┌──────────┐ ┌──────────┐
│   FSDP   │   │   FSDP   │  │   FSDP   │ │   vLLM   │
│  Model   │   │  Model   │  │  Model   │ │ Engine   │
└──────────┘   └──────────┘  └──────────┘ └──────────┘
```

### 2. 核心组件

#### Driver Process (驱动进程)
- **位置**: [`verl/trainer/main.py:Runner`](verl/trainer/main.py)
- **职责**:
  - 初始化tokenizer和processor
  - 创建Ray Worker Groups
  - 创建Reward Manager
  - 创建DataLoader
  - 实例化RayPPOTrainer并调用`fit()`

#### RayPPOTrainer (主训练器)
- **位置**: [`verl/trainer/ray_trainer.py:RayPPOTrainer`](verl/trainer/ray_trainer.py)
- **职责**:
  - 训练循环管理
  - Worker初始化和协调
  - 优势计算
  - 检查点管理
  - 验证

#### FSDPWorker (工作节点)
- **位置**: [`verl/workers/fsdp_workers.py:FSDPWorker`](verl/workers/fsdp_workers.py)
- **职责**:
  - 加载和封装模型（Actor、Critic、RefPolicy）
  - 使用FSDP进行分布式训练
  - 管理vLLM rollout引擎
  - 参数和优化器offload管理

#### Reward Manager (奖励管理器)
- **类型**: `BatchFunctionRewardManager`, `SequentialFunctionRewardManager`, `LLMBatchFunctionRewardManager`
- **职责**:
  - 加载奖励函数
  - 批量计算奖励
  - 返回奖励和详细指标

---

## RL Agent的作用

### 1. 主要目标

训练一个视觉-语言模型（VLM），使其能够：
1. **理解图表内容**：分析柱状图、折线图、饼图等
2. **使用工具**：根据需要调用视觉工具进行局部注意力聚焦
3. **准确回答问题**：基于图表信息回答数值或分类问题
4. **学习策略**：通过强化学习优化工具使用和答案生成策略

### 2. 状态空间 (State Space)

- **输入**: 包含图像和文本提示
  - `images`: PIL图像对象（处理后）
  - `prompt`: 格式化的用户问题
  - `metadata`: 图表的边界框信息（bbox）

### 3. 动作空间 (Action Space)

- **离散动作**（在rollout中采样）：
  - 生成文本token序列
  - 可能包含Python代码块（工具调用）

- **工具调用**（从生成文本解析）：
  ```python
  focus_on_columns_with_mask(columns_bbox, image)
  focus_on_rows_with_highlight(rows_bbox, image)
  focus_on_x_values_with_draw(x_values_bbox, image)
  # ... 等等
  ```

### 4. 奖励函数 (Reward Function)

位置: [`examples/reward_function/refocus.py:compute_score`](examples/reward_function/refocus.py)

```python
def compute_score(predicts, ground_truths, format_weight=0.1):
    """
    计算预测答案和真实答案的匹配度

    返回:
        overall: 整体分数 (0.0 - 1.0)
    """
    # 解析"FINAL ANSWER:"后的答案
    # 处理多个可能的答案（用||分隔）
    # 数值答案使用相似度评分
    # 分类答案使用精确匹配
```

### 5. 策略网络 (Policy Network)

- **基础模型**: Qwen2.5-VL-3B/7B（可配置）
- **架构**: Vision-Language Transformer
  - Vision Tower: 处理图像输入
  - Language Model: 生成文本
- **可训练参数**:
  - 通常训练所有参数
  - 可选择冻结vision tower

---

## 训练流程详解

### 完整训练循环

```
┌─────────────────────────────────────────────────────────────────────┐
│                      训练主循环 (fit())                          │
│                                                              │
│  1. 加载检查点（如果存在）                                     │
│  2. 训练前验证（可选）                                       │
│  3. 训练循环 (for step in training_steps):                      │
│     3.1 数据加载                                             │
│     3.2 Rollout（生成）                                       │
│     3.3 奖励计算                                              │
│     3.4 优势计算                                              │
│     3.5 Actor更新                                              │
│     3.6 Critic更新（如果使用）                                   │
│     3.7 验证和保存                                            │
│  4. 训练后验证                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细步骤说明

#### 步骤1: 数据加载

```python
# 在ray_trainer.py:fit()中
for batch_dict in train_dataloader:
    batch = DataProto.from_single_dict(batch_dict)

    # 添加UID用于GRPO/RLOO分组
    batch.non_tensor_batch["uid"] = uuid4()
```

数据包含:
- `input_ids`: Tokenized prompt
- `attention_mask`: 注意力掩码
- `position_ids`: 位置ID（Qwen2-VL使用多维度rope）
- `multi_modal_data`: 图像数据
- `metadata`: bbox信息
- `ground_truth`: 真实答案

#### 步骤2: Rollout（生成）

```python
# ray_trainer.py:fit()中的gen阶段
gen_batch = batch.pop(
    batch_keys=["input_ids", "attention_mask", "position_ids"],
    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
)

# 调用vLLM进行快速生成
gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch)
```

生成的配置:
- `temperature`: 1.0（训练）, 0.5（验证）
- `n`: 每个prompt生成5个样本（GRPO需要）
- `top_p`: 0.99
- `max_tokens`: max_response_length

#### 步骤2.5: 两阶段Rollout（工具执行）

这是本项目的核心创新点：

```python
# ray_trainer.py:get_second_rollout_batch()

# 第一阶段：解析和执行工具
for idx, output_text in enumerate(output_texts):
    parsed = tool_parser.parse(output_text)

    if parsed["status"]:  # 包含工具调用
        # 执行Python代码
        exec(parsed["content"], tool_context)

        # 获取编辑后的图像
        edited_image = captured_output

        # 构建第二阶段prompt
        second_prompt = original_prompt + "\n" + tool_output + "\nOBSERVATION: ..."

        # 生成第二阶段回答
        second_output = model.generate(second_prompt, [original_image, edited_image])
```

**工具执行上下文**:
```python
context = {
    "image_1": original_image,
    "focus_on_columns_with_mask": focus_on_columns_with_mask,
    "focus_on_rows_with_highlight": focus_on_rows_with_highlight,
    # ... 其他工具函数
    "columns_bbox": metadata["columns_bbox"],
    "rows_bbox": metadata["rows_bbox"]
}
```

#### 步骤3: 奖励计算

```python
# 批量计算奖励
reward_tensor, reward_metrics = ray.get(reward_fn.compute_reward.remote(batch))

# 奖励形状: (batch_size, response_length)
# 对response维度求和得到每个样本的总奖励
batch.batch["token_level_scores"] = reward_tensor
```

奖励指标:
- `overall`: 整体准确率（主要训练信号）
- 格式分数（可选）

#### 步骤4: 优势计算

```python
# ray_trainer.py:fit()中的adv阶段

# 1. 计算旧策略log_prob
old_log_probs = actor_rollout_wg.compute_log_probs(batch)

# 2. 计算参考策略log_prob（用于KL惩罚）
if use_reference_policy:
    ref_log_probs = ref_policy_wg.compute_ref_log_probs(batch)

# 3. 计算value（如果使用critic）
if use_critic:
    values = critic_wg.compute_values(batch)

# 4. 应用KL惩罚
if use_reference_policy and not use_kl_loss:
    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl, kl_penalty)
    # token_level_rewards = token_level_scores - kl_coef * kl

# 5. 计算优势
batch = compute_advantage(batch, adv_estimator, gamma, lam)
```

#### 步骤5: Actor更新

```python
# ray_trainer.py中
actor_output = actor_rollout_wg.update_actor(batch)
```

Actor更新内部（在`dp_actor.py`）:
```python
# 1. 计算新的log_prob
log_probs = actor_model(batch)

# 2. 计算ratio = exp(log_prob - old_log_prob)
ratio = exp(log_probs - old_log_probs)

# 3. PPO裁剪
ratio_clipped = clip(ratio, 1 - clip_ratio, 1 + clip_ratio)

# 4. DAPO dual-clip（创新）
ratio_dual = clip_ratio_dual
pg_loss = -min(adv * ratio, adv * ratio_clipped, adv * ratio_dual)

# 5. 反向传播和优化
pg_loss.backward()
optimizer.step()
```

#### 步骤6: Critic更新（可选）

```python
critic_output = critic_wg.update_critic(batch)
```

Critic更新:
```python
# MSE loss on returns
vf_loss = (vpreds - returns)^2
vf_loss.backward()
critic_optimizer.step()
```

---

## 关键组件说明

### 1. Worker角色系统

VeRL使用角色枚举来区分不同类型的worker:

```python
class Role(IntEnum):
    Actor = auto()                    # 策略网络（训练）
    Rollout = auto()                  # 生成引擎
    ActorRollout = auto()             # 合并角色（Actor + Rollout）
    Critic = auto()                  # 价值网络
    RefPolicy = auto()                # 参考策略（冻结）
    RewardModel = auto()               # 奖励模型
    ActorRolloutRef = auto()          # Actor+Rollout+Ref合并
```

### 2. 数据协议 (DataProto)

位置: [`verl/protocol.py`](verl/protocol.py)

VeRL使用自定义的`DataProto`类来传递数据，它包含:

```python
class DataProto:
    batch: TensorDict           # 张量数据（在GPU上）
        - input_ids
        - attention_mask
        - position_ids
        - responses (生成的token)
        - response_mask
        - old_log_probs
        - ref_log_probs
        - values
        - advantages
        - returns
        - token_level_scores

    non_tensor_batch: dict       # 非张量数据（在CPU上）
        - prompt
        - query
        - ground_truth
        - metadata
        - uid
        - figure_path
        - figure_id

    meta_info: dict              # 元信息
        - global_token_num
        - temperature
        - eos_token_id
```

### 3. FSDP配置

FSDP (Fully Sharded Data Parallel)配置项:

```yaml
worker.actor.fsdp:
  enable_full_shard: true        # FULL_SHARD vs SHARD_GRAD_OP
  enable_cpu_offload: false       # 参数CPU offload
  enable_rank0_init: true        # Rank 0初始化后广播

worker.actor.offload:
  offload_params: true           # 训练时offload参数
  offload_optimizer: true       # 训练时offload优化器
```

内存优化:
- `offload_params=true + offload_optimizer=true`: 最大化CPU使用，最小化GPU
- `offload_params=false + offload_optimizer=false`: 最大化GPU使用

### 4. vLLM Rollout

位置: [`verl/workers/rollout/vllm_rollout_spmd.py`](verl/workers/rollout/vllm_rollout_spmd.py)

vLLM配置:
```yaml
worker.rollout:
  n: 5                          # 每个prompt生成样本数
  temperature: 1.0                # 采样温度
  top_p: 0.99
  gpu_memory_utilization: 0.4       # GPU内存利用率
  tensor_parallel_size: 2             # TP大小
  limit_images: 2                  # 最大图像数
```

vLLM优化:
- PagedAttention: 高效的KV cache管理
- Tensor Parallel: 模型权重切分
- CUDA Graphs: 加速生成

---

## 两阶段Rollout机制

### 设计动机

对于图表问答任务，直接回答往往不准确，因为：
1. 图表包含大量信息，需要局部聚焦
2. 数值比较需要精确读取
3. 不同的问题类型需要不同的处理策略

两阶段rollout让模型可以：
1. 第一阶段：分析问题，决定是否需要工具，调用工具生成聚焦后的图像
2. 第二阶段：基于原图和编辑后的图像生成最终答案

### 实现流程

#### 阶段1: 工具生成和执行

```python
# ray_trainer.py:get_second_rollout_batch()

# 1. 解析工具调用
parsed = parser.parse(output_text)
# 返回: {"status": True, "content": python_code}

# 2. 执行工具代码
exec(python_code, context)
# context包含所有工具函数和bbox信息

# 3. 捕获编辑后的图像
edited_image = captured_output
```

#### 工具函数

位置: [`verl/tooluse/tools.py`](verl/tooluse/tools.py)

可用工具:
```python
# Mask工具 - 遮罩非聚焦区域
focus_on_columns_with_mask(columns_bbox, image)
focus_on_rows_with_mask(rows_bbox, image)
focus_on_x_values_with_mask(x_values_bbox, image)
focus_on_y_values_with_mask(y_values_bbox, image)

# Draw工具 - 绘制聚焦区域边框
focus_on_columns_with_draw(columns_bbox, image)
focus_on_rows_with_draw(rows_bbox, image)
focus_on_x_values_with_draw(x_values_bbox, image)
focus_on_y_values_with_draw(y_values_bbox, image)

# Highlight工具 - 高亮聚焦区域
focus_on_columns_with_highlight(columns_bbox, image)
focus_on_rows_with_highlight(rows_bbox, image)
focus_on_x_values_with_highlight(x_values_bbox, image)
focus_on_y_values_with_highlight(y_values_bbox, image)
```

#### 阶段2: 最终答案生成

```python
# 1. 构建第二阶段prompt
second_prompt = (
    original_prompt + "\n" +
    tool_output + "\n" +
    "OBSERVATION: Execution success. The output is as follows:" +
    "\n<the image outputs of the code is added as the second image>"
)

# 2. 处理双图像输入
images = [original_image, edited_image]
model_inputs = processor(images, [second_prompt], ...)

# 3. 生成最终答案
second_output = model.generate(model_inputs)
```

#### 奖励调整

工具调用会影响奖励：
- 成功的工具调用: `penalty += 1`（鼓励使用工具）
- 失败的工具调用: `penalty -= 10`（严重惩罚）
- 无工具调用: `penalty = 0`（中立）

最终奖励 = 答案准确率 + penalty_weight * penalty

---

## 奖励系统

### 奖励函数类型

位置: [`verl/workers/reward/`](verl/workers/reward/) 和 [`examples/reward_function/`](examples/reward_function/)

支持4种奖励计算模式:

#### 1. Batch Reward

```python
class BatchFunctionRewardManager:
    def compute_reward(self, batch: DataProto) -> DataProto:
        # 一次性计算整个batch的奖励
        scores = reward_fn(prompts, responses, ground_truths)
        return DataProto.from_dict({"token_level_scores": scores})
```

#### 2. Sequential Reward

```python
class SequentialFunctionRewardManager:
    def compute_reward(self, batch: DataProto) -> DataProto:
        # 逐个计算奖励
        for i in range(len(batch)):
            score = reward_fn(prompt, response, ground_truth)
            scores.append(score)
        return DataProto.from_dict({"token_level_scores": scores})
```

#### 3. LLM Batch Reward

```python
class LLMBatchFunctionRewardManager:
    def compute_reward(self, batch: DataProto) -> DataProto:
        # 使用外部LLM批量评估
        prompts = self._build_prompts(batch)
        llm_outputs = self.llm_client.batch_generate(prompts)
        scores = self._parse_llm_outputs(llm_outputs)
        return DataProto.from_dict({"token_level_scores": scores})
```

#### 4. LLM Double Batch Reward

```python
class LLMDoubleBatchFunctionRewardManager:
    def compute_reward(self, batch: DataProto) -> DataProto:
        # 两阶段LLM评估
        # 第一阶段：评估工具使用
        # 第二阶段：评估最终答案
        ...
```

### ChartQA奖励函数

位置: [`examples/reward_function/refocus.py:compute_score`](examples/reward_function/refocus.py)

```python
def compute_score(predicts: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
    """
    输入:
        predicts: 模型生成的答案列表
        ground_truths: 真实答案列表

    输出:
        [{"overall": 0.0-1.0}, ...]
    """
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        # 1. 提取"FINAL ANSWER:"后的内容
        answers = re.findall(r'FINAL ANSWER:\s*(.*?)(?=\.?\s|\.?$)', predict)

        if len(answers) > 0:
            answer = answers[0]
            sub_gts = ground_truth.split("|||")

            correct_answers = 0
            for ans in answer.split('||'):
                for gt in sub_gts:
                    if is_number(gt) and is_number(ans):
                        # 数值相似度
                        score = 1 - abs(float(gt) - float(ans)) / max(abs(gt), abs(ans))
                    else:
                        # 字符串精确匹配
                        score = 1.0 if ans == gt else 0.0
                    correct_answers += max(candidate_scores)

            overall_score = correct_answers / len(sub_gts)
        else:
            overall_score = 0.0

        scores.append({"overall": overall_score})

    return scores
```

### 在线过滤 (Online Filtering)

当`algorithm.online_filtering: true`时：

```python
# ray_trainer.py:_make_batch_data_online_filtering()

# 1. 生成rollout
while len(batch) < target_size:
    new_batch = rollout_and_reward()

    # 2. 计算每个UID的平均分数
    uid2mean = {}
    for uid, scores in zip(uids, reward_scores):
        uid2mean[uid] = np.mean(scores)

    # 3. 过滤异常值
    kept_uids = [
        uid for uid, avg_score in uid2mean.items()
        if filter_low < avg_score < filter_high
    ]

    # 4. 保留有效样本
    batch = batch[kept_uids]
```

这防止极端奖励值影响训练稳定性。

---

## 优势估计算法

VeRL支持多种优势估计方法，位置: [`verl/trainer/core_algos.py`](verl/trainer/core_algos.py)

### 1. GAE (Generalized Advantage Estimation)

```python
def compute_gae_advantage_return(
    token_level_rewards: Tensor,  # (bs, response_length)
    values: Tensor,                # (bs, response_length)
    response_mask: Tensor,
    gamma: float = 1.0,           # 折扣因子
    lam: float = 1.0,              # GAE lambda
):
    # 反向计算GAE
    lastgaelam = 0
    advantages = []
    for t in reversed(range(response_length)):
        nextvalues = values[:, t+1] if t < response_length-1 else 0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages.append(lastgaelam)

    advantages = stack(advantages[::-1])
    returns = advantages + values

    # 归一化
    advantages = masked_whiten(advantages, response_mask)

    return advantages, returns
```

**使用场景**: 需要Critic网络时

### 2. GRPO (Group Relative Policy Optimization)

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    index: Tensor,  # UID数组
    eps: float = 1e-6
):
    # 1. 计算每个样本的总奖励
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)

    # 2. 按UID分组
    id2score = defaultdict(list)
    for i, score, uid in zip(range(len(scores)), scores, index):
        id2score[uid].append(score)

    # 3. 计算每组的均值和标准差
    id2mean, id2std = {}, {}
    for uid, score_list in id2score.items():
        id2mean[uid] = mean(score_list)
        id2std[uid] = std(score_list)

    # 4. 标准化优势
    for i, score, uid in zip(range(len(scores)), scores, index):
        scores[i] = (score - id2mean[uid]) / (id2std[uid] + eps)

    # 5. 广播到token级别
    returns = scores.unsqueeze(-1) * response_mask

    return returns, returns  # advantages = returns
```

**特点**:
- 不需要Critic网络
- 同一prompt的多个样本相互比较
- 自动基线（组内均值）
- 适合outcome-level奖励（单值奖励）

**使用场景**: ChartQA项目使用

### 3. RLOO (Leave-One-Out)

```python
def compute_rloo_outcome_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    index: Tensor
):
    scores = token_level_rewards.sum(dim=-1)

    # 按UID分组
    id2score = defaultdict(list)
    for i, score, uid in zip(range(len(scores)), scores, index):
        id2score[uid].append(score)

    # 计算组内总和
    id2sum = {}
    for uid, score_list in id2score.items():
        id2sum[uid] = sum(score_list)

    # LOO: (score - (sum - score)) / (n - 1)
    for i, score, uid in zip(range(len(scores)), scores, index):
        sample_num = len(id2score[uid])
        baseline = (id2sum[uid] - score) / (sample_num - 1)
        scores[i] = score - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns
```

### 4. REINFORCE++

```python
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    gamma: float
):
    # 折扣累积
    returns = zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(response_length)):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        running_return = running_return * response_mask[:, t]

    # 归一化
    advantages = masked_whiten(returns, response_mask)

    return advantages, returns
```

### 5. ReMax

```python
def compute_remax_outcome_advantage(
    token_level_rewards: Tensor,
    reward_baselines: Tensor,  # 预先计算的基线
    response_mask: Tensor
):
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns
```

---

## 配置说明

### 配置层级结构

VeRL使用OmegaConf进行配置管理，支持三层配置：

```python
# 1. 默认配置 (dataclass)
default_config = PPOConfig()

# 2. YAML配置
file_config = OmegaConf.load("config.yaml")

# 3. CLI参数
cli_args = OmegaConf.from_cli()

# 合并: CLI > YAML > Default
final_config = OmegaConf.merge(default_config, file_config, cli_args)
```

### 关键配置参数

#### 数据配置

```yaml
data:
  train_files: path/to/train.parquet
  val_files: path/to/val.parquet
  prompt_key: prompt
  answer_key: answer
  image_key: images
  max_prompt_length: 16384
  max_response_length: 8192
  rollout_batch_size: 16        # vLLM batch size
  val_batch_size: 512
  format_prompt: ./format_prompt/chartQA.jinja
  max_pixels: 4194304           # 图像最大像素
  min_pixels: 262144            # 图像最小像素
```

#### 算法配置

```yaml
algorithm:
  adv_estimator: grpo            # 优势估计方法
  disable_kl: false
  use_kl_loss: true           # KL作为loss而非penalty
  kl_penalty: low_var_kl        # KL惩罚类型
  kl_coef: 0.01              # KL系数
  online_filtering: true         # 在线过滤
  filter_low: 0.01             # 过滤下限
  filter_high: 0.99            # 过滤上限
  gamma: 1.0                  # 折扣因子
  lam: 0.95                   # GAE lambda
```

#### Worker配置

```yaml
worker:
  actor:
    global_batch_size: 8
    micro_batch_size_per_device_for_update: 1    # 梯度累积
    micro_batch_size_per_device_for_experience: 8  # vLLM batch
    max_grad_norm: 1.0
    ppo_epochs: 4
    ulysses_sequence_parallel_size: 1
    model:
      model_path: Qwen/Qwen2.5-VL-3B-Instruct
      enable_gradient_checkpointing: true
      freeze_vision_tower: false
    optim:
      lr: 1e-6
      weight_decay: 0.01
      strategy: adamw
      lr_warmup_ratio: 0.0
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
    offload:
      offload_params: true
      offload_optimizer: true

  rollout:
    n: 5                          # GRPO采样数
    temperature: 1.0
    top_p: 0.99
    gpu_memory_utilization: 0.4
    tensor_parallel_size: 2
    limit_images: 2
    val_override_config:
      temperature: 0.5            # 验证时降低温度
      n: 1                        # 验证时单样本

  ref:
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
    offload:
      offload_params: false

  reward:
    reward_type: llm_batch         # 计算模式
    reward_function: ./refocus.py:compute_score
    double_reward: false            # 是否进行两阶段奖励
```

#### 训练器配置

```yaml
trainer:
  total_epochs: 2000
  max_steps: 2000
  project_name: mini_chartQA
  logger: ["console", "wandb"]
  nnodes: 1
  n_gpus_per_node: 4
  val_freq: 2                    # 验证频率
  val_before_train: true
  val_only: false
  val_generations_to_log: 3
  save_freq: 5
  save_limit: 20
  save_checkpoint_path: ./checkpoints
  load_checkpoint_path: null
```

### 内存优化策略

根据GPU内存大小调整配置：

#### 小GPU (≤24GB)

```yaml
worker.actor:
  global_batch_size: 4
  micro_batch_size_per_device_for_update: 1
worker.rollout:
  gpu_memory_utilization: 0.4
  tensor_parallel_size: 1
worker.actor.offload:
  offload_params: true
  offload_optimizer: true
worker.actor.fsdp:
  enable_cpu_offload: false
```

#### 中等GPU (40-48GB)

```yaml
worker.actor:
  global_batch_size: 8
  micro_batch_size_per_device_for_update: 2
worker.rollout:
  gpu_memory_utilization: 0.5
  tensor_parallel_size: 2
worker.actor.offload:
  offload_params: false
  offload_optimizer: false
```

#### 大GPU (80GB+)

```yaml
worker.actor:
  global_batch_size: 16
  micro_batch_size_per_device_for_update: 4
worker.rollout:
  gpu_memory_utilization: 0.7
  tensor_parallel_size: 4
worker.actor.offload:
  offload_params: false
  offload_optimizer: false
```

---

## 总结

VeRL框架为多模态RLHF训练提供了完整的解决方案：

1. **分布式训练**: Ray + FSDP + vLLM
2. **灵活算法**: 支持GAE/GRPO/RLOO/REINFORCE++等
3. **工具使用**: 两阶段rollout支持工具调用
4. **高效实现**: vLLM rollout, FSDP优化, CPU offload
5. **易于配置**: 层次化配置系统

对于ChartQA任务，该系统能够：
- 训练模型理解复杂图表
- 学习何时使用视觉工具
- 优化答案生成策略
- 处理多模态输入

通过调整配置参数，可以在不同硬件规模上高效训练。
