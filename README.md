# RLHF 算法核心组件手撕练习代码库

本项目是一个针对大语言模型对齐（Alignment）阶段核心算法的纯 PyTorch 手撕模板，包括 **PPO**、**DPO** 和 **GRPO**。

此模板屏蔽了庞大的工程组件和真实的 LLM 模型加载逻辑，采用 `Mock` (模拟) 机制构建了极简的强化学习环境，旨在帮助学习者专注于**算法核心逻辑与数学公式的代码实现**。所有的核心算法逻辑均以 `TODO` 形式留空，非常适合作为进阶学习和巩固 RLHF 相关算法的练习材料。

---

## 📂 文件目录与手撕内容详解

### 1. `ppo.py` - Proximal Policy Optimization

基于 OpenAI InstructGPT 论文思路的 PPO-RLHF 架构。
采用标准的 `sample() → process() → update()` 强化学习流水线结构。

**需要手撕的核心内容（TODOs）：**

- **Token-level KL 散度计算 (`_compute_kl`)**：计算当前策略与参考模型在生成序列上的逐 Token KL 散度。
- **逐 Token 奖励重组 (`_build_token_rewards`)**：将句末的标量总 Reward 与每一步的 KL 惩罚结合，形成 PPO 需要的密集的逐 Token 奖励信号。
- **广义优势估计 GAE (`_compute_gae`)**：计算每一步的 Advantage（优势函数）和 Return（目标回报）。
- **策略网络损失 (`_policy_loss`)**：带裁切的 PPO 代理目标函数 (Clipped Surrogate Objective Loss)。
- **价值网络损失 (`_value_loss`)**：带裁切的 Value Function MSE 损失。
- **主流程组装 (`main`)**：将采样、数据加工与多 Epoch 梯度更新组合成完整的 PPO 训练循环。

### 2. `dpo.py` - Direct Preference Optimization

基于 Rafailov et al. 2023 的 DPO 算法。
避开了复杂的 Actor-Critic 强化学习架构，直接在偏好数据 (Chosen / Rejected) 上优化策略模型。

**需要手撕的核心内容（TODOs）：**

- **序列 Log Prob 聚合 (`_sequence_log_prob`)**：配合 mask 屏蔽掉 padding 部分，将逐 Token的 log probability 聚合为序列级的总 log probability。
- **DPO Bradley-Terry 损失 (`_dpo_loss`)**：利用 Reference Model 的 Log Prob 作为基准，推导隐式奖励 (Implicit reward)，并计算最终对比学习损失。
- **隐式 Reward 验证 (`extract_implicit_reward`)**：从训练完成的策略模型和参考模型中抽取隐式奖励，验证模型是否成功学会了偏好。
- **主流程组装 (`main`)**：加载偏好数据，计算 Log Prob 差值，梯度更新的精简循环。

### 3. `grpo.py` - Group Relative Policy Optimization

基于 DeepSeek 团队的 GRPO 算法 (DeepSeekMath, 2024)。
去除了传统的 Value 模型，通过同一 Prompt 多次采样的组内相对表现来计算 Advantage，大幅降低了显存开销。

**需要手撕的核心内容（TODOs）：**

- **组内相对优势 (`_group_advantages`)**：使用简单的数学统计 (均值与标准差)，对针对同一 prompt 采样生成的 $G$ 个 Response 进行组内标准化打分。
- **GRPO 策略损失 (`_policy_loss`)**：类似于 PPO，但 Advantage 退化为了序列级别（而非逐 Token）。
- **KL 惩罚项 (`_kl_penalty`)**：DeepSeek 方式的 KL 散度约束，作为惩罚项加入最终 loss 中。
- **主流程组装 (`main`)**：包含 Prompt 扩充、重复采样成组、组内相对评价及利用 KL 惩罚更新等流程。

### 4. `rlhf_env.py` - 模拟环境与基础设施 (完全自包含，无需修改)

本文件为上述三个算法提供了运行土壤。

- 包括极简的词表生成 (`MockTokenizer`)
- 包含极简的 Transformer / RNN 替代模型（如 `MockPolicyModel`, `MockReferenceModel`, `MockValueModel`, `MockRewardModel`）
- 提供了自动回归推理采样工具 (`generate_response`) 和 数据构造工具 (`build_prompt_batch`, `build_preference_pairs`)。

---

## 👩‍💻 如何使用此模板进行练习？

1. **阅读环境定义**：首先大致浏览 `rlhf_env.py`，了解数据的输入输出 Shape, 比如所有模型支持在 `[Batch_Size, Sequence_Length]` 的张量上进行模拟 forward。
2. **定位 TODO**：打开 `ppo.py` / `dpo.py` / `grpo.py`，搜索关键字 `TODO`。
3. **补充核心逻辑**：根据每处 TODO 附近的详细注释及其数学含义提示，利用 PyTorch (`torch.xxx`) 完成缺失的逻辑。
4. **运行测试**：在各自模块中编写完缺失代码及 `main()` 之后，直接执行 `python ppo.py` (或 dpo/grpo)，如果没有报错并能观察到 reward / margin 逐渐上升及 loss 的下降，说明手撕成功！
