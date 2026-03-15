# RLHF 手撕练习模板

> **版本说明**
> - **RLHF-EASY** — TODO 注释包含完整的逐步伪代码提示，适合初次接触 RLHF 工程实现的学习者
> - **RLHF-HARD** — TODO 注释只给数学公式和数据流，适合有一定手撕基础、希望独立推导的学习者

本模板以 **HuggingFace TRL** 和 **DeepSeek OpenRLHF** 的真实代码结构为蓝本，复现 PPO、DPO、GRPO 三种主流 RLHF 算法的核心工程逻辑。

---

## 前置要求

```bash
pip install torch
```

无需 GPU，Mock 模型在 CPU 上即可运行完整流程。

---

## 文件结构

```
RLHF-EASY/   (或 RLHF-HARD/)
├── rlhf_env.py   # 基础组件库（Tokenizer、Dataset、DataLoader、模型接口）
├── ppo.py        # PPO-RLHF Trainer
├── dpo.py        # DPO Trainer
└── grpo.py       # GRPO Trainer
```

### 为什么要有 `rlhf_env.py`？

真实训练中组件（tokenizer、dataloader、模型）是现成的，框架的工程复杂度隐藏在库代码里。  
`rlhf_env.py` 用 Mock 组件**模拟这些接口**，让你聚焦在 RLHF 算法逻辑本身，而不是重写 tokenizer。

| 组件 | 对应的真实库组件 |
|------|----------------|
| `MockTokenizer` | `AutoTokenizer` (HuggingFace) |
| `MockPolicyModel` | `AutoModelForCausalLM` |
| `MockReferenceModel` | deepcopy + freeze 的 SFT checkpoint |
| `MockRewardModel` | `AutoModelForSequenceClassification` (num_labels=1) |
| `MockValueModel` | `AutoModelForCausalLMWithValueHead` 的 value head |
| `RawPromptDataset` + `ppo_collate_fn` | `Dataset` + `DataCollatorWithPadding` |
| `RawPreferenceDataset` + `dpo_collate_fn` | TRL `DPODataCollatorWithPadding` |

---

## 快速开始

**第一步：验证框架正确**

```bash
python rlhf_env.py
```

看到 `✅ 框架结构正常` 说明环境无问题（此时 TODO 尚未完成，框架骨架本身是正确的）。

**第二步：按顺序完成 TODO**

建议按以下顺序，每完成一个 TODO 就运行对应文件验证：

```
[TODO-A] → [TODO-B] → [TODO-C] → PPO TODO-1~7 → DPO TODO-1~4 → GRPO TODO-1~7
```

---

## TODO 完整列表

### `rlhf_env.py`（三个算法共用，必须先完成）

| TODO | 函数 | 核心概念 |
|------|------|---------|
| **TODO-A** | `ppo_collate_fn` | Prompt **左填充**。Causal LM 从最右侧有效 token 续写，右填充会导致 generate 崩坏 |
| **TODO-B** | `dpo_collate_fn` | `[prompt + response]` 拼接；label 中 prompt 部分设 -100（不计入 loss）；右填充 |
| **TODO-C** | `MockPolicyModel.get_log_probs` | Causal LM shift（预测 t+1）→ log_softmax → gather；用 labels 或 action_mask 构建 loss_mask |

> `get_log_probs` 是整个框架最核心的函数，PPO 和 DPO 都依赖它，务必先搞懂再继续。

### `ppo.py`

| TODO | 函数 | 核心概念 |
|------|------|---------|
| **TODO-1** | `rollout()` | 生成 response → 构建 action_mask（cumsum 技巧）→ 收集 old log_probs / ref log_probs / rewards / values |
| **TODO-2** | `process()` | 逐 token KL 惩罚 → terminal reward scatter 到最后有效位 → 调用 GAE |
| **TODO-3** | `update()` | ppo_epochs 次循环重计算 new_log_probs → policy loss + value loss → 两个 optimizer 分别更新 |
| **TODO-4** | `_compute_gae()` | GAE 递推：δ_t = r_t + γV_{t+1} - V_t，A_t = δ_t + γλA_{t+1}；最后归一化 |
| **TODO-5** | `_policy_loss()` | Clipped Surrogate：-min(r·A, clip(r, 1±ε)·A)，只对有效 token 取均值 |
| **TODO-6** | `_value_loss()` | Clipped Value Loss：0.5·max((V_new-R)², (clip(V_new, V_old±ε)-R)²) |
| **TODO-7** | `main()` | 完整训练循环 |

### `dpo.py`

| TODO | 函数 | 核心概念 |
|------|------|---------|
| **TODO-1** | `_sequence_log_prob()` | per-token → 序列级均值聚合（消除 length bias） |
| **TODO-2** | `_dpo_loss()` | Bradley-Terry：h = β·(logr_chosen - logr_rejected)；L = -log σ(h) |
| **TODO-3** | `train_step()` | policy + ref 各自算 log_prob → DPO loss → 反向传播 |
| **TODO-4** | `main()` | 完整训练循环 |
| **TODO-5** | `extract_implicit_reward()` | r*(x,y) = β·(log π - log π_ref)，理解 DPO 隐式 reward |

### `grpo.py`

| TODO | 函数 | 核心概念 |
|------|------|---------|
| **TODO-1** | `rollout()` | `repeat_interleave(G)` 展开 → G 次独立采样 → action_mask → log_probs slice |
| **TODO-2** | `process()` | reward 打分 → 调用 _group_advantages |
| **TODO-3** | `_group_advantages()` | 组内标准化：A_i = (r_i - μ_group) / (σ_group + ε) |
| **TODO-4** | `update()` | 重计算 new_log_probs → policy loss + KL → 更新 |
| **TODO-5** | `_policy_loss()` | 与 PPO 一致，但 advantage 是 per-sequence 标量，需广播到 token 维度 |
| **TODO-6** | `_kl_penalty()` | f-divergence：exp(x) - x - 1，其中 x = log π_ref - log π_new |
| **TODO-7** | `main()` | 完整训练循环 |

---

## 关键工程细节

### 1. log_probs 的 shift 与 slice

```
full_ids = [prompt(T_p tokens), response(T_r tokens)]   → [B, T_p+T_r]
get_log_probs(full_ids) 内部做 shift，返回                → [B, T_p+T_r-1]

response 的 T_r 个 log prob 位于 [T_p-1, T_p+T_r-2]:
old_log_probs = full_log_probs[:, T_p-1:]              → [B, T_r]
```

这是官方 TRL / OpenRLHF 的标准做法：**在 rollout 里 slice，下游全部用 `[B, T_r]`**。

### 2. action_mask 的构建

```python
is_eos = (response_ids == eos_token_id)
action_mask = (is_eos.cumsum(dim=1) <= 1).long()
# EOS token 本身保留（cumsum=1），EOS 之后的 pad 清零（cumsum>1）
```

### 3. PPO vs DPO 的 padding 方向

| 场景 | 方向 | 原因 |
|------|------|------|
| PPO/GRPO prompt | **左填充** | generate() 从最右侧有效 token 续写 |
| DPO chosen/rejected | **右填充** | 静态数据，不需要 generate |

### 4. DPO labels 的 -100 masking

```
input_ids: [BOS, p1, p2, r1, r2, EOS]
labels:    [-100, -100, -100, r1, r2, EOS]
           ← prompt 不计 loss →  ← response 计 loss →
```

PyTorch CrossEntropyLoss 的 `ignore_index=-100` 自动跳过这些位置。

### 5. GRPO 的 f-divergence KL vs PPO 的近似 KL

| | 公式 | 特性 |
|---|---|---|
| PPO | `log π_old - log π_ref` | 近似，策略差异大时不准 |
| GRPO | `exp(x) - x - 1`，x = log π_ref - log π_new | 无偏估计，天然非负，更稳定 |

---

## 三种算法对比

| | PPO | DPO | GRPO |
|---|---|---|---|
| 需要 Reward Model | ✅ | ❌（隐式） | ✅ |
| 需要 Value Model | ✅ | ❌ | ❌ |
| 需要在线采样 | ✅ | ❌ | ✅ |
| 数据类型 | prompt | (prompt, chosen, rejected) | prompt |
| Advantage 来源 | GAE | — | 组内相对奖励 |

---

## 调试建议

1. **先跑通 `rlhf_env.py`**，确认 `get_log_probs` 的输出形状正确：
   - 输入 `full_ids [B, T]`，输出 `[B, T-1]`，slice 后 `[B, T_r]`

2. **逐 TODO 验证**，不要一次完成所有再运行，出错难以定位

3. **常见 shape 错误**：
   - `action_mask [B, T_r]` vs `log_probs [B, T_r]` 维度不一致 → 检查是否正确 slice
   - GAE 里 `values` 和 `token_rewards` 维度不匹配 → 确认两者都是 `[B, T_r]`

4. **验证 DPO 收敛**：训练后 `chosen_reward > rejected_reward` 的比例应接近 1.0

5. **验证 GRPO 的组内对比**：打印同一 prompt 的 G 个 response 和对应 advantage，
   reward 高的 advantage 应 > 0，reward 低的应 < 0
