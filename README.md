# LLM 手撕练习模板

从零实现 Transformer 和三种主流 RLHF 算法，对齐 HuggingFace TRL / Llama / DeepSeek OpenRLHF 官方做法。

---

## 仓库结构

```
.
├── Transformer-EASY/     # Decoder-Only Transformer 手撕（提示含答案）
│   ├── transformer.py
│   └── README.md
│
├── Transformer-HARD/     # Decoder-Only Transformer 手撕（只有公式和数据流）
│   └── transformer.py
│
├── RLHF-EASY/            # PPO · DPO · GRPO 手撕（提示含答案）
│   ├── rlhf_env.py
│   ├── ppo.py
│   ├── dpo.py
│   ├── grpo.py
│   └── README.md
│
└── RLHF-HARD/            # PPO · DPO · GRPO 手撕（只有公式和数据流）
    ├── rlhf_env.py
    ├── ppo.py
    ├── dpo.py
    ├── grpo.py
    └── README.md
```

---

## EASY vs HARD

| | EASY | HARD |
|---|---|---|
| **TODO 注释** | 逐步伪代码 + 正确答案（注释形式） | 仅公式 / 极简数据流，无答案 |
| **适合人群** | 初次接触该算法，需要对照实现 | 有一定基础，希望独立推导 |
| **实现难度** | ★★☆☆☆ | ★★★★☆ |

两个版本**逻辑完全一致**，代码空白部分相同，只有注释详细程度不同。

---

## 模块一：Transformer-EASY / HARD

从零手撕一个 **Llama 风格 Decoder-Only Transformer**，单文件跑通训练步 + 自回归推理。

**手撕内容（共 12 步）：**

| 步骤 | 内容 |
|------|------|
| Step 1 | RMSNorm |
| Step 2 | Safe Softmax（数值稳定） |
| Step 3 | RoPE 旋转位置编码 |
| Step 4 | Multi-Head Attention + KV Cache |
| Step 5 | SwiGLU FFN |
| Step 6 | Transformer Block（Pre-Norm） |
| Step 7 | 完整 Transformer 模型 |
| Step 8 | Cross-Entropy Loss（手动 log-sum-exp） |
| Step 9 | Top-K / Top-P Sampling |
| Step 10 | BPE Tokenizer |
| Step 11 | 训练步验证（forward → loss → backward） |
| Step 12 | 推理验证（Prefill + KV Cache Decode Loop） |

**快速开始：**
```bash
cd Transformer-EASY    # 或 Transformer-HARD
pip install torch
python transformer.py  # 全部 TODO 完成后应输出 🎉 全部测试通过
```

---

## 模块二：RLHF-EASY / HARD

基于 HuggingFace TRL 和 DeepSeek OpenRLHF 真实结构，手撕三种主流 RLHF 算法。

### 三种算法对比

| | PPO | DPO | GRPO |
|---|---|---|---|
| 数据 | prompt | (prompt, chosen, rejected) | prompt |
| 在线采样 | ✅ | ❌ | ✅ |
| Reward Model | ✅ | ❌（隐式） | ✅ |
| Value Model | ✅ | ❌ | ❌ |
| Advantage 来源 | GAE | — | 组内相对奖励 |

### 文件职责

| 文件 | 内容 |
|------|------|
| `rlhf_env.py` | MockTokenizer · Dataset · DataLoader · PolicyModel · RewardModel · ValueModel |
| `ppo.py` | rollout → process(GAE) → update(Clipped Surrogate + Value Loss) |
| `dpo.py` | Bradley-Terry Loss · 隐式 reward 提取 |
| `grpo.py` | 组采样 → 组内标准化 → Clipped Surrogate + f-divergence KL |

### 学习顺序

```
rlhf_env.py               ← 必须先完成，三个算法共用
  [TODO-A] ppo_collate_fn   — Prompt 左填充
  [TODO-B] dpo_collate_fn   — 拼接 + 右填充 + labels(-100)
  [TODO-C] get_log_probs    — 统一 log prob 入口（核心）

→ ppo.py    (rollout · process · GAE · policy/value loss)
→ dpo.py    (sequence log prob · Bradley-Terry · train_step)
→ grpo.py   (组扩展 · 组内标准化 · f-divergence KL)
```

**快速开始：**
```bash
cd RLHF-EASY    # 或 RLHF-HARD
pip install torch
python rlhf_env.py     # 验证 Mock 环境正常
python ppo.py          # TODO 全部完成后可运行
python dpo.py
python grpo.py
```

---

## 前置要求

```bash
pip install torch
```

无需 GPU，所有 Mock 模型在 CPU 上即可完整运行。

---

## 使用建议

1. **选择版本**：首次接触某算法选 EASY，有基础想挑战选 HARD
2. **按顺序实现**：每完成一个 TODO 立即运行验证，不要一次写完再跑
3. **先跑 EASY，再挑战 HARD**：用 EASY 的答案理解原理，再在 HARD 中独立推导
4. **对照官方实现**：
   - Transformer → [LlamaModel (HuggingFace)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
   - RLHF → [TRL PPOTrainer](https://github.com/huggingface/trl) · [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
