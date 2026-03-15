# Decoder-Only Transformer 手撕练习模板 (EASY)

从零实现一个 **Llama 风格**的 Decoder-Only Transformer，单文件跑通：
- 训练步：forward → loss → backward → optimizer.step()
- 推理步：BPE 编码 → Prefill → KV Cache Decode Loop → BPE 解码

> **EASY 版**：每个 TODO 下方紧跟正确实现代码，建议先盖住答案自己写，再对照。

---

## 前置要求

```bash
pip install torch
```

无需 GPU，CPU 即可完整跑通。

---

## 快速开始

```bash
python transformer.py
```

正常输出：

```
模型参数量: 17,805,824
测试一：训练模式 (forward → loss → backward)   ← ✅
测试二：推理生成模式 (Prefill + KV Cache Decode) ← ✅
🎉 全部测试通过！
```

---

## 文件结构（按学习顺序）

| 步骤 | 类/函数 | 内容 |
|------|---------|------|
| **第 0 步** | `ModelConfig` | 超参数配置 |
| **第 1 步** | `RMSNorm` | Root Mean Square 归一化 |
| **第 2 步** | `safe_softmax` | 数值稳定的 Softmax |
| **第 3 步** | `RotaryPositionalEmbedding` | RoPE 旋转位置编码 |
| **第 4 步** | `MultiHeadAttention` | 多头注意力 + KV Cache |
| **第 5 步** | `SwiGLU_FFN` | Gated Feed-Forward Network |
| **第 6 步** | `TransformerBlock` | Pre-Norm Decoder Layer |
| **第 7 步** | `Transformer` | 完整模型 (embedding → blocks → lm_head) |
| **第 8 步** | `compute_loss` | 手动 Cross-Entropy (log-sum-exp 稳定化) |
| **第 9 步** | `sample_top_k_top_p` | Top-K / Top-P / Temperature 采样 |
| **第 10 步** | `BPETokenizer` | Byte-Pair Encoding 分词器 |
| **第 11 步** | `test_training_step` | 串联训练链路 |
| **第 12 步** | `test_generation_step` | 串联推理链路 |

---

## 各步骤核心概念

### Step 1 — RMSNorm
```
RMSNorm(x) = x / sqrt(E[x²] + ε) × weight
```
vs LayerNorm：去掉了减均值的中心化步骤，更轻量。Llama 全程使用 RMSNorm。

### Step 2 — Safe Softmax
```
softmax(x_i) = exp(x_i − max(x)) / Σ exp(x_j − max(x))
```
先减最大值，防止 `exp()` 上溢。数学上与原 Softmax 完全等价。

### Step 3 — RoPE
```
inv_freq = 1 / 10000^(2i/d_k)
freqs    = outer(positions, inv_freq)    # (S, d_k/2)
x_rot    = x × cos(freqs) + rotate_half(x) × sin(freqs)
```
- 只对 Q、K 施加，V 不需要
- `rotate_half(a, b) → (−b, a)` 对应 2D 旋转
- cos/sin 预计算并缓存（`register_buffer`）
- 效果：QKᵀ 内积自然包含相对位置信息

### Step 4 — Multi-Head Attention
```
流程: Linear(x) → view(B,S,H,d_k).T → RoPE → [KV Cache concat] → QKᵀ/√d_k → mask → softmax → AV → Linear
```
关键细节：
- Reshape：`(B,S,D) → (B,H,S,d_k)` 用 `view + transpose(1,2)`
- 因果掩码：`triu(diagonal=1)` 生成上三角，只有 prefill (S_q>1) 需要
- KV Cache：在 dim=2（sequence 维度）拼接历史 K, V

### Step 5 — SwiGLU FFN
```
FFN(x) = W_down × (SiLU(W_gate × x) ⊙ W_up × x)
```
vs 普通 FFN：引入门控机制，三个矩阵，表达能力更强（Llama2/3 默认）。

### Step 6 — TransformerBlock (Pre-Norm)
```
h      = x + MHA(RMSNorm(x))        ← 先 Norm 再 Attention
output = h + FFN(RMSNorm(h))        ← 先 Norm 再 FFN
```
Pre-Norm 相比 Post-Norm 训练更稳定，是现代 LLM 的标准结构。

### Step 7 — Transformer.forward
```
(B, S) → Embedding → [TransformerBlock × n_layers] → RMSNorm → Linear → (B, S, vocab)
```
KV Cache：每层返回 `(K, V)`，下次 decode 时传入以避免重复计算。

### Step 8 — Cross-Entropy Loss (内部 shift)
```
logits[:, :-1, :] 预测 input_ids[:, 1:]   ← 自动右移，无需手动处理
loss = -mean( log_softmax(logits)[target] )
```
`log_softmax` 采用 `x - max(x) - log(Σ exp(x-max(x)))` 防止溢出。

### Step 9 — Top-K / Top-P Sampling
| 策略 | 做法 |
|------|------|
| Temperature | logits / T，T→0 更确定，T→∞ 更随机 |
| Top-K | 只保留最高 K 个 logit，其余置 -inf |
| Top-P | 按降序累积概率，保留刚好超过 p 的最小集合 |

### Step 10 — BPE Tokenizer
```
训练: 从 256 单字节出发，反复合并最高频相邻对，记录 merges 规则
编码: UTF-8 字节 → 反复应用 merges 规则直到不能再合并
解码: id → bytes (vocab) → bytes.join → UTF-8 decode
```

### Step 11/12 — 训练 & 推理测试
```python
# 训练步：
logits, _ = model(input_ids)         # forward
loss = compute_loss(logits, input_ids)
optimizer.zero_grad(); loss.backward(); optimizer.step()

# 推理步（KV Cache）：
# Prefill: 所有 prompt token 一次性 forward，获取初始 past_kv
logits, past_kv = model(prompt_ids)
next_token = sample(logits[:, -1, :])
# Decode Loop: 每步只输入 1 个 token，携带 KV Cache
for _ in range(max_new_tokens):
    logits, past_kv = model(next_token, past_key_values=past_kv)
    next_token = sample(logits[:, -1, :])
```

---

## 与 HuggingFace Llama 的对应关系

| 本文件 | HuggingFace Llama |
|--------|-------------------|
| `RMSNorm` | `LlamaRMSNorm` |
| `RotaryPositionalEmbedding` | `LlamaRotaryEmbedding` |
| `MultiHeadAttention` | `LlamaAttention` |
| `SwiGLU_FFN` | `LlamaMLP` |
| `TransformerBlock` | `LlamaDecoderLayer` |
| `Transformer` | `LlamaModel` + `LlamaForCausalLM` |
| `compute_loss` | `LlamaForCausalLM.compute_loss` |

---

## 调试建议

1. **逐步运行**：每完成一个 Step（RMSNorm → RoPE → MHA），单独构造张量测试 shape
2. **shape 检查**：
   - `q.shape == (B, n_heads, S, d_k)` after reshape
   - `scores.shape == (B, n_heads, S_q, S_kv)` after matmul
   - `loss.shape == torch.Size([])` (标量)
3. **常见错误**：
   - `transpose` 后忘 `contiguous()` 再 `view` → 报错
   - RoPE 应用在 V 上 → 不报错但结果错误
   - `compute_loss` 调用时不经过 shift 直接传 raw targets → loss 训练意义错误（本版已在内部 shift）
4. **KV Cache 验证**：
   - prefill (S=N) 的 `logits[:, -1, :]` 应等于 decode (S=1，携带 past_kv) 的 `logits[:, 0, :]`
