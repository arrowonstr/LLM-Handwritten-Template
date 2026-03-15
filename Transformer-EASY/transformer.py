"""
手写 Decoder-Only Transformer (Llama 风格) — EASY 版
=====================================================
目标: 单文件跑通 forward → 训练一步 → 自回归生成

手写部分: RMSNorm · Safe Softmax · RoPE · Multi-Head Attention
         KV Cache · SwiGLU FFN · Cross-Entropy Loss
         Top-K/Top-P Sampling · BPE Tokenizer
调包部分: nn.Linear · nn.Embedding · F.silu · torch.matmul

EASY 版使用说明:
  每个 TODO 的「答案」以注释形式给出，理解后自己在下方实现。
  完整跑通后可运行: python transformer.py
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第 0 步: 模型配置
# ============================================================================
@dataclass
class ModelConfig:
    d_model:        int   = 512    # 隐藏层维度
    n_heads:        int   = 8      # 注意力头数
    n_layers:       int   = 4      # Transformer Block 层数
    vocab_size:     int   = 1000   # 词表大小
    max_seq_len:    int   = 128    # 最大序列长度
    ffn_hidden_dim: int   = 2048   # FFN 中间层维度 (≈ 4 × d_model)
    norm_eps:       float = 1e-6   # RMSNorm 的 epsilon


# ============================================================================
# 第 1 步: RMSNorm
# ============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama 风格).

    公式: RMSNorm(x) = x / sqrt(mean(x²) + ε) × weight
    vs LayerNorm: 去掉了减均值的中心化步骤，更轻量。
    输入输出 shape 不变: (B, S, D) → (B, S, D)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO 1-a: 定义可学习参数 weight，shape=(d_model,)，初始化为全 1
        # 答案: self.weight = nn.Parameter(torch.ones(d_model))
        raise NotImplementedError("TODO 1-a: 定义 self.weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, S, D)   Returns: (B, S, D)"""
        # TODO 1-b: 计算 RMS，对最后一维 (dim=-1) 计算，需要 keepdim=True
        # 答案: mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        #       rms     = torch.sqrt(mean_sq + self.eps)

        # TODO 1-c: 归一化并乘以可学习参数，返回结果
        # 答案: return (x / rms) * self.weight
        raise NotImplementedError("TODO 1-b/c: 实现 RMSNorm forward")


# ============================================================================
# 第 2 步: Safe Softmax (数值稳定版)
# ============================================================================
def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """手写数值稳定的 Softmax.

    公式: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    先减最大值防止 exp() 上溢，数学上与原 Softmax 完全等价。
    """
    # TODO 2: 实现 safe softmax
    # 答案: x_max = x.max(dim=dim, keepdim=True)[0]
    #       exp_x = torch.exp(x - x_max)
    #       return exp_x / exp_x.sum(dim=dim, keepdim=True)
    raise NotImplementedError("TODO 2: 实现 safe_softmax")


# ============================================================================
# 第 3 步: Rotary Positional Embedding (RoPE)
# ============================================================================
class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 — 对 Q、K 施加旋转，使注意力分数隐含相对位置信息.

    核心公式: x_rotated = x × cos + rotate_half(x) × sin
    优势: 相对位置信息直接编码进 QKᵀ 内积，外推性强。
    """

    def __init__(self, d_k: int, max_seq_len: int):
        """
        Args:
            d_k:         每个注意力头的维度 (d_model // n_heads)
            max_seq_len: 最大序列长度，预计算 cos/sin 缓存
        """
        super().__init__()

        # TODO 3-a: 计算频率向量 inv_freq，shape: (d_k//2,)
        # 公式: 1 / 10000^(2i/d_k)，i=0,2,4,...,d_k-2
        # 答案: inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))

        # TODO 3-b: 预计算并缓存 cos/sin
        # 1. t = torch.arange(max_seq_len)                   → (max_seq_len,)
        # 2. freqs = torch.outer(t, inv_freq)                → (max_seq_len, d_k//2)
        # 3. self.register_buffer("cos_cached", freqs.cos()) → 不参与梯度，随模型移动设备
        # 4. self.register_buffer("sin_cached", freqs.sin())
        raise NotImplementedError("TODO 3-a/b: 实现 RoPE __init__")

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将最后一维一分为二，做 2D 旋转: (a, b) → (−b, a).

        Args:
            x: (..., d_k)
        Returns:
            (..., d_k)
        """
        # TODO 3-c: 实现 rotate_half
        # 答案: half = x.shape[-1] // 2
        #       return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        raise NotImplementedError("TODO 3-c: 实现 rotate_half")

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:            (B, n_heads, S, d_k)  Q 或 K
            position_ids: (S,)  位置索引（KV Cache 推理时不从 0 开始）
        Returns:
            (B, n_heads, S, d_k)
        """
        # TODO 3-d: 用 position_ids 索引缓存，扩展维度后应用旋转
        # 答案: cos = torch.cat([self.cos_cached[position_ids]] * 2, dim=-1)[None, None]
        #       sin = torch.cat([self.sin_cached[position_ids]] * 2, dim=-1)[None, None]
        #       return x * cos + self.rotate_half(x) * sin
        raise NotImplementedError("TODO 3-d: 实现 RoPE forward")


# ============================================================================
# 第 4 步: Multi-Head Attention + KV Cache
# ============================================================================
class MultiHeadAttention(nn.Module):
    """多头注意力机制 + KV Cache 支持.

    完整流程:
      Linear投影 → 多头切分 → RoPE(Q,K) → KV Cache拼接
      → scaled dot-product → 因果掩码 → Softmax → AV → Concat → 输出投影

    注意: V 不需要 RoPE！
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k     = config.d_model // config.n_heads
        self.d_model = config.d_model
        # 四个线性投影（bias=False 对齐 Llama）
        self.wq   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.d_k, config.max_seq_len)

    def forward(
        self,
        x:            torch.Tensor,
        position_ids: torch.Tensor,
        past_kv:      Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:            (B, S, D)
            position_ids: (S,)
            past_kv:      (past_K, past_V) 各 (B, n_heads, past_S, d_k)
        Returns:
            output:     (B, S_q, D)
            present_kv: (K, V)
        """
        B, S, D = x.shape

        # ── 4-a 线性投影 (直接调包) ───────────────────────────────────────────
        q = self.wq(x)   # (B, S, D)
        k = self.wk(x)
        v = self.wv(x)

        # ── 4-b Reshape 切分多头 ─────────────────────────────────────────────
        # TODO 4-b: (B, S, D) → (B, n_heads, S, d_k)
        # 答案: q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        #        (k, v 同理)
        raise NotImplementedError("TODO 4-b: reshape 切分多头")

        # ── 4-c 对 Q, K 施加 RoPE (V 不需要) ─────────────────────────────────
        # TODO 4-c: q = self.rope(q, position_ids)
        # 答案同上，k 同理

        # ── 4-d KV Cache 拼接 ────────────────────────────────────────────────
        # TODO 4-d: 在 dim=2 (sequence 维度) 拼接历史 K, V
        # 答案: if past_kv is not None:
        #           past_k, past_v = past_kv
        #           k = torch.cat([past_k, k], dim=2)
        #           v = torch.cat([past_v, v], dim=2)
        #        present_kv = (k, v)

        # ── 4-e Scaled Dot-Product ───────────────────────────────────────────
        # TODO 4-e: scores = QKᵀ / √d_k，shape: (B, n_heads, S_q, S_kv)
        # 答案: scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ── 4-f 因果掩码 (Causal Mask) ───────────────────────────────────────
        # 只有 prefill (S_q > 1) 需要；decode 阶段 S_q=1，新 token 可看全历史
        # 答案: if S_q > 1:
        #           S_q, S_kv = q.size(2), k.size(2)
        #           mask = torch.triu(torch.ones(S_kv, S_kv, device=q.device), diagonal=1).bool()
        #           scores = scores.masked_fill(mask[-S_q:, :], float('-inf'))

        # ── 4-g Softmax ───────────────────────────────────────────────────────
        # 答案: attn_weights = safe_softmax(scores, dim=-1)

        # ── 4-h AV 加权求和 ───────────────────────────────────────────────────
        # TODO 4-h: output = attn_weights @ v，shape: (B, n_heads, S_q, d_k)
        # 答案: output = torch.matmul(attn_weights, v)

        # ── 4-i Concat 多头 + 输出投影 ────────────────────────────────────────
        # TODO 4-i: (B, n_heads, S_q, d_k) → transpose(1,2) → contiguous → view → wo
        # 答案: output = output.transpose(1, 2).contiguous().view(B, S_q, D)
        #        output = self.wo(output)
        #        return output, present_kv


# ============================================================================
# 第 5 步: SwiGLU Feed-Forward Network
# ============================================================================
class SwiGLU_FFN(nn.Module):
    """SwiGLU FFN — Llama2/3 标准 FFN.

    公式: FFN(x) = W_down × (SiLU(W_gate × x) ⊙ W_up × x)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO 5-a: 定义三个线性层（均无 bias）
        # w_gate, w_up:  d_model → ffn_hidden_dim
        # w_down:        ffn_hidden_dim → d_model
        # 答案: self.w_gate = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        #        self.w_up   = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        #        self.w_down = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)
        raise NotImplementedError("TODO 5-a: 定义 SwiGLU 三个线性层")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 5-b: 实现 SwiGLU 前向
        # 答案: gate = F.silu(self.w_gate(x))
        #       return self.w_down(gate * self.w_up(x))
        raise NotImplementedError("TODO 5-b: 实现 SwiGLU forward")


# ============================================================================
# 第 6 步: Transformer Block (Decoder Layer)
# ============================================================================
class TransformerBlock(nn.Module):
    """Pre-Norm Decoder Block.

    结构: RMSNorm → MHA → 残差 → RMSNorm → FFN → 残差
    Pre-Norm (先 Norm 后 attention) 比 Post-Norm 训练更稳定，Llama 使用此方式。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO 6-a: 初始化两个 RMSNorm、MHA、FFN
        # 答案: self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        #        self.attn      = MultiHeadAttention(config)
        #        self.ffn_norm  = RMSNorm(config.d_model, config.norm_eps)
        #        self.ffn       = SwiGLU_FFN(config)
        raise NotImplementedError("TODO 6-a: 初始化 TransformerBlock 组件")

    def forward(
        self,
        x:            torch.Tensor,
        position_ids: torch.Tensor,
        past_kv:      Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO 6-b: 注意力子层: Pre-Norm + MHA + 残差
        # 答案: attn_out, present_kv = self.attn(self.attn_norm(x), position_ids, past_kv)
        #        h = x + attn_out

        # TODO 6-c: FFN 子层: Pre-Norm + FFN + 残差
        # 答案: output = h + self.ffn(self.ffn_norm(h))
        #        return output, present_kv
        raise NotImplementedError("TODO 6-b/c: 实现 TransformerBlock forward")


# ============================================================================
# 第 7 步: 完整 Transformer 模型
# ============================================================================
class Transformer(nn.Module):
    """Decoder-Only Transformer (Llama 风格).

    输入: (B, S) token IDs
    输出: (B, S, vocab_size) logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO 7-a: 初始化各组件
        # 答案: self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        #        self.layers          = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        #        self.final_norm      = RMSNorm(config.d_model, config.norm_eps)
        #        self.lm_head         = nn.Linear(config.d_model, config.vocab_size, bias=False)
        raise NotImplementedError("TODO 7-a: 初始化 Transformer 组件")

    def forward(
        self,
        input_ids:       torch.Tensor,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids:       (B, S)
            past_key_values: list[n_layers] of (past_K, past_V) | None
        Returns:
            logits:             (B, S, vocab_size)
            present_key_values: list[n_layers] of (K, V)
        """
        # TODO 7-b: 完整 forward
        # 1. h = self.token_embedding(input_ids)               → (B, S, D)
        # 2. past_len = past_key_values[0][0].size(2) if (past_key_values and past_key_values[0]) else 0
        #    position_ids = torch.arange(past_len, past_len + S, device=input_ids.device)
        # 3. if past_key_values is None: past_key_values = [None] * n_layers
        # 4. present_key_values = []
        #    for i, layer in enumerate(self.layers):
        #        h, present_kv = layer(h, position_ids, past_key_values[i])
        #        present_key_values.append(present_kv)
        # 5. h = self.final_norm(h)
        #    logits = self.lm_head(h)
        #    return logits, present_key_values
        raise NotImplementedError("TODO 7-b: 实现 Transformer forward")


# ============================================================================
# 第 8 步: Cross-Entropy Loss (手动实现，不用 F.cross_entropy)
# ============================================================================
def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """自回归语言模型 Cross-Entropy Loss.

    公式: L = -1/N Σ log p(x_{t+1} | x_{≤t})

    Args:
        logits:   (B, S, vocab_size)  模型输出
        input_ids:(B, S)              输入 token ID

    注意: 内部自动做 shift。logits[:,:-1,:] 预测 input_ids[:,1:]。
    无需手动传入 targets。
    """
    # TODO 8-a: shift — 对齐自回归目标
    # logits[:, :-1, :] 用 t=0..S-2 时刻预测 t=1..S-1 的真实 token
    # 答案: logits_s = logits[:, :-1, :].contiguous()   # (B, S-1, V)
    #        targets  = input_ids[:, 1:].contiguous()    # (B, S-1)

    # TODO 8-b: 展平为 (N, V) 和 (N,)，N = B*(S-1)
    # 答案: logits_flat  = logits_s.view(-1, V)
    #        targets_flat = targets.view(-1)

    # TODO 8-c: 数值稳定的 log_softmax
    # 答案: x_max       = logits_flat.max(dim=-1, keepdim=True)[0]
    #        x_shifted   = logits_flat - x_max
    #        log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=-1, keepdim=True))
    #        log_probs   = x_shifted - log_sum_exp             # (N, V)

    # TODO 8-d: 用 gather 取出目标 token 的 log 概率，取负均值
    # 答案: target_log_probs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    #        return -target_log_probs.mean()
    raise NotImplementedError("TODO 8: 实现 compute_loss")


# ============================================================================
# 第 9 步: Top-K / Top-P Sampling
# ============================================================================
def sample_top_k_top_p(
    logits:      torch.Tensor,
    top_k:       int   = 50,
    top_p:       float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Top-K 与 Top-P (Nucleus) 采样.

    Args:
        logits:      (B, vocab_size) 当前步最后位置的 logits
        top_k:       保留概率最高的 k 个候选（0 表示禁用）
        top_p:       在累积概率达到 p 的最小候选集内采样（0.0 表示禁用）
        temperature: 温度，越小越确定，越大越随机
    Returns:
        next_token: (B, 1)
    """
    # TODO 9-a: temperature 缩放
    # 答案: logits = logits / max(temperature, 1e-8)

    # TODO 9-b: Top-K 截断
    # 只保留最高 k 个，其余置 -inf
    # 答案: if top_k > 0:
    #           k   = min(top_k, logits.size(-1))
    #           kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    #           logits = logits.masked_fill(logits < kth, float('-inf'))

    # TODO 9-c: Top-P (Nucleus) 截断
    # 降序排列 → 计算 softmax 累积概率 → 把累积 > p 的位置置 -inf（右移一位保留边界 token）
    # 答案: if top_p > 0.0:
    #           sorted_logits, sorted_idx = logits.sort(descending=True)
    #           cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    #           to_remove = cum_probs > top_p
    #           to_remove[..., 1:] = to_remove[..., :-1].clone()
    #           to_remove[..., 0]  = False
    #           sorted_logits[to_remove] = float('-inf')
    #           logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    # TODO 9-d: softmax 归一化后多项式采样
    # 答案: return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    raise NotImplementedError("TODO 9: 实现 sample_top_k_top_p")


# ============================================================================
# 第 10 步: 极简 BPE Tokenizer
# ============================================================================
class BPETokenizer:
    """Byte-Pair Encoding 分词器 — 理解现代 LLM 真实分词逻辑.

    基于 UTF-8 字节序列，从 256 个单字节出发，反复合并最高频相邻对。
    """

    def __init__(self):
        self.vocab:  dict = {}   # id → bytes
        self.merges: dict = {}   # (id1, id2) → new_id

    def train(self, text: str, vocab_size: int):
        """用给定语料训练 BPE 词表.

        Args:
            text:       训练语料
            vocab_size: 目标词表大小 (≥ 256)
        """
        # TODO 10-a: 初始化基础字节词表 (共 256 个单字节 id)
        # 答案: ids = list(text.encode("utf-8"))
        #        self.vocab = {i: bytes([i]) for i in range(256)}

        # TODO 10-b: 反复合并，直到词表大小达到 vocab_size
        # for _ in range(vocab_size - 256):
        #   1. 统计相邻对频次: counts[(a, b)] = ...
        #   2. best = max(counts, key=counts.get)
        #   3. new_id = max(self.vocab) + 1
        #   4. 在 ids 中把所有连续 (best[0], best[1]) 替换为 new_id
        #   5. self.merges[best] = new_id
        #      self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
        raise NotImplementedError("TODO 10-a/b: 实现 BPE train")

    def encode(self, text: str) -> List[int]:
        """文本 → token ID 列表."""
        # TODO 10-c: 转字节 → 反复应用 merges 规则直到不能再合并
        # 答案: ids = list(text.encode("utf-8"))
        #        changed = True
        #        while changed:
        #            changed = False
        #            for pair, new_id in self.merges.items():
        #                new_ids, i = [], 0
        #                while i < len(ids):
        #                    if i + 1 < len(ids) and (ids[i], ids[i+1]) == pair:
        #                        new_ids.append(new_id); i += 2; changed = True
        #                    else:
        #                        new_ids.append(ids[i]); i += 1
        #                ids = new_ids
        #        return ids
        raise NotImplementedError("TODO 10-c: 实现 BPE encode")

    def decode(self, ids: List[int]) -> str:
        """token ID 列表 → 文本."""
        # TODO 10-d: 通过 self.vocab 把每个 id 转为 bytes，拼接后 UTF-8 decode
        # 答案: return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
        raise NotImplementedError("TODO 10-d: 实现 BPE decode")


# ============================================================================
# 第 11 步: 测试 — 训练步
# ============================================================================
def test_training_step(model: Transformer, config: ModelConfig):
    """测试模型训练时的单步行为: forward → loss → backward → optimizer step."""
    print("\n" + "=" * 60)
    print("测试一：训练模式 (forward → loss → backward)")
    print("=" * 60)
    model.train()

    batch_size, seq_len = 2, 16

    # TODO 11-a: 构造随机训练序列
    # raw = randint(B, seq_len+1)，input_ids = raw[:, :-1]
    # compute_loss 内部会自动 shift，所以直接传 input_ids 即可
    # 答案: raw       = torch.randint(0, config.vocab_size, (batch_size, seq_len + 1))
    #        input_ids = raw[:, :-1]

    # TODO 11-b: forward + loss（不要用 no_grad！需要梯度）
    # 答案: logits, _ = model(input_ids)
    #        loss = compute_loss(logits, input_ids)

    # TODO 11-c: backward + optimizer step
    # 答案: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #        optimizer.zero_grad()
    #        loss.backward()
    #        optimizer.step()

    # 打印结果:
    # print(f"  输入 shape:  {input_ids.shape}")
    # print(f"  logits shape:{logits.shape}")
    # print(f"  Loss 值:     {loss.item():.4f}")
    # print("  ✅ backward + optimizer.step() 正常运行")
    raise NotImplementedError("TODO 11: 实现 test_training_step")


# ============================================================================
# 第 12 步: 测试 — 推理生成 (Prefill + Decode Loop + KV Cache)
# ============================================================================
def test_generation_step(model: Transformer, config: ModelConfig):
    """测试推理阶段: BPE 编码 → Prefill → Decode Loop → BPE 解码."""
    print("\n" + "=" * 60)
    print("测试二：推理生成模式 (Prefill + KV Cache Decode)")
    print("=" * 60)
    model.eval()

    # BPE 训练
    corpus = ("transformers are beautiful and they transform NLP. "
              "transformer architecture is the backbone of LLMs.") * 50
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300)
    bpe_vocab_size = len(tokenizer.vocab)  # 采样时截断到此范围，防止 decode KeyError

    start_text = "transform"
    input_ids  = torch.tensor([tokenizer.encode(start_text)])  # (1, S_prompt)
    max_new_tokens = 10

    with torch.no_grad():
        # TODO 12-a: Prefill — 整个 prompt 一次性 forward
        # 答案: logits, past_kv   = model(input_ids)
        #        next_token        = sample_top_k_top_p(logits[:, -1, :bpe_vocab_size])
        #        generated_ids     = [next_token.item()]

        # TODO 12-b: Decode Loop — 每步只输入新 token，携带 KV Cache
        # 答案: for _ in range(max_new_tokens - 1):
        #            logits, past_kv = model(next_token, past_key_values=past_kv)
        #            next_token      = sample_top_k_top_p(logits[:, -1, :bpe_vocab_size])
        #            generated_ids.append(next_token.item())
        pass

    # TODO 12-c: BPE 解码并打印
    # 答案: generated_text = tokenizer.decode(generated_ids)
    #        print(f"  Prompt:    '{start_text}'")
    #        print(f"  Generated: '{generated_text}'")
    raise NotImplementedError("TODO 12: 实现 test_generation_step")


# ============================================================================
# Main
# ============================================================================
def main():
    config = ModelConfig()
    model  = Transformer(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    test_training_step(model, config)
    test_generation_step(model, config)

    print("\n🎉 全部测试通过！")


if __name__ == "__main__":
    main()
