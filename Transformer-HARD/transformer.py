"""
手写 Decoder-Only Transformer (Llama 风格) — HARD 版
=====================================================
目标: 单文件跑通 forward → 训练一步 → 自回归生成

手写部分: RMSNorm · Safe Softmax · RoPE · Multi-Head Attention
         KV Cache · SwiGLU FFN · Cross-Entropy Loss
         Top-K/Top-P Sampling · BPE Tokenizer
调包部分: nn.Linear · nn.Embedding · F.silu · torch.matmul
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
    d_model:        int   = 512
    n_heads:        int   = 8
    n_layers:       int   = 4
    vocab_size:     int   = 1000
    max_seq_len:    int   = 128
    ffn_hidden_dim: int   = 2048
    norm_eps:       float = 1e-6


# ============================================================================
# 第 1 步: RMSNorm
# ============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm(x) = x / RMS(x) × weight
    RMS(x) = sqrt( mean(x²) + ε )
    输入输出: (B, S, D) → (B, S, D)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO 1-a: 可学习缩放参数，(d_model,)
        raise NotImplementedError("TODO 1-a")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1-b/c: 计算 RMS → 归一化 → 缩放
        raise NotImplementedError("TODO 1-b/c")


# ============================================================================
# 第 2 步: Safe Softmax
# ============================================================================
def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定 Softmax.

    softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    """
    # TODO 2: 减最大值 → exp → 归一化
    raise NotImplementedError("TODO 2")


# ============================================================================
# 第 3 步: Rotary Positional Embedding (RoPE)
# ============================================================================
class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码，只作用于 Q 和 K。

    inv_freq_i = 1 / 10000^(2i/d_k)
    freqs[t,i] = t × inv_freq_i
    x_rot = x × cos(freqs) + rotate_half(x) × sin(freqs)
    """

    def __init__(self, d_k: int, max_seq_len: int):
        super().__init__()
        # TODO 3-a/b: 计算 inv_freq，预计算 cos/sin 缓存并 register_buffer
        # cos/sin 缓存 shape: (max_seq_len, d_k//2)
        raise NotImplementedError("TODO 3-a/b")

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """(a, b) → (−b, a)，对最后一维操作"""
        # TODO 3-c
        raise NotImplementedError("TODO 3-c")

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        x:            (B, n_heads, S, d_k)
        position_ids: (S,)
        → (B, n_heads, S, d_k)
        """
        # TODO 3-d: 索引缓存 → cat 拼到 d_k → unsqueeze 广播 → 应用旋转公式
        raise NotImplementedError("TODO 3-d")


# ============================================================================
# 第 4 步: Multi-Head Attention + KV Cache
# ============================================================================
class MultiHeadAttention(nn.Module):
    """Multi-Head Causal Self-Attention with KV Cache.

    数据流:
      x [B,S,D]
        → wq/wk/wv → view(B,S,H,d_k).T → [B,H,S,d_k]
        → RoPE(Q,K) [V 不做 RoPE]
        → cat(past_K, K, dim=2)  [KV Cache]
        → QKᵀ/√d_k → causal mask → softmax → AV
        → transpose → view(B,S,D) → wo
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k     = config.d_model // config.n_heads
        self.d_model = config.d_model
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
        x: (B, S, D)  → output: (B, S_q, D),  present_kv: (K, V)
        """
        B, S, D = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # TODO 4-b: reshape → (B, n_heads, S, d_k)
        # TODO 4-c: RoPE(Q, K)，V 不做
        # TODO 4-d: KV Cache — dim=2 拼接，记录 present_kv
        # TODO 4-e: scores = QKᵀ / √d_k
        # TODO 4-f: 因果掩码，S_q > 1 时才需要（triu，取末 S_q 行）
        # TODO 4-g: safe_softmax → attn_weights
        # TODO 4-h: attn_weights @ V
        # TODO 4-i: concat 多头 → wo → return output, present_kv
        raise NotImplementedError("TODO 4: 实现 MultiHeadAttention forward")


# ============================================================================
# 第 5 步: SwiGLU FFN
# ============================================================================
class SwiGLU_FFN(nn.Module):
    """FFN(x) = W_down × (SiLU(W_gate × x) ⊙ W_up × x)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO 5-a: w_gate, w_up: D→ffn_dim; w_down: ffn_dim→D; 均无 bias
        raise NotImplementedError("TODO 5-a")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 5-b
        raise NotImplementedError("TODO 5-b")


# ============================================================================
# 第 6 步: Transformer Block
# ============================================================================
class TransformerBlock(nn.Module):
    """Pre-Norm Decoder Block.

    h = x + MHA( RMSNorm(x) )
    o = h + FFN( RMSNorm(h) )
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO 6-a: attn_norm, attn, ffn_norm, ffn
        raise NotImplementedError("TODO 6-a")

    def forward(
        self,
        x:            torch.Tensor,
        position_ids: torch.Tensor,
        past_kv:      Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO 6-b: Pre-Norm + MHA + 残差
        # TODO 6-c: Pre-Norm + FFN + 残差，return output, present_kv
        raise NotImplementedError("TODO 6-b/c")


# ============================================================================
# 第 7 步: 完整 Transformer
# ============================================================================
class Transformer(nn.Module):
    """Decoder-Only Transformer.  (B, S) → (B, S, vocab_size)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO 7-a: token_embedding, layers, final_norm, lm_head
        raise NotImplementedError("TODO 7-a")

    def forward(
        self,
        input_ids:       torch.Tensor,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        数据流:
          input_ids → embedding → [Block × L] → RMSNorm → lm_head → logits
          position_ids 从 past_len 开始，支持 KV Cache 推理
        """
        # TODO 7-b: embedding → position_ids → 层循环 → final_norm → lm_head
        raise NotImplementedError("TODO 7-b")


# ============================================================================
# 第 8 步: Cross-Entropy Loss
# ============================================================================
def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """自回归 CE Loss，内部自动 shift。

    L = -mean( log p(x_{t+1} | x_{≤t}) )
    log_softmax 用 log-sum-exp 技巧保证数值稳定。

    Args:
        logits:   (B, S, V)
        input_ids:(B, S)   — 内部 shift，无需手动处理 targets
    """
    # TODO 8-a: logits[:,:-1,:] 对齐 input_ids[:,1:]
    # TODO 8-b: view(-1, V) 展平
    # TODO 8-c: x_max 减去 → log Σ exp → log_probs
    # TODO 8-d: gather target → 取负均值
    raise NotImplementedError("TODO 8")


# ============================================================================
# 第 9 步: Top-K / Top-P Sampling
# ============================================================================
def sample_top_k_top_p(
    logits:      torch.Tensor,
    top_k:       int   = 50,
    top_p:       float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    temperature → Top-K → Top-P → multinomial
    返回 (B, 1)
    """
    # TODO 9-a: / temperature
    # TODO 9-b: topk 界标值，低于界标的置 -inf
    # TODO 9-c: 降序 cumsum；累积 > p 的置 -inf，注意右移一位保留边界 token
    # TODO 9-d: softmax → multinomial
    raise NotImplementedError("TODO 9")


# ============================================================================
# 第 10 步: BPE Tokenizer
# ============================================================================
class BPETokenizer:
    """Byte-Pair Encoding — 从 256 字节出发，反复合并最高频相邻对。"""

    def __init__(self):
        self.vocab:  dict = {}   # id → bytes
        self.merges: dict = {}   # (id1, id2) → new_id

    def train(self, text: str, vocab_size: int):
        # TODO 10-a: ids = UTF-8 字节列表；vocab = {i: bytes([i]) for i in range(256)}
        # TODO 10-b: 循环 vocab_size-256 次：统计相邻对 → 合并最高频 → 更新 merges/vocab
        raise NotImplementedError("TODO 10-a/b")

    def encode(self, text: str) -> List[int]:
        # TODO 10-c: UTF-8 字节 → 反复按 merges 顺序合并，直到无可合并
        raise NotImplementedError("TODO 10-c")

    def decode(self, ids: List[int]) -> str:
        # TODO 10-d: vocab[id] → bytes → join → UTF-8 decode
        raise NotImplementedError("TODO 10-d")


# ============================================================================
# 第 11 步: 训练测试
# ============================================================================
def test_training_step(model: Transformer, config: ModelConfig):
    print("\n" + "=" * 60)
    print("测试一：训练模式 (forward → loss → backward)")
    print("=" * 60)
    model.train()

    # TODO 11-a: 构造 (B, S) input_ids
    # TODO 11-b: forward → compute_loss（不要用 no_grad）
    # TODO 11-c: zero_grad → backward → step
    raise NotImplementedError("TODO 11")


# ============================================================================
# 第 12 步: 推理测试
# ============================================================================
def test_generation_step(model: Transformer, config: ModelConfig):
    """Prefill: 整个 prompt 一次 forward 获取 past_kv
    Decode:  每步输入单 token，携带 past_kv，避免重复计算
    """
    print("\n" + "=" * 60)
    print("测试二：推理生成模式 (Prefill + KV Cache Decode)")
    print("=" * 60)
    model.eval()

    corpus = ("transformers are beautiful and they transform NLP. "
              "transformer architecture is the backbone of LLMs.") * 50
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300)
    bpe_vocab_size = len(tokenizer.vocab)

    start_text = "transform"
    input_ids  = torch.tensor([tokenizer.encode(start_text)])
    max_new_tokens = 10

    with torch.no_grad():
        # TODO 12-a: Prefill — model(input_ids) → logits, past_kv → sample first token
        # TODO 12-b: Decode Loop — model(next_token, past_kv) × max_new_tokens-1
        # 注意: sample 时截断 logits[:, -1, :bpe_vocab_size] 防止 decode KeyError
        pass

    # TODO 12-c: tokenizer.decode → print
    raise NotImplementedError("TODO 12")


# ============================================================================
# Main
# ============================================================================
def main():
    config = ModelConfig()
    model  = Transformer(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    test_training_step(model, config)
    test_generation_step(model, config)
    print("\n🎉 全部测试通过！")


if __name__ == "__main__":
    main()
