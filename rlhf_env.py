"""
RLHF Environment: Mock Models & Utilities
==========================================
为 PPO / DPO / GRPO 三个练习提供统一的模拟环境。
所有 "模型" 均用简单张量运算模拟，不需要真正训练。

核心组件:
  1. MockTokenizer       — 模拟 tokenizer (word-level)
  2. MockPolicyModel     — 模拟策略模型 (可训练参数)
  3. MockReferenceModel  — 冻结的参考模型
  4. MockRewardModel     — 模拟奖励模型 (输出标量 reward)
  5. MockValueModel      — 模拟价值函数模型 (PPO 用)
  6. 辅助函数: sample_from_policy, generate_responses, build_prompt_batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ==============================================================================
# 0. 超参数 & 配置
# ==============================================================================
@dataclass
class RLHFConfig:
    """全局超参数，三个算法共用"""
    vocab_size: int = 50          # 简化词表大小
    hidden_dim: int = 64          # 隐层维度
    max_seq_len: int = 20         # 最大序列长度
    pad_token_id: int = 0         # PAD token
    eos_token_id: int = 1         # EOS token
    bos_token_id: int = 2         # BOS token
    temperature: float = 1.0      # 采样温度
    top_k: int = 0                # top-k 采样 (0 = 不使用)
    batch_size: int = 4           # 每批 prompt 数
    lr: float = 1e-4              # 学习率


# ==============================================================================
# 1. MockTokenizer
# ==============================================================================
class MockTokenizer:
    """
    极简分词器：把句子按空格分词，维护一个 word→id 映射表。
    """
    def __init__(self, vocab_size: int = 50):
        # 构建一个简单词表
        base_words = [
            "<pad>", "<eos>", "<bos>",
            "the", "a", "is", "are", "was", "to", "of",
            "good", "bad", "great", "nice", "helpful", "harmful",
            "answer", "question", "AI", "model", "human",
            "I", "you", "we", "it", "this", "that",
            "can", "will", "do", "not", "very", "much",
            "think", "know", "like", "want", "need",
            "yes", "no", "hello", "world", "thank",
            "safe", "unsafe", "correct", "wrong", "true", "false",
        ]
        self.vocab_size = min(vocab_size, len(base_words))
        self.word2id = {w: i for i, w in enumerate(base_words[:self.vocab_size])}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(2)  # <bos>
        for word in text.lower().split():
            tokens.append(self.word2id.get(word, 3))  # 未知词映射到 'the'
        return tokens

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == 0:
                continue  # skip <pad>
            if i == 1:
                break      # stop at <eos>
            words.append(self.id2word.get(i, "<unk>"))
        return " ".join(words)


# ==============================================================================
# 2. MockPolicyModel (可训练)
# ==============================================================================
class MockPolicyModel(nn.Module):
    """
    模拟策略模型。
    输入: token_ids [B, T]
    输出: logits   [B, T, V]   — 每个位置预测下一个 token 的未归一化分数
    
    内部只是一个 Embedding + 简单线性层，不是真正的 Transformer，
    但接口和输出形状与真实 LLM 一致。
    """
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                       padding_idx=config.pad_token_id)
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]  token id 序列
        Returns:
            logits: [B, T, V]  每个位置的 next-token logits
        """
        x = self.embedding(input_ids)           # [B, T, H]
        x = F.relu(self.fc1(x))                 # [B, T, H]
        logits = self.fc2(x)                    # [B, T, V]
        return logits

    def get_log_probs(self, input_ids: torch.Tensor, 
                      response_ids: torch.Tensor) -> torch.Tensor:
        """
        计算 response 每个 token 的 log probability。
        
        Args:
            input_ids:    [B, T_prompt]     prompt token ids
            response_ids: [B, T_response]   response token ids
        Returns:
            log_probs: [B, T_response]  每个 response token 的 log prob
        """
        # 拼接 prompt + response
        full_ids = torch.cat([input_ids, response_ids], dim=1)  # [B, T_all]
        logits = self.forward(full_ids)  # [B, T_all, V]
        
        # 取 response 部分对应位置的 logits
        # logits[:, T_prompt-1 : T_all-1, :] 预测 response_ids
        T_prompt = input_ids.shape[1]
        response_logits = logits[:, T_prompt - 1 : -1, :]  # [B, T_response, V]
        
        # log_softmax 然后 gather
        log_probs_all = F.log_softmax(response_logits, dim=-1)  # [B, T_response, V]
        log_probs = log_probs_all.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, T_response]
        
        return log_probs


# ==============================================================================
# 3. MockReferenceModel (冻结)
# ==============================================================================
class MockReferenceModel(nn.Module):
    """
    参考模型，结构与 PolicyModel 完全一致，但参数冻结。
    通常是 SFT 后的初始模型副本。
    """
    def __init__(self, policy_model: MockPolicyModel):
        super().__init__()
        # 深拷贝策略模型的参数
        import copy
        self.model = copy.deepcopy(policy_model)
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def get_log_probs(self, input_ids: torch.Tensor,
                      response_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.get_log_probs(input_ids, response_ids)


# ==============================================================================
# 4. MockRewardModel
# ==============================================================================
class MockRewardModel(nn.Module):
    """
    模拟奖励模型。
    输入: prompt + response 的 token_ids
    输出: 标量 reward (越高越好)
    
    这里用简单规则模拟奖励：
      - 包含 "good", "great", "helpful" 等正面词 → 高奖励
      - 包含 "bad", "harmful", "unsafe" 等负面词 → 低奖励
      - 加上一些学习到的偏好 (线性层)
    """
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                       padding_idx=config.pad_token_id)
        self.fc = nn.Linear(config.hidden_dim, 1)
        
        # 正面/负面词 id (基于 MockTokenizer 的词表)
        self.positive_ids = {10, 12, 13, 14, 43, 44}  # good, great, nice, helpful, safe, correct
        self.negative_ids = {11, 15, 45, 46}           # bad, harmful, unsafe, wrong

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]  完整序列 (prompt + response)
        Returns:
            rewards: [B]  每个样本的标量奖励
        """
        x = self.embedding(input_ids)       # [B, T, H]
        x = x.mean(dim=1)                   # [B, H]  全局平均池化
        reward = self.fc(x).squeeze(-1)     # [B]
        
        # 加上基于规则的奖励 bonus
        with torch.no_grad():
            for b in range(input_ids.shape[0]):
                token_set = set(input_ids[b].tolist())
                bonus = 0.0
                bonus += 0.5 * len(token_set & self.positive_ids)
                bonus -= 0.5 * len(token_set & self.negative_ids)
                reward[b] = reward[b] + bonus
        
        return reward

    def get_reward(self, prompt_ids: torch.Tensor, 
                   response_ids: torch.Tensor) -> torch.Tensor:
        """便捷方法：分别传入 prompt 和 response"""
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        with torch.no_grad():
            return self.forward(full_ids)


# ==============================================================================
# 5. MockValueModel (PPO 专用)
# ==============================================================================
class MockValueModel(nn.Module):
    """
    价值函数模型 V(s)。
    输入: 当前已生成的 token 序列
    输出: 每个时间步的状态价值估计
    """
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                       padding_idx=config.pad_token_id)
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]
        Returns:
            values: [B, T]  每个位置的状态价值
        """
        x = self.embedding(input_ids)
        x = F.relu(self.fc1(x))
        values = self.fc2(x).squeeze(-1)    # [B, T]
        return values


# ==============================================================================
# 6. 采样 & 生成工具函数
# ==============================================================================
def sample_from_logits(logits: torch.Tensor, 
                       temperature: float = 1.0,
                       top_k: int = 0) -> torch.Tensor:
    """
    从 logits 采样一个 token。
    
    Args:
        logits: [B, V] 或 [V]
        temperature: 采样温度
        top_k: top-k 过滤 (0 = 全词表采样)
    Returns:
        sampled token ids: [B] 或 []
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        # 只保留 top-k 的 logits，其余设为 -inf
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_val = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, 
                             torch.full_like(logits, float('-inf')), 
                             logits)

    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampled


def generate_response(policy_model: MockPolicyModel,
                      prompt_ids: torch.Tensor,
                      max_new_tokens: int = 15,
                      temperature: float = 1.0,
                      top_k: int = 0,
                      eos_token_id: int = 1) -> torch.Tensor:
    """
    自回归生成 response tokens。
    
    Args:
        policy_model: 策略模型
        prompt_ids: [B, T_prompt]
        max_new_tokens: 最多生成多少个新 token
        temperature, top_k: 采样参数
        eos_token_id: 终止 token
    Returns:
        response_ids: [B, T_response]  (不含 prompt)
    """
    policy_model.eval()
    B = prompt_ids.shape[0]
    generated = prompt_ids.clone()
    response_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = policy_model(generated)         # [B, T, V]
            next_logits = logits[:, -1, :]           # [B, V]
            next_token = sample_from_logits(next_logits, temperature, top_k)  # [B]
            response_tokens.append(next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            # 检查是否所有序列都生成了 EOS
            if (next_token == eos_token_id).all():
                break
    
    policy_model.train()
    response_ids = torch.stack(response_tokens, dim=1)  # [B, T_response]
    return response_ids


# ==============================================================================
# 7. 构建 Prompt Batch
# ==============================================================================
def build_prompt_batch(tokenizer: MockTokenizer, 
                       config: RLHFConfig) -> torch.Tensor:
    """
    生成一批模拟 prompt，用于采样。
    
    Returns:
        prompt_ids: [B, T_prompt]  已 padding 对齐
    """
    prompts = [
        "the question is",
        "I want to know",
        "can you answer",
        "this is a question",
    ]
    
    batch_size = min(config.batch_size, len(prompts))
    encoded = [tokenizer.encode(p) for p in prompts[:batch_size]]
    
    # Padding 到相同长度
    max_len = max(len(e) for e in encoded)
    padded = []
    for e in encoded:
        padded.append(e + [config.pad_token_id] * (max_len - len(e)))
    
    return torch.tensor(padded, dtype=torch.long)


def build_preference_pairs(tokenizer: MockTokenizer,
                           config: RLHFConfig) -> Dict[str, torch.Tensor]:
    """
    为 DPO 构建偏好对 (chosen / rejected)。
    
    Returns:
        dict with keys:
            'prompt_ids':   [B, T_prompt]
            'chosen_ids':   [B, T_resp]
            'rejected_ids': [B, T_resp]
    """
    data = [
        {
            "prompt": "the question is",
            "chosen": "the answer is good helpful",
            "rejected": "the answer is bad harmful",
        },
        {
            "prompt": "I want to know",
            "chosen": "I think this is great safe",
            "rejected": "I think this is wrong unsafe",
        },
        {
            "prompt": "can you answer",
            "chosen": "yes I can this is correct",
            "rejected": "no I can not very bad",
        },
        {
            "prompt": "this is a question",
            "chosen": "the answer is nice helpful great",
            "rejected": "the answer is bad wrong harmful",
        },
    ]
    
    batch_size = min(config.batch_size, len(data))
    data = data[:batch_size]
    
    def pad_sequences(sequences, pad_id):
        max_len = max(len(s) for s in sequences)
        return [s + [pad_id] * (max_len - len(s)) for s in sequences]
    
    prompts = [tokenizer.encode(d["prompt"]) for d in data]
    chosens = [tokenizer.encode(d["chosen"], add_bos=False) for d in data]
    rejecteds = [tokenizer.encode(d["rejected"], add_bos=False) for d in data]
    
    prompts = pad_sequences(prompts, config.pad_token_id)
    chosens = pad_sequences(chosens, config.pad_token_id)
    rejecteds = pad_sequences(rejecteds, config.pad_token_id)
    
    return {
        'prompt_ids': torch.tensor(prompts, dtype=torch.long),
        'chosen_ids': torch.tensor(chosens, dtype=torch.long),
        'rejected_ids': torch.tensor(rejecteds, dtype=torch.long),
    }


# ==============================================================================
# 8. 测试 & 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RLHF Environment 测试")
    print("=" * 60)
    
    config = RLHFConfig()
    tokenizer = MockTokenizer(config.vocab_size)
    
    # 创建模型
    policy = MockPolicyModel(config)
    ref_model = MockReferenceModel(policy)
    reward_model = MockRewardModel(config)
    value_model = MockValueModel(config)
    
    # 编码测试
    prompt = "the question is"
    ids = tokenizer.encode(prompt)
    print(f"\n[Tokenizer] '{prompt}' → {ids}")
    print(f"[Tokenizer] decode: '{tokenizer.decode(ids)}'")
    
    # 生成测试
    prompt_batch = build_prompt_batch(tokenizer, config)
    print(f"\n[Prompt Batch] shape: {prompt_batch.shape}")
    
    response = generate_response(policy, prompt_batch, max_new_tokens=8)
    print(f"[Response] shape: {response.shape}")
    for i in range(prompt_batch.shape[0]):
        p = tokenizer.decode(prompt_batch[i].tolist())
        r = tokenizer.decode(response[i].tolist())
        print(f"  Prompt: '{p}' → Response: '{r}'")
    
    # Reward 测试
    rewards = reward_model.get_reward(prompt_batch, response)
    print(f"\n[Rewards] {rewards}")
    
    # Log Probs 测试
    log_probs = policy.get_log_probs(prompt_batch, response)
    print(f"[Log Probs] shape: {log_probs.shape}")
    
    ref_log_probs = ref_model.get_log_probs(prompt_batch, response)
    print(f"[Ref Log Probs] shape: {ref_log_probs.shape}")
    
    # Value 测试
    full_seq = torch.cat([prompt_batch, response], dim=1)
    values = value_model(full_seq)
    print(f"[Values] shape: {values.shape}")
    
    # DPO 偏好对测试
    pref = build_preference_pairs(tokenizer, config)
    print(f"\n[Preference Pairs]")
    print(f"  prompt_ids:   {pref['prompt_ids'].shape}")
    print(f"  chosen_ids:   {pref['chosen_ids'].shape}")
    print(f"  rejected_ids: {pref['rejected_ids'].shape}")
    
    print("\n✅ 环境测试通过！可以开始 PPO / DPO / GRPO 练习。")
