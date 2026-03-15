"""
RLHF 环境 — 对齐官方实现的手撕基础模块
==========================================
目标: 以 HuggingFace Transformers / TRL 的真实代码结构为蓝本，
     搭建一套可以直接用于手撕 PPO / DPO / GRPO 的 Mock 环境。

"真实做法" 的三个核心原则:
  1. Padding 在 DataLoader 的 collate_fn 里『动态』完成，而非写死在数据集里。
  2. 模型接受统一的 (input_ids, attention_mask)，一切遮蔽通过 mask 完成。
  3. log_probs 只有一个统一入口，用 labels (=-100) 或 action_mask 过滤无效位置。

本文件提供:
  ① MockTokenizer           — 模拟 tokenizer，接口对齐 HuggingFace
  ② MockCausalLM            — 模拟 Causal LM，forward 返回 logits [B, T, V]
  ③ RawPromptDataset        — PPO/GRPO 用，每条样本只有 prompt (string)
  ④ RawPreferenceDataset    — DPO 用，每条样本 (prompt, chosen, rejected)
  ⑤ ppo_collate_fn          — [TODO-A] PPO 数据收集器
  ⑥ dpo_collate_fn          — [TODO-B] DPO 数据收集器
  ⑦ MockPolicyModel         — 封装 MockCausalLM，提供 generate() 和 [TODO-C] get_log_probs()

学习者需要完成的 TODO:
  [TODO-A] ppo_collate_fn
  [TODO-B] dpo_collate_fn
  [TODO-C] MockPolicyModel.get_log_probs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 0. 全局配置
# ==============================================================================
@dataclass
class RLHFConfig:
    """全局超参数，三个算法共用"""
    vocab_size:    int   = 50
    hidden_dim:    int   = 64
    pad_token_id:  int   = 0
    eos_token_id:  int   = 1
    bos_token_id:  int   = 2
    temperature:   float = 1.0
    max_new_tokens: int  = 15
    batch_size:    int   = 4
    lr:            float = 1e-4


# ==============================================================================
# 1. MockTokenizer — 对齐 HuggingFace Tokenizer 接口
# ==============================================================================
class MockTokenizer:
    """
    极简 word-level tokenizer，接口模拟 HuggingFace AutoTokenizer。

    关键属性 (与 HuggingFace 一致):
        pad_token_id, eos_token_id, bos_token_id
        padding_side: 'left' | 'right'  ← 由调用者在 collate_fn 里按需设置

    关键方法:
        __call__(texts, padding, truncation, max_length, return_tensors)
            → {'input_ids': Tensor, 'attention_mask': Tensor}
        encode(text) → List[int]
        decode(ids)  → str
        batch_decode(ids_list) → List[str]
    """
    def __init__(self, vocab_size: int = 50):
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
        self.vocab_size    = min(vocab_size, len(base_words))
        self.word2id       = {w: i for i, w in enumerate(base_words[:self.vocab_size])}
        self.id2word       = {i: w for w, i in self.word2id.items()}
        self.pad_token_id  = 0
        self.eos_token_id  = 1
        self.bos_token_id  = 2
        self.padding_side  = "right"

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for w in text.lower().split():
            ids.append(self.word2id.get(w, 3))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        for i in ids:
            if i == self.pad_token_id and skip_special_tokens:
                continue
            if i == self.eos_token_id:
                if not skip_special_tokens:
                    words.append("<eos>")
                break
            words.append(self.id2word.get(i, "<unk>"))
        return " ".join(words)

    def batch_decode(self, ids_list, skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]

    def __call__(self,
                 texts: List[str],
                 padding: bool = True,
                 truncation: bool = True,
                 max_length: int = 64,
                 add_bos: bool = True,
                 add_eos: bool = False,
                 return_tensors: str = "pt"
                 ) -> Dict[str, torch.Tensor]:
        """
        批量编码，返回 {'input_ids': Tensor[B,T], 'attention_mask': Tensor[B,T]}。
        padding_side 由 self.padding_side 控制 ('left' 或 'right')。
        """
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        if truncation:
            encoded = [e[:max_length] for e in encoded]

        if not padding:
            return {"input_ids": encoded, "attention_mask": [[1]*len(e) for e in encoded]}

        max_len = max(len(e) for e in encoded)
        input_ids_list  = []
        attention_masks = []

        for e in encoded:
            pad_len = max_len - len(e)
            mask    = [1] * len(e) + [0] * pad_len

            if self.padding_side == "right":
                padded = e + [self.pad_token_id] * pad_len
            else:
                padded = [self.pad_token_id] * pad_len + e
                mask   = [0] * pad_len + [1] * len(e)

            input_ids_list.append(padded)
            attention_masks.append(mask)

        return {
            "input_ids":      torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


# ==============================================================================
# 2. Dataset — 原始数据，不含任何 Tensor/Padding
# ==============================================================================
class RawPromptDataset(Dataset):
    """PPO / GRPO 使用。Dataset 里存原始字符串，padding 在 collate_fn 里动态完成。"""
    def __init__(self):
        self.prompts: List[str] = [
            "the question is",
            "I want to know",
            "can you answer",
            "this is a question",
            "I think the answer",
            "do you know what",
            "please tell me",
            "what is the",
        ]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> Dict[str, str]:
        return {"prompt": self.prompts[idx]}


class RawPreferenceDataset(Dataset):
    """DPO 使用。每条数据是 (prompt, chosen, rejected) 字符串三元组。"""
    def __init__(self):
        self.data = [
            {
                "prompt":   "the question is",
                "chosen":   "the answer is good helpful",
                "rejected": "the answer is bad harmful",
            },
            {
                "prompt":   "I want to know",
                "chosen":   "I think this is great safe",
                "rejected": "I think this is wrong unsafe",
            },
            {
                "prompt":   "can you answer",
                "chosen":   "yes I can this is correct",
                "rejected": "no I can not very bad",
            },
            {
                "prompt":   "this is a question",
                "chosen":   "the answer is nice helpful great",
                "rejected": "the answer is bad wrong harmful",
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, str]:
        return self.data[idx]


# ==============================================================================
# 3. collate_fn — 动态 Padding (核心 TODO)
# ==============================================================================

def ppo_collate_fn(tokenizer: MockTokenizer, config: RLHFConfig):
    """
    工厂函数：返回一个 PPO/GRPO 专用的 collate_fn。

    数据流:
      List[{'prompt': str}]  →  {'input_ids': [B, T_p], 'attention_mask': [B, T_p]}

    ====================================================================
    [TODO-A] 实现 ppo_collate_fn

    为什么 prompt 必须左填充？
      Causal LM 的 generate() 从序列最右侧的有效 token 开始续写。
      若右填充，最右边是 <pad>，生成会立即崩坏。

    提示:
      tokenizer.padding_side = ?
      tokenizer(texts, padding=True, return_tensors='pt')
    ====================================================================
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("[TODO-A] 请实现 ppo_collate_fn")
    return collate


def dpo_collate_fn(tokenizer: MockTokenizer, config: RLHFConfig):
    """
    工厂函数：返回一个 DPO 专用的 collate_fn。

    数据流:
      List[{'prompt': str, 'chosen': str, 'rejected': str}]
      →  {
           'chosen_input_ids':      [B, T_c],
           'chosen_attention_mask': [B, T_c],
           'chosen_labels':         [B, T_c],
           'rejected_input_ids':    [B, T_r],
           'rejected_attention_mask': [B, T_r],
           'rejected_labels':       [B, T_r],
         }

    ====================================================================
    [TODO-B] 实现 dpo_collate_fn

    构建原则 (来自 TRL DPOTrainer._tokenize_row):
      input_ids  = [BOS, p_1..p_n, r_1..r_m, EOS]
      labels     = [-100 × (n+1), r_1..r_m, EOS]   ← prompt 部分不计 loss
      pad 位置的 labels 也设为 -100

    填充方向: 右填充 (chosen 和 rejected 各自对齐各自批次内最长)
    ====================================================================
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("[TODO-B] 请实现 dpo_collate_fn")
    return collate


# ==============================================================================
# 4. MockCausalLM — 模拟 LLM 主体，学习者不需要修改这里
# ==============================================================================
class MockCausalLM(nn.Module):
    """
    模拟 Causal Language Model。
    接口对齐 HuggingFace AutoModelForCausalLM:
        forward(input_ids, attention_mask) → logits [B, T, V]
    """
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                      padding_idx=config.pad_token_id)
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        x = self.embedding(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


# ==============================================================================
# 5. MockPolicyModel — 策略模型封装
# ==============================================================================
class MockPolicyModel(nn.Module):
    """
    策略模型 (可训练)。封装 MockCausalLM，提供:
      - forward(input_ids, attention_mask) → logits [B, T, V]
      - generate(...)  → response_ids [B, T_r]   (已实现)
      - get_log_probs(...)                         [TODO-C]
    """
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.lm = MockCausalLM(config)

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        return self.lm(input_ids, attention_mask)

    @torch.no_grad()
    def generate(self,
                 input_ids:      torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 15,
                 temperature:    float = 1.0,
                 do_sample:      bool = True
                 ) -> torch.Tensor:
        """
        自回归生成 (已实现)。接口对齐 HuggingFace model.generate()。
        Returns:
            response_ids: [B, T_r]  只含新生成的 token，EOS 之后填 <pad>
        """
        self.eval()
        B = input_ids.shape[0]
        cur_ids  = input_ids.clone()
        cur_mask = attention_mask.clone() if attention_mask is not None \
                   else torch.ones_like(input_ids)

        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        response_tokens: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            logits      = self.lm(cur_ids, cur_mask)
            next_logits = logits[:, -1, :]

            if do_sample:
                next_logits = next_logits / max(temperature, 1e-8)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = next_logits.argmax(dim=-1)

            next_token = torch.where(finished,
                                     torch.full_like(next_token, self.config.pad_token_id),
                                     next_token)
            response_tokens.append(next_token)

            cur_ids  = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
            new_mask = (~finished).long().unsqueeze(1)
            cur_mask = torch.cat([cur_mask, new_mask], dim=1)

            finished = finished | (next_token == self.config.eos_token_id)
            if finished.all():
                break

        self.train()
        return torch.stack(response_tokens, dim=1)

    # ── get_log_probs — [TODO-C] ────────────────────────────────────────────
    def get_log_probs(self,
                      input_ids:      torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels:         Optional[torch.Tensor] = None,
                      action_mask:    Optional[torch.Tensor] = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一的 log probability 入口 (PPO 和 DPO 共用)。

        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]
            labels:         [B, T]   DPO 场景，-100 表示忽略位置
            action_mask:    [B, T_r] PPO 场景，仅覆盖 response 部分

        Returns:
            per_token_log_probs: [B, T-1]   无效位置清零
            loss_mask:           [B, T-1]   1=需计算, 0=忽略

        =====================================================================
        [TODO-C] 请实现 get_log_probs

        Causal LM log prob:
          log p(x_t | x_{<t})  对所有 t ∈ [1, T-1]

        关键操作:
          shift: logits[:, :-1, :] 预测 input_ids[:, 1:]
          loss_mask 二选一:
            labels    → shift_labels[:, 1:] != -100
            action_mask → 左 pad (T_p-1) 个 0 对齐到 T-1 维度

          log_prob = log_softmax(shift_logits).gather(shift_targets)
          无效位置 × 0

        注意: 不在此处做序列级 sum/mean
        =====================================================================
        """
        raise NotImplementedError("[TODO-C] 请实现 get_log_probs")


# ==============================================================================
# 6. MockReferenceModel — 冻结的参考模型
# ==============================================================================
class MockReferenceModel(nn.Module):
    """参考模型，参数冻结。通常是 SFT checkpoint 的深拷贝。"""
    def __init__(self, policy_model: MockPolicyModel):
        super().__init__()
        self.model = copy.deepcopy(policy_model)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    @torch.no_grad()
    def get_log_probs(self,
                      input_ids:      torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels:         Optional[torch.Tensor] = None,
                      action_mask:    Optional[torch.Tensor] = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.get_log_probs(input_ids, attention_mask, labels, action_mask)


# ==============================================================================
# 7. MockRewardModel — 奖励模型
# ==============================================================================
class MockRewardModel(nn.Module):
    """奖励模型。输入完整序列，输出标量 reward [B]。"""
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                      padding_idx=config.pad_token_id)
        self.fc = nn.Linear(config.hidden_dim, 1)
        self.positive_ids = {10, 12, 13, 14, 43, 44}
        self.negative_ids = {11, 15, 45, 46}

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        else:
            x = x.mean(dim=1)
        reward = self.fc(x).squeeze(-1)

        with torch.no_grad():
            for b in range(input_ids.shape[0]):
                token_set = set(input_ids[b].tolist())
                bonus = 0.5 * len(token_set & self.positive_ids) \
                      - 0.5 * len(token_set & self.negative_ids)
                reward[b] = reward[b] + bonus

        return reward

    @torch.no_grad()
    def score(self,
              prompt_ids:    torch.Tensor,
              prompt_mask:   torch.Tensor,
              response_ids:  torch.Tensor,
              response_mask: torch.Tensor) -> torch.Tensor:
        input_ids      = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        return self.forward(input_ids, attention_mask)


# ==============================================================================
# 8. MockValueModel — PPO 专用价值函数
# ==============================================================================
class MockValueModel(nn.Module):
    """价值函数 V(s_t)，PPO 专用。输出 [B, T]。"""
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                      padding_idx=config.pad_token_id)
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 1)

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ==============================================================================
# 9. DataLoader 构建辅助函数
# ==============================================================================
def make_ppo_dataloader(config: RLHFConfig, tokenizer: MockTokenizer) -> DataLoader:
    dataset = RawPromptDataset()
    collate = ppo_collate_fn(tokenizer, config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)


def make_dpo_dataloader(config: RLHFConfig, tokenizer: MockTokenizer) -> DataLoader:
    dataset = RawPreferenceDataset()
    collate = dpo_collate_fn(tokenizer, config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)


# ==============================================================================
# 10. 快速验证
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RLHF Env 快速自测 (仅验证框架结构，TODO 需要学习者完成)")
    print("=" * 60)

    config    = RLHFConfig()
    tokenizer = MockTokenizer(config.vocab_size)

    texts = ["the question is", "I want to know what"]
    tokenizer.padding_side = "left"
    out = tokenizer(texts, padding=True, return_tensors="pt")
    print(f"\n[Tokenizer - left pad]")
    print(f"  input_ids:      {out['input_ids']}")
    print(f"  attention_mask: {out['attention_mask']}")

    tokenizer.padding_side = "right"
    out = tokenizer(texts, padding=True, return_tensors="pt")
    print(f"\n[Tokenizer - right pad]")
    print(f"  input_ids:      {out['input_ids']}")
    print(f"  attention_mask: {out['attention_mask']}")

    ppo_ds = RawPromptDataset()
    dpo_ds = RawPreferenceDataset()
    print(f"\n[Dataset] PPO prompts: {len(ppo_ds)}, DPO pairs: {len(dpo_ds)}")
    print(f"  sample PPO: {ppo_ds[0]}")
    print(f"  sample DPO: {dpo_ds[0]}")

    policy = MockPolicyModel(config)
    ref    = MockReferenceModel(policy)
    reward = MockRewardModel(config)
    value  = MockValueModel(config)

    dummy_ids  = torch.randint(2, config.vocab_size, (2, 8))
    dummy_mask = torch.ones(2, 8, dtype=torch.long)
    logits = policy(dummy_ids, dummy_mask)
    print(f"\n[PolicyModel] forward logits shape: {logits.shape}")

    response = policy.generate(dummy_ids, dummy_mask, max_new_tokens=5)
    print(f"[PolicyModel] generate response shape: {response.shape}")

    r = reward.score(dummy_ids, dummy_mask, response, torch.ones_like(response))
    print(f"[RewardModel] score shape: {r.shape}")

    print("\n✅ 框架结构正常。请完成 [TODO-A] [TODO-B] [TODO-C] 后继续 PPO/DPO/GRPO 练习。")
