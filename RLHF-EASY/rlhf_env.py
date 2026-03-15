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
  ⑤ ppo_collate_fn          — [TODO-A] PPO 数据收集器，需实现『Prompt 左填充』
  ⑥ dpo_collate_fn          — [TODO-B] DPO 数据收集器，需实现『拼接 + 右填充 + labels』
  ⑦ MockPolicyModel         — 封装 MockCausalLM，提供 generate() 和 TODO get_log_probs()

学习者需要完成的 TODO:
  [TODO-A] ppo_collate_fn:        Prompt 左填充 + attention_mask
  [TODO-B] dpo_collate_fn:        chosen/rejected 拼接 + 右填充 + labels (-100) + attention_mask
  [TODO-C] MockPolicyModel.get_log_probs: 统一 log prob 入口 (用 labels 或 action_mask 屏蔽)
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
    vocab_size:    int   = 50      # 简化词表大小
    hidden_dim:    int   = 64      # 隐层维度
    pad_token_id:  int   = 0       # <pad>
    eos_token_id:  int   = 1       # <eos>
    bos_token_id:  int   = 2       # <bos>
    temperature:   float = 1.0
    max_new_tokens: int  = 15      # 生成时最大新 token 数
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
        self.padding_side  = "right"   # 调用者在 collate_fn 里按需临时修改

    # ── 底层 encode / decode ────────────────────────────────────────────────
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for w in text.lower().split():
            ids.append(self.word2id.get(w, 3))  # 未知词 → 'the'
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

    # ── 模拟 HuggingFace tokenizer.__call__ ─────────────────────────────────
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
            # 不 pad，直接返回（长度可能不一，无法转 Tensor）
            return {"input_ids": encoded, "attention_mask": [[1]*len(e) for e in encoded]}

        max_len = max(len(e) for e in encoded)
        input_ids_list  = []
        attention_masks = []

        for e in encoded:
            pad_len = max_len - len(e)
            mask    = [1] * len(e) + [0] * pad_len  # 先按右填充建 mask

            if self.padding_side == "right":
                padded = e + [self.pad_token_id] * pad_len
            else:  # left padding
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
    """
    PPO / GRPO 使用。每条数据只有一个 prompt 字符串。
    注意: Dataset 里存的是『原始字符串』，padding 在 collate_fn 里动态完成。
    """
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
    """
    DPO 使用。每条数据是 (prompt, chosen_response, rejected_response) 字符串三元组。
    注意: Dataset 里存的是『原始字符串』，padding/labels 在 collate_fn 里动态完成。
    """
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
# 3. collate_fn — 动态 Padding (核心 TODO，贴近官方 DataCollator)
# ==============================================================================

def ppo_collate_fn(tokenizer: MockTokenizer, config: RLHFConfig):
    """
    工厂函数：返回一个 PPO/GRPO 专用的 collate_fn。

    collate_fn 的作用 (等价于 HuggingFace DataCollatorWithPadding):
      - 输入: List[{'prompt': str}]  ← 一个 mini-batch 的原始样本
      - 输出: {'input_ids': [B, T_p], 'attention_mask': [B, T_p]}

    ====================================================================
    [TODO-A] 实现 ppo_collate_fn

    官方真实做法: Prompt 必须『左填充 (left padding)』，
    因为 Causal LM 的 generate() 是从最后一个有效 token 继续生成的。
    如果用右填充，最右边的 token 是 <pad>，生成会立即崩坏。

    步骤:
      1. 从 batch 里提取所有 prompt 字符串:
            texts = [item['prompt'] for item in batch]

      2. 把 tokenizer.padding_side 临时设为 'left'
            tokenizer.padding_side = 'left'

      3. 调用 tokenizer(texts, padding=True, return_tensors='pt')
         得到 input_ids [B, T_p] 和 attention_mask [B, T_p]

      4. 返回 {'input_ids': ..., 'attention_mask': ...}

    注意: attention_mask 中, 1 = 真实 token, 0 = <pad>
          在 Transformer attention 中，pad 位置的 attention score 设为 -inf，
          从而不影响其他位置的 representation。
    ====================================================================
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "[TODO-A] 请实现 ppo_collate_fn:\n"
            "  1. 提取 texts = [item['prompt'] for item in batch]\n"
            "  2. 设置 tokenizer.padding_side = 'left'\n"
            "  3. 调用 tokenizer(texts, ...) 获取 input_ids, attention_mask\n"
            "  4. 返回 dict"
        )
    return collate


def dpo_collate_fn(tokenizer: MockTokenizer, config: RLHFConfig):
    """
    工厂函数：返回一个 DPO 专用的 collate_fn。

    collate_fn 的作用 (等价于 TRL DPODataCollatorWithPadding):
      - 输入:  List[{'prompt': str, 'chosen': str, 'rejected': str}]
      - 输出:  {
                  'chosen_input_ids':      [B, T_c],
                  'chosen_attention_mask': [B, T_c],
                  'chosen_labels':         [B, T_c],   ← prompt 部分为 -100，pad 部分为 -100
                  'rejected_input_ids':      [B, T_r],
                  'rejected_attention_mask': [B, T_r],
                  'rejected_labels':         [B, T_r],
               }

    ====================================================================
    [TODO-B] 实现 dpo_collate_fn

    官方真实做法 (来自 TRL DPOTrainer._tokenize_row):

    Step 1 — 拼接 (Concatenate)
      对每个样本，把 prompt token ids 和 response token ids 拼在一起:
        [BOS, p1, p2, ..., r1, r2, ..., EOS]
      这是 DPO 训练时模型接受的完整序列。

    Step 2 — 构建 Labels (关键!)
      labels 是 input_ids 的一份拷贝，然后:
        a. 把所有 prompt 对应位置设为 -100  →  不计算 prompt 的 loss
        b. 把所有 <pad> 对应位置设为 -100   →  不计算 pad 的 loss
      PyTorch CrossEntropyLoss 的 ignore_index=-100 会自动跳过这些位置。

    Step 3 — 右填充 (Right Padding)
      DPO 使用静态数据，用右填充即可（不像 PPO 需要 left pad）。
      tokenizer.padding_side = 'right'
      对 chosen 序列按最长者对齐，attention_mask 中 <pad>=0, 有效token=1。

    具体步骤:
      for each item in batch:
        p_ids = tokenizer.encode(item['prompt'], add_bos=True, add_eos=False)
        c_ids = tokenizer.encode(item['chosen'],  add_bos=False, add_eos=True)
        r_ids = tokenizer.encode(item['rejected'], add_bos=False, add_eos=True)

        c_input = p_ids + c_ids                       # 拼接
        c_label = [-100]*len(p_ids) + c_ids           # prompt 部分 = -100
        r_input = p_ids + r_ids
        r_label = [-100]*len(p_ids) + r_ids

      然后对 chosen_inputs / rejected_inputs 分别做右填充:
        - pad <pad_token_id> 到最长
        - labels 的 pad 位置也要设为 -100

      最后构建 attention_mask (1=有效, 0=pad) 并返回所有 6 个张量。
    ====================================================================
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "[TODO-B] 请实现 dpo_collate_fn:\n"
            "  1. 拼接 prompt + chosen/rejected → input_ids\n"
            "  2. 构建 labels: prompt 部分 = -100, response 部分 = token_ids\n"
            "  3. 右填充对齐，pad 位置的 labels 也设为 -100\n"
            "  4. 构建 attention_mask (1=有效, 0=pad)\n"
            "  5. 返回 6 个 Tensor"
        )
    return collate


# ==============================================================================
# 4. MockCausalLM — 模拟 LLM 主体，学习者不需要修改这里
# ==============================================================================
class MockCausalLM(nn.Module):
    """
    模拟 Causal Language Model。
    接口对齐 HuggingFace AutoModelForCausalLM:
        forward(input_ids, attention_mask) → logits [B, T, V]

    attention_mask 中 0 的位置在 Embedding 之后被遮蔽 (× 0)，
    模拟真实 Transformer 中 attention 对 <pad> 的屏蔽效果。
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
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]   1=有效, 0=pad (optional)
        Returns:
            logits: [B, T, V]
        """
        x = self.embedding(input_ids)             # [B, T, H]
        if attention_mask is not None:
            # 用 mask 屏蔽 <pad> 位置的 hidden state，
            # 模拟 Transformer 中 padding 位置不参与 attention 的效果
            x = x * attention_mask.unsqueeze(-1).float()
        x = F.relu(self.fc1(x))                   # [B, T, H]
        logits = self.fc2(x)                       # [B, T, V]
        return logits


# ==============================================================================
# 5. MockPolicyModel — 策略模型封装，提供 generate() 和 get_log_probs() 接口
# ==============================================================================
class MockPolicyModel(nn.Module):
    """
    策略模型 (可训练)。封装 MockCausalLM，提供:
      - forward(input_ids, attention_mask) → logits
      - generate(input_ids, attention_mask, ...)  → response_ids [B, T_r]
      - get_log_probs(input_ids, attention_mask, labels)  → [TODO-C]

    接口设计对齐 HuggingFace AutoModelForCausalLM + GenerationMixin。
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

    # ── generate (已实现，不需要修改) ─────────────────────────────────────
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

        Args:
            input_ids:      [B, T_p]   左填充过的 prompt
            attention_mask: [B, T_p]   对应的 mask
            max_new_tokens: 最多生成多少个新 token
            temperature:    采样温度
            do_sample:      True=随机采样, False=greedy

        Returns:
            response_ids: [B, T_r]   只含新生成的 token (不含 prompt)
                          EOS 之后的位置自动填 <pad>

        注意: 这里返回的 response 可能长度不一 (有人早停)，
              实际 HuggingFace 会对整个批次取最长并右填充。
        """
        self.eval()
        B = input_ids.shape[0]
        cur_ids  = input_ids.clone()
        cur_mask = attention_mask.clone() if attention_mask is not None \
                   else torch.ones_like(input_ids)

        # 记录每个序列是否已经生成了 EOS
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        response_tokens: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            logits    = self.lm(cur_ids, cur_mask)  # [B, T, V]
            next_logits = logits[:, -1, :]           # [B, V]

            if do_sample:
                next_logits = next_logits / max(temperature, 1e-8)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            else:
                next_token = next_logits.argmax(dim=-1)  # [B]

            # 已结束的序列强制输出 <pad>
            next_token = torch.where(finished,
                                     torch.full_like(next_token, self.config.pad_token_id),
                                     next_token)
            response_tokens.append(next_token)

            # 更新 cur_ids / cur_mask
            cur_ids  = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
            new_mask = (~finished).long().unsqueeze(1)  # pad → 0
            cur_mask = torch.cat([cur_mask, new_mask], dim=1)

            # 检查 EOS
            finished = finished | (next_token == self.config.eos_token_id)
            if finished.all():
                break

        self.train()
        response_ids = torch.stack(response_tokens, dim=1)  # [B, T_r]
        return response_ids

    # ── get_log_probs — [TODO-C] ────────────────────────────────────────────
    def get_log_probs(self,
                      input_ids:      torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels:         Optional[torch.Tensor] = None,
                      action_mask:    Optional[torch.Tensor] = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一的 log probability 入口。

        这是整个 RLHF 框架里最关键的函数之一。
        真实的 TRL/OpenRLHF 内部只维护这『一个』函数，用于 PPO 和 DPO 两种场景。

        Args:
            input_ids:      [B, T]   完整序列 (对 PPO 是 [prompt+response]，对 DPO 是 [prompt+chosen/rejected])
            attention_mask: [B, T]   1=有效token, 0=<pad>
            labels:         [B, T]   (DPO 场景) prompt 部分=-100, pad 部分=-100, response 部分=token_id
                                     传入时，函数用 labels !=-100 构建 loss_mask
            action_mask:    [B, T_r] (PPO 场景) 只覆盖 response 部分，1=有效, 0=pad 或 EOS 后
                                     传入时，函数将其扩展对齐序列长度使用
                                     labels 和 action_mask 传其中一个即可。

        Returns:
            per_token_log_probs: [B, T-1]  每个位置的 log p(token_t | tokens_<t)
                                            无效位置(pad/prompt/后续pad)被置 0
            loss_mask:           [B, T-1]  1=有效 (需要计算), 0=忽略

        =====================================================================
        [TODO-C] 请实现统一的 get_log_probs

        官方真实做法 (参考 TRL DPOTrainer._forward 和 make_experience):

        Step 1 — 前向传播
            logits = self.forward(input_ids, attention_mask)  # [B, T, V]

        Step 2 — 错位 (Causal LM 的核心: 用前 T-1 个 token 预测后 T-1 个 token)
            shift_logits = logits[:, :-1, :]    # [B, T-1, V]
                           ↑ 位置 0..T-2 的 hidden state 用来预测
            shift_targets = input_ids[:, 1:]    # [B, T-1]
                           ↑ 位置 1..T-1 的真实 token id

        Step 3 — 构建 loss_mask (二选一)
          如果传入的是 labels:
            shift_labels = labels[:, 1:]        # [B, T-1]  (同样错位)
            loss_mask = (shift_labels != -100)  # True=有效, False=无效
            # 临时把 -100 替换为 0 (防止 gather 报越界错误)
            safe_targets = shift_labels.clone()
            safe_targets[~loss_mask] = 0

          如果传入的是 action_mask:
            # action_mask 形状通常是 [B, T_r]，需要对齐到 [B, T-1]
            # 方式: 在前面补 (T-1-T_r) 列的 0 (对应 prompt 部分)
            T_prompt = input_ids.shape[1] - action_mask.shape[1] - 1
            loss_mask = F.pad(action_mask, (T_prompt, 0), value=0)  # [B, T-1]
            safe_targets = shift_targets

        Step 4 — 计算 log_softmax 并 gather 真实 token 的对数概率
            log_probs_all = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
            per_token_log_probs = log_probs_all.gather(
                dim=-1, index=safe_targets.unsqueeze(-1)
            ).squeeze(-1)                                          # [B, T-1]

        Step 5 — 把无效位置清零
            per_token_log_probs = per_token_log_probs * loss_mask.float()

        Step 6 — 返回
            return per_token_log_probs, loss_mask

        提示:
          - loss_mask 的 dtype 可以是 bool 或 long，后续 .float() 转换即可
          - 不要在这里做序列级的 sum/mean，留给 Trainer 做（因为 PPO 和 DPO 聚合方式不同）
          - 注意 Step 3 中 shift_labels 的错位和 labels 的错位要一致
        =====================================================================
        """
        raise NotImplementedError(
            "[TODO-C] 请实现 MockPolicyModel.get_log_probs:\n"
            "  Step 1: forward → logits [B, T, V]\n"
            "  Step 2: 错位 → shift_logits [B, T-1, V], shift_targets [B, T-1]\n"
            "  Step 3: 用 labels 或 action_mask 构建 loss_mask\n"
            "  Step 4: log_softmax + gather → per_token_log_probs [B, T-1]\n"
            "  Step 5: 无效位置清零\n"
            "  Step 6: return per_token_log_probs, loss_mask"
        )


# ==============================================================================
# 6. MockReferenceModel — 冻结的参考模型（结构同 PolicyModel）
# ==============================================================================
class MockReferenceModel(nn.Module):
    """
    参考模型，参数冻结。通常是 SFT checkpoint 的深拷贝。
    接口与 MockPolicyModel 完全一致，所有方法都包裹在 torch.no_grad() 中。
    """
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
        """与 PolicyModel.get_log_probs 接口完全一致，但包裹在 no_grad() 中。"""
        return self.model.get_log_probs(input_ids, attention_mask, labels, action_mask)


# ==============================================================================
# 7. MockRewardModel — 奖励模型
# ==============================================================================
class MockRewardModel(nn.Module):
    """
    奖励模型。
    输入: 完整序列的 input_ids + attention_mask
    输出: 标量 reward [B]

    接口对齐 HuggingFace AutoModelForSequenceClassification (num_labels=1) 的习惯。
    """
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                      padding_idx=config.pad_token_id)
        self.fc = nn.Linear(config.hidden_dim, 1)
        # 正面/负面词集合（基于 MockTokenizer 词表位置）
        self.positive_ids = {10, 12, 13, 14, 43, 44}  # good, great, nice, helpful, safe, correct
        self.negative_ids = {11, 15, 45, 46}           # bad, harmful, unsafe, wrong

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]  (用于 mask 平均池化中的 pad 位置)
        Returns:
            rewards: [B]
        """
        x = self.embedding(input_ids)           # [B, T, H]
        if attention_mask is not None:
            # Masked mean pooling: 只对有效 token 求均值
            mask_f = attention_mask.unsqueeze(-1).float()   # [B, T, 1]
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        else:
            x = x.mean(dim=1)                   # [B, H]
        reward = self.fc(x).squeeze(-1)         # [B]

        with torch.no_grad():
            for b in range(input_ids.shape[0]):
                token_set = set(input_ids[b].tolist())
                bonus = 0.5 * len(token_set & self.positive_ids) \
                      - 0.5 * len(token_set & self.negative_ids)
                reward[b] = reward[b] + bonus

        return reward

    @torch.no_grad()
    def score(self,
              prompt_ids:      torch.Tensor,
              prompt_mask:     torch.Tensor,
              response_ids:    torch.Tensor,
              response_mask:   torch.Tensor) -> torch.Tensor:
        """
        PPO/GRPO 专用便捷方法: 分别传入 prompt 和 response 的 ids/mask，
        拼接后打分。

        Args:
            prompt_ids:    [B, T_p]
            prompt_mask:   [B, T_p]
            response_ids:  [B, T_r]
            response_mask: [B, T_r]
        Returns:
            rewards: [B]
        """
        input_ids      = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        return self.forward(input_ids, attention_mask)


# ==============================================================================
# 8. MockValueModel — PPO 专用价值函数 V(s)
# ==============================================================================
class MockValueModel(nn.Module):
    """
    价值函数 V(s_t)，PPO 专用。
    接口对齐 TRL AutoModelForCausalLMWithValueHead 的 value head 输出习惯。
    """
    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                      padding_idx=config.pad_token_id)
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 1)

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]
        Returns:
            values: [B, T]  每个时间步的状态价值估计
        """
        x = self.embedding(input_ids)              # [B, T, H]
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        x = F.relu(self.fc1(x))                    # [B, T, H]
        values = self.fc2(x).squeeze(-1)            # [B, T]
        return values


# ==============================================================================
# 9. DataLoader 构建辅助函数
# ==============================================================================
def make_ppo_dataloader(config: RLHFConfig,
                        tokenizer: MockTokenizer) -> DataLoader:
    """
    构建 PPO/GRPO 的 DataLoader。
    collate_fn 中完成左填充。
    """
    dataset = RawPromptDataset()
    collate = ppo_collate_fn(tokenizer, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )


def make_dpo_dataloader(config: RLHFConfig,
                        tokenizer: MockTokenizer) -> DataLoader:
    """
    构建 DPO 的 DataLoader。
    collate_fn 中完成拼接、右填充、labels 构建。
    """
    dataset = RawPreferenceDataset()
    collate = dpo_collate_fn(tokenizer, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )


# ==============================================================================
# 10. 快速验证 (rlhf_env 自测)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RLHF Env 快速自测 (仅验证框架结构，TODO 需要学习者完成)")
    print("=" * 60)

    config    = RLHFConfig()
    tokenizer = MockTokenizer(config.vocab_size)

    # ── Tokenizer 测试 ──────────────────────────────────────────────────────
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

    # ── Dataset / DataLoader 结构测试 ───────────────────────────────────────
    ppo_ds = RawPromptDataset()
    dpo_ds = RawPreferenceDataset()
    print(f"\n[Dataset] PPO prompts: {len(ppo_ds)}, DPO pairs: {len(dpo_ds)}")
    print(f"  sample PPO: {ppo_ds[0]}")
    print(f"  sample DPO: {dpo_ds[0]}")

    # ── Model 结构测试 ──────────────────────────────────────────────────────
    policy = MockPolicyModel(config)
    ref    = MockReferenceModel(policy)
    reward = MockRewardModel(config)
    value  = MockValueModel(config)

    dummy_ids  = torch.randint(2, config.vocab_size, (2, 8))
    dummy_mask = torch.ones(2, 8, dtype=torch.long)
    logits = policy(dummy_ids, dummy_mask)
    print(f"\n[PolicyModel] forward logits shape: {logits.shape}")  # [2, 8, V]

    response = policy.generate(dummy_ids, dummy_mask, max_new_tokens=5)
    print(f"[PolicyModel] generate response shape: {response.shape}")  # [2, 5]

    r = reward.score(dummy_ids, dummy_mask, response,
                     torch.ones_like(response))
    print(f"[RewardModel] score shape: {r.shape}")  # [2]

    print("\n✅ 框架结构正常。请完成 [TODO-A] [TODO-B] [TODO-C] 后继续 PPO/DPO/GRPO 练习。")
