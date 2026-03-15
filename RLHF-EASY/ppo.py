"""
PPO (Proximal Policy Optimization) for RLHF — 手撕练习
========================================================
基于 InstructGPT / TRL PPOTrainer 的真实工程结构。

流程:
  PPOTrainer
    ├── rollout()     Phase 1: 采样 — 生成 response，构建 action_mask，收集 log_probs / values
    ├── process()     Phase 2: 加工 — KL → token_rewards → GAE
    └── update()      Phase 3: 训练 — Clipped Surrogate + Value Loss (多轮 epoch)

关键工程点 (对齐官方):
  - DataLoader 中用 ppo_collate_fn 完成 Prompt 左填充 [TODO-A]
  - generate() 返回 response_ids + 构建 action_mask (response EOS 后清零) [TODO-1]
  - get_log_probs 使用 action_mask 屏蔽 prompt/pad，只算 response 部分 [TODO-C]
  - ValueModel 只取 response 对应的时间步, 对齐 action_mask [TODO-2]

核心手撕项:
  [TODO-A]  rlhf_env.py ppo_collate_fn         ← Prompt 左填充
  [TODO-C]  rlhf_env.py get_log_probs          ← 统一 log prob
  [TODO-1]  rollout() 中构建 action_mask
  [TODO-2]  rollout() 中提取 response values
  [TODO-3]  process() → KL + token_rewards + GAE
  [TODO-4]  update()  → PPO multi-epoch update
  [TODO-5]  _compute_gae()
  [TODO-6]  _policy_loss()
  [TODO-7]  _value_loss()
  [TODO-8]  main() 主循环
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader

from rlhf_env import (
    RLHFConfig, MockTokenizer,
    MockPolicyModel, MockReferenceModel,
    MockRewardModel, MockValueModel,
    make_ppo_dataloader,
)


# ==============================================================================
# 超参数
# ==============================================================================
@dataclass
class PPOConfig:
    clip_eps:       float = 0.2
    vf_clip_eps:    float = 0.2
    gamma:          float = 1.0
    lam:            float = 0.95
    kl_coef:        float = 0.1
    vf_coef:        float = 0.5
    ppo_epochs:     int   = 4
    num_iterations: int   = 10


# ==============================================================================
# PPOTrainer
# ==============================================================================
class PPOTrainer:
    """
    PPO-RLHF Trainer (对齐 TRL PPOTrainer 的整体结构)。

    使用方式 (在 main() 中完成):
        trainer   = PPOTrainer(config, ppo_config)
        for batch in dataloader:
            rollout_data   = trainer.rollout(batch)
            processed_data = trainer.process(rollout_data)
            metrics        = trainer.update(rollout_data, processed_data)
    """

    def __init__(self, config: RLHFConfig, ppo_config: PPOConfig):
        self.config     = config
        self.ppo_config = ppo_config

        self.policy_model = MockPolicyModel(config)
        self.ref_model    = MockReferenceModel(self.policy_model)
        self.reward_model = MockRewardModel(config)
        self.value_model  = MockValueModel(config)
        self.tokenizer    = MockTokenizer(config.vocab_size)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_model.parameters(), lr=config.lr)
        self.value_optimizer  = torch.optim.Adam(
            self.value_model.parameters(), lr=config.lr)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Rollout (采样)
    # ─────────────────────────────────────────────────────────────────────────
    def rollout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        采样阶段：从 DataLoader 拿到左填充好的 Prompt，
        用当前策略生成 Response，并收集所有后续计算所需张量。

        Args:
            batch: ppo_collate_fn 返回的 dict，包含:
                   'input_ids'      [B, T_p]   左填充过的 prompt
                   'attention_mask' [B, T_p]   对应 mask

        Returns:
            rollout_data: dict 包含
                'prompt_ids':      [B, T_p]
                'prompt_mask':     [B, T_p]
                'response_ids':    [B, T_r]
                'action_mask':     [B, T_r]   1=有效token(含EOS), 0=EOS后的pad
                'old_log_probs':   [B, T_r]   已从 full log_probs 中 slice 出 response 部分
                'ref_log_probs':   [B, T_r]   (同上)
                'rewards':         [B]        reward model 打分
                'old_values':      [B, T_r]   value model 估计

        =========================================================================
        [TODO-1] 请实现 rollout()

        Step 1 — 提取 prompt 信息
            prompt_ids  = batch['input_ids']       # [B, T_p]
            prompt_mask = batch['attention_mask']   # [B, T_p]

        Step 2 — 用策略生成 response (不记录梯度)
            with torch.no_grad():
                response_ids = self.policy_model.generate(
                    prompt_ids, prompt_mask, max_new_tokens=self.config.max_new_tokens)
                # response_ids: [B, T_r]，EOS 之后自动为 <pad>

        Step 3 — 构建 action_mask [关键!]
            action_mask 标记 response 中『有效』的 token（从第一个到 EOS，含 EOS）。
            EOS 之后的 <pad> 全部为 0。

            做法:
                # 找到每个序列第一个 EOS 的位置
                is_eos = (response_ids == self.config.eos_token_id)  # [B, T_r]
                # 利用 cumsum: EOS 出现后置位为 False
                # cumsum 在 EOS 位置变成 >=1，在 EOS 之后变为 >=2
                # 技巧: (cumsum <= 1) 给 EOS 之前(含EOS)置 True
                eos_cumsum = is_eos.long().cumsum(dim=1)
                action_mask = (eos_cumsum <= 1).long()   # [B, T_r]
                # 如果一个序列完全没有 EOS，action_mask 全为 1 (全有效)

        Step 4 — 构建完整序列的 input_ids 和 attention_mask，供 get_log_probs 使用
            full_ids  = torch.cat([prompt_ids, response_ids], dim=1)    # [B, T_p+T_r]
            full_mask = torch.cat([prompt_mask,
                                   action_mask], dim=1)                  # [B, T_p+T_r]
            注意: full_mask 中 prompt 部分用 prompt_mask (可能含 left-pad=0)

        Step 5 — 计算 old_log_probs / ref_log_probs (旧策略快照, detach)
            ★ 官方做法: get_log_probs 返回 [B, T_p+T_r-1]，立即 slice 出 response 部分
              response 的 T_r 个 log prob 从位置 T_p-1 开始，所以 [:, T_p-1:] 正好取出 [B, T_r]

            with torch.no_grad():
                T_p = prompt_ids.shape[1]

                full_lp, _  = self.policy_model.get_log_probs(
                    full_ids, full_mask, action_mask=action_mask)  # [B, T_p+T_r-1]
                old_log_probs = full_lp[:, T_p - 1:]              # [B, T_r]  ← slice!

                full_rlp, _ = self.ref_model.get_log_probs(
                    full_ids, full_mask, action_mask=action_mask)  # [B, T_p+T_r-1]
                ref_log_probs = full_rlp[:, T_p - 1:]             # [B, T_r]  ← slice!

        Step 7 — Reward Model 打分
            with torch.no_grad():
                rewards = self.reward_model.score(
                    prompt_ids, prompt_mask, response_ids, action_mask)

        Step 8 — Value Model 估计 V(s_t)，只取 response 部分的时间步
            with torch.no_grad():
                all_values = self.value_model(full_ids, full_mask)  # [B, T_p+T_r]
                # 只取 response 对应的时间步:
                T_p = prompt_ids.shape[1]
                old_values = all_values[:, T_p:]                    # [B, T_r]
                # 用 action_mask 过滤掉 response 中 EOS 后的 pad:
                old_values = old_values * action_mask.float()       # [B, T_r]

        Step 9 — 打包返回
        =========================================================================
        """
        raise NotImplementedError("[TODO-1] 请实现 PPOTrainer.rollout()")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Process (加工)
    # ─────────────────────────────────────────────────────────────────────────
    def process(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        加工阶段: KL → token_rewards → GAE → advantages / returns

        Args:
            rollout_data: rollout() 返回的 dict

        Returns:
            processed_data: {'advantages': [B, T_r], 'returns': [B, T_r]}

        =========================================================================
        [TODO-3] 请实现 process()

        Step 1 — 从 rollout_data 中取出数据:
            old_log_probs = rollout_data['old_log_probs']   # [B, T_r]
            ref_log_probs = rollout_data['ref_log_probs']   # [B, T_r]
            rewards       = rollout_data['rewards']          # [B]
            old_values    = rollout_data['old_values']       # [B, T_r]
            action_mask   = rollout_data['action_mask']      # [B, T_r]
            B             = action_mask.shape[0]             # ← 供下面的 scatter 使用

        Step 2 — 逐 Token KL 散度 (PPO-style 近似):
            kl = old_log_probs - ref_log_probs   # [B, T_r] ,  ≈ KL(π_old || π_ref)
            注意: 由于 get_log_probs 已经把 prompt/pad 位置清零了，
                  这里的 kl 自然也在无效位置为 0，不需要额外 masking。

        Step 3 — 构建逐 Token 奖励:
            token_rewards = torch.zeros_like(old_values)          # [B, T_r]
            # 把总奖励加到每个序列『最后一个有效 token』的位置
            # last_valid_idx[i] = action_mask[i].sum() - 1
            last_idx = action_mask.sum(dim=1) - 1                 # [B]
            token_rewards[torch.arange(B), last_idx] += rewards   # scatter 到最后一步
            token_rewards -= self.ppo_config.kl_coef * kl         # 每步都有 KL 惩罚

        Step 4 — GAE:
            advantages, returns = self._compute_gae(
                token_rewards, old_values, action_mask,
                self.ppo_config.gamma, self.ppo_config.lam)

        Step 5 — 返回
        =========================================================================
        """
        raise NotImplementedError("[TODO-3] 请实现 PPOTrainer.process()")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Update (训练)
    # ─────────────────────────────────────────────────────────────────────────
    def update(self,
               rollout_data:   Dict[str, torch.Tensor],
               processed_data: Dict[str, torch.Tensor]
               ) -> Dict[str, float]:
        """
        训练阶段: PPO 多 epoch 更新 policy 和 value 网络。

        Args:
            rollout_data:   rollout() 返回的 dict
            processed_data: process() 返回的 dict

        Returns:
            metrics: {'policy_loss', 'value_loss', 'mean_reward', 'mean_kl'}

        =========================================================================
        [TODO-4] 请实现 PPOTrainer.update()

        从数据中取出:
            prompt_ids    = rollout_data['prompt_ids']
            prompt_mask   = rollout_data['prompt_mask']
            response_ids  = rollout_data['response_ids']
            action_mask   = rollout_data['action_mask']
            old_log_probs = rollout_data['old_log_probs']
            old_values    = rollout_data['old_values']
            rewards       = rollout_data['rewards']
            advantages    = processed_data['advantages']
            returns       = processed_data['returns']

        多 epoch 循环 (ppo_epochs 次):
            for epoch in range(self.ppo_config.ppo_epochs):

              1. 重新计算 new_log_probs (需要梯度!):
                 full_ids  = cat([prompt_ids, response_ids], dim=1)   # [B, T_p+T_r]
                 full_mask = cat([prompt_mask, action_mask], dim=1)   # [B, T_p+T_r]
                 T_p = prompt_ids.shape[1]
                 full_lp, _ = self.policy_model.get_log_probs(
                     full_ids, full_mask, action_mask=action_mask)    # [B, T_p+T_r-1]
                 new_log_probs = full_lp[:, T_p - 1:]                 # [B, T_r]  ← slice!

              2. 重新估计 new_values (需要梯度!):
                 T_p = prompt_ids.shape[1]
                 all_v = self.value_model(full_ids, full_mask)
                 new_values = all_v[:, T_p:] * action_mask.float()

              3. policy_loss = self._policy_loss(new_log_probs, old_log_probs,
                                                 advantages, action_mask,
                                                 self.ppo_config.clip_eps)

              4. value_loss  = self._value_loss(new_values, old_values,
                                                returns, action_mask,
                                                self.ppo_config.vf_clip_eps)

              5. loss = policy_loss + self.ppo_config.vf_coef * value_loss

              6. zero_grad → backward → step (policy_optimizer + value_optimizer)

        返回 metrics (各项 loss 的平均值, mean_reward 等)
        =========================================================================
        """
        raise NotImplementedError("[TODO-4] 请实现 PPOTrainer.update()")

    # =========================================================================
    # 内部方法 — 手撕核心
    # =========================================================================

    def _compute_gae(self,
                     token_rewards: torch.Tensor,
                     values:        torch.Tensor,
                     action_mask:   torch.Tensor,
                     gamma:         float,
                     lam:           float
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE)。

        Args:
            token_rewards: [B, T_r]
            values:        [B, T_r]  (已用 action_mask 清零 pad 位置)
            action_mask:   [B, T_r]  1=有效, 0=pad
            gamma, lam:    float

        Returns:
            advantages: [B, T_r]
            returns:    [B, T_r]   = advantages + values

        =========================================================================
        [TODO-5] 请实现 GAE

        真实 TRL 的 GAE 实现考虑了 action_mask:
          last_gae = 0
          for t = T-1, T-2, ..., 0:
              # 只有有效位置才参与递推
              next_value = values[:, t+1] * action_mask[:, t+1] if t < T-1 else 0
              delta = token_rewards[:, t] + gamma * next_value - values[:, t]
              last_gae = delta + gamma * lam * last_gae * (action_mask[:, t+1] if t < T-1 else 0)
              advantages[:, t] = last_gae * action_mask[:, t]

        最后:
          returns    = advantages + values
          # 只对有效位置做标准化
          valid_adv  = advantages[action_mask.bool()]
          advantages = (advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)
          advantages = advantages * action_mask.float()
        =========================================================================
        """
        raise NotImplementedError("[TODO-5] 请实现 _compute_gae()")

    def _policy_loss(self,
                     new_log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages:    torch.Tensor,
                     action_mask:   torch.Tensor,
                     clip_eps:      float) -> torch.Tensor:
        """
        PPO Clipped Surrogate Objective。

        Args:
            new_log_probs: [B, T_r]
            old_log_probs: [B, T_r]  (detached)
            advantages:    [B, T_r]
            action_mask:   [B, T_r]
            clip_eps:      float

        Returns:
            policy_loss: scalar

        =========================================================================
        [TODO-6] 请实现 PPO policy loss

        与之前版本相比，关键改动: 要用 action_mask 过滤无效位置！

          ratio = exp(new_log_probs - old_log_probs)   # [B, T_r]
          surr1 = ratio * advantages
          surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
          per_token_loss = -min(surr1, surr2)           # [B, T_r]

          # 只对有效 token 取均值 (而非直接 .mean()，否则 pad 位置会稀释梯度)
          loss = (per_token_loss * action_mask.float()).sum() \
                 / action_mask.float().sum().clamp(min=1)

        直觉:
          不做 action_mask 过滤的 .mean() 会把 pad 位置的 0 也算进去，
          使得真实 response 的梯度被稀释，等效学习率降低。
        =========================================================================
        """
        raise NotImplementedError("[TODO-6] 请实现 _policy_loss()")

    def _value_loss(self,
                    new_values:  torch.Tensor,
                    old_values:  torch.Tensor,
                    returns:     torch.Tensor,
                    action_mask: torch.Tensor,
                    vf_clip_eps: float) -> torch.Tensor:
        """
        Clipped Value Function Loss。

        Args:
            new_values:  [B, T_r]
            old_values:  [B, T_r]
            returns:     [B, T_r]
            action_mask: [B, T_r]
            vf_clip_eps: float

        Returns:
            value_loss: scalar

        =========================================================================
        [TODO-7] 请实现 clipped value loss

          vf_loss1 = (new_values - returns) ** 2
          clipped  = old_values + clamp(new_values - old_values, -ε, ε)
          vf_loss2 = (clipped - returns) ** 2
          per_token_loss = 0.5 * max(vf_loss1, vf_loss2)     # [B, T_r]

          # 同样用 action_mask 只对有效位置取均值
          loss = (per_token_loss * action_mask.float()).sum() \
                 / action_mask.float().sum().clamp(min=1)
        =========================================================================
        """
        raise NotImplementedError("[TODO-7] 请实现 _value_loss()")


# ==============================================================================
# [TODO-8] 主流程
# ==============================================================================
def main():
    """
    PPO-RLHF 完整训练主流程。

    =========================================================================
    [TODO-8] 请实现完整主流程

    步骤:
      1. 初始化
         config    = RLHFConfig()
         ppo_cfg   = PPOConfig()
         trainer   = PPOTrainer(config, ppo_cfg)
         tokenizer = trainer.tokenizer  (或新建 MockTokenizer)

      2. 构建 DataLoader (注意: 内部已自动使用 ppo_collate_fn 的 left padding)
         dataloader = make_ppo_dataloader(config, tokenizer)

      3. 训练循环
         for iteration in range(ppo_cfg.num_iterations):
             for batch in dataloader:
                 rollout_data   = trainer.rollout(batch)
                 processed_data = trainer.process(rollout_data)
                 metrics        = trainer.update(rollout_data, processed_data)
             打印 metrics

      4. 训练后打印最终采样结果
    =========================================================================
    """
    raise NotImplementedError("[TODO-8] 请实现 PPO main()")


if __name__ == "__main__":
    main()
