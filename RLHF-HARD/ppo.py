"""
PPO (Proximal Policy Optimization) for RLHF — 手撕练习 [HARD]
===============================================================
基于 InstructGPT / TRL PPOTrainer 的真实工程结构。

流程:
  PPOTrainer
    ├── rollout()     Phase 1: 采样
    ├── process()     Phase 2: KL → token_rewards → GAE
    └── update()      Phase 3: Clipped Surrogate + Value Loss

核心手撕项:
  [TODO-A]  rlhf_env.py ppo_collate_fn
  [TODO-C]  rlhf_env.py get_log_probs
  [TODO-1]  rollout()
  [TODO-2]  process()
  [TODO-3]  update()
  [TODO-4]  _compute_gae()
  [TODO-5]  _policy_loss()
  [TODO-6]  _value_loss()
  [TODO-7]  main()
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
    PPO-RLHF Trainer (对齐 TRL PPOTrainer)。

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
    # Phase 1: Rollout
    # ─────────────────────────────────────────────────────────────────────────
    def rollout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        采样阶段：生成 response，收集所有后续训练所需张量。

        Args:
            batch: ppo_collate_fn 输出
                   'input_ids'      [B, T_p]
                   'attention_mask' [B, T_p]

        Returns:
            rollout_data:
                'prompt_ids':    [B, T_p]
                'prompt_mask':   [B, T_p]
                'response_ids':  [B, T_r]
                'action_mask':   [B, T_r]   EOS 及之前=1, 之后=0
                'old_log_probs': [B, T_r]   response 部分 log π_old
                'ref_log_probs': [B, T_r]   response 部分 log π_ref
                'rewards':       [B]
                'old_values':    [B, T_r]   value model 估计，pad 位清零

        =========================================================================
        [TODO-1] 请实现 rollout()

        数据流:
          prompt → generate → response_ids [B, T_r]
          action_mask: cumsum(is_eos) <= 1  (含 EOS，EOS 后清零)
          full_ids = cat([prompt, response])
          old_log_probs = get_log_probs(full_ids, action_mask=action_mask)[:, T_p-1:]
          old_values    = value_model(full_ids)[:, T_p:] * action_mask

        所有 get_log_probs / reward / value 调用均在 no_grad() 下进行
        =========================================================================
        """
        raise NotImplementedError("[TODO-1] 请实现 PPOTrainer.rollout()")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Process
    # ─────────────────────────────────────────────────────────────────────────
    def process(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        加工阶段: KL 惩罚 → 逐 token 奖励 → GAE

        Returns:
            {'advantages': [B, T_r], 'returns': [B, T_r]}

        =========================================================================
        [TODO-2] 请实现 process()

        per_token_kl[t]    = old_log_probs[t] - ref_log_probs[t]
        token_rewards[t]   = -kl_coef * kl[t]
        token_rewards[last_valid] += r        ← terminal reward scatter

        last_valid_idx = action_mask.sum(dim=1) - 1

        advantages, returns = _compute_gae(token_rewards, old_values, action_mask)
        =========================================================================
        """
        raise NotImplementedError("[TODO-2] 请实现 PPOTrainer.process()")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Update
    # ─────────────────────────────────────────────────────────────────────────
    def update(self,
               rollout_data:   Dict[str, torch.Tensor],
               processed_data: Dict[str, torch.Tensor]
               ) -> Dict[str, float]:
        """
        多 epoch 更新 policy 和 value 网络。

        Returns:
            metrics: {'policy_loss', 'value_loss', 'mean_reward', 'mean_kl'}

        =========================================================================
        [TODO-3] 请实现 update()

        每轮 epoch:
          full_ids  = cat([prompt, response])
          new_log_probs = get_log_probs(full_ids, action_mask=action_mask)[:, T_p-1:]
          new_values    = value_model(full_ids)[:, T_p:] * action_mask

          loss = _policy_loss(...) + vf_coef * _value_loss(...)
          两个 optimizer 各自 zero_grad → backward → step
        =========================================================================
        """
        raise NotImplementedError("[TODO-3] 请实现 PPOTrainer.update()")

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
            values:        [B, T_r]  pad 位已清零
            action_mask:   [B, T_r]
            gamma, lam:    float

        Returns:
            advantages: [B, T_r]
            returns:    [B, T_r]  = advantages + values

        =========================================================================
        [TODO-4] 请实现 _compute_gae()

        递推公式 (从右到左):
          δ_t = r_t + γ · V(s_{t+1}) · mask_{t+1} - V(s_t)
          A_t = δ_t + γλ · A_{t+1} · mask_{t+1}
          A_t = A_t · mask_t   (pad 位清零)

        最后对有效位置做归一化:
          A[valid] = (A[valid] - μ) / (σ + ε)
        =========================================================================
        """
        raise NotImplementedError("[TODO-4] 请实现 _compute_gae()")

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
            old_log_probs: [B, T_r]  detached
            advantages:    [B, T_r]
            action_mask:   [B, T_r]

        =========================================================================
        [TODO-5] 请实现 _policy_loss()

          r_t = exp(log π_new - log π_old)
          L = -E[ min(r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t) ]

        仅对 action_mask=1 的位置取均值
        =========================================================================
        """
        raise NotImplementedError("[TODO-5] 请实现 _policy_loss()")

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
            old_values:  [B, T_r]  detached
            returns:     [B, T_r]
            action_mask: [B, T_r]

        =========================================================================
        [TODO-6] 请实现 _value_loss()

          v_clipped = clip(V_new, V_old-ε, V_old+ε)
          L = 0.5 · E[ max((V_new - R)², (v_clipped - R)²) ]

        仅对 action_mask=1 的位置取均值
        =========================================================================
        """
        raise NotImplementedError("[TODO-6] 请实现 _value_loss()")


# ==============================================================================
# [TODO-7] 主流程
# ==============================================================================
def main():
    """
    =========================================================================
    [TODO-7] 请实现完整主流程

    trainer → dataloader → for iteration: rollout → process → update → log
    =========================================================================
    """
    raise NotImplementedError("[TODO-7] 请实现 PPO main()")


if __name__ == "__main__":
    main()
