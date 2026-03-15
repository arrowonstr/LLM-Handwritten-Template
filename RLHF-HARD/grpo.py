"""
GRPO (Group Relative Policy Optimization) — 手撕练习 [HARD]
============================================================
基于 DeepSeek-R1 论文 (2024) 和 TRL GRPOTrainer。

流程:
  GRPOTrainer
    ├── rollout()   Phase 1: 组采样，每个 prompt 生成 G 个 response
    ├── process()   Phase 2: reward 打分 + 组内标准化 → advantage
    └── update()    Phase 3: Clipped Surrogate + f-divergence KL

vs PPO:
  - 无 Value Model，用组内相对奖励估计 advantage
  - advantage 是 per-sequence 标量，广播到 response 所有 token
  - KL: f-divergence 无偏估计 exp(x) - x - 1，而非 PPO 的近似

核心手撕项:
  [TODO-A]  rlhf_env.py ppo_collate_fn
  [TODO-C]  rlhf_env.py get_log_probs
  [TODO-1]  rollout()
  [TODO-2]  process()
  [TODO-3]  _group_advantages()
  [TODO-4]  update()
  [TODO-5]  _policy_loss()
  [TODO-6]  _kl_penalty()
  [TODO-7]  main()
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

from rlhf_env import (
    RLHFConfig, MockTokenizer,
    MockPolicyModel, MockReferenceModel,
    MockRewardModel,
    make_ppo_dataloader,
)


# ==============================================================================
# 超参数
# ==============================================================================
@dataclass
class GRPOConfig:
    group_size:     int   = 4
    clip_eps:       float = 0.2
    kl_coef:        float = 0.04
    num_epochs:     int   = 1
    num_iterations: int   = 10


# ==============================================================================
# GRPOTrainer
# ==============================================================================
class GRPOTrainer:
    """
    GRPO Trainer (对齐 TRL GRPOTrainer / DeepSeek OpenRLHF)。

    使用方式 (在 main() 中完成):
        trainer    = GRPOTrainer(config, grpo_config)
        dataloader = make_ppo_dataloader(config, trainer.tokenizer)
        for iteration in range(grpo_config.num_iterations):
            for batch in dataloader:
                rollout_data   = trainer.rollout(batch)
                processed_data = trainer.process(rollout_data)
                metrics        = trainer.update(rollout_data, processed_data)
    """

    def __init__(self, config: RLHFConfig, grpo_config: GRPOConfig):
        self.config      = config
        self.grpo_config = grpo_config

        self.policy_model = MockPolicyModel(config)
        self.ref_model    = MockReferenceModel(self.policy_model)
        self.reward_model = MockRewardModel(config)
        self.tokenizer    = MockTokenizer(config.vocab_size)
        self.optimizer    = torch.optim.Adam(
            self.policy_model.parameters(), lr=config.lr)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Rollout
    # ─────────────────────────────────────────────────────────────────────────
    def rollout(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        组采样: 每个 prompt 生成 G 个 response。

        Args:
            batch: ppo_collate_fn 输出
                   'input_ids'      [B, T_p]
                   'attention_mask' [B, T_p]

        Returns:
            'prompt_ids':    [B*G, T_p]
            'prompt_mask':   [B*G, T_p]
            'response_ids':  [B*G, T_r]
            'action_mask':   [B*G, T_r]
            'old_log_probs': [B*G, T_r]
            'ref_log_probs': [B*G, T_r]

        =========================================================================
        [TODO-1] 请实现 rollout()

        组展开:
          expanded = prompt.repeat_interleave(G, dim=0)   → [B*G, T_p]

        action_mask:
          cumsum(is_eos) <= 1   (含 EOS，EOS 后清零)

        log_probs:
          get_log_probs(full_ids, action_mask=action_mask)[:, T_p-1:]  → [B*G, T_r]
        =========================================================================
        """
        raise NotImplementedError("[TODO-1] 请实现 GRPOTrainer.rollout()")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Process
    # ─────────────────────────────────────────────────────────────────────────
    def process(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        reward 打分 + 组内标准化。

        Returns:
            {'rewards': [B*G], 'advantages': [B*G]}

        =========================================================================
        [TODO-2] 请实现 process()

          rewards    = reward_model.score(prompt, response)   → [B*G]
          advantages = _group_advantages(rewards, B, G)       → [B*G]
        =========================================================================
        """
        raise NotImplementedError("[TODO-2] 请实现 GRPOTrainer.process()")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Update
    # ─────────────────────────────────────────────────────────────────────────
    def update(self,
               rollout_data:   Dict[str, torch.Tensor],
               processed_data: Dict[str, torch.Tensor]
               ) -> Dict[str, float]:
        """
        GRPO 更新: Clipped Surrogate + f-divergence KL。

        Returns:
            {'policy_loss', 'kl_loss', 'mean_reward'}

        =========================================================================
        [TODO-4] 请实现 update()

        每轮 epoch:
          new_log_probs = get_log_probs(full_ids, action_mask)[:, T_p-1:]
          loss = _policy_loss(...) + kl_coef * _kl_penalty(...)
          optimizer: zero_grad → backward → step
        =========================================================================
        """
        raise NotImplementedError("[TODO-4] 请实现 GRPOTrainer.update()")

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _group_advantages(self,
                          rewards:    torch.Tensor,
                          batch_size: int,
                          group_size: int) -> torch.Tensor:
        """
        组内相对优势。

        Args:
            rewards:    [B*G]

        Returns:
            advantages: [B*G]

        =========================================================================
        [TODO-3] 请实现 _group_advantages()

          grouped = rewards.view(B, G)
          A_i = (r_i - μ_group) / (σ_group + ε)

        避免 std=0 的数值问题
        =========================================================================
        """
        raise NotImplementedError("[TODO-3] 请实现 _group_advantages()")

    def _policy_loss(self,
                     new_log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages:    torch.Tensor,
                     action_mask:   torch.Tensor,
                     clip_eps:      float) -> torch.Tensor:
        """
        GRPO Clipped Surrogate Objective。

        Args:
            new_log_probs: [B*G, T_r]
            old_log_probs: [B*G, T_r]  detached
            advantages:    [B*G]       per-sequence 标量
            action_mask:   [B*G, T_r]

        =========================================================================
        [TODO-5] 请实现 _policy_loss()

          r_t = exp(log π_new - log π_old)
          adv 广播: advantages.unsqueeze(1)  → [B*G, 1]
          L = -E[ min(r_t · A,  clip(r_t, 1-ε, 1+ε) · A) ]

        仅对 action_mask=1 的位置取均值
        =========================================================================
        """
        raise NotImplementedError("[TODO-5] 请实现 _policy_loss()")

    def _kl_penalty(self,
                    new_log_probs: torch.Tensor,
                    ref_log_probs: torch.Tensor,
                    action_mask:   torch.Tensor) -> torch.Tensor:
        """
        GRPO f-divergence KL 惩罚 (DeepSeekMath)。

        Args:
            new_log_probs: [B*G, T_r]
            ref_log_probs: [B*G, T_r]
            action_mask:   [B*G, T_r]

        =========================================================================
        [TODO-6] 请实现 _kl_penalty()

        f-divergence 无偏估计 (x = log π_ref - log π_new):
          KL ≈ e^x - x - 1  ≥ 0,  x=0 时取等

        对比 PPO 的近似 KL = log(π/π_ref)，此估计在策略差异大时更准确
        =========================================================================
        """
        raise NotImplementedError("[TODO-6] 请实现 _kl_penalty()")


# ==============================================================================
# [TODO-7] 主流程
# ==============================================================================
def main():
    """
    =========================================================================
    [TODO-7] 请实现完整主流程

    trainer → dataloader → iteration 循环 → rollout → process → update → log
    可打印同一 prompt 的 G 个 response 及其 reward/advantage 观察组内对比效果
    =========================================================================
    """
    raise NotImplementedError("[TODO-7] 请实现 GRPO main()")


if __name__ == "__main__":
    main()
