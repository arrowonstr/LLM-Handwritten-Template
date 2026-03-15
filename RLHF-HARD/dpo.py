"""
DPO (Direct Preference Optimization) — 手撕练习 [HARD]
======================================================
基于 Rafailov et al. 2023，对齐 TRL DPOTrainer。

流程:
  DPOTrainer
    ├── _sequence_log_prob()   per-token → 序列级
    ├── _dpo_loss()            Bradley-Terry loss
    ├── train_step()           单步训练
    └── extract_implicit_reward()

核心手撕项:
  [TODO-B]  rlhf_env.py dpo_collate_fn
  [TODO-C]  rlhf_env.py get_log_probs
  [TODO-1]  _sequence_log_prob()
  [TODO-2]  _dpo_loss()
  [TODO-3]  train_step()
  [TODO-4]  main()
  [TODO-5]  extract_implicit_reward()
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from rlhf_env import (
    RLHFConfig, MockTokenizer,
    MockPolicyModel, MockReferenceModel,
    make_dpo_dataloader,
)


# ==============================================================================
# 超参数
# ==============================================================================
@dataclass
class DPOConfig:
    beta:            float = 0.1
    label_smoothing: float = 0.0
    num_epochs:      int   = 20
    log_interval:    int   = 5


# ==============================================================================
# DPOTrainer
# ==============================================================================
class DPOTrainer:
    """
    DPO Trainer (对齐 TRL DPOTrainer)。

    使用方式 (在 main() 中完成):
        trainer    = DPOTrainer(config, dpo_config)
        dataloader = make_dpo_dataloader(config, trainer.tokenizer)
        for epoch in range(dpo_config.num_epochs):
            for batch in dataloader:
                metrics = trainer.train_step(batch)
    """

    def __init__(self, config: RLHFConfig, dpo_config: DPOConfig):
        self.config     = config
        self.dpo_config = dpo_config

        self.policy_model = MockPolicyModel(config)
        self.ref_model    = MockReferenceModel(self.policy_model)
        self.tokenizer    = MockTokenizer(config.vocab_size)
        self.optimizer    = torch.optim.Adam(
            self.policy_model.parameters(), lr=config.lr)

    # ─────────────────────────────────────────────────────────────────────────
    # 单步训练
    # ─────────────────────────────────────────────────────────────────────────
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        DPO 单步训练 (对应 TRL DPOTrainer.compute_loss)。

        Args:
            batch: dpo_collate_fn 输出，包含 chosen/rejected 各自的
                   input_ids [B, T], attention_mask [B, T], labels [B, T]

        Returns:
            {'loss', 'reward_margin', 'accuracy'}

        =========================================================================
        [TODO-3] 请实现 train_step()

        数据流:
          policy:  get_log_probs(chosen, labels=chosen_lbl) → pc_logps, pc_mask
                   get_log_probs(rejected, labels=rejected_lbl) → pr_logps, pr_mask
          ref (no_grad): 同上 → rc_logps, rr_logps

          序列级聚合: _sequence_log_prob(per_token_logps, mask) → [B]

          DPO loss: _dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

        监控:
          reward_margin = (chosen_rewards - rejected_rewards).mean()
          accuracy      = (chosen_rewards > rejected_rewards).float().mean()
        =========================================================================
        """
        raise NotImplementedError("[TODO-3] 请实现 DPOTrainer.train_step()")

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _sequence_log_prob(self,
                           per_token_log_probs: torch.Tensor,
                           loss_mask: torch.Tensor) -> torch.Tensor:
        """
        per-token log probs 聚合为序列级。

        Args:
            per_token_log_probs: [B, T-1]  无效位清零
            loss_mask:           [B, T-1]

        Returns:
            seq_log_prob: [B]

        =========================================================================
        [TODO-1] 请实现 _sequence_log_prob()

          log p(y|x) = (Σ_t log p_t · mask_t) / Σ_t mask_t

        均值聚合以消除长度偏差 (length bias)
        =========================================================================
        """
        raise NotImplementedError("[TODO-1] 请实现 _sequence_log_prob()")

    def _dpo_loss(self,
                  policy_chosen_logps:   torch.Tensor,
                  policy_rejected_logps: torch.Tensor,
                  ref_chosen_logps:      torch.Tensor,
                  ref_rejected_logps:    torch.Tensor,
                  beta:                  float,
                  label_smoothing:       float = 0.0,
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DPO Bradley-Terry Loss。

        Args:
            *_logps: [B]  序列级 log prob
            beta: KL 约束强度
            label_smoothing: Azar et al. 2024 robust DPO

        Returns:
            loss, chosen_rewards [B], rejected_rewards [B]

        =========================================================================
        [TODO-2] 请实现 _dpo_loss()

        隐式 reward:
          r(x,y) = β · (log π(y|x) - log π_ref(y|x))

        Bradley-Terry preference logit:
          h = β · [(log π_c - log π_ref_c) - (log π_r - log π_ref_r)]

        标准 DPO (label_smoothing=0):
          L = -log σ(h)

        Robust DPO:
          L = -(1-ls)·log σ(h) - ls·log σ(-h)
        =========================================================================
        """
        raise NotImplementedError("[TODO-2] 请实现 _dpo_loss()")

    # ─────────────────────────────────────────────────────────────────────────
    # 分析工具
    # ─────────────────────────────────────────────────────────────────────────
    def extract_implicit_reward(self,
                                input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
        """
        从策略中提取隐式 reward。

        Returns:
            implicit_reward: [B]

        =========================================================================
        [TODO-5] 请实现 extract_implicit_reward()

          r*(x,y) = β · (log π(y|x) - log π_ref(y|x))

        用 _sequence_log_prob 做序列级聚合，全程 no_grad
        =========================================================================
        """
        raise NotImplementedError("[TODO-5] 请实现 extract_implicit_reward()")


# ==============================================================================
# [TODO-4] 主流程
# ==============================================================================
def main():
    """
    =========================================================================
    [TODO-4] 请实现完整主流程

    trainer → dataloader → epoch 循环 → train_step → log
    训练前后可打印 implicit reward margin 观察训练效果
    =========================================================================
    """
    raise NotImplementedError("[TODO-4] 请实现 DPO main()")


if __name__ == "__main__":
    main()
