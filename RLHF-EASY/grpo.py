"""
GRPO (Group Relative Policy Optimization) — 手撕练习
=====================================================
基于 DeepSeek-R1 论文 (2024) 和 TRL GRPOTrainer 的真实工程结构。

流程:
  GRPOTrainer
    ├── rollout()          Phase 1: 组采样 — 每个 prompt 生成 G 个 response，构建 action_mask
    ├── process()          Phase 2: 打分 + 组内标准化 → advantage
    └── update()           Phase 3: Clipped objective + f-divergence KL

vs PPO 的关键差异:
  - 不需要 Value Model → 节省显存
  - 用组内相对奖励代替 V(s_t) 估计 advantage
  - KL 使用 f-divergence 无偏估计: exp(x) - x - 1 ≥ 0
  - advantage 是 per-sequence 标量 (广播到整个 response)

核心手撕项:
  [TODO-A]  rlhf_env.py ppo_collate_fn          ← Prompt 左填充 (与 PPO 共用)
  [TODO-C]  rlhf_env.py get_log_probs           ← 统一 log prob
  [TODO-1]  rollout(): repeat_interleave + generate + action_mask
  [TODO-2]  process(): reward 打分 + 调用 _group_advantages
  [TODO-3]  _group_advantages(): 组内标准化
  [TODO-4]  update(): Clipped Surrogate + f-divergence KL + 梯度更新
  [TODO-5]  _policy_loss(): Clipped Surrogate (with action_mask)
  [TODO-6]  _kl_penalty(): f-divergence KL
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
    make_ppo_dataloader,    # GRPO 的 prompt DataLoader 与 PPO 共用
)


# ==============================================================================
# 超参数
# ==============================================================================
@dataclass
class GRPOConfig:
    group_size:     int   = 4     # G — 每个 prompt 生成 G 个 response
    clip_eps:       float = 0.2
    kl_coef:        float = 0.04  # β
    num_epochs:     int   = 1     # 每次 rollout 后训练轮数
    num_iterations: int   = 10


# ==============================================================================
# GRPOTrainer
# ==============================================================================
class GRPOTrainer:
    """
    GRPO Trainer (对齐 TRL GRPOTrainer / DeepSeek OpenRLHF 结构)。

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
        组采样: 每个 prompt 生成 G 个 response，收集 log_probs 和 action_mask。

        Args:
            batch: ppo_collate_fn 返回的 dict
                   'input_ids'      [B, T_p]
                   'attention_mask' [B, T_p]

        Returns:
            rollout_data: dict 包含
                'prompt_ids':      [B*G, T_p]
                'prompt_mask':     [B*G, T_p]
                'response_ids':    [B*G, T_r]
                'action_mask':     [B*G, T_r]
                'old_log_probs':   [B*G, T_r]   已从 full log_probs 中 slice 出 response 部分
                'ref_log_probs':   [B*G, T_r]   (同上)

        =========================================================================
        [TODO-1] 请实现 GRPOTrainer.rollout()

        Step 1 — 组展开 (Group Expansion):
            每个 prompt 复制 G 份，让模型对同一 prompt 独立采样 G 次。
            这是 GRPO 的核心机制：同组 G 个 response 用于组内相对比较。

            prompt_ids  = batch['input_ids']        # [B, T_p]
            prompt_mask = batch['attention_mask']   # [B, T_p]
            G = self.grpo_config.group_size

            # repeat_interleave: [p0, p0, ..(G次), p1, p1, ..(G次), ...]
            expanded_ids  = prompt_ids.repeat_interleave(G, dim=0)   # [B*G, T_p]
            expanded_mask = prompt_mask.repeat_interleave(G, dim=0)  # [B*G, T_p]

        Step 2 — 生成 response (由于 do_sample=True，同 prompt 的 G 次结果各不同):
            with torch.no_grad():
                response_ids = self.policy_model.generate(
                    expanded_ids, expanded_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True)    # [B*G, T_r]

        Step 3 — 构建 action_mask (与 PPO 相同):
            is_eos    = (response_ids == self.config.eos_token_id)
            eos_cumsum = is_eos.long().cumsum(dim=1)
            action_mask = (eos_cumsum <= 1).long()   # [B*G, T_r]

        Step 4 — 计算 old_log_probs 和 ref_log_probs:
            full_ids  = cat([expanded_ids, response_ids], dim=1)   # [B*G, T_p+T_r]
            full_mask = cat([expanded_mask, action_mask], dim=1)   # [B*G, T_p+T_r]

            ★ 官方做法: get_log_probs 返回 [B*G, T_p+T_r-1]，立即 slice 出 response 部分
              T_p = expanded_ids.shape[1]
              response 的 T_r 个 log prob 从位置 T_p-1 开始:

            with torch.no_grad():
                full_lp, _  = self.policy_model.get_log_probs(
                    full_ids, full_mask, action_mask=action_mask)   # [B*G, T_p+T_r-1]
                old_log_probs = full_lp[:, T_p - 1:]               # [B*G, T_r]  ← slice!

                full_rlp, _ = self.ref_model.get_log_probs(
                    full_ids, full_mask, action_mask=action_mask)   # [B*G, T_p+T_r-1]
                ref_log_probs = full_rlp[:, T_p - 1:]              # [B*G, T_r]  ← slice!

        Step 5 — 打包返回
        =========================================================================
        """
        raise NotImplementedError("[TODO-1] 请实现 GRPOTrainer.rollout()")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Process
    # ─────────────────────────────────────────────────────────────────────────
    def process(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        打分 + 组内标准化 → per-sequence advantage。

        Args:
            rollout_data: rollout() 返回的 dict

        Returns:
            {'rewards': [B*G], 'advantages': [B*G]}

        =========================================================================
        [TODO-2] 请实现 GRPOTrainer.process()

        Step 1 — 取出需要的张量:
            prompt_ids   = rollout_data['prompt_ids']    # [B*G, T_p]
            prompt_mask  = rollout_data['prompt_mask']   # [B*G, T_p]
            response_ids = rollout_data['response_ids']  # [B*G, T_r]
            action_mask  = rollout_data['action_mask']   # [B*G, T_r]

        Step 2 — Reward 打分 (RewardModel 只关心 prompt 和 response):
            rewards = self.reward_model.score(
                prompt_ids, prompt_mask, response_ids, action_mask)  # [B*G]

        Step 3 — 组内标准化:
            G = self.grpo_config.group_size
            B = prompt_ids.shape[0] // G
            advantages = self._group_advantages(rewards, B, G)    # [B*G]

        Step 4 — 返回:
            return {'rewards': rewards, 'advantages': advantages}
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
        GRPO 更新: Clipped Surrogate + f-divergence KL penalty。

        Args:
            rollout_data:   rollout() 返回的 dict
            processed_data: process() 返回的 dict

        Returns:
            metrics: {'policy_loss', 'kl_loss', 'mean_reward'}

        =========================================================================
        [TODO-4] 请实现 GRPOTrainer.update()

        Step 1 — 取出所有需要的张量:
            prompt_ids    = rollout_data['prompt_ids']
            prompt_mask   = rollout_data['prompt_mask']
            response_ids  = rollout_data['response_ids']
            action_mask   = rollout_data['action_mask']
            old_log_probs = rollout_data['old_log_probs']   # detached
            ref_log_probs = rollout_data['ref_log_probs']   # detached
            advantages    = processed_data['advantages']    # [B*G]

        Step 2 — 拼接完整序列 (与 rollout 一致):
            full_ids  = torch.cat([prompt_ids, response_ids], dim=1)  # [B*G, T_p+T_r]
            full_mask = torch.cat([prompt_mask, action_mask], dim=1)  # [B*G, T_p+T_r]

        Step 3 — 多轮更新 (num_epochs 次对同一批 rollout 数据迭代):
            for _ in range(self.grpo_config.num_epochs):

              3a. 用当前策略重计算 log_probs (需要梯度):
                T_p = prompt_ids.shape[1]
                full_lp, _ = self.policy_model.get_log_probs(
                    full_ids, full_mask, action_mask=action_mask)   # [B*G, T_p+T_r-1]
                new_log_probs = full_lp[:, T_p - 1:]               # [B*G, T_r]  ← slice!

              3b. 计算 policy loss (调用 _policy_loss):
                p_loss = self._policy_loss(
                    new_log_probs, old_log_probs, advantages,
                    action_mask, self.grpo_config.clip_eps)

              3c. 计算 KL penalty (调用 _kl_penalty):
                kl = self._kl_penalty(new_log_probs, ref_log_probs, action_mask)

              3d. 合并 loss:
                loss = p_loss + self.grpo_config.kl_coef * kl

              3e. 梯度更新:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        Step 4 — 返回 metrics (各轮均值):
            n = self.grpo_config.num_epochs
            return {
                'policy_loss': total_p_loss / n,
                'kl_loss':     total_kl     / n,
                'mean_reward': processed_data['rewards'].mean().item(),
            }
        =========================================================================
        """
        raise NotImplementedError("[TODO-4] 请实现 GRPOTrainer.update()")

    # =========================================================================
    # 内部方法 — 手撕核心
    # =========================================================================

    def _group_advantages(self,
                          rewards:    torch.Tensor,
                          batch_size: int,
                          group_size: int) -> torch.Tensor:
        """
        组内相对优势。

        Args:
            rewards:    [B*G]
            batch_size: B
            group_size: G

        Returns:
            advantages: [B*G]

        =========================================================================
        [TODO-3] 请实现 _group_advantages()

        GRPO 用组内相对奖励代替 Value Model：
          同组 G 个 response 中：
            reward 高于组均值 → advantage > 0 → 增大概率
            reward 低于组均值 → advantage < 0 → 减小概率

        步骤:
          grouped = rewards.view(batch_size, group_size)              # [B, G]
          mu  = grouped.mean(dim=1, keepdim=True)   # [B, 1]
          std = grouped.std(dim=1, keepdim=True)    # [B, 1]
          advantages = ((grouped - mu) / (std + 1e-8)).view(-1)  # [B*G]

        注意: 若一组内所有 reward 相同，std=0，除法会产生 NaN/Inf。
              用 std + 1e-8 防止数值问题。
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
            new_log_probs: [B*G, T_r]   已 slice，每个 response token 各一个 log prob
            old_log_probs: [B*G, T_r]   (detached)
            advantages:    [B*G]         per-sequence (标量)
            action_mask:   [B*G, T_r]    1=有效, 0=pad
            clip_eps:      float

        Returns:
            policy_loss: scalar

        =========================================================================
        [TODO-5] 请实现 GRPO _policy_loss()

        log_probs 已在 rollout/update 里 slice 到 [B*G, T_r]，
        与 action_mask [B*G, T_r] 形状完全一致，直接做 elementwise 运算即可。

        步骤:
          # 广播 advantage 到 token 维度
          adv = advantages.unsqueeze(1)                    # [B*G, 1]

          ratio = exp(new_log_probs - old_log_probs)       # [B*G, T_r]
          surr1 = ratio * adv
          surr2 = clamp(ratio, 1-ε, 1+ε) * adv
          per_token_loss = -min(surr1, surr2)              # [B*G, T_r]

          # 直接用 action_mask，无需 F.pad
          amask = action_mask.float()                      # [B*G, T_r]
          loss  = (per_token_loss * amask).sum() / amask.sum().clamp(min=1)
        =========================================================================
        """
        raise NotImplementedError("[TODO-5] 请实现 _policy_loss()")

    def _kl_penalty(self,
                    new_log_probs: torch.Tensor,
                    ref_log_probs: torch.Tensor,
                    action_mask:   torch.Tensor) -> torch.Tensor:
        """
        GRPO f-divergence KL 惩罚 (来自 DeepSeekMath 论文)。

        Args:
            new_log_probs: [B*G, T_r]   已 slice
            ref_log_probs: [B*G, T_r]   已 slice
            action_mask:   [B*G, T_r]

        Returns:
            kl_penalty: scalar

        =========================================================================
        [TODO-6] 请实现 _kl_penalty()

        GRPO 使用与 PPO 不同的 KL 估计器:
            KL(π_ref || π) 的 f-divergence 无偏估计:
              D_KL ≈ exp(log_ratio) - log_ratio - 1
              其中 log_ratio = ref_log_probs - new_log_probs  (= log π_ref/π)

        数学: e^x - x - 1 ≥ 0 ∀x，x=0 时取等，天然非负，更稳定。
        对比 PPO 的 log(π/π_ref)：PPO 的近似估计在策略差异大时不够准确。

        步骤:
          log_ratio = ref_log_probs - new_log_probs          # [B*G, T_r]
          kl        = torch.exp(log_ratio) - log_ratio - 1   # [B*G, T_r]
          # 直接用 action_mask，形状匹配，无需 pad
          amask     = action_mask.float()                    # [B*G, T_r]
          kl_loss   = (kl * amask).sum() / amask.sum().clamp(min=1)
        =========================================================================
        """
        raise NotImplementedError("[TODO-6] 请实现 _kl_penalty()")


# ==============================================================================
# [TODO-7] 主流程
# ==============================================================================
def main():
    """
    GRPO 完整训练主流程。

    =========================================================================
    [TODO-7] 请实现完整主流程

    步骤:
      1. 初始化
         config    = RLHFConfig()
         grpo_cfg  = GRPOConfig()
         trainer   = GRPOTrainer(config, grpo_cfg)

      2. 构建 DataLoader (与 PPO 共用 ppo_collate_fn，Prompt 左填充)
         dataloader = make_ppo_dataloader(config, trainer.tokenizer)

      3. 训练循环
         for iteration in range(grpo_cfg.num_iterations):
             for batch in dataloader:
                 rollout_data   = trainer.rollout(batch)
                 processed_data = trainer.process(rollout_data)
                 metrics        = trainer.update(rollout_data, processed_data)
             打印 metrics

      4. 可以打印同一 prompt 的 G 个不同 response 及其 reward/advantage，
         观察组内相对奖励的效果。
    =========================================================================
    """
    raise NotImplementedError("[TODO-7] 请实现 GRPO main()")


if __name__ == "__main__":
    main()
