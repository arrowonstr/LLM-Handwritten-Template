"""
GRPO (Group Relative Policy Optimization) — 手撕练习
=====================================================
基于 DeepSeek 的 GRPO 算法 (DeepSeekMath, 2024)。

结构:
  GRPOTrainer 类
    ├── __init__()              初始化模型、优化器
    ├── sample()                Phase 1: 组采样 (每个 prompt 生成 G 个 response)
    ├── process()               Phase 2: 打分 + 组内标准化 → advantage
    ├── update()                Phase 3: Clipped objective + KL 惩罚
    ├── _group_advantages()     内部: 组内 advantage 归一化
    ├── _policy_loss()          内部: Clipped Surrogate Objective
    └── _kl_penalty()           内部: GRPO 形式 KL 散度

GRPO vs PPO:
  - 不需要 Value Model → 节省显存
  - 用组内比较 (self-play) 代替 V(s) 估计 advantage
  - KL 使用 f-divergence 无偏估计: e^x - x - 1

核心手撕项:
  TODO1: sample()            — 组采样 (repeat_interleave + generate)
  TODO2: process()           — 打分 + 组内标准化
  TODO3: update()            — Clipped objective + KL + 梯度更新
  TODO4: _group_advantages   — 组内标准化
  TODO5: _policy_loss        — Clipped Objective (sequence-level advantage)
  TODO6: _kl_penalty         — GRPO KL 估计器
  TODO7: main()              — 组装完整训练主循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

from rlhf_env import (
    RLHFConfig, MockTokenizer, MockPolicyModel, MockReferenceModel,
    MockRewardModel,
    generate_response, build_prompt_batch
)


# ==============================================================================
# 超参数
# ==============================================================================
@dataclass
class GRPOConfig:
    group_size: int = 4          # G — 每个 prompt 的 response 数
    clip_eps: float = 0.2        # clip 范围
    kl_coef: float = 0.04        # KL 惩罚系数 β
    num_epochs: int = 1          # 每次 rollout 训练轮数 (通常 1)
    num_iterations: int = 10     # 外循环迭代数
    max_new_tokens: int = 12     # 生成最大 token 数


# ==============================================================================
# GRPOTrainer
# ==============================================================================
class GRPOTrainer:
    """
    GRPO Trainer.

    使用方式 (你需要在 main() 中手写):
        trainer = GRPOTrainer(config, grpo_config)
        for iteration in ...:
            sample_data     = trainer.sample(prompt_ids)
            processed_data  = trainer.process(sample_data, prompt_ids)
            metrics         = trainer.update(sample_data, processed_data)
    """

    def __init__(self, config: RLHFConfig, grpo_config: GRPOConfig):
        self.config = config
        self.grpo_config = grpo_config

        # 模型 (注意: 没有 Value Model!)
        self.policy_model = MockPolicyModel(config)
        self.ref_model = MockReferenceModel(self.policy_model)
        self.reward_model = MockRewardModel(config)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(), lr=config.lr)

        # tokenizer
        self.tokenizer = MockTokenizer(config.vocab_size)

    # ------------------------------------------------------------------
    # Phase 1: 组采样
    # ------------------------------------------------------------------
    def sample(self, prompt_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        为每个 prompt 生成 G 个 response，并收集 old log probs。

        Args:
            prompt_ids: [B, T_p]

        Returns:
            sample_data: dict 包含
                'expanded_prompt_ids': [B*G, T_p]
                'response_ids':        [B*G, T_r]
                'old_log_probs':       [B*G, T_r]  (detached)
                'ref_log_probs':       [B*G, T_r]  (detached)

        =====================================================================
        TODO1.1: 实现组采样

        提示:
          1. 复制 prompt:
             expanded = prompt_ids.repeat_interleave(group_size, dim=0)
             [B, T_p] → [B*G, T_p]
             例: B=2, G=3 → [p0,p0,p0, p1,p1,p1]

          2. 生成 response (随机采样 → 同一 prompt 的 G 条不同):
             response_ids = generate_response(
                 self.policy_model, expanded, max_new_tokens=...)

          3. 收集 old_log_probs (detach) 和 ref_log_probs:
             with torch.no_grad():
                 old_log_probs = self.policy_model.get_log_probs(...)
                 ref_log_probs = self.ref_model.get_log_probs(...)

          4. 打包返回
        =====================================================================
        """
        raise NotImplementedError("TODO1.1: 实现组采样")

    # ------------------------------------------------------------------
    # Phase 2: 加工
    # ------------------------------------------------------------------
    def process(self, sample_data: Dict[str, torch.Tensor],
                prompt_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        打分 + 组内标准化 → 计算 advantage。

        Args:
            sample_data: sample() 返回的字典
            prompt_ids:  [B, T_p]  原始 prompt (用于确定 B)

        Returns:
            processed_data: dict 包含
                'rewards':    [B*G]    原始 reward
                'advantages': [B*G]    组内标准化后的 advantage

        =====================================================================
        TODO2.1: 实现加工

        提示:
          1. 从 sample_data 取出 expanded_prompt_ids 和 response_ids
          2. 用 reward model 打分:
             rewards = self.reward_model.get_reward(
                 expanded_prompt_ids, response_ids)    # [B*G]
          3. 组内标准化:
             B = prompt_ids.shape[0]
             G = self.grpo_config.group_size
             advantages = self._group_advantages(rewards, B, G)
          4. 返回 {'rewards': ..., 'advantages': ...}
        =====================================================================
        """
        raise NotImplementedError("TODO2.1: 实现加工")

    # ------------------------------------------------------------------
    # Phase 3: 训练
    # ------------------------------------------------------------------
    def update(self, sample_data: Dict[str, torch.Tensor],
               processed_data: Dict[str, torch.Tensor]
               ) -> Dict[str, float]:
        """
        GRPO 更新: clipped objective + KL penalty。

        Args:
            sample_data:    sample() 返回的字典
            processed_data: process() 返回的字典

        Returns:
            metrics: { 'policy_loss', 'kl_loss', 'total_loss',
                       'mean_reward', 'mean_advantage' }

        =====================================================================
        TODO3.1: 实现 GRPO 训练更新

        提示:
          从 sample_data 取: expanded_prompt_ids, response_ids,
                             old_log_probs, ref_log_probs
          从 processed_data 取: advantages

          多 epoch 循环 (num_epochs 次, GRPO 通常 1):
            for epoch in range(self.grpo_config.num_epochs):

              1. 用当前策略重新计算 log probs (保留梯度!):
                 new_log_probs = self.policy_model.get_log_probs(
                     expanded_prompt_ids, response_ids)

              2. 计算 policy loss:
                 p_loss = self._policy_loss(
                     new_log_probs, old_log_probs, advantages, clip_eps)

              3. 计算 KL 惩罚:
                 kl_loss = self._kl_penalty(new_log_probs, ref_log_probs)

              4. 总损失:
                 total_loss = p_loss + kl_coef * kl_loss

              5. 反向传播:
                 self.optimizer.zero_grad()
                 total_loss.backward()
                 self.optimizer.step()

          返回 metrics dict
        =====================================================================
        """
        raise NotImplementedError("TODO3.1: 实现 GRPO 训练更新")

    # ==================================================================
    # 内部方法: 手撕核心
    # ==================================================================

    # ------------------------------------------------------------------
    # TODO4: 组内 Advantage
    # ------------------------------------------------------------------
    def _group_advantages(self, rewards: torch.Tensor,
                          batch_size: int,
                          group_size: int) -> torch.Tensor:
        """
        组内相对优势: 用组内 reward 标准化代替 Value Model。

        Args:
            rewards:    [B*G]
            batch_size: B
            group_size: G

        Returns:
            advantages: [B*G]

        =====================================================================
        TODO4.1: 实现组内 Advantage

        提示:
          1. reshape: grouped = rewards.view(B, G)
          2. 组均值: μ = grouped.mean(dim=1, keepdim=True)    # [B, 1]
             组标准差: σ = grouped.std(dim=1, keepdim=True)   # [B, 1]
          3. 标准化: advantages = (grouped - μ) / (σ + 1e-8)
          4. flatten: advantages = advantages.view(-1)

        直觉:
          同组内 reward 高于均值 → advantage > 0 → 增大概率
          同组内 reward 低于均值 → advantage < 0 → 减小概率
        =====================================================================
        """
        raise NotImplementedError("TODO4.1: 实现组内 Advantage")

    # ------------------------------------------------------------------
    # TODO5: GRPO Clipped Objective
    # ------------------------------------------------------------------
    def _policy_loss(self, new_log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     clip_eps: float) -> torch.Tensor:
        """
        Clipped policy loss (PPO 风格, 但 advantage 是 per-sequence)。

        Args:
            new_log_probs: [B*G, T]
            old_log_probs: [B*G, T]  (detached)
            advantages:    [B*G]     序列级 advantage
            clip_eps:      float

        Returns:
            policy_loss: scalar

        =====================================================================
        TODO5.1: 实现 GRPO Clipped Objective

        提示:
          与 PPO 几乎一致, 但 advantage 是 per-sequence 不是 per-token。

          1. 扩展 advantage 到 token 维度:
             adv = advantages.unsqueeze(1)             # [B*G, 1]
             (broadcasting → [B*G, T])
          2. ratio = exp(new_log_probs - old_log_probs) # [B*G, T]
          3. surr1 = ratio * adv
             surr2 = clamp(ratio, 1-ε, 1+ε) * adv
          4. loss = -min(surr1, surr2).mean()
        =====================================================================
        """
        raise NotImplementedError("TODO5.1: 实现 GRPO Clipped Objective")

    # ------------------------------------------------------------------
    # TODO6: GRPO KL 散度
    # ------------------------------------------------------------------
    def _kl_penalty(self, new_log_probs: torch.Tensor,
                    ref_log_probs: torch.Tensor) -> torch.Tensor:
        """
        GRPO 形式的 KL 散度惩罚。

        Args:
            new_log_probs: [B*G, T]
            ref_log_probs: [B*G, T]

        Returns:
            kl_loss: scalar

        =====================================================================
        TODO6.1: 实现 GRPO KL 散度惩罚

        提示:
          GRPO 使用 f-divergence 无偏估计器 (与 PPO 的 log(π/π_ref) 不同):
            D_KL ≈ exp(log_ratio) - log_ratio - 1
            其中 log_ratio = ref_log_probs - new_log_probs  即 log(π_ref/π)

          数学: e^x - x - 1 ≥ 0 ∀x, 且 x=0 时取等 (凸函数性质)
          所以这个估计器天然非负。

          步骤:
            1. log_ratio = ref_log_probs - new_log_probs
            2. kl = exp(log_ratio) - log_ratio - 1       # [B*G, T]
            3. kl_loss = kl.mean()                        # scalar
        =====================================================================
        """
        raise NotImplementedError("TODO6.1: 实现 GRPO KL 散度惩罚")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def generate_sample(self, prompt_ids: torch.Tensor,
                        max_new_tokens: int = 8) -> str:
        """生成一条样例"""
        with torch.no_grad():
            resp = generate_response(
                self.policy_model, prompt_ids[:1],
                max_new_tokens=max_new_tokens)
            reward = self.reward_model.get_reward(prompt_ids[:1], resp[:1])
        p = self.tokenizer.decode(prompt_ids[0].tolist())
        r = self.tokenizer.decode(resp[0].tolist())
        return f"'{p}' → '{r}' (reward={reward.item():.3f})"

    def show_group(self, prompt_ids: torch.Tensor):
        """展示一个 prompt 的组采样结果"""
        single = prompt_ids[:1]
        sample_data = self.sample(single)
        processed_data = self.process(sample_data, single)
        G = self.grpo_config.group_size
        p = self.tokenizer.decode(single[0].tolist())
        print(f"  Prompt: '{p}'")
        for g in range(G):
            r = self.tokenizer.decode(
                sample_data['response_ids'][g].tolist())
            rw = processed_data['rewards'][g].item()
            adv = processed_data['advantages'][g].item()
            print(f"    [{g}] '{r}' | reward={rw:.3f} | advantage={adv:.3f}")


# ==============================================================================
# TODO7: 主流程 — 请自行完成
# ==============================================================================
def main():
    """
    GRPO 完整训练主流程。

    =========================================================================
    TODO7.1: 实现完整主流程

    提示 (你需要自行组装以下步骤):

      1. 初始化
         - 创建 RLHFConfig, GRPOConfig
         - 创建 GRPOTrainer
         - 创建 MockTokenizer (用于打印, 或直接用 trainer.tokenizer)

      2. 训练前: 查看初始生成质量
         prompt_ids = build_prompt_batch(tokenizer, config)
         print(trainer.generate_sample(prompt_ids))

      3. 训练循环 (例如 num_iterations 次)
         for iteration in range(grpo_config.num_iterations):
             a. 构建 prompt batch:
                prompt_ids = build_prompt_batch(...)
             b. 采样 (关键! 每个 prompt 生成 G 个 response):
                sample_data = trainer.sample(prompt_ids)
             c. 加工 (打分 + 组内标准化):
                processed_data = trainer.process(sample_data, prompt_ids)
             d. 更新 (clipped objective + KL):
                metrics = trainer.update(sample_data, processed_data)
             e. 打印指标 (每隔几轮)

      4. 训练后: 查看生成质量 + 展示组采样
         print(trainer.generate_sample(prompt_ids))
         trainer.show_group(prompt_ids)
    =========================================================================
    """
    raise NotImplementedError("TODO7.1: 请自行完成 GRPO 主流程")


if __name__ == "__main__":
    main()
