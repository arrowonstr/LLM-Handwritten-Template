"""
PPO (Proximal Policy Optimization) for RLHF — 手撕练习
========================================================
基于 InstructGPT / OpenAI 的 PPO-RLHF 流程。

结构:
  PPOTrainer 类
    ├── __init__()          初始化模型、优化器
    ├── sample()            Phase 1: 采样 (Rollout)
    ├── process()           Phase 2: 加工 (KL → Token Reward → GAE)
    ├── update()            Phase 3: 训练 (PPO multi-epoch update)
    ├── _compute_kl()       内部: 逐 token KL 散度
    ├── _build_token_rewards()  内部: KL 惩罚融入 reward
    ├── _compute_gae()      内部: GAE
    ├── _policy_loss()      内部: Clipped Surrogate Objective
    └── _value_loss()       内部: Clipped Value Loss

核心手撕项:
  TODO1: sample()             — 采样: 生成 response + 收集 probs/values
  TODO2: process()            — 加工: KL → Token Reward → GAE
  TODO3: update()             — 训练: PPO multi-epoch update
  TODO4: _compute_kl          — KL 散度 (per-token)
  TODO5: _build_token_rewards — 逐 token reward 构建
  TODO6: _compute_gae         — GAE 递推
  TODO7: _policy_loss         — PPO Clipped Objective
  TODO8: _value_loss          — Clipped Value Loss
  TODO9: main()               — 组装完整训练主循环 (采样 → 加工 → 更新)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

from rlhf_env import (
    RLHFConfig, MockTokenizer, MockPolicyModel, MockReferenceModel,
    MockRewardModel, MockValueModel,
    generate_response, build_prompt_batch
)


# ==============================================================================
# 超参数
# ==============================================================================
@dataclass
class PPOConfig:
    clip_eps: float = 0.2        # PPO clip 范围
    vf_clip_eps: float = 0.2     # Value function clip 范围
    gamma: float = 1.0           # 折扣因子 (RLHF 中通常 1.0)
    lam: float = 0.95            # GAE λ
    kl_coef: float = 0.1         # KL 惩罚系数
    vf_coef: float = 0.5         # value loss 权重
    ppo_epochs: int = 4          # 每次 rollout 后 PPO 更新轮数
    max_new_tokens: int = 15     # 生成最大 token 数


# ==============================================================================
# PPOTrainer
# ==============================================================================
class PPOTrainer:
    """
    PPO-RLHF Trainer.

    使用方式 (你需要在 main() 中手写):
        trainer = PPOTrainer(config, ppo_config)
        for iteration in ...:
            rollout_data   = trainer.sample(prompt_ids)
            processed_data = trainer.process(rollout_data)
            metrics        = trainer.update(rollout_data, processed_data)
    """

    def __init__(self, config: RLHFConfig, ppo_config: PPOConfig):
        self.config = config
        self.ppo_config = ppo_config

        # 模型
        self.policy_model = MockPolicyModel(config)
        self.ref_model = MockReferenceModel(self.policy_model)
        self.reward_model = MockRewardModel(config)
        self.value_model = MockValueModel(config)

        # 优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy_model.parameters(), lr=config.lr)
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=config.lr)

        # tokenizer (用于打印)
        self.tokenizer = MockTokenizer(config.vocab_size)

    # ------------------------------------------------------------------
    # Phase 1: 采样
    # ------------------------------------------------------------------
    def sample(self, prompt_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        采样阶段：用当前策略为每个 prompt 生成 response，
        并收集后续计算所需的所有张量。

        Args:
            prompt_ids: [B, T_p]

        Returns:
            rollout_data: dict 包含
                'prompt_ids':    [B, T_p]
                'response_ids':  [B, T_r]
                'old_log_probs': [B, T_r]   log π_old(a_t|s_t)
                'ref_log_probs': [B, T_r]   log π_ref(a_t|s_t)
                'rewards':       [B]        reward model 总奖励
                'old_values':    [B, T_r]   V(s_t) 估计

        =====================================================================
        TODO1.1: 实现采样

        提示:
          1. 用 generate_response() 生成 response
             (传入 self.policy_model, prompt_ids, max_new_tokens)
          2. 用当前策略计算 old_log_probs (需要 detach / no_grad,
             因为这是 "旧" 策略的快照, 后续 update 时不能回传梯度)
             self.policy_model.get_log_probs(prompt_ids, response_ids)
          3. 用参考模型计算 ref_log_probs
             self.ref_model.get_log_probs(prompt_ids, response_ids)
          4. 用 reward model 打分
             self.reward_model.get_reward(prompt_ids, response_ids)
          5. 用 value model 估计状态价值
             - 拼接 full_seq = cat([prompt_ids, response_ids], dim=1)
             - all_values = self.value_model(full_seq)    # [B, T_all]
             - 只取 response 部分: old_values = all_values[:, T_p - 1:-1]
             - 也需要 no_grad
          6. 打包成 dict 返回
        =====================================================================
        """
        raise NotImplementedError("TODO1.1: 实现采样")

    # ------------------------------------------------------------------
    # Phase 2: 加工
    # ------------------------------------------------------------------
    def process(self, rollout_data: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        加工阶段：从 rollout 数据计算优势函数和回报。
        
        流程: KL → token rewards → GAE

        Args:
            rollout_data: sample() 返回的 dict

        Returns:
            processed_data: dict 包含
                'advantages': [B, T_r]
                'returns':    [B, T_r]

        =====================================================================
        TODO2.1: 实现加工

        提示:
          从 rollout_data 中取出:
            old_log_probs, ref_log_probs, rewards, old_values

          Step 1: 计算逐 token KL
            kl = self._compute_kl(old_log_probs, ref_log_probs)

          Step 2: 构建逐 token reward (融入 KL 惩罚)
            token_rewards = self._build_token_rewards(
                rewards, kl, self.ppo_config.kl_coef)

          Step 3: 计算 GAE
            advantages, returns = self._compute_gae(
                token_rewards, old_values,
                self.ppo_config.gamma, self.ppo_config.lam)

          打包返回 {'advantages': ..., 'returns': ...}
        =====================================================================
        """
        raise NotImplementedError("TODO2.1: 实现加工")

    # ------------------------------------------------------------------
    # Phase 3: 训练
    # ------------------------------------------------------------------
    def update(self, rollout_data: Dict[str, torch.Tensor],
               processed_data: Dict[str, torch.Tensor]
               ) -> Dict[str, float]:
        """
        训练阶段：PPO 多 epoch 更新 policy 和 value 网络。

        Args:
            rollout_data:   sample() 返回的 dict
            processed_data: process() 返回的 dict

        Returns:
            metrics: dict { 'policy_loss', 'value_loss', 'mean_reward', 'mean_kl' }

        =====================================================================
        TODO3.1: 实现 PPO 训练更新

        提示:
          从 rollout_data 取: prompt_ids, response_ids, old_log_probs, old_values
          从 processed_data 取: advantages, returns
          T_p = prompt_ids.shape[1]    (prompt 长度, 用于切 values)

          多 epoch 循环 (ppo_epochs 次):
            for epoch in range(self.ppo_config.ppo_epochs):

              1. 用当前 (已更新的) 策略重新计算 log probs:
                 new_log_probs = self.policy_model.get_log_probs(
                     prompt_ids, response_ids)
                 (注意: 这里不能 detach, 需要梯度!)

              2. 用当前 value model 重新估计 values:
                 full_seq = cat([prompt_ids, response_ids], dim=1)
                 new_values = self.value_model(full_seq)[:, T_p:]

              3. 计算 policy loss:
                 p_loss = self._policy_loss(
                     new_log_probs, old_log_probs, advantages, clip_eps)

              4. 计算 value loss:
                 v_loss = self._value_loss(
                     new_values, old_values, returns, vf_clip_eps)

              5. 合并 loss:
                 loss = p_loss + vf_coef * v_loss

              6. 反向传播 & 更新:
                 self.policy_optimizer.zero_grad()
                 self.value_optimizer.zero_grad()
                 loss.backward()
                 self.policy_optimizer.step()
                 self.value_optimizer.step()

          最后返回 metrics dict (累计 loss 取平均, mean_reward, mean_kl 等)
        =====================================================================
        """
        raise NotImplementedError("TODO3.1: 实现 PPO 训练更新")

    # ==================================================================
    # 内部方法: 手撕核心
    # ==================================================================

    # ------------------------------------------------------------------
    # TODO4: 逐 Token KL 散度
    # ------------------------------------------------------------------
    def _compute_kl(self, log_probs: torch.Tensor,
                    ref_log_probs: torch.Tensor) -> torch.Tensor:
        """
        计算策略与参考模型之间的逐 token KL 散度。

        Args:
            log_probs:     [B, T]  log π(a_t | s_t)
            ref_log_probs: [B, T]  log π_ref(a_t | s_t)

        Returns:
            kl: [B, T]  每个 token 的 KL 散度 (非负)

        =====================================================================
        TODO4.1: 实现逐 token KL 散度

        提示:
          - 简化形式: KL ≈ log π(a|s) - log π_ref(a|s)
          - 即 log(π / π_ref)，是 KL 在被采样动作上的无偏估计
          - 来自 KL(π||π_ref) = E_π[ log(π/π_ref) ]，
            我们取期望下的单个样本
          - 结果 shape: [B, T]
        =====================================================================
        """
        raise NotImplementedError("TODO4.1: 计算逐 token KL 散度")

    # ------------------------------------------------------------------
    # TODO5: 逐 Token Reward 构建
    # ------------------------------------------------------------------
    def _build_token_rewards(self, rewards: torch.Tensor,
                             kl_per_token: torch.Tensor,
                             kl_coef: float) -> torch.Tensor:
        """
        将 KL 惩罚融入逐 token 奖励信号。

        reward model 给标量奖励，PPO 需要逐 token 奖励。
        做法: 总奖励放最后一步，KL 惩罚放每步。

        Args:
            rewards:      [B]     总奖励 (标量)
            kl_per_token: [B, T]  逐 token KL
            kl_coef:      float   KL 惩罚系数

        Returns:
            token_rewards: [B, T]

        =====================================================================
        TODO5.1: 构建逐 token reward

        提示:
          1. 初始化全零 [B, T] 张量
          2. 最后一个时间步 += rewards (总奖励)
          3. 每个时间步 -= kl_coef * kl_per_token (KL 惩罚)
        =====================================================================
        """
        raise NotImplementedError("TODO5.1: 构建逐 token reward")

    # ------------------------------------------------------------------
    # TODO6: GAE
    # ------------------------------------------------------------------
    def _compute_gae(self, token_rewards: torch.Tensor,
                     values: torch.Tensor,
                     gamma: float, lam: float
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE)。

        Args:
            token_rewards: [B, T]  逐 token 奖励
            values:        [B, T]  V(s_t) 估计
            gamma:         float   折扣因子
            lam:           float   GAE λ

        Returns:
            advantages: [B, T]  GAE 优势
            returns:    [B, T]  returns = advantages + values

        =====================================================================
        TODO6.1: 实现 GAE

        提示:
          从后往前逐步递推:
            last_gae = 0
            for t = T-1, T-2, ..., 0:
                next_value = values[:, t+1] if t < T-1 else 0
                δ_t = rewards[:, t] + γ * next_value - values[:, t]
                last_gae = δ_t + γ * λ * last_gae
                advantages[:, t] = last_gae

          计算完后:
            returns = advantages + values
            advantages = (advantages - mean) / (std + 1e-8)   # 标准化
        =====================================================================
        """
        raise NotImplementedError("TODO6.1: 实现 GAE")

    # ------------------------------------------------------------------
    # TODO7: PPO Clipped Objective
    # ------------------------------------------------------------------
    def _policy_loss(self, new_log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     clip_eps: float) -> torch.Tensor:
        """
        PPO 的 clipped surrogate objective。

        Args:
            new_log_probs: [B, T]  log π_new(a|s)
            old_log_probs: [B, T]  log π_old(a|s) (detached)
            advantages:    [B, T]  GAE 优势
            clip_eps:      float   ε

        Returns:
            policy_loss: scalar (要最小化)

        =====================================================================
        TODO7.1: 实现 PPO Clipped Objective

        提示:
          1. ratio = exp(new_log_probs - old_log_probs)
          2. surr1 = ratio * advantages
          3. surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
          4. loss  = -min(surr1, surr2).mean()

          直觉:
            A>0 (好动作): ratio 最多涨到 1+ε → 限制贪婪
            A<0 (坏动作): ratio 最低降到 1-ε → 限制过激
        =====================================================================
        """
        raise NotImplementedError("TODO7.1: 实现 PPO Clipped Objective")

    # ------------------------------------------------------------------
    # TODO8: Clipped Value Loss
    # ------------------------------------------------------------------
    def _value_loss(self, new_values: torch.Tensor,
                    old_values: torch.Tensor,
                    returns: torch.Tensor,
                    vf_clip_eps: float) -> torch.Tensor:
        """
        Clipped value function loss。

        Args:
            new_values:  [B, T]  当前 V(s)
            old_values:  [B, T]  rollout 时的 V(s) (detached)
            returns:     [B, T]  GAE returns
            vf_clip_eps: float   ε

        Returns:
            value_loss: scalar

        =====================================================================
        TODO8.1: 实现 Clipped Value Loss

        提示:
          1. vf_loss1 = (new_values - returns)²
          2. clipped_values = old_values + clamp(new_values - old_values, -ε, ε)
          3. vf_loss2 = (clipped_values - returns)²
          4. loss = 0.5 * max(vf_loss1, vf_loss2).mean()
        =====================================================================
        """
        raise NotImplementedError("TODO8.1: 实现 Clipped Value Loss")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def generate_sample(self, prompt_ids: torch.Tensor,
                        max_new_tokens: int = 10) -> str:
        """用当前策略生成一条样例 (用于打印监控)"""
        with torch.no_grad():
            resp = generate_response(
                self.policy_model, prompt_ids[:1],
                max_new_tokens=max_new_tokens)
        p = self.tokenizer.decode(prompt_ids[0].tolist())
        r = self.tokenizer.decode(resp[0].tolist())
        return f"'{p}' → '{r}'"


# ==============================================================================
# TODO9: 主流程 — 请自行完成
# ==============================================================================
def main():
    """
    PPO-RLHF 完整训练主流程。

    =========================================================================
    TODO9.1: 实现完整主流程

    提示 (你需要自行组装以下步骤):

      1. 初始化
         - 创建 RLHFConfig, PPOConfig
         - 创建 PPOTrainer
         - 创建 MockTokenizer (用于打印)

      2. 训练循环 (例如 num_iterations = 10)
         for iteration in range(num_iterations):
             a. 构建 prompt batch:
                prompt_ids = build_prompt_batch(tokenizer, config)
             b. 采样:
                rollout_data = trainer.sample(prompt_ids)
             c. 加工:
                processed_data = trainer.process(rollout_data)
             d. 更新:
                metrics = trainer.update(rollout_data, processed_data)
             e. 打印指标和生成样例 (每隔几轮)

      3. 完成后打印最终生成效果
    =========================================================================
    """
    raise NotImplementedError("TODO9.1: 请自行完成 PPO 主流程")


if __name__ == "__main__":
    main()
