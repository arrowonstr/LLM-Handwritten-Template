"""
DPO (Direct Preference Optimization) — 手撕练习
=================================================
基于 Rafailov et al. 2023 的 DPO 算法。

结构:
  DPOTrainer 类
    ├── __init__()                初始化模型、优化器
    ├── prepare_data()            Phase 1: 准备偏好对数据
    ├── compute_log_probs()       Phase 2: 计算 chosen/rejected 的 log probs
    ├── update()                  Phase 3: 计算 DPO loss 并更新
    ├── _sequence_log_prob()      内部: 序列级 log prob 聚合
    ├── _dpo_loss()               内部: Bradley-Terry DPO 损失
    └── extract_implicit_reward() 工具: 提取隐式 reward

DPO 数学推导:
  RLHF 目标:      max_π E[r(x,y)] - β·KL(π || π_ref)
  最优策略:        π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)
  反解 reward:     r(x,y) = β·log(π*(y|x) / π_ref(y|x)) + const
  代入 BT 模型:    P(y_w ≻ y_l) = σ(r(y_w) - r(y_l))
  DPO 损失:        L = -E[log σ(β·(Δlog_chosen - Δlog_rejected))]

核心手撕项:
  TODO1: compute_log_probs()    — 计算 chosen/rejected 的序列级 log probs
  TODO2: update()               — 计算 DPO loss + 梯度更新 + 指标计算
  TODO3: _sequence_log_prob     — 序列级 log prob 聚合
  TODO4: _dpo_loss              — DPO Bradley-Terry Loss
  TODO5: extract_implicit_reward — 隐式 reward 提取
  TODO6: main()                 — 组装完整训练主循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

from rlhf_env import (
    RLHFConfig, MockTokenizer, MockPolicyModel, MockReferenceModel,
    build_preference_pairs
)


# ==============================================================================
# 超参数
# ==============================================================================
@dataclass
class DPOConfig:
    beta: float = 0.1            # β — 控制偏离参考模型的程度
    label_smoothing: float = 0.0 # 可选: 标签平滑 (robust DPO)
    num_epochs: int = 20         # 训练 epoch 数
    log_interval: int = 5        # 日志打印间隔


# ==============================================================================
# DPOTrainer
# ==============================================================================
class DPOTrainer:
    """
    DPO Trainer.

    使用方式 (你需要在 main() 中手写):
        trainer = DPOTrainer(config, dpo_config)
        data    = trainer.prepare_data()
        for epoch in ...:
            log_prob_data = trainer.compute_log_probs(data)
            metrics       = trainer.update(log_prob_data)
    """

    def __init__(self, config: RLHFConfig, dpo_config: DPOConfig):
        self.config = config
        self.dpo_config = dpo_config

        # 模型
        self.policy_model = MockPolicyModel(config)
        self.ref_model = MockReferenceModel(self.policy_model)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(), lr=config.lr)

        # tokenizer (用于打印)
        self.tokenizer = MockTokenizer(config.vocab_size)

    # ------------------------------------------------------------------
    # Phase 1: 准备数据
    # ------------------------------------------------------------------
    def prepare_data(self) -> Dict[str, torch.Tensor]:
        """
        加载并返回偏好对数据。

        Returns:
            dict with 'prompt_ids', 'chosen_ids', 'rejected_ids'
        """
        return build_preference_pairs(self.tokenizer, self.config)

    # ------------------------------------------------------------------
    # Phase 2: 计算 Log Probabilities
    # ------------------------------------------------------------------
    def compute_log_probs(self, data: Dict[str, torch.Tensor]
                          ) -> Dict[str, torch.Tensor]:
        """
        分别计算 policy 和 ref_model 对 chosen/rejected 的序列级 log prob。

        Args:
            data: prepare_data() 返回的字典

        Returns:
            dict 包含:
                'policy_chosen_logps':   [B]
                'policy_rejected_logps': [B]
                'ref_chosen_logps':      [B]   (detached)
                'ref_rejected_logps':    [B]   (detached)

        =====================================================================
        TODO1.1: 实现 log probs 计算

        提示:
          从 data 中取出: prompt_ids, chosen_ids, rejected_ids

          对 chosen:
            1. 用 self.policy_model.get_log_probs(prompt_ids, chosen_ids)
               得到逐 token log probs [B, T]
            2. 构建 mask: chosen_mask = (chosen_ids != pad_token_id).float()
            3. 调用 self._sequence_log_prob(逐token, mask) 聚合为 [B]
            4. 同样用 self.ref_model.get_log_probs() 计算 ref 的

          对 rejected:
            同上流程

          注意:
            - policy 的 log probs 需要保留梯度 (不要 detach)
            - ref 的 log probs 要 .detach() (冻结模型不需要梯度)
            - pad_token_id = self.config.pad_token_id
        =====================================================================
        """
        raise NotImplementedError("TODO1.1: 实现 log probs 计算")

    # ------------------------------------------------------------------
    # Phase 3: 训练更新
    # ------------------------------------------------------------------
    def update(self, log_prob_data: Dict[str, torch.Tensor]
               ) -> Dict[str, float]:
        """
        计算 DPO loss 并执行一步梯度更新。

        Args:
            log_prob_data: compute_log_probs() 返回的字典

        Returns:
            metrics: { 'loss', 'reward_margin', 'accuracy' }

        =====================================================================
        TODO2.1: 实现 DPO 训练更新

        提示:
          1. 调用 self._dpo_loss() 计算 loss 和隐式 rewards:
             loss, chosen_rewards, rejected_rewards = self._dpo_loss(
                 log_prob_data['policy_chosen_logps'],
                 log_prob_data['policy_rejected_logps'],
                 log_prob_data['ref_chosen_logps'],
                 log_prob_data['ref_rejected_logps'],
                 self.dpo_config.beta,
                 self.dpo_config.label_smoothing)

          2. 反向传播:
             self.optimizer.zero_grad()
             loss.backward()
             self.optimizer.step()

          3. 计算监控指标:
             reward_margin = (chosen_rewards - rejected_rewards).mean()
             accuracy = (chosen_rewards > rejected_rewards)的比例

          4. 返回 metrics dict
        =====================================================================
        """
        raise NotImplementedError("TODO2.1: 实现 DPO 训练更新")

    # ==================================================================
    # 内部方法: 手撕核心
    # ==================================================================

    # ------------------------------------------------------------------
    # TODO3: 序列级 Log Probability
    # ------------------------------------------------------------------
    def _sequence_log_prob(self, per_token_log_probs: torch.Tensor,
                           mask: torch.Tensor = None) -> torch.Tensor:
        """
        将逐 token log prob 聚合为序列级 log prob。

        Args:
            per_token_log_probs: [B, T]
            mask:                [B, T]  (1=有效, 0=padding)

        Returns:
            seq_log_prob: [B]

        =====================================================================
        TODO3.1: 实现序列级 log probability

        提示:
          - DPO 原论文用求和: log π(y|x) = Σ_t log π(y_t|x,y_{<t})
          - 实际最常用做法: 对有效长度求均值 (避免长序列的 log prob 偏小)
          - 如果有 mask:
              masked = per_token_log_probs * mask
              seq_log_prob = masked.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
          - 如果没 mask:
              seq_log_prob = per_token_log_probs.mean(dim=-1)
        =====================================================================
        """
        raise NotImplementedError("TODO3.1: 实现序列级 log probability")

    # ------------------------------------------------------------------
    # TODO4: DPO Loss
    # ------------------------------------------------------------------
    def _dpo_loss(self,
                  policy_chosen_logps: torch.Tensor,
                  policy_rejected_logps: torch.Tensor,
                  ref_chosen_logps: torch.Tensor,
                  ref_rejected_logps: torch.Tensor,
                  beta: float,
                  label_smoothing: float = 0.0,
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 DPO 损失。

        Args:
            policy_chosen_logps:   [B]
            policy_rejected_logps: [B]
            ref_chosen_logps:      [B]
            ref_rejected_logps:    [B]
            beta:                  float
            label_smoothing:       float

        Returns:
            loss:             scalar
            chosen_rewards:   [B]  (detached, 用于监控)
            rejected_rewards: [B]  (detached, 用于监控)

        =====================================================================
        TODO4.1: 实现 DPO Loss

        提示:
          1. Log-ratio:
             chosen_logratios  = policy_chosen_logps  - ref_chosen_logps
             rejected_logratios = policy_rejected_logps - ref_rejected_logps
          2. Logits (隐式 reward 差):
             logits = β * (chosen_logratios - rejected_logratios)
          3. Loss:
             若 label_smoothing == 0:
               loss = -F.logsigmoid(logits).mean()
             否则 (robust DPO):
               loss = -((1-ls)*F.logsigmoid(logits) + ls*F.logsigmoid(-logits)).mean()
          4. 隐式 reward (监控用, detach):
             chosen_rewards  = β * chosen_logratios.detach()
             rejected_rewards = β * rejected_logratios.detach()
        =====================================================================
        """
        raise NotImplementedError("TODO4.1: 实现 DPO Loss")

    # ------------------------------------------------------------------
    # TODO5: 隐式 Reward 提取
    # ------------------------------------------------------------------
    def extract_implicit_reward(self, prompt_ids: torch.Tensor,
                                response_ids: torch.Tensor) -> torch.Tensor:
        """
        从训练好的策略中提取隐式 reward。

        Args:
            prompt_ids:   [B, T_p]
            response_ids: [B, T_r]

        Returns:
            implicit_reward: [B]

        =====================================================================
        TODO5.1: 隐式 Reward

        提示:
          r(x,y) = β * (log π(y|x) - log π_ref(y|x))
          
          1. 获取 policy 逐 token log probs → _sequence_log_prob
          2. 获取 ref 逐 token log probs → _sequence_log_prob
          3. return β * (policy_seq - ref_seq)
          
          注意用 torch.no_grad()
        =====================================================================
        """
        raise NotImplementedError("TODO5.1: 提取隐式 Reward")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def print_preference_pair(self, data: Dict[str, torch.Tensor], idx: int):
        """打印第 idx 个偏好对"""
        p = self.tokenizer.decode(data['prompt_ids'][idx].tolist())
        c = self.tokenizer.decode(data['chosen_ids'][idx].tolist())
        r = self.tokenizer.decode(data['rejected_ids'][idx].tolist())
        print(f"  Prompt:   '{p}'")
        print(f"  Chosen:   '{c}'")
        print(f"  Rejected: '{r}'")


# ==============================================================================
# TODO6: 主流程 — 请自行完成
# ==============================================================================
def main():
    """
    DPO 完整训练主流程。

    =========================================================================
    TODO6.1: 实现完整主流程

    提示 (你需要自行组装以下步骤):

      1. 初始化
         - 创建 RLHFConfig, DPOConfig
         - 创建 DPOTrainer

      2. 准备数据
         data = trainer.prepare_data()
         (可选: 打印几个偏好对)

      3. 训练前: 查看初始隐式 reward
         chosen_r  = trainer.extract_implicit_reward(prompt_ids, chosen_ids)
         rejected_r = trainer.extract_implicit_reward(prompt_ids, rejected_ids)
         此时 margin 应接近 0 (策略 ≈ 参考模型)

      4. 训练循环
         for epoch in range(dpo_config.num_epochs):
             log_prob_data = trainer.compute_log_probs(data)
             metrics       = trainer.update(log_prob_data)
             (每隔几轮打印指标)

      5. 训练后: 再次查看隐式 reward
         此时 chosen reward 应 > rejected reward
         accuracy 应接近 1.0
    =========================================================================
    """
    raise NotImplementedError("TODO6.1: 请自行完成 DPO 主流程")


if __name__ == "__main__":
    main()
