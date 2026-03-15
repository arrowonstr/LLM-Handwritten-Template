"""
DPO (Direct Preference Optimization) — 手撕练习
=================================================
基于 Rafailov et al. 2023，对齐 TRL DPOTrainer 的真实工程结构。

流程:
  DPOTrainer
    ├── _sequence_log_prob()   内部: per-token log probs 聚合为序列级
    ├── _dpo_loss()            内部: Bradley-Terry DPO 损失
    ├── compute_ref_logps()    预计算 (可选, RLHF 通常只做一次)
    └── train_step()           单步训练: forward → log_probs → loss → update

关键工程点 (对齐官方 TRL):
  - DataLoader 中用 dpo_collate_fn 完成拼接、右填充、labels 构建 [TODO-B]
  - get_log_probs 统一入口，用 labels 中的 -100 构建 loss_mask [TODO-C]
  - Reference model log probs 可以提前缓存 (预计算) 节省显存

核心手撕项:
  [TODO-B]  rlhf_env.py dpo_collate_fn          ← 拼接 + 右填充 + labels
  [TODO-C]  rlhf_env.py get_log_probs           ← 统一 log prob
  [TODO-1]  _sequence_log_prob()                ← per-token → 序列级
  [TODO-2]  _dpo_loss()                         ← DPO Bradley-Terry loss
  [TODO-3]  train_step()                        ← 单步训练
  [TODO-4]  main()                              ← 主循环
  [TODO-5]  extract_implicit_reward()           ← 从策略反解隐式 reward
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
    DPO Trainer (对齐 TRL DPOTrainer 的整体结构)。

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
            batch: dpo_collate_fn 返回的 dict，包含:
                'chosen_input_ids':      [B, T_c]
                'chosen_attention_mask': [B, T_c]
                'chosen_labels':         [B, T_c]   prompt/pad 位置 = -100
                'rejected_input_ids':      [B, T_r]
                'rejected_attention_mask': [B, T_r]
                'rejected_labels':         [B, T_r]

        Returns:
            metrics: {'loss', 'reward_margin', 'accuracy'}

        =========================================================================
        [TODO-3] 请实现 DPOTrainer.train_step()

        Step 1 — 取出数据:
            chosen_ids   = batch['chosen_input_ids']
            chosen_mask  = batch['chosen_attention_mask']
            chosen_lbl   = batch['chosen_labels']
            rejected_ids = batch['rejected_input_ids']
            rejected_mask= batch['rejected_attention_mask']
            rejected_lbl = batch['rejected_labels']

        Step 2 — 计算 policy 的 per-token log probs (需要梯度):
            pc_logps, pc_mask = self.policy_model.get_log_probs(
                chosen_ids, chosen_mask, labels=chosen_lbl)
            pr_logps, pr_mask = self.policy_model.get_log_probs(
                rejected_ids, rejected_mask, labels=rejected_lbl)

          聚合为序列级:
            policy_chosen_logps   = self._sequence_log_prob(pc_logps, pc_mask)
            policy_rejected_logps = self._sequence_log_prob(pr_logps, pr_mask)

        Step 3 — 计算 ref 的 per-token log probs (不需要梯度):
            with torch.no_grad():
                rc_logps, rc_mask = self.ref_model.get_log_probs(
                    chosen_ids, chosen_mask, labels=chosen_lbl)
                rr_logps, rr_mask = self.ref_model.get_log_probs(
                    rejected_ids, rejected_mask, labels=rejected_lbl)

            ref_chosen_logps   = self._sequence_log_prob(rc_logps, rc_mask)
            ref_rejected_logps = self._sequence_log_prob(rr_logps, rr_mask)

        Step 4 — DPO Loss:
            loss, chosen_rewards, rejected_rewards = self._dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                self.dpo_config.beta, self.dpo_config.label_smoothing)

        Step 5 — 反向传播:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        Step 6 — 计算监控指标:
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        Step 7 — 返回 {'loss', 'reward_margin', 'accuracy'}
        =========================================================================
        """
        raise NotImplementedError("[TODO-3] 请实现 DPOTrainer.train_step()")

    # =========================================================================
    # 内部方法 — 手撕核心
    # =========================================================================

    def _sequence_log_prob(self,
                           per_token_log_probs: torch.Tensor,
                           loss_mask: torch.Tensor) -> torch.Tensor:
        """
        将 per-token log probs 聚合为序列级 log prob。

        Args:
            per_token_log_probs: [B, T-1]  (get_log_probs 输出，无效位置已清零)
            loss_mask:           [B, T-1]  1=有效, 0=无效 (bool 或 long)

        Returns:
            seq_log_prob: [B]

        =========================================================================
        [TODO-1] 请实现 _sequence_log_prob()

        DPO 原论文用求和，但实践中通常用『对有效 token 取均值』以避免
        长序列 log prob 天然偏小的偏差 (length bias)。

        做法:
            mask_f = loss_mask.float()                           # [B, T-1]
            seq_log_prob = (per_token_log_probs * mask_f).sum(dim=-1) \
                           / mask_f.sum(dim=-1).clamp(min=1)    # [B]

        注意: per_token_log_probs 的无效位置已经被 get_log_probs 清零了，
             所以直接 sum 不会引入错误；但仍需要 mask_f 用于计算有效长度。
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
            policy_chosen_logps:   [B]
            policy_rejected_logps: [B]
            ref_chosen_logps:      [B]
            ref_rejected_logps:    [B]
            beta, label_smoothing: float

        Returns:
            loss:             scalar
            chosen_rewards:   [B]  (detached, 仅用于监控)
            rejected_rewards: [B]  (detached)

        =========================================================================
        [TODO-2] 请实现 DPO Loss

        Step 1 — Log-ratio (隐式 reward 的差值):
            chosen_logratios   = policy_chosen_logps   - ref_chosen_logps
            rejected_logratios = policy_rejected_logps - ref_rejected_logps

        Step 2 — Logits (Bradley-Terry 的 preference logit):
            logits = beta * (chosen_logratios - rejected_logratios)

        Step 3 — Loss:
            若 label_smoothing == 0 (标准 DPO):
                loss = -F.logsigmoid(logits).mean()
            否则 (robust DPO, Azar et al. 2024):
                ls = label_smoothing
                loss = (-(1-ls) * F.logsigmoid(logits)
                        - ls * F.logsigmoid(-logits)).mean()

        Step 4 — 隐式 reward (用于监控, 不涉及梯度):
            chosen_rewards   = beta * chosen_logratios.detach()    # [B]
            rejected_rewards = beta * rejected_logratios.detach()  # [B]

        直觉:
            DPO 不需要显式 reward model。
            通过 log(π/π_ref) 可以反解出当前策略对应的『隐式 reward』。
            训练目标是让 chosen 对应的隐式 reward > rejected 的。
        =========================================================================
        """
        raise NotImplementedError("[TODO-2] 请实现 _dpo_loss()")

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────────────────
    def extract_implicit_reward(self,
                                input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
        """
        从训练好的策略中提取隐式 reward: r(x,y) = β * (log π(y|x) - log π_ref(y|x))

        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]
            labels:         [B, T]  (-100 用于屏蔽 prompt/pad)

        Returns:
            implicit_reward: [B]

        =========================================================================
        [TODO-5] 请实现 extract_implicit_reward()

        DPO 的核心理论洞见: 最优策略隐含了一个 reward model，形式为:
            r*(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)
        去掉配分函数 Z(x) 后，可以用 log-ratio 作为隐式 reward 的代理:
            r_implicit(x, y) = β * (log π(y|x) - log π_ref(y|x))

        步骤:
          with torch.no_grad():
            Step 1 — 计算 policy 的序列 log prob:
              p_logps, p_mask = self.policy_model.get_log_probs(
                  input_ids, attention_mask, labels=labels)
              policy_seq = self._sequence_log_prob(p_logps, p_mask)   # [B]

            Step 2 — 计算 ref 的序列 log prob:
              r_logps, r_mask = self.ref_model.get_log_probs(
                  input_ids, attention_mask, labels=labels)
              ref_seq = self._sequence_log_prob(r_logps, r_mask)       # [B]

          Step 3 — 隐式 reward:
            return self.dpo_config.beta * (policy_seq - ref_seq)       # [B]

        注意: 此函数依赖 _sequence_log_prob (TODO-1)，需先完成 TODO-1。
        =========================================================================
        """
        raise NotImplementedError("[TODO-5] 请实现 extract_implicit_reward()")


# ==============================================================================
# [TODO-4] 主流程
# ==============================================================================
def main():
    """
    DPO 完整训练主流程。

    =========================================================================
    [TODO-4] 请实现完整主流程

    步骤:
      1. 初始化
         config    = RLHFConfig()
         dpo_cfg   = DPOConfig()
         trainer   = DPOTrainer(config, dpo_cfg)

      2. 构建 DataLoader (内部已用 dpo_collate_fn 完成拼接、右填充、labels)
         dataloader = make_dpo_dataloader(config, trainer.tokenizer)

      3. 训练前: 打印初始隐式 reward margin (应接近 0)

      4. 训练循环
         for epoch in range(dpo_cfg.num_epochs):
             for batch in dataloader:
                 metrics = trainer.train_step(batch)
             每隔 log_interval 轮打印 metrics

      5. 训练后: 打印最终隐式 reward margin (chosen > rejected)
    =========================================================================
    """
    raise NotImplementedError("[TODO-4] 请实现 DPO main()")


if __name__ == "__main__":
    main()
