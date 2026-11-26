from __future__ import annotations


from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from sample_factory.algo.learning.learner import BaseLearner, DefaultLearner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.torch_utils import masked_select
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionDistribution, Config, PolicyID


class DGBatchLossLearner(DefaultLearner):
    """DefaultLearner + Jannek's encoder_batch_loss, applied to the DG pre-threshold activity,
    + optional CA3->DG feedback prediction loss (dg_feedback_pred_loss)."""

    def _get_dg_projection(self):
        encoder = getattr(self.actor_critic, "encoder", None)  # shared weights
        if encoder is None:
            encoder = self.actor_critic.actor_encoder  # separate weights: DG of the actor
        return encoder.DG_projection

    def _get_core(self):
        core = getattr(self.actor_critic, "core", None)  # shared weights
        if core is None:
            core = self.actor_critic.actor_core  # separate weights: core of the actor
        return core

    def _dg_batch_loss(self, valids: Tensor, num_invalids: int):
        dg = self._get_dg_projection()
        pre_threshold = dg.pre_threshold  # (batch, Hippo_n_feature), BN output from this minibatch's forward pass
        dg.pre_threshold = None  # don't keep the graph alive on the module
        if pre_threshold is None:
            return 0
        #pre_threshold = masked_select(pre_threshold, valids, num_invalids)
        pre_threshold = pre_threshold[valids]  # keep only valid rows; stays (n_valid, 16)
        if pre_threshold.shape[0] == 0:
            return 0
        # Penalty for not used sequences in this mini batch
        mask_active_now = pre_threshold > dg.intercept
        batch_mask = torch.logical_not(torch.any(mask_active_now, dim=0))  # (Hippo_n_feature,)
        # push the largest pre-threshold value of each silent feature up towards the threshold
        batch_max = pre_threshold.max(dim=0).values
        batch_penalty = (torch.relu(dg.intercept - batch_max) * batch_mask).sum() / (batch_mask.sum() + 1e-6)
        # for W&B
        self._dg_stats = dict(
            dg_batch_penalty=batch_penalty.detach(),
            dg_silent_features=batch_mask.sum().detach(),             # how many of 16 never fired in this minibatch
            dg_active_fraction=mask_active_now.float().mean().detach(),  # overall DG sparsity, should stay ~0.025
        )
        return batch_penalty

    def _feedback_pred_loss(self):
        """CA3->DG feedback prediction loss from this minibatch's core forward (None if not available)."""
        core = self._get_core()
        get_loss = getattr(core, "feedback_pred_loss", None)
        if get_loss is None:
            return None
        loss = get_loss()
        if loss is None:
            return None
        self._fb_stats = dict(dg_feedback_pred_mse=loss.detach())
        return loss

    def _record_summaries(self, train_loop_vars):
        stats = super()._record_summaries(train_loop_vars)
        for key, value in getattr(self, "_dg_stats", {}).items():
            stats[key] = value.item()
        for key, value in getattr(self, "_fb_stats", {}).items():
            stats[key] = value.item()
        return stats

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        ) = super()._calculate_losses(mb, num_invalids)

        if getattr(self.cfg, "DG_batch_loss", False):
            batch_penalty = self._dg_batch_loss(mb.valids, num_invalids)
            policy_loss = policy_loss + self.cfg.DG_batch_loss_coeff * batch_penalty

        if getattr(self.cfg, "dg_feedback_pred_loss", False):
            fb_loss = self._feedback_pred_loss()
            if fb_loss is not None:
                policy_loss = policy_loss + self.cfg.dg_feedback_pred_loss_coeff * fb_loss

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries


def make_hipposlam_learner(
    cfg: Config, env_info: EnvInfo, policy_versions_tensor: Tensor, policy_id: PolicyID, param_server: ParameterServer
) -> BaseLearner:
    if getattr(cfg, "DG_batch_loss", False) or getattr(cfg, "dg_feedback_pred_loss", False):
        return DGBatchLossLearner(cfg, env_info, policy_versions_tensor, policy_id, param_server)
    return DefaultLearner(cfg, env_info, policy_versions_tensor, policy_id, param_server)