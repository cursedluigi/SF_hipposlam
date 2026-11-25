"""

    x̂_{t+1} = (W_CA3 + L·R(a_t)) · x_t

One step t of the rollout is:

    o_t                      observation at step t
    u_t   = DG(o_t)          DG output at step t, 16 values            (npz: dg[t])
    x_t   = W_CA3 x_{t-1}    CA3 core state AFTER u_t has been injected (npz: x[t], first 1136 dims)
            with u_t         -> u_t sits in slot 0 of x_t
    a_t   = policy(x_t, …)   action chosen in the same forward pass     (npz: action[t])
    env.step(a_t) -> o_{t+1}

So the next state is

    x_{t+1} = ca3_step(x_t, u_{t+1}),   u_{t+1} = DG(o_{t+1}),   o_{t+1} caused by a_t

and the predictor uses what is known at decision time (x_t, a_t) to guess the one unknown, u_{t+1}:

    û_{t+1}  = L · R(a_t) · x_t          L: (n_feature, r),  R(a): (r, D),  D = n_feature * (L+R-1)
    x̂_{t+1} = ca3_step(x_t, û_{t+1})

A transition (x_t, a_t, u_{t+1}, x_{t+1}) only exists when t and t+1 are in the same episode,
i.e. episode_id[t] == episode_id[t+1]. The last step of each episode has no transition.

Why predicting u_{t+1} is enough
--------------------------------
W_CA3 (shift by one slot, clear slot 0, add the new DG input to slots 0..R-1) is fixed and
reproduced exactly by `ca3_step`. Given x_t, the only unknown part of x_{t+1} is u_{t+1}.
The error on x_{t+1} is non-zero only in slots 0..R-1, where it equals (u_{t+1} - û_{t+1})
repeated R times, so the loss on the 16 DG targets has the same minimiser as the loss on the
full CA3 state (full-state squared error = R × `prediction_loss`).

Training rules
--------------
- `delta_update`: the delta rule (Widrow-Hoff / LMS). L is fixed, only R(a_t) changes:
      ΔR(a_t) = lr · (Lᵀ e_{t+1}) · x_tᵀ,   e_{t+1} = u_{t+1} - û_{t+1}
  With L = identity (rank == n_feature, `fixed_L="identity"`) this is the textbook delta rule on a
  full 16 × D matrix W(a) per action. With a random fixed L of rank r < 16, predictions can only
  lie in the r-dim column space of L, which caps how well any target can be fitted.
  Batch size 1, steps in time order = online delta rule; batch > 1 = average of per-transition
  delta updates.
- `prediction_loss` + autograd + an optimiser (e.g. Adam) trains L and R together. That is NOT the
  delta rule; keep it only as a comparison.

x_t is the core part of the state only (bypass dropped) and is treated as fixed input (no gradient
into DG; offline it is a stored array anyway).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Must match the training config (BypassSS, --Hippo_n_feature=16 --Hippo_L=64 --Hippo_R=8)
HIPPO_N_FEATURE = 16
HIPPO_L = 64
HIPPO_R = 8
EXPANDED_LENGTH = HIPPO_R + HIPPO_L - 1  # 71
CORE_SIZE = HIPPO_N_FEATURE * EXPANDED_LENGTH  # 1136 (rnn_size 1149 = 1136 + 13 bypass)
N_ACTIONS = 5  # REDUCED_ACTION_SET in dmlab_gym.py


def ca3_step(x_t: torch.Tensor, u_tp1: torch.Tensor, R: int = HIPPO_R) -> torch.Tensor:
    """W_CA3: x_{t+1} = ca3_step(x_t, u_{t+1}). Same update as SimpleSequenceWithBypassCore.forward
    (single-step branch), core part only.

    x_t: (B, n_feature * expanded_length), u_tp1: (B, n_feature). Returns x_{t+1} core, same shape.
    """
    B, n_f = u_tp1.shape
    s = x_t.view(B, n_f, -1).roll(shifts=1, dims=-1).clone()  # shift every slot one step along
    s[:, :, 0] = 0  # roll wrapped the old last slot into slot 0: erase it
    s[:, :, :R] += u_tp1.unsqueeze(-1)  # inject u_{t+1} into slots 0..R-1
    return s.view(B, -1)


def dg_from_state(x_t: torch.Tensor, n_feature: int = HIPPO_N_FEATURE) -> torch.Tensor:
    """u_t = slot 0 of x_t (Phase 1 check [1]). Applied to x_{t+1} it gives the target u_{t+1}."""
    return x_t.view(x_t.shape[0], n_feature, -1)[:, :, 0]


class ActionCondCA3Predictor(nn.Module):
    """û_{t+1} = L · R(a_t) · x_t.

    mode="action": one R per action (the model under test)
    mode="shared": a single R for all actions (action-free learned baseline)
    mode="zero":   L·R ≡ 0, i.e. pure W_CA3 (û_{t+1} = 0; trivial baseline)

    fixed_L="random":   L ~ N(0, 1/rank), trainable unless train_L=False
    fixed_L="identity": L = I (requires rank == n_feature); û_{t+1} = R(a_t) x_t, full W(a) per action
    """

    def __init__(
        self,
        rank: int,
        mode: str = "action",
        n_actions: int = N_ACTIONS,
        n_feature: int = HIPPO_N_FEATURE,
        core_size: int = CORE_SIZE,
        fixed_L: str = "random",
        train_L: bool = True,
    ):
        super().__init__()
        assert mode in ("action", "shared", "zero")
        assert fixed_L in ("random", "identity")
        self.mode = mode
        self.n_feature = n_feature
        n_R = n_actions if mode == "action" else 1
        if fixed_L == "identity":
            assert rank == n_feature, "identity L needs rank == n_feature"
            L0 = torch.eye(n_feature)
        else:
            L0 = torch.randn(n_feature, rank) / rank**0.5
        # R starts at zero: û_{t+1} starts at 0 (= W_CA3 alone) and both factors still get gradient.
        self.L = nn.Parameter(L0)
        self.R = nn.Parameter(torch.zeros(n_R, rank, core_size))
        if mode == "zero" or not train_L:
            self.L.requires_grad_(False)
        if mode == "zero":
            self.R.requires_grad_(False)

    def _R_index(self, a_t: torch.Tensor) -> torch.Tensor:
        return a_t if self.mode == "action" else torch.zeros_like(a_t)

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Returns û_{t+1}: (B, n_feature), from x_t (B, D) and a_t (B,)."""
        if self.mode == "zero":
            return x_t.new_zeros(x_t.shape[0], self.n_feature)
        R_a = self.R[self._R_index(a_t)]  # (B, r, D)
        z = torch.bmm(R_a, x_t.detach().unsqueeze(-1)).squeeze(-1)  # (B, r)
        return z @ self.L.t()  # (B, n_feature)

    def predict_next(self, x_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """x̂_{t+1} = (W_CA3 + L·R(a_t)) x_t, full core state."""
        return ca3_step(x_t, self(x_t, a_t))

    @torch.no_grad()
    def delta_update(self, x_t: torch.Tensor, a_t: torch.Tensor, u_tp1: torch.Tensor, lr: float) -> torch.Tensor:
        """One delta-rule step on R(a_t); L is not changed.

            e_{t+1}   = u_{t+1} - û_{t+1}                      (B, n_feature)
            ΔR(a_t)  += lr · (Lᵀ e_{t+1}) x_tᵀ / B             each transition updates only its own action's R

        x_t: (B, D), a_t: (B,), u_tp1: (B, n_feature). B = 1 in time order is the online rule.
        Returns the per-transition squared error (B,) measured BEFORE the update.
        Stability (batch-mean form): lr < 2 / λ_max(E[x xᵀ]) · 1/σ_max(L)², see `safe_delta_lr`.
        """
        assert self.mode != "zero"
        e_tp1 = u_tp1 - self(x_t, a_t)  # (B, n_feature)
        proj = e_tp1 @ self.L  # (B, r)   = (Lᵀ e)ᵀ
        dR = proj.unsqueeze(-1) * x_t.unsqueeze(1)  # (B, r, D)  outer product per transition
        self.R.data.index_add_(0, self._R_index(a_t), (lr / x_t.shape[0]) * dR)
        return (e_tp1**2).sum(dim=1)


def safe_delta_lr(model: ActionCondCA3Predictor, x_t: torch.Tensor, scale: float = 1.0) -> float:
    """lr = scale / (mean ‖x_t‖² · σ_max(L)²). mean ‖x‖² = trace E[x xᵀ] ≥ λ_max, so scale < 2 is stable."""
    sx = (x_t**2).sum(dim=1).mean().item()
    sL = torch.linalg.matrix_norm(model.L.detach(), ord=2).item() ** 2
    return scale / (sx * sL + 1e-12)


def prediction_loss(u_hat_tp1: torch.Tensor, u_tp1: torch.Tensor) -> torch.Tensor:
    """Squared error on the 16 DG targets u_{t+1}, summed over features, averaged over transitions."""
    return ((u_tp1 - u_hat_tp1) ** 2).sum(dim=1).mean()


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------
# Key names in transitions_sXXXX.npz (Phase 1 collection script)
KEYS = dict(
    x="x",  # (N_steps, 1149) x[t]: CA3 state after injecting u_t = DG(o_t), same forward pass that chose a_t
    a="action",  # (N_steps,) int, a_t: action chosen from x[t]
    ep="episode_id",  # (N_steps,) int, episode index
)


def load_transitions(path: str, keys: dict = KEYS):
    """Per-transition arrays, never pairing across episode boundaries:

    x_t   = x[t]            (core)
    a_t   = action[t]       chosen from x_t
    u_tp1 = slot 0 of x[t+1] = DG(o_{t+1}), the prediction target
    x_tp1 = x[t+1]          (core)
    ep    = episode_id[t]
    """
    d = np.load(path)
    x = d[keys["x"]][:, :CORE_SIZE].astype(np.float32)  # drop the 13 bypass dims
    a = d[keys["a"]].astype(np.int64)
    ep = d[keys["ep"]].astype(np.int64)
    same_ep = ep[:-1] == ep[1:]  # t and t+1 in the same episode
    x_t, x_tp1 = x[:-1][same_ep], x[1:][same_ep]
    u_tp1 = x_tp1.reshape(len(x_tp1), HIPPO_N_FEATURE, -1)[:, :, 0].copy()
    return dict(x_t=x_t, a_t=a[:-1][same_ep], u_tp1=u_tp1, x_tp1=x_tp1, ep=ep[:-1][same_ep])


def split_by_episode(data: dict, heldout_frac: float = 0.2, seed: int = 0):
    """Whole episodes go to train or held-out, never both."""
    eps = np.unique(data["ep"])
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_held = max(1, int(round(heldout_frac * len(eps))))
    held = np.isin(data["ep"], eps[:n_held])
    pick = lambda m: {k: torch.from_numpy(v[m]) for k, v in data.items()}
    return pick(~held), pick(held)