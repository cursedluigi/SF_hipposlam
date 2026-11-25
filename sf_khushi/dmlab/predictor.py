"""
Phase 2 — per-feature DG-event predictor (HippoSLAM / prevaction-dg)

undecided design choice marked `PLACEHOLDER(...)`.
It runs end-to-end on synthetic data so you can check shapes before the real
.npz from build_event_dataset.py is wired in:

    python predictor.py --synthetic --variant B
    python predictor.py --synthetic --variant A
    python predictor.py --data /path/to/events.npz --variant B

Pieces
  1. Config                      — all hyperparameters + open decisions
  2. CA3 shift register          — fixed dynamics x' = S x + J u (paper eq. 1-4)
  3. Per-feature targets         — the dataset rework (16 Δt + censoring flags)
  4. Dataset loading             — real .npz (PLACEHOLDER keys) or synthetic
  5. DGEventPredictor            — shared trunk, Variant A or B heads
  6. Losses                      — L_pred for A / B, dead-feature hook
  7. Imagined rollout + L_vc     — Option 3 auxiliary loss (not yet wired into SF)
  8. Offline training + sanity check vs. a trivial baseline
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Phase2Config:
    n_features: int = 16
    R: int = 8                       # units injected per input
    L: int = 64                      # theta cycles of persistence
    bypass_dim: int = 13             # depth (10) + instructions (3), skips CA3
    n_actions: int = 5               # reduced action set, Table A1
    action_embed_dim: int = 8        # same as prev-action embedding
    fire_threshold: float = 2.43     # DG threshold (Phase 0 event definition)

    # --- open design decisions ----------------------------------------------
    # PLACEHOLDER(variant): "A" = one-step firing (BCE), "B" = time-to-event hazard.
    variant: str = "B"
    # PLACEHOLDER(stop_grad): stop gradients from the predictor into CA3/DG input?
    #   True keeps the predictor from reshaping DG toward "easy to predict" features.
    stop_grad_input: bool = True
    # PLACEHOLDER(dead_features): features that never fire in any checkpoint.
    dead_features: Tuple[int, ...] = (1, 4, 6)
    #   "mask"       -> excluded from L_pred (current default)
    #   "pre_thresh" -> auxiliary regression on pre-threshold activity
    #                   (needs Xiaoxiong's answer + re-extracted data)
    dead_feature_mode: str = "mask"
    # PLACEHOLDER(magnitude): firings range ~0.04-1.05.
    #   Variant A injects a fixed magnitude; Variant B regresses it.
    fixed_magnitude: float = 0.5
    # PLACEHOLDER(reward_head): predict r̂ in imagination? If False, r̂ = 0
    #   (sparse goal reward; biases imagined values down by ~gamma^k).
    use_reward_head: bool = False

    # --- Variant B: Δt bins (steps) -----------------------------------------
    # PLACEHOLDER(dt_bins): bin b covers [edges[b], edges[b+1]-1]; last bin = "after horizon".
    #   Bin 0 must be exactly {1} so P(fire at next step) is readable for imagination.
    dt_edges: Tuple[int, ...] = (1, 2, 3, 5, 9, 17, 33, 65)

    # --- network --------------------------------------------------------------
    hidden: int = 256
    n_layers: int = 2
    dropout: float = 0.2             # PLACEHOLDER(regularisation): tune on real data

    # --- loss weights (PROPOSALS, not confirmed) -----------------------------
    omega: float = 40.0              # Variant A pos_weight (~1/0.025)
    alpha: float = 1.0               # Variant B magnitude weight
    beta: float = 0.5                # L_aux = L_pred + beta * L_vc
    lambda_aux: float = 0.0          # weight of L_aux inside the APPO loss (0 = baseline)
    k_imagine: int = 8               # imagination horizon
    n_rollouts: int = 1              # rollouts per start state
    gamma: float = 0.99

    # --- training --------------------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 512
    epochs: int = 5
    val_frac: float = 0.1
    seed: int = 1111

    @property
    def seq_len(self) -> int:        # ℓ = L + R - 1 = 71
        return self.L + self.R - 1

    @property
    def ca3_dim(self) -> int:        # 16 * 71 = 1136
        return self.n_features * self.seq_len

    @property
    def n_dt_bins(self) -> int:      # K finite bins + 1 overflow bin
        return len(self.dt_edges)

    @property
    def live_features(self) -> torch.Tensor:
        m = torch.ones(self.n_features, dtype=torch.bool)
        if self.dead_feature_mode == "mask":
            m[list(self.dead_features)] = False
        return m


def ca3_step(x: torch.Tensor, u: torch.Tensor, cfg: Phase2Config) -> torch.Tensor:
    """x_{t+1} = S x_t + J u_t, per feature.  x: [B, F*ℓ], u: [B, F]."""
    B = x.shape[0]
    xs = x.view(B, cfg.n_features, cfg.seq_len)
    out = torch.zeros_like(xs)
    out[..., 1:] = xs[..., :-1]              # S: shift one slot along the sequence
    out[..., : cfg.R] += u.unsqueeze(-1)     # J: inject into the first R slots
    return out.view(B, cfg.ca3_dim)


def ca3_advance(x: torch.Tensor, n: int, cfg: Phase2Config) -> torch.Tensor:
    """Advance n steps with no DG input (for event-jumping imagination, Phase 3)."""
    B = x.shape[0]
    xs = x.view(B, cfg.n_features, cfg.seq_len)
    out = torch.zeros_like(xs)
    if n < cfg.seq_len:
        out[..., n:] = xs[..., : cfg.seq_len - n]
    return out.view(B, cfg.ca3_dim)


def dg_input_from_state(x_next: torch.Tensor, cfg: Phase2Config) -> torch.Tensor:
    """Read u_t back from x_{t+1} (slot 0 of each feature; offset i*71).
    Matches the rnn_states[t+1] convention verified by spot_check.py."""
    return x_next.view(-1, cfg.n_features, cfg.seq_len)[..., 0]


def build_per_feature_targets(u: np.ndarray, ep_end: np.ndarray, cfg: Phase2Config) -> Dict[str, np.ndarray]:
    """
    u      : [T, F] DG input at each step (0 = no firing, >0 = magnitude)
    ep_end : [T] bool, True on the last step of an episode
    For every t and feature f, the next firing strictly after t:
      dt[t,f]        steps until it fires (>=1), or remaining steps if censored
      censored[t,f]  episode ended before f fired again
      magnitude[t,f] magnitude of that firing (0 if censored)
      fired_next[t,f] 1 if f fires at t+1 (Variant A label)
      remaining[t]   steps left in the episode after t
    Rows are never dropped (decision 2026-10-06).
    PLACEHOLDER(rollout_boundaries): episodes cut at a rollout-batch boundary are
      treated as ended here. Use the pending-state carryover in build_event_dataset.py
      if the logged data contains truncated episodes.
    """
    T, Fn = u.shape
    fired = u > 0
    dt = np.zeros((T, Fn), dtype=np.int64)
    censored = np.zeros((T, Fn), dtype=bool)
    magnitude = np.zeros((T, Fn), dtype=np.float32)
    remaining = np.zeros(T, dtype=np.int64)

    next_fire = np.full(Fn, -1, dtype=np.int64)    # index of next firing after t
    next_mag = np.zeros(Fn, dtype=np.float32)
    ep_last = T - 1
    for t in range(T - 1, -1, -1):
        if ep_end[t]:                                # new episode (scanning backwards)
            next_fire[:] = -1
            ep_last = t
        remaining[t] = ep_last - t
        has = next_fire >= 0
        dt[t, has] = next_fire[has] - t
        magnitude[t, has] = next_mag[has]
        censored[t, ~has] = True
        dt[t, ~has] = remaining[t]
        # step t's own firing becomes "next" for t-1
        f_now = fired[t]
        next_fire[f_now] = t
        next_mag[f_now] = u[t, f_now]

    fired_next = np.zeros((T, Fn), dtype=np.float32)
    fired_next[:-1] = fired[1:] & ~ep_end[:-1, None]
    return dict(dt=dt, censored=censored, magnitude=magnitude,
                fired_next=fired_next, remaining=remaining)


def dt_to_bin(dt: torch.Tensor, cfg: Phase2Config) -> torch.Tensor:
    """Map Δt (>=1) to bin index; Δt >= last edge -> overflow bin."""
    edges = torch.tensor(cfg.dt_edges, device=dt.device)
    idx = torch.bucketize(dt, edges, right=True) - 1
    return idx.clamp(0, cfg.n_dt_bins - 1)


def censor_mask(remaining: torch.Tensor, cfg: Phase2Config) -> torch.Tensor:
    """[B, K+1] bool: bins still possible given "no firing within `remaining` steps".
    Bins that partly overlap the observed window are counted as possible
    (slight approximation; finer bins near small Δt reduce it)."""
    edges = list(cfg.dt_edges)
    upper = torch.tensor([e - 1 for e in edges[1:]] + [10**9], device=remaining.device)
    return upper.unsqueeze(0) > remaining.unsqueeze(1)


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.t = tensors
        self.n = len(next(iter(tensors.values())))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.t.items()}


def load_event_dataset(path: str, cfg: Phase2Config) -> Dict[str, torch.Tensor]:
    """
    PLACEHOLDER(npz_keys): adjust to what build_event_dataset.py actually writes.
    Assumed keys (one row per step t, episodes contiguous):
      ca3      [T, 1136 or 1149]  rnn_states[t+1] slice -> state at t   (float)
      action   [T]                a_t                                  (int)
      dg_u     [T, 16]            post-threshold DG input at t         (float)
      ep_end   [T]                last step of episode                 (bool)
      reward   [T]                r_t (only needed for a reward head)  (float)
      bypass   [T, 13]            optional, needed for the policy/critic in imagination
      dg_pre   [T, 16]            optional, pre-threshold DG (dead-feature loss)
    """
    d = np.load(path)
    ca3 = d["ca3"][:, : cfg.ca3_dim].astype(np.float32)   # drop bypass if present
    tg = build_per_feature_targets(d["dg_u"], d["ep_end"].astype(bool), cfg)
    out = dict(
        ca3=torch.from_numpy(ca3),
        action=torch.from_numpy(d["action"].astype(np.int64)),
        **{k: torch.from_numpy(v) for k, v in tg.items()},
    )
    for opt in ("reward", "bypass", "dg_pre"):
        if opt in d:
            out[opt] = torch.from_numpy(d[opt].astype(np.float32))
    return out


def make_synthetic_dataset(cfg: Phase2Config, n_episodes: int = 60, ep_len: int = 300) -> Dict[str, torch.Tensor]:
    """Fake data that runs through the real CA3 dynamics, so shapes and the
    target logic are exercised. Learnable structure: firing depends on the
    action, and feature f is ~6x more likely within 12 steps after feature f-1
    fired (a "route" — visible in the CA3 state)."""
    rng = np.random.default_rng(cfg.seed)
    p_base = rng.uniform(0.01, 0.03, size=cfg.n_features)
    p_base[list(cfg.dead_features)] = 0.0
    action_bias = rng.uniform(0.3, 2.0, size=(cfg.n_actions, cfg.n_features))
    xs, acts, us, ends = [], [], [], []
    for _ in range(n_episodes):
        x = torch.zeros(1, cfg.ca3_dim)
        last_fire = np.full(cfg.n_features, -100)
        for t in range(ep_len):
            a = rng.integers(cfg.n_actions)
            boost = np.where(t - np.roll(last_fire, 1) <= 12, 6.0, 1.0)
            fire = rng.random(cfg.n_features) < np.clip(p_base * action_bias[a] * boost, 0, 1)
            last_fire[fire] = t
            u = np.where(fire, rng.uniform(0.04, 1.05, cfg.n_features), 0.0).astype(np.float32)
            xs.append(x[0].clone()); acts.append(a); us.append(u); ends.append(t == ep_len - 1)
            x = ca3_step(x, torch.from_numpy(u)[None], cfg)
    u = np.stack(us)
    tg = build_per_feature_targets(u, np.array(ends), cfg)
    return dict(ca3=torch.stack(xs), action=torch.tensor(acts),
                **{k: torch.from_numpy(v) for k, v in tg.items()})


class DGEventPredictor(nn.Module):
    def __init__(self, cfg: Phase2Config, action_embed: Optional[nn.Embedding] = None):
        super().__init__()
        self.cfg = cfg
        # PLACEHOLDER(action_embed): pass the agent's prev-action nn.Embedding to reuse it
        #   (confirmed preference). A fresh table is used when none is given.
        self.action_embed = action_embed or nn.Embedding(cfg.n_actions, cfg.action_embed_dim)
        layers, d = [], cfg.ca3_dim + cfg.action_embed_dim
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden
        self.trunk = nn.Sequential(*layers)
        Fn = cfg.n_features
        if cfg.variant == "A":
            self.fire_head = nn.Linear(d, Fn)
        elif cfg.variant == "B":
            self.hazard_head = nn.Linear(d, Fn * cfg.n_dt_bins)
            self.mag_head = nn.Linear(d, Fn)
        else:
            raise ValueError(cfg.variant)
        if cfg.dead_feature_mode == "pre_thresh":
            self.pre_head = nn.Linear(d, Fn)        # PLACEHOLDER(dead_features)
        if cfg.use_reward_head:
            self.reward_head = nn.Linear(d, 1)      # PLACEHOLDER(reward_head)

    def forward(self, ca3: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.cfg.stop_grad_input:
            ca3 = ca3.detach()
        z = self.trunk(torch.cat([ca3, self.action_embed(action)], dim=-1))
        out: Dict[str, torch.Tensor] = {}
        if self.cfg.variant == "A":
            out["fire_logit"] = self.fire_head(z)                                 # [B, F]
        else:
            out["hazard_logit"] = self.hazard_head(z).view(-1, self.cfg.n_features, self.cfg.n_dt_bins)
            out["magnitude"] = F.softplus(self.mag_head(z))                       # [B, F]
        if hasattr(self, "pre_head"):
            out["pre"] = self.pre_head(z)
        if hasattr(self, "reward_head"):
            out["reward"] = self.reward_head(z).squeeze(-1)
        return out

    @torch.no_grad()
    def sample_next_input(self, ca3, action) -> torch.Tensor:
        """Sparse DG input for the next step, used in imagination. [B, F]"""
        out = self.forward(ca3, action)
        live = self.cfg.live_features.to(ca3.device).float()
        if self.cfg.variant == "A":
            fire = torch.bernoulli(torch.sigmoid(out["fire_logit"]))
            return fire * self.cfg.fixed_magnitude * live
        # Variant B, stepped: P(fire at next step) = mass of bin 0 (= {Δt=1}).
        # PLACEHOLDER(event_jump): sample a Δt per feature and use ca3_advance()
        #   to jump between events instead (Phase 3); needs a policy for the skipped steps.
        p1 = torch.softmax(out["hazard_logit"], -1)[..., 0]
        fire = torch.bernoulli(p1)
        return fire * out["magnitude"] * live


def loss_pred(model: DGEventPredictor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    cfg = model.cfg
    out = model(batch["ca3"], batch["action"])
    live = cfg.live_features.to(batch["ca3"].device)
    logs: Dict[str, float] = {}

    if cfg.variant == "A":
        tgt = batch["fired_next"].float()
        pw = torch.full((cfg.n_features,), cfg.omega, device=tgt.device)
        bce = F.binary_cross_entropy_with_logits(out["fire_logit"], tgt, pos_weight=pw, reduction="none")
        loss = bce[:, live].sum(-1).mean()
        logs["bce"] = loss.item()
    else:
        logq = torch.log_softmax(out["hazard_logit"], -1)              # [B, F, K+1]
        cens = batch["censored"]                                        # [B, F]
        bins = dt_to_bin(batch["dt"], cfg)                              # [B, F]
        nll_unc = -logq.gather(-1, bins.unsqueeze(-1)).squeeze(-1)
        allowed = censor_mask(batch["remaining"], cfg).unsqueeze(1)     # [B, 1, K+1]
        nll_cen = -torch.logsumexp(logq.masked_fill(~allowed, -1e9), -1)
        nll = torch.where(cens, nll_cen, nll_unc)
        mag_err = (out["magnitude"] - batch["magnitude"]) ** 2 * (~cens).float()
        per = nll + cfg.alpha * mag_err
        loss = per[:, live].sum(-1).mean()
        logs.update(nll=nll[:, live].sum(-1).mean().item(),
                    mag_mse=(mag_err[:, live].sum() / (~cens[:, live]).sum().clamp(min=1)).item())

    if cfg.dead_feature_mode == "pre_thresh":
        # PLACEHOLDER(dead_features): Xiaoxiong's "loss on pre-threshold activity" —
        #   predictor-side interpretation. Needs dg_pre in the dataset.
        if "dg_pre" in batch:
            dead = list(cfg.dead_features)
            pre_loss = F.mse_loss(out["pre"][:, dead], batch["dg_pre"][:, dead])
            loss = loss + pre_loss
            logs["pre_mse"] = pre_loss.item()

    if cfg.use_reward_head and "reward" in batch:
        r_loss = F.mse_loss(out["reward"], batch["reward"])   # PLACEHOLDER(reward_head) target alignment
        loss = loss + r_loss
        logs["reward_mse"] = r_loss.item()
    return loss, logs


# Imagined rollout + value consistency  (Option 3 auxiliary loss)

PolicyFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]   # (ca3, bypass) -> action logits
ValueFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]    # (ca3, bypass) -> V [B]


def make_actor_critic_fns(actor_critic) -> Tuple[PolicyFn, ValueFn]:
    """
    PLACEHOLDER(actor_critic): wrap the Sample Factory model.
      core output = concat(ca3 [1136], bypass [13]) -> MlpDecoderJit -> heads.
      Roughly:
        h = actor_critic.decoder(torch.cat([ca3, bypass], -1))
        logits = actor_critic.action_parameterization(h)[0]  # check SF version
        value  = actor_critic.critic_linear(h).squeeze(-1)
    """
    raise NotImplementedError("wire to the trained actor_critic")


def imagined_return(model: DGEventPredictor, policy_fn: PolicyFn, value_fn: ValueFn,
                    ca3_0: torch.Tensor, bypass_0: torch.Tensor) -> torch.Tensor:
    """k-step imagined target G = Σ γ^j r̂_j + γ^k V(x̂_k). Returns [B], no grad."""
    cfg = model.cfg
    Gs = []
    with torch.no_grad():
        for _ in range(cfg.n_rollouts):
            x, G, disc = ca3_0, torch.zeros(ca3_0.shape[0], device=ca3_0.device), 1.0
            # PLACEHOLDER(bypass): bypass (depth + instructions) is held at its real value
            #   for the whole rollout. Alternatives: zero it, or predict it.
            byp = bypass_0
            for _ in range(cfg.k_imagine):
                a = torch.distributions.Categorical(logits=policy_fn(x, byp)).sample()
                if cfg.use_reward_head:
                    G = G + disc * model(x, a)["reward"]
                u = model.sample_next_input(x, a)
                x = ca3_step(x, u, cfg)
                disc *= cfg.gamma
            Gs.append(G + disc * value_fn(x, byp))
    return torch.stack(Gs).mean(0)


def aux_loss(model: DGEventPredictor, batch: Dict[str, torch.Tensor],
             policy_fn: PolicyFn, value_fn: ValueFn) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_aux = L_pred + β·L_vc.  To use inside APPO: L += cfg.lambda_aux * L_aux.
    PLACEHOLDER(sf_integration): hook into the Sample Factory learner loss; the
    batch there comes from the replayed rollout (CA3 = rnn_states[t+1] convention)."""
    cfg = model.cfg
    l_pred, logs = loss_pred(model, batch)
    bypass = batch.get("bypass", torch.zeros(batch["ca3"].shape[0], cfg.bypass_dim, device=batch["ca3"].device))
    G = imagined_return(model, policy_fn, value_fn, batch["ca3"], bypass)
    v = value_fn(batch["ca3"], bypass)
    l_vc = F.mse_loss(v, G)          # G is already stop-gradient
    logs["vc"] = l_vc.item()
    return l_pred + cfg.beta * l_vc, logs



# 8. Offline training + sanity check

def trivial_baseline_loss(train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor], cfg: Phase2Config) -> float:
    """Same loss for a predictor that ignores state and action (base rates).
    The model must beat this, or it's learning nothing useful."""
    live = cfg.live_features
    if cfg.variant == "A":
        p = train["fired_next"].float().mean(0).clamp(1e-4, 1 - 1e-4)
        # best constant under pos_weight ω: logit = log(ω p / (1 - p))
        logit = torch.log(cfg.omega * p / (1 - p)).expand_as(val["fired_next"])
        pw = torch.full((cfg.n_features,), cfg.omega)
        bce = F.binary_cross_entropy_with_logits(logit, val["fired_next"].float(), pos_weight=pw, reduction="none")
        return bce[:, live].sum(-1).mean().item()
    # B: empirical bin histogram per feature from uncensored rows (+1 smoothing)
    bins = dt_to_bin(train["dt"], cfg)
    hist = torch.ones(cfg.n_features, cfg.n_dt_bins)
    for f in range(cfg.n_features):
        b = bins[~train["censored"][:, f], f]
        hist[f] += torch.bincount(b, minlength=cfg.n_dt_bins).float()
    logq = torch.log(hist / hist.sum(-1, keepdim=True)).unsqueeze(0).expand(len(val["dt"]), -1, -1)
    vb = dt_to_bin(val["dt"], cfg)
    nll_unc = -logq.gather(-1, vb.unsqueeze(-1)).squeeze(-1)
    allowed = censor_mask(val["remaining"], cfg).unsqueeze(1)
    nll_cen = -torch.logsumexp(logq.masked_fill(~allowed, -1e9), -1)
    nll = torch.where(val["censored"], nll_cen, nll_unc)
    return nll[:, live].sum(-1).mean().item()


def split(data: Dict[str, torch.Tensor], cfg: Phase2Config):
    # Contiguous split so val episodes are unseen (rows within an episode are correlated).
    n = len(data["ca3"]); cut = int(n * (1 - cfg.val_frac))
    return ({k: v[:cut] for k, v in data.items()}, {k: v[cut:] for k, v in data.items()})


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval(); tot, n = {}, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        loss, logs = loss_pred(model, b)
        for k, v in {"loss": loss.item(), **logs}.items():
            tot[k] = tot.get(k, 0.0) + v * len(b["ca3"])
        n += len(b["ca3"])
    model.train()
    return {k: v / n for k, v in tot.items()}


def train(cfg: Phase2Config, data: Dict[str, torch.Tensor], out_path: Optional[str] = None):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = split(data, cfg)
    print(f"[data] train {len(tr['ca3'])}  val {len(va['ca3'])}  variant {cfg.variant}  "
          f"live features {cfg.live_features.sum().item()}/{cfg.n_features}")
    fire_rate = (tr["fired_next"][:, cfg.live_features] > 0).float().mean().item()
    cens_rate = tr["censored"][:, cfg.live_features].float().mean().item()
    print(f"[data] firing rate {fire_rate:.4f}  censored rows {cens_rate:.3f}")

    tl = torch.utils.data.DataLoader(EventDataset(tr), batch_size=cfg.batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(EventDataset(va), batch_size=cfg.batch_size)
    model = DGEventPredictor(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    base = trivial_baseline_loss(tr, va, cfg)
    key = "loss" if cfg.variant == "A" else "nll"
    best, best_state, best_ep = float("inf"), None, 0

    for ep in range(cfg.epochs):
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            loss, _ = loss_pred(model, b)
            opt.zero_grad(); loss.backward(); opt.step()
        ev = evaluate(model, vl, device)
        print(f"[epoch {ep+1}] val " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items())
              + f"  | trivial baseline {base:.4f}")
        if ev[key] < best:
            best, best_ep = ev[key], ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)   # keep the best-validation epoch
    print(f"[sanity] best epoch {best_ep}: val {key} {best:.4f} vs state-blind baseline {base:.4f} -> "
          f"{'BEATS' if best < base else 'does NOT beat'} baseline")
    if out_path:
        torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, out_path)
        print(f"[saved] {out_path}")
    return model


def smoke_test_aux(cfg: Phase2Config, model: DGEventPredictor, data: Dict[str, torch.Tensor]):
    """Runs aux_loss with stand-in policy/critic MLPs (shapes only)."""
    pol = nn.Linear(cfg.ca3_dim + cfg.bypass_dim, cfg.n_actions)
    val = nn.Linear(cfg.ca3_dim + cfg.bypass_dim, 1)
    policy_fn = lambda x, b: pol(torch.cat([x, b], -1))
    value_fn = lambda x, b: val(torch.cat([x, b], -1)).squeeze(-1)
    dev = next(model.parameters()).device
    batch = {k: v[:64].to(dev) for k, v in data.items()}
    pol.to(dev); val.to(dev)
    loss, logs = aux_loss(model, batch, policy_fn, value_fn)
    loss.backward()
    print(f"[aux smoke] L_aux {loss.item():.4f}  " + "  ".join(f"{k} {v:.4f}" for k, v in logs.items()))


def self_test(cfg: Phase2Config):
    """Checks the CA3 dynamics and target builder on tiny hand-made cases."""
    x = torch.zeros(1, cfg.ca3_dim); u = torch.zeros(1, cfg.n_features); u[0, 3] = 0.7
    x1 = ca3_step(x, u, cfg)
    assert torch.allclose(dg_input_from_state(x1, cfg), u), "slot-0 read-back failed"
    s = x1.view(cfg.n_features, cfg.seq_len)
    assert s[3, : cfg.R].eq(0.7).all() and s[3, cfg.R:].eq(0).all()
    x5 = ca3_advance(x1, 5, cfg).view(cfg.n_features, cfg.seq_len)
    assert x5[3, 5 : 5 + cfg.R].eq(0.7).all() and x5[3, :5].eq(0).all()
    xr = x1
    for _ in range(5):
        xr = ca3_step(xr, torch.zeros_like(u), cfg)
    assert torch.allclose(xr.view(cfg.n_features, -1), x5), "advance != repeated step"
    gone = ca3_advance(x1, cfg.seq_len, cfg)
    assert gone.abs().sum() == 0, "activity should leave after ℓ steps"

    # targets: 1 episode, 6 steps, feature 0 fires at t=2 (mag .5), never again
    uu = np.zeros((6, cfg.n_features), np.float32); uu[2, 0] = 0.5
    tg = build_per_feature_targets(uu, np.array([0, 0, 0, 0, 0, 1], bool), cfg)
    assert list(tg["dt"][:2, 0]) == [2, 1] and not tg["censored"][:2, 0].any()
    assert tg["censored"][2:, 0].all() and list(tg["remaining"]) == [5, 4, 3, 2, 1, 0]
    assert tg["fired_next"][1, 0] == 1 and tg["fired_next"][:, 0].sum() == 1
    assert list(dt_to_bin(torch.tensor([1, 2, 3, 4, 5, 64, 65, 500]), cfg)) == [0, 1, 2, 2, 3, 6, 7, 7]
    print("[self-test] CA3 dynamics, targets and binning OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="event .npz from build_event_dataset.py")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--variant", choices=["A", "B"], default="B")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg = Phase2Config(variant=args.variant, epochs=args.epochs)
    self_test(cfg)
    if args.data:
        data = load_event_dataset(args.data, cfg)
    elif args.synthetic:
        data = make_synthetic_dataset(cfg)
    else:
        p.error("pass --data PATH or --synthetic")
    model = train(cfg, data, args.out)
    smoke_test_aux(cfg, model, data)


if __name__ == "__main__":
    main()