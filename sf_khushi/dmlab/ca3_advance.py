"""
Deterministic CA3 shift-register advance for imagined rollouts (Phase 3).

The CA3 core (SimpleSequenceWithBypassCore in custom_core.py) is a fixed linear
shift register, so moving its state forward is exact, not learned. Between two
DG events no input arrives, so the state only shifts. At the event step the DG
input of each firing feature is added to that feature's first R slots.

State layout (the flat rnn_state, width = core.get_out_size(), e.g. 1149):

    [ core: F * ell values, viewed as (F, ell), feature-major | bypass: b values ]
      F = Hippo_n_feature (16), ell = R + L - 1 (71), b = bypass width (13)

One real core step (copied from SimpleSequenceWithBypassCore.forward):

    core = core.roll(1, dim=-1); core[..., 0] = 0; core[..., :R] += u
    bypass = current bypass input

Note the "+=": slots 1..R-1 hold (shifted old value + new input). A single
firing therefore occupies R slots, and two firings less than R steps apart
overlap and sum. `step` and `advance` reproduce this exactly.

INDEX CONVENTION (all functions here work in *state index* space):
    advance(s_k, dt, u) == s_{k+dt}
where u is the DG input of the step that produced s_{k+dt}, and no feature fired
in the steps producing s_{k+1} .. s_{k+dt-1}. Because rnn_states[t+1] holds
timestep t's DG activity, check that the dt stored by build_event_dataset.py
follows this same convention before wiring the predictor in (see --check).

Bypass (depth + instructions): nothing predicts it. `bypass="hold"` keeps the
last real value (default), `bypass="zero"` zeros it. This is an open question
for Xiaoxiong; switch with the argument.

Usage
-----
Library:
    from sf_khushi.dmlab.ca3_advance import CA3Layout, step, shift, advance, imagine_event
    layout = CA3Layout.from_core(actor_critic.core)
    s_next = advance(s, dt, u, layout)                         # one known event
    s_next, dt_star, fired = imagine_event(s, dt_f, mag_f, layout)  # per-feature predictions

Validate against real states from one episode (consecutive rnn_states, e.g.
saved from spot_check.py with np.save):
    python -m sf_khushi.dmlab.ca3_advance --check states.npy
    python -m sf_khushi.dmlab.ca3_advance --check data.npz --key rnn_states --dones-key dones
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

IntLike = Union[int, Tensor]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CA3Layout:
    n_feature: int = 16
    R: int = 8
    L: int = 64
    bypass: int = 13

    @property
    def length(self) -> int:  # ell, slots per feature
        return self.R + self.L - 1

    @property
    def core_size(self) -> int:
        return self.n_feature * self.length

    @property
    def total(self) -> int:
        return self.core_size + self.bypass

    @classmethod
    def from_core(cls, core) -> "CA3Layout":
        """Read the layout off a live SimpleSequenceWithBypassCore."""
        return cls(
            n_feature=int(core.Hippo_n_feature),
            R=int(core.R),
            L=int(core.L),
            bypass=int(core.bypass_size),
        )

    def split(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """(B, total) -> core (B, F, ell), bypass (B, b). Returns views, not copies."""
        if state.dim() != 2 or state.shape[1] != self.total:
            raise ValueError(
                f"state must be (B, {self.total}), got {tuple(state.shape)}. "
                f"If the width is 3533, the state was sized from cfg.rnn_size "
                f"(maybe_overwrite_rnn_size bug); size it from core.get_out_size()."
            )
        core = state[:, : self.core_size].view(state.shape[0], self.n_feature, self.length)
        return core, state[:, self.core_size :]

    def join(self, core: Tensor, bypass: Tensor) -> Tensor:
        return torch.cat([core.reshape(core.shape[0], self.core_size), bypass], dim=1)


def _as_batch(state: Tensor) -> Tuple[Tensor, bool]:
    if state.dim() == 1:
        return state.unsqueeze(0), True
    return state, False


def _new_bypass(old: Tensor, bypass: Union[str, Tensor]) -> Tensor:
    if isinstance(bypass, Tensor):
        return bypass.to(old).expand_as(old).clone()
    if bypass == "hold":
        return old.clone()
    if bypass == "zero":
        return torch.zeros_like(old)
    raise ValueError(f"bypass must be 'hold', 'zero' or a tensor, got {bypass!r}")


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def shift(state: Tensor, n: IntLike, layout: CA3Layout, bypass: Union[str, Tensor] = "hold") -> Tensor:
    """n zero-input steps. n may be an int or a (B,) int tensor (one per batch row).

    Closed form: new[..., j] = old[..., j - n] for j >= n, else 0. Values shifted
    past slot ell-1 fall off the end, so n >= ell empties the register.
    """
    state, squeeze = _as_batch(state)
    core, byp = layout.split(state)
    B, ell = core.shape[0], layout.length

    n = torch.as_tensor(n, device=state.device, dtype=torch.long)
    if n.dim() == 0:
        n = n.expand(B)
    if (n < 0).any():
        raise ValueError("shift amount must be >= 0")

    src = torch.arange(ell, device=state.device).unsqueeze(0) - n.unsqueeze(1)  # (B, ell)
    valid = (src >= 0).unsqueeze(1)                                               # (B, 1, ell)
    idx = src.clamp(min=0).unsqueeze(1).expand(B, layout.n_feature, ell)
    new_core = torch.gather(core, 2, idx) * valid

    out = layout.join(new_core, _new_bypass(byp, bypass))
    return out.squeeze(0) if squeeze else out


def step(state: Tensor, u: Tensor, layout: CA3Layout, bypass: Union[str, Tensor] = "hold") -> Tensor:
    """One real core step with DG input u (B, F) or (F,). Matches the core's forward."""
    state, squeeze = _as_batch(state)
    u = u.to(state).reshape(state.shape[0], layout.n_feature)
    out = shift(state, 1, layout, bypass=bypass)
    core, _ = layout.split(out)
    core[:, :, : layout.R] += u.unsqueeze(-1)
    return out.squeeze(0) if squeeze else out


def advance(
    state: Tensor,
    dt: IntLike,
    u: Optional[Tensor],
    layout: CA3Layout,
    bypass: Union[str, Tensor] = "hold",
) -> Tensor:
    """Jump to the next event: dt-1 silent steps, then one step with input u.

    advance(s_k, dt, u) == s_{k+dt}, given no firing in between (see module docstring).
    dt >= 1, int or (B,) tensor. u is (B, F); u=None means no event (pure shift by dt).
    """
    dt_t = torch.as_tensor(dt, device=state.device, dtype=torch.long)
    if (dt_t < 1).any():
        raise ValueError("dt must be >= 1 (dt=1 means the event is the very next step)")
    if u is None:
        return shift(state, dt_t, layout, bypass=bypass)
    return step(shift(state, dt_t - 1, layout, bypass=bypass), u, layout, bypass=bypass)


def imagine_event(
    state: Tensor,
    dt_per_feature: Tensor,
    magnitude: Tensor,
    layout: CA3Layout,
    horizon: Optional[int] = None,
    bypass: Union[str, Tensor] = "hold",
) -> Tuple[Tensor, Tensor, Tensor]:
    """One imagined step from per-feature predictions (earliest event wins).

    dt_per_feature: (B, F) predicted steps until each feature next fires; use inf
                    (or any value > horizon) for "does not fire before episode end".
    magnitude:      (B, F) DG value to inject for each feature if it fires.
    horizon:        cap for rows where nothing fires (default ell: register empties).

    Returns (new_state, dt_star (B,), fired (B, F) bool). Features tied at dt_star
    all fire together. Rows with no event are shifted by `horizon` and get no
    injection (fired all False); the caller should treat them as terminal/censored.
    """
    state, squeeze = _as_batch(state)
    horizon = layout.length if horizon is None else int(horizon)
    dt_f = dt_per_feature.to(state).reshape(state.shape[0], layout.n_feature)
    # Round to whole steps first so features predicted at e.g. 4.2 and 3.9 tie at 4.
    dt_f = dt_f.round().clamp(min=1)
    dt_f = torch.where(dt_f > horizon, torch.full_like(dt_f, float("inf")), dt_f)

    dt_min, _ = dt_f.min(dim=1)
    has_event = torch.isfinite(dt_min)
    dt_star = torch.where(has_event, dt_min, torch.full_like(dt_min, horizon)).round().long().clamp(min=1)
    fired = (dt_f == dt_min.unsqueeze(1)) & has_event.unsqueeze(1)
    u = magnitude.to(state).reshape_as(dt_f) * fired

    new_state = advance(state, dt_star, u, layout, bypass=bypass)
    if squeeze:
        return new_state.squeeze(0), dt_star.squeeze(0), fired.squeeze(0)
    return new_state, dt_star, fired


def dg_input_from_states(states: Tensor, layout: CA3Layout) -> Tensor:
    """Recover the DG input that produced each state: slot 0 of every feature.

    (slot 0 is zeroed then gets +u, so it equals u exactly.) (T, total) -> (T, F).
    """
    core, _ = layout.split(states)
    return core[:, :, 0].clone()


# ---------------------------------------------------------------------------
# Validation against real consecutive states
# ---------------------------------------------------------------------------
def check_states(states: Tensor, layout: CA3Layout, dones: Optional[Tensor] = None, atol: float = 1e-5) -> bool:
    """Replay real consecutive states (T, total) and compare. Prints a report.

    1. one-step:    step(s_t, u_{t+1}) == s_{t+1}  (core part; bypass is new input, skipped)
    2. event jumps: for consecutive firing steps k < m, advance(s_k, m-k, u_m) == s_m
    3. silent tail: from the last event, shift reproduces every later state

    Slot 0 is where u is read from, so it matches by construction; the test is
    whether every other slot (the shifted history) lines up.
    """
    states = states.double()
    T = states.shape[0]
    u = dg_input_from_states(states, layout)

    # Episode segments: a done at t means s_{t+1} starts a new episode.
    breaks = set()
    if dones is not None:
        breaks = {int(t) + 1 for t in torch.nonzero(dones.reshape(-1)).flatten().tolist()}

    def core_of(x: Tensor) -> Tensor:
        return layout.split(x.unsqueeze(0) if x.dim() == 1 else x)[0]

    ok = True
    # 1. one-step
    pairs = [t for t in range(T - 1) if (t + 1) not in breaks]
    if pairs:
        idx = torch.tensor(pairs)
        pred = step(states[idx], u[idx + 1], layout)
        err = (core_of(pred) - core_of(states[idx + 1])).abs().amax(dim=(1, 2))
        n_bad = int((err > atol).sum())
        print(f"[1] one-step:    {len(pairs) - n_bad}/{len(pairs)} match   (max err {err.max().item():.2e})")
        ok &= n_bad == 0

    # 2/3. event jumps within each segment
    seg_starts = [0] + sorted(b for b in breaks if 0 < b < T)
    seg_ends = seg_starts[1:] + [T]
    jumps = bad_jumps = tail = bad_tail = 0
    max_err = 0.0
    fires = (u.abs() > 0).any(dim=1)
    for a, b in zip(seg_starts, seg_ends):
        ev = [t for t in range(a, b) if fires[t]]
        for k, m in zip(ev[:-1], ev[1:]):
            pred = advance(states[k], m - k, u[m], layout)
            e = (core_of(pred) - core_of(states[m])).abs().max().item()
            max_err = max(max_err, e)
            jumps += 1
            bad_jumps += e > atol
        if ev:
            last = ev[-1]
            for t in range(last + 1, b):
                e = (core_of(shift(states[last], t - last, layout)) - core_of(states[t])).abs().max().item()
                max_err = max(max_err, e)
                tail += 1
                bad_tail += e > atol
    print(f"[2] event jumps: {jumps - bad_jumps}/{jumps} match")
    print(f"[3] silent tail: {tail - bad_tail}/{tail} match   (max err over 2+3 {max_err:.2e})")
    print(f"    {int(fires.sum())} firing steps over {T} states, {len(seg_starts)} episode segment(s)")
    if jumps == 0:
        print("    WARNING: fewer than 2 firing steps, so the event-jump test did not run.")
    ok &= bad_jumps == 0 and bad_tail == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def _load(path: str, key: Optional[str]):
    import numpy as np

    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        return data, None
    keys = list(data.keys())
    k = key or ("rnn_states" if "rnn_states" in keys else keys[0])
    print(f"loading '{k}' from {path} (keys: {keys})")
    return data[k], data


def main():
    p = argparse.ArgumentParser(description="Validate the CA3 advance against real consecutive rnn_states.")
    p.add_argument("--check", required=True, help=".npy (T, total) or .npz file of consecutive states")
    p.add_argument("--key", default=None, help="array name inside the .npz (default rnn_states)")
    p.add_argument("--dones-key", default=None, help="optional episode-done array (T,) inside the .npz")
    p.add_argument("--n-feature", type=int, default=16)
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--bypass", type=int, default=13)
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()

    layout = CA3Layout(args.n_feature, args.R, args.L, args.bypass)
    arr, npz = _load(args.check, args.key)
    states = torch.as_tensor(arr).reshape(-1, arr.shape[-1])
    if states.shape[1] > layout.total:
        print(f"note: state width {states.shape[1]} > {layout.total}; using the first {layout.total} columns "
              f"(assumes the extra width is rnn_size-override padding at the end)")
        states = states[:, : layout.total]
    dones = torch.as_tensor(npz[args.dones_key]) if (npz is not None and args.dones_key) else None
    print(f"layout: F={layout.n_feature} R={layout.R} L={layout.L} ell={layout.length} "
          f"bypass={layout.bypass} total={layout.total}; states {tuple(states.shape)}")
    raise SystemExit(0 if check_states(states, layout, dones, args.atol) else 1)


if __name__ == "__main__":
    main()