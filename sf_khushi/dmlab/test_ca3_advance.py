"""
Tests for ca3_advance.py against the real SimpleSequenceWithBypassCore.

Run from the repo root:
    python -m pytest sf_khushi/dmlab/test_ca3_advance.py -q
or without pytest:
    python sf_khushi/dmlab/test_ca3_advance.py

On the cluster sample_factory is installed and the real ModelCore is used. If it
is missing (e.g. a laptop), a minimal stand-in is injected so custom_core.py still
imports; the core's forward logic itself is always the real one.
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _ensure_sample_factory():
    try:
        import sample_factory.model.core  # noqa: F401
        return
    except ImportError:
        pass
    import logging

    class ModelCore(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

    mods = {
        "sample_factory": types.ModuleType("sample_factory"),
        "sample_factory.model": types.ModuleType("sample_factory.model"),
        "sample_factory.model.core": types.ModuleType("sample_factory.model.core"),
        "sample_factory.utils": types.ModuleType("sample_factory.utils"),
        "sample_factory.utils.typing": types.ModuleType("sample_factory.utils.typing"),
        "sample_factory.utils.utils": types.ModuleType("sample_factory.utils.utils"),
    }
    core = mods["sample_factory.model.core"]
    core.ModelCore = core.ModelCoreIdentity = core.ModelCoreRNN = ModelCore
    mods["sample_factory.utils.typing"].Config = object
    mods["sample_factory.utils.utils"].log = logging.getLogger("stub")
    sys.modules.update(mods)


def _load_custom_core():
    _ensure_sample_factory()
    spec = importlib.util.spec_from_file_location("custom_core_under_test", os.path.join(HERE, "custom_core.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from ca3_advance import CA3Layout, advance, dg_input_from_states, imagine_event, shift, step, check_states  # noqa: E402

custom_core = _load_custom_core()
F, R, L, BYP = 16, 8, 64, 13
LAYOUT = CA3Layout(F, R, L, BYP)


def make_core():
    cfg = SimpleNamespace(Hippo_R=R, Hippo_L=L, Hippo_n_feature=F)
    return custom_core.SimpleSequenceWithBypassCore(cfg, F + BYP)


def sparse_inputs(T, B, p=0.03, seed=0):
    """DG-like inputs: mostly zero, occasional positive continuous values."""
    g = torch.Generator().manual_seed(seed)
    fires = torch.rand(T, B, F, generator=g) < p
    vals = torch.rand(T, B, F, generator=g) * 1.5 + 0.02
    dg = vals * fires
    byp = torch.randn(T, B, BYP, generator=g)
    return torch.cat([dg, byp], dim=-1)


def rollout_real(core, x, s0):
    """Run the real core one step at a time. Returns (T+1, B, total) states."""
    states = [s0]
    s = s0
    with torch.no_grad():
        for t in range(x.shape[0]):
            _, s = core(x[t], s.clone())
            states.append(s)
    return torch.stack(states)


def test_layout_matches_core():
    core = make_core()
    assert CA3Layout.from_core(core) == LAYOUT
    assert LAYOUT.total == core.get_out_size() == 1149


def test_step_matches_real_core():
    core = make_core()
    T, B = 200, 6
    x = sparse_inputs(T, B, p=0.1)
    S = rollout_real(core, x, torch.zeros(B, LAYOUT.total))
    for t in range(T):
        pred = step(S[t], x[t, :, :F], LAYOUT, bypass=x[t, :, F:])
        assert torch.allclose(pred, S[t + 1], atol=1e-6), f"mismatch at t={t}"


def test_step_matches_packed_sequence_path():
    """The learner uses the PackedSequence branch; check that one too."""
    core = make_core()
    T, B = 50, 4
    x = sparse_inputs(T, B, p=0.1, seed=3)
    packed = nn.utils.rnn.pack_padded_sequence(x, torch.full((B,), T), enforce_sorted=False)
    with torch.no_grad():
        _, final = core(packed, torch.zeros(B, LAYOUT.total))
    s = torch.zeros(B, LAYOUT.total)
    for t in range(T):
        s = step(s, x[t, :, :F], LAYOUT, bypass=x[t, :, F:])
    assert torch.allclose(s, final, atol=1e-6)


def test_advance_equals_silent_steps_then_event():
    core = make_core()
    B = 5
    x0 = sparse_inputs(30, B, p=0.2, seed=1)
    S = rollout_real(core, x0, torch.zeros(B, LAYOUT.total))
    s = S[-1]
    for dt in [1, 2, 7, 8, 9, 40, 70, 71, 72, 150]:
        u = torch.rand(B, F) * (torch.rand(B, F) < 0.2)
        silent = torch.zeros(dt, B, F + BYP)
        silent[-1, :, :F] = u
        real = rollout_real(core, silent, s)[-1]
        pred = advance(s, dt, u, LAYOUT)
        assert torch.allclose(LAYOUT.split(pred)[0], LAYOUT.split(real)[0], atol=1e-6), f"dt={dt}"


def test_overlapping_events_accumulate():
    """Two firings < R steps apart overlap in slots and must sum, like the real core."""
    core = make_core()
    s = torch.zeros(1, LAYOUT.total)
    u1 = torch.zeros(1, F); u1[0, 3] = 1.0
    u2 = torch.zeros(1, F); u2[0, 3] = 0.5
    s1 = advance(s, 1, u1, LAYOUT)
    s2 = advance(s1, 3, u2, LAYOUT)
    x = torch.zeros(4, 1, F + BYP); x[0, :, :F] = u1; x[3, :, :F] = u2
    real = rollout_real(core, x, s)[-1]
    assert torch.allclose(LAYOUT.split(s2)[0], LAYOUT.split(real)[0])
    assert LAYOUT.split(s2)[0][0, 3, 3].item() == 1.5  # overlap slot


def test_per_row_dt():
    B = 4
    s = torch.zeros(B, LAYOUT.total)
    s = advance(s, 1, torch.rand(B, F), LAYOUT)
    dts = torch.tensor([1, 5, 30, 80])
    u = torch.rand(B, F)
    batched = advance(s, dts, u, LAYOUT)
    for i in range(B):
        single = advance(s[i], int(dts[i]), u[i], LAYOUT)
        assert torch.allclose(batched[i], single)


def test_shift_empties_after_ell():
    s = advance(torch.zeros(LAYOUT.total), 1, torch.ones(F), LAYOUT)
    assert LAYOUT.split(shift(s, LAYOUT.length - 1, LAYOUT).unsqueeze(0))[0].abs().sum() > 0
    assert LAYOUT.split(shift(s, LAYOUT.length, LAYOUT).unsqueeze(0))[0].abs().sum() == 0


def test_bypass_modes():
    s = torch.zeros(1, LAYOUT.total); s[0, -BYP:] = 2.0
    assert torch.all(advance(s, 3, None, LAYOUT, bypass="hold")[0, -BYP:] == 2.0)
    assert torch.all(advance(s, 3, None, LAYOUT, bypass="zero")[0, -BYP:] == 0.0)


def test_imagine_event_earliest_wins():
    s = torch.zeros(2, LAYOUT.total)
    inf = float("inf")
    dt_f = torch.full((2, F), inf)
    dt_f[0, 2], dt_f[0, 5], dt_f[0, 9] = 6.2, 5.8, 12.0   # 2 and 5 tie at 6
    mag = torch.full((2, F), 0.7)
    new, dt_star, fired = imagine_event(s, dt_f, mag, LAYOUT)
    assert dt_star.tolist() == [6, LAYOUT.length]
    assert fired[0].nonzero().flatten().tolist() == [2, 5]
    assert not fired[1].any()
    core = LAYOUT.split(new)[0]
    assert torch.allclose(core[0, 2, :R], torch.full((R,), 0.7))
    assert core[0, 9].abs().sum() == 0
    assert core[1].abs().sum() == 0


def test_check_states_on_real_rollout():
    core = make_core()
    x = sparse_inputs(300, 1, p=0.03, seed=7)
    S = rollout_real(core, x, torch.zeros(1, LAYOUT.total))[:, 0]
    assert torch.allclose(dg_input_from_states(S[1:], LAYOUT), x[:, 0, :F])
    assert check_states(S, LAYOUT)
    # A corrupted state must be caught.
    bad = S.clone(); bad[150, 5] += 0.3
    assert not check_states(bad, LAYOUT)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")