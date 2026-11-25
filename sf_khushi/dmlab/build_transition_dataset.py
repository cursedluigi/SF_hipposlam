"""

Runs the trained policy through real episodes and stores ONE row per step.

-------------------------------------------------------------------------
Index convention (why x_t is the POST-injection state)
-------------------------------------------------------------------------
At step t the core takes the state from step t-1, shifts it, and injects
u_t = DG(obs_t). SimpleSequenceWithBypassCore returns that new state as both
its output and new_rnn_state, so the decoder and therefore the action a_t
sees this post-injection state. We store it as x_t.

    x_t     = new_rnn_states at step t     (= rnn_states[t+1] in SF's buffer)
    u_t     = DG input of step t           (= slot 0 of each feature in x_t)
    a_t     = action chosen from x_t
    x_{t+1} = next row's x                 (contains u_{t+1} = DG(obs_{t+1}),
                                            the first DG input a_t can affect)

So a transition is (x[i], action[i], x[i+1], dg[i+1], reward[i]) for every
row i with last[i] == False. u_t itself does NOT depend on a_t (obs_t was
seen before a_t was chosen); the target that does is dg[i+1].

-------------------------------------------------------------------------
Saved arrays (.npz), N = total steps over all episodes
-------------------------------------------------------------------------
    x          (N, 1149) float32  post-injection CA3 state (core + bypass)
    dg         (N, 16)   float32  DG input of that step (== slot 0 of x)
    action     (N,)      int64    action a_t chosen from x_t
    reward     (N,)      float32  shaped reward from env.step(a_t)
    episode_id (N,)      int64
    t          (N,)      int64    step index within the episode
    last       (N,)      bool     True on the final step (no x_{t+1})
    ep_len     (E,)      int64    steps per episode
    ep_success (E,)      bool     goal reached (any positive reward)

Added for the linear pre-threshold predictor (DG_name=batchnorm_relu,
DG = ReLU(BN(z) - intercept), BN affine=False, eval-mode running stats):
    dg_prebn      (N, 16) float32  z = DG linear output, before BatchNorm
    dg_prerelu    (N, 16) float32  BN(z) - intercept;  ReLU(dg_prerelu) == dg
    policy_action (N,)    int64    action the policy sampled
    is_random     (N,)    bool     True if action[i] was replaced by a uniform random one
    bn_running_mean, bn_running_var (16,), bn_eps, dg_intercept, epsilon  (scalars)
"action" is always the EXECUTED action (the one passed to env.step).

Use load_transitions() below to get (x, a, x_next, dg_next, r) split by episode.

Usage (from the repo root, same flags as spot_check.py):
    PYTHONPATH=$PWD python sf_khushi/dmlab/build_transition_dataset.py \\
        --env openfield_map2_fixed_loc3 \\
        --experiment baseline_envfix_v4_s4444 \\
        --train_dir /home/fr/fr_kb470/SF_hipposlam/train_dir/train_dir \\
        --device cpu --no_render \\
        --num_episodes 50 --out transitions_s4444.npz

Extra flags (parsed before the rest goes to Sample Factory's parser):
    --num_episodes N   episodes to collect (default 20)
    --out PATH         output file (default transitions.npz)
    --epsilon E        prob. of replacing the policy action by a uniform random one (default 0.0)
    --eps_seed S       RNG seed for the epsilon draws (default 0)
"""

import sys

import numpy as np
import torch

N_FEATURES = 16


# ---------------------------------------------------------------------------
# Collection (needs Sample Factory + DMLab; only imported when actually run)
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    from sf_khushi.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components

    argv = sys.argv[1:] if argv is None else argv
    num_episodes, out_path, epsilon, eps_seed, rest = 20, "transitions.npz", 0.0, 0, []
    i = 0
    while i < len(argv):
        if argv[i] == "--num_episodes":
            num_episodes = int(argv[i + 1]); i += 2
        elif argv[i] == "--out":
            out_path = argv[i + 1]; i += 2
        elif argv[i] == "--epsilon":
            epsilon = float(argv[i + 1]); i += 2
        elif argv[i] == "--eps_seed":
            eps_seed = int(argv[i + 1]); i += 2
        else:
            rest.append(argv[i]); i += 1
    register_dmlab_components()
    return num_episodes, out_path, epsilon, eps_seed, parse_dmlab_args(argv=rest, evaluation=True)


class DGTap:
    """Forward hooks on DG_projection: z = linear(x), u = BN(z) - intercept (last call wins)."""

    def __init__(self, actor_critic):
        hits = [(n, m) for n, m in actor_critic.named_modules() if n.endswith("DG_projection")]
        assert len(hits) == 1, f"expected one DG_projection, found {[n for n, _ in hits]}"
        name, self.dg = hits[0]
        for attr in ("linear", "batchnorm1d", "intercept"):
            assert hasattr(self.dg, attr), f"{name} ({type(self.dg).__name__}) has no '{attr}' -- DG_name=batchnorm_relu?"
        assert not self.dg.batchnorm1d.training, "BN in train mode -- call actor_critic.eval() first"
        self.z = self.u = None
        self._h = [
            self.dg.linear.register_forward_hook(lambda m, i, o: setattr(self, "z", o.detach())),
            self.dg.batchnorm1d.register_forward_hook(lambda m, i, o: setattr(self, "u", o.detach() - self.dg.intercept)),
        ]
        print(f"[prebn] hooked {name} ({type(self.dg).__name__}), intercept={self.dg.intercept}")

    def pop(self):
        assert self.z is not None and self.u is not None, "no DG forward since last pop()"
        z, u, self.z, self.u = self.z, self.u, None, None
        return z, u

    def bn_params(self):
        bn = self.dg.batchnorm1d
        return {
            "bn_running_mean": bn.running_mean.detach().cpu().numpy().astype(np.float32),
            "bn_running_var": bn.running_var.detach().cpu().numpy().astype(np.float32),
            "bn_eps": np.float32(bn.eps),
            "dg_intercept": np.float32(self.dg.intercept),
        }

    def remove(self):
        for h in self._h:
            h.remove()


@torch.no_grad()
def collect_episode(actor_critic, env, env_info, device, tap, epsilon=0.0, rng=None, max_steps=2000):
    """One episode with the stochastic policy (the real interaction distribution).

    With epsilon > 0, each step's action is replaced by a uniform random action with
    probability epsilon. The baseline core does not read the action, so x_t / rnn_states
    are unaffected; only a_t (and hence what comes next) changes.
    """
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor

    assert env.num_agents == 1, "written for the single-agent eval setup (cfg.num_envs=1)"
    obs, _ = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_size = actor_critic.core.get_out_size()  # NOT get_rnn_size(cfg)
    rnn_states = torch.zeros([1, rnn_size], dtype=torch.float32, device=device)

    num_actions = int(env_info.action_space.n)
    xs, dgs, zs, pres, acts, pacts, rands, rews = [], [], [], [], [], [], [], []
    for _ in range(max_steps):
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
        out = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
        actions = out["actions"]
        new_rnn_states = out["new_rnn_states"]

        # DG input of this step, straight from the encoder (same as spot_check.py).
        # The tap is popped right after this call, so z/u come from the same forward.
        u = actor_critic.encoder(normalized_obs)[:, :N_FEATURES]
        z_pre, u_pre = tap.pop()

        # epsilon-greedy: swap the sampled action for a uniform random one
        policy_a = int(actions.reshape(-1)[0].item())
        is_rand = epsilon > 0 and torch.rand(1, generator=rng).item() < epsilon
        if is_rand:
            a = int(torch.randint(0, num_actions, (1,), generator=rng).item())
            actions = torch.full_like(actions, a)
        else:
            a = policy_a

        xs.append(new_rnn_states[0].cpu().numpy().copy())   # x_t: post-injection
        dgs.append(u[0].cpu().numpy().copy())
        zs.append(z_pre[0].cpu().numpy().copy())
        pres.append(u_pre[0].cpu().numpy().copy())
        acts.append(a)                                      # executed action
        pacts.append(policy_a)
        rands.append(is_rand)

        actions_ = unsqueeze_tensor(actions, dim=-1) if actions.ndim == 1 else actions
        obs, rew, terminated, truncated, _ = env.step(preprocess_actions(env_info, actions_))
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        rews.append(float(rew.reshape(-1)[0].item()))

        rnn_states = new_rnn_states
        if bool(terminated.any()) or bool(truncated.any()):
            break

    return {
        "x": np.stack(xs).astype(np.float32),
        "dg": np.stack(dgs).astype(np.float32),
        "action": np.array(acts, dtype=np.int64),
        "reward": np.array(rews, dtype=np.float32),
        "dg_prebn": np.stack(zs).astype(np.float32),
        "dg_prerelu": np.stack(pres).astype(np.float32),
        "policy_action": np.array(pacts, dtype=np.int64),
        "is_random": np.array(rands, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Pure numpy/torch helpers (no Sample Factory needed; testable offline)
# ---------------------------------------------------------------------------
def episodes_to_arrays(episodes):
    """Concatenate per-episode dicts into the flat arrays that get saved."""
    lens = [len(e["action"]) for e in episodes]
    keys = ("x", "dg", "action", "reward", "dg_prebn", "dg_prerelu", "policy_action", "is_random")
    out = {k: np.concatenate([e[k] for e in episodes]) for k in keys}
    out["episode_id"] = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(lens)])
    out["t"] = np.concatenate([np.arange(n, dtype=np.int64) for n in lens])
    out["last"] = np.concatenate([np.arange(n) == n - 1 for n in lens])
    out["ep_len"] = np.array(lens, dtype=np.int64)
    out["ep_success"] = np.array([bool((e["reward"] > 0).any()) for e in episodes])
    return out


def check_transitions(data, layout, atol=1e-5):
    """Two checks before anything trains on this file.

    1. dg[i] == slot 0 of every feature in x[i]   (u_t really is inside x_t)
    2. step(x[i], dg[i+1]) == x[i+1] on the core part, for every non-last row
       (x_{t+1} is exactly one fixed shift + injection away from x_t, so the
       pairing x[i] -> x[i+1] is right and nothing is off by one)
    """
    from ca3_advance import dg_input_from_states, step

    x = torch.as_tensor(data["x"]).double()
    dg = torch.as_tensor(data["dg"]).double()
    ok = True

    err1 = (dg_input_from_states(x, layout) - dg).abs().max().item()
    print(f"[1] dg == slot 0 of x:        max err {err1:.2e}")
    ok &= err1 <= atol

    i = np.nonzero(~data["last"])[0]
    if len(i):
        idx = torch.as_tensor(i)
        pred = step(x[idx], dg[idx + 1], layout)
        err = (layout.split(pred)[0] - layout.split(x[idx + 1])[0]).abs().amax(dim=(1, 2))
        n_bad = int((err > atol).sum())
        print(f"[2] step(x_t, u_t+1) == x_t+1: {len(i) - n_bad}/{len(i)} transitions match "
              f"(max err {err.max().item():.2e})")
        ok &= n_bad == 0

    # 3. ReLU(dg_prerelu) == dg   (pre-threshold values belong to the same rows)
    pre = torch.as_tensor(data["dg_prerelu"]).double()
    err3 = (torch.relu(pre) - dg).abs().max().item()
    print(f"[3] ReLU(dg_prerelu) == dg:     max err {err3:.2e}")
    ok &= err3 <= atol

    # 4. dg_prerelu == (dg_prebn - mean) / sqrt(var + eps) - intercept   (fixed eval-mode BN)
    z = torch.as_tensor(data["dg_prebn"]).double()
    mu = torch.as_tensor(data["bn_running_mean"]).double()
    var = torch.as_tensor(data["bn_running_var"]).double()
    pre_hat = (z - mu) / torch.sqrt(var + float(data["bn_eps"])) - float(data["dg_intercept"])
    err4 = (pre_hat - pre).abs().max().item()
    print(f"[4] BN(dg_prebn) - c == prerelu: max err {err4:.2e}")
    ok &= err4 <= 1e-4   # float32 BN on the model side

    print("CHECK:", "PASS" if ok else "FAIL -- do not train on this file")
    return ok


def load_transitions(path, heldout_frac=0.2, seed=0):
    """Return (train, heldout) dicts of transitions, split by whole episodes.

    Each dict has x (n, D), action (n,), x_next (n, D), dg_next (n, 16),
    reward (n,), episode_id (n,), and -- for dumps made with this version --
    dg_prerelu_next (n, 16), dg_prebn_next (n, 16), is_random (n,), policy_action (n,).
    is_random[i] refers to action[i] (the action of the transition).
    Older dumps without these keys still load.
    """
    d = np.load(path)
    rows = np.nonzero(~d["last"])[0]
    ep_ids = np.unique(d["episode_id"])
    rng = np.random.RandomState(seed)
    held = set(rng.choice(ep_ids, size=max(1, int(round(heldout_frac * len(ep_ids)))), replace=False).tolist())
    is_held = np.isin(d["episode_id"][rows], list(held))

    def take(r):
        out = {
            "x": d["x"][r], "action": d["action"][r], "x_next": d["x"][r + 1],
            "dg_next": d["dg"][r + 1], "reward": d["reward"][r], "episode_id": d["episode_id"][r],
        }
        if "dg_prerelu" in d.files:
            out["dg_prerelu_next"] = d["dg_prerelu"][r + 1]
            out["dg_prebn_next"] = d["dg_prebn"][r + 1]
            out["is_random"] = d["is_random"][r]
            out["policy_action"] = d["policy_action"][r]
        return out

    return take(rows[~is_held]), take(rows[is_held])


# ---------------------------------------------------------------------------
def main():
    from ca3_advance import CA3Layout
    from eval_utils import build_model_and_env

    num_episodes, out_path, epsilon, eps_seed, cfg = parse_args()
    cfg, actor_critic, env, env_info, device = build_model_and_env(cfg)
    actor_critic.eval()
    layout = CA3Layout.from_core(actor_critic.core)
    tap = DGTap(actor_critic)
    rng = torch.Generator().manual_seed(eps_seed)
    print(f"epsilon={epsilon}, eps_seed={eps_seed}")

    episodes = []
    for ep in range(num_episodes):
        e = collect_episode(actor_critic, env, env_info, device, tap, epsilon=epsilon, rng=rng)
        fired = (e["dg"] > 0).mean(axis=0)
        print(f"episode {ep}: T={len(e['action'])} steps, success={bool((e['reward'] > 0).any())}, "
              f"active features={int((fired > 0).sum())}/16, mean rate of active={fired[fired > 0].mean() if (fired > 0).any() else 0:.2%}, "
              f"random actions={e['is_random'].mean():.1%}")
        episodes.append(e)
    env.close()
    tap.remove()

    data = episodes_to_arrays(episodes)
    data.update(tap.bn_params())
    data["epsilon"] = np.float32(epsilon)
    n_trans = int((~data["last"]).sum())
    print(f"\n{len(data['action'])} steps, {n_trans} transitions, {num_episodes} episodes, "
          f"success {data['ep_success'].mean():.0%}")
    rate = (data["dg"] > 0).mean(axis=0)
    print("per-feature firing rate:", " ".join(f"{r:.1%}" for r in rate))
    counts = np.bincount(data["action"], minlength=int(env_info.action_space.n))
    print("executed action counts:", " ".join(f"{a}:{c}" for a, c in enumerate(counts)))

    ok = check_transitions(data, layout)
    np.savez_compressed(out_path, **data)
    print(f"Saved to {out_path}" + ("" if ok else "  (CHECK FAILED -- see above)"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)   # non-zero exit stops the sbatch before Gate 1 runs