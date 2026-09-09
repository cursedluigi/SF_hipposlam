"""
compare_planner_vs_baseline.py

Phase-2 evaluation for the HippoSLAM one-step lookahead planner (plan_one_step.py).

Runs `--max_num_episodes` episodes of a trained checkpoint in one or both modes and
reports the difference:

    "off"       : normal trained policy head chooses every action (mirrors
                  sample_factory.enjoy.enjoy() exactly, including --eval_deterministic).
    "onestep"   : plan_one_step()'s argmax overrides the action every step.

------------------------------------------------------------------------------------
PROVENANCE -- what's confirmed vs. still a guess
------------------------------------------------------------------------------------
The setup/loop below (make_env, load_state_dict, rnn_states sizing via get_rnn_size,
action_mask popping, actions -> unsqueeze_tensor -> preprocess_actions -> env.step,
dones via make_dones, per-agent rnn_states reset on done) is copied structurally from
the real source of sample_factory.enjoy.enjoy() (Sample Factory, alex-petrenko/sample-
factory). This is no longer a reconstruction --
it's the same calls in the same order, minus the render/video/wandb/hf-push branches
enjoy() has that this script doesn't need, plus a mode branch at the action-selection
point where enjoy()'s loop just does `policy_outputs = actor_critic(...)`.

Still genuinely unconfirmed, because nothing in enjoy() itself tells us this:

  SUCCESS SIGNAL. enjoy() has NO concept of "success" or "goal reached" anywhere --
  it only tracks episode reward and infos[i].get("true_objective", reward). Whatever
  marks "the agent reached the goal" (vs. hitting the frame cap) must live in
  dmlab_gym.py / DmlabGymEnv_custom's info dict, or nowhere explicit at all (in which
  case you'd infer success from the reward's sparse terminal bonus instead). grep
  dmlab_gym.py for how it flags goal-reach, or check what an `info` dict actually
  contains at episode end, before trusting get_success() below -- it's still a
  guess-with-fallback exactly as before.

  ACTION_MASK. enjoy() pops "action_mask" from obs *if present*. I don't know whether
  HippoSLAM's action space uses masking (your 5-action reduced IMPALA set suggests
  probably not, since all 5 are presumably always legal) -- but plan_one_step() as
  written does NOT pass action_mask through to model.encoder/core/decoder at all. If
  your obs dict does contain "action_mask", plan_one_step()'s imagined branches would
  silently ignore it while the real "off"-mode forward pass would not, which would
  make the two modes subtly incomparable. Confirm action_mask is absent (or unused)
  for this env before trusting a head-to-head comparison.

  env.num_agents. Assumed 1 (single-agent nav task per the paper), but the loop below
  is written generically over env.num_agents like the real enjoy() is, so it should
  hold even if that assumption is wrong.

------------------------------------------------------------------------------------
Usage -- same flags as enjoy_hipposlam.py, plus --modes
------------------------------------------------------------------------------------
    python compare_planner_vs_baseline.py \\
        --env openfield_map2_fixed_loc3 \\
        --experiment prev_action_dg_v1 \\
        --train_dir /home/fr/fr_kb470/SF_hipposlam/sf_khushi/train_dir \\
        --max_num_episodes 50 \\
        --max_num_frames 900 \\
        --no_render \\
        --modes off onestep

(--max_num_episodes, --max_num_frames, --no_render, --eval_deterministic, --device,
--train_dir, --experiment are all pre-existing Sample Factory / dmlab args, same ones
enjoy_hipposlam.py already accepts -- --modes is the only new flag this script adds.)
"""

import sys

import numpy as np
import torch

# Same directory as plan_one_step.py is assumed.
from plan_one_step import plan_one_step

from sf_khushi.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components

from sample_factory.algo.learning.learner import BaseLearner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict


def parse_args(argv=None):
    """
    Splits out our one custom flag (--modes) before handing the rest to
    parse_dmlab_args, exactly as enjoy_hipposlam.py does with its own args.
    """
    argv = sys.argv[1:] if argv is None else argv

    modes = ["off", "onestep"]
    remaining_argv = []
    i = 0
    while i < len(argv):
        if argv[i] == "--modes":
            modes = []
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                modes.append(argv[i])
                i += 1
        else:
            remaining_argv.append(argv[i])
            i += 1

    for m in modes:
        assert m in ("off", "onestep"), f"unknown mode {m!r}"

    register_dmlab_components()
    cfg = parse_dmlab_args(argv=remaining_argv, evaluation=True)

    return modes, cfg


def make_env(cfg):
    """Mirrors enjoy.make_env(), minus render_mode (we never render for this comparison)."""
    return make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
    )


def load_state_dict(cfg, actor_critic, device):
    """Copied verbatim from sample_factory.enjoy.load_state_dict, using BaseLearner (not Learner)."""
    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = BaseLearner.get_checkpoints(BaseLearner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = BaseLearner.load_checkpoint(checkpoints, device)
    if checkpoint_dict:
        actor_critic.load_state_dict(checkpoint_dict["model"])
    else:
        raise RuntimeError("Could not load checkpoint")


def get_success(info, episode_frame_count, max_frames):
    """
    UNCONFIRMED -- see provenance note at top of file. Guessing common key names,
    falling back to "episode ended before the frame cap". The fallback is NOT
    reliable if any other early-termination condition exists in dmlab_gym.py.
    """
    for key in ("success", "reached_goal", "goal_reached", "is_success"):
        if key in info:
            return bool(info[key])
    return episode_frame_count < max_frames


def build_model_and_env(cfg):
    """Mirrors the setup portion of enjoy() -- everything before its while-loop."""
    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert cfg.env_frameskip % eval_env_frameskip == 0, (
        f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    )
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    cfg.num_envs = 1

    env = make_env(cfg)
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    load_state_dict(cfg, actor_critic, device)

    return cfg, actor_critic, env, env_info, device


@torch.no_grad()
def run_eval(cfg, actor_critic, env, env_info, device, mode, num_episodes, max_frames):
    """
    Same frame-stream structure as enjoy()'s while-loop: continuously step until
    `num_episodes` episodes have completed, resetting per-agent rnn_states (not the
    env -- envs auto-reset internally) on each done, exactly as enjoy() does.

    mode == "off":     identical to enjoy()'s action selection, including
                        --eval_deterministic handling via argmax_actions.
    mode == "onestep": plan_one_step()'s argmax overrides the action. The matching
                        imagined_state becomes the next rnn_states for that agent --
                        valid because plan_one_step() computed it from the real
                        encoder/core with that exact action already swapped in, i.e.
                        identical to what a real forward pass with that prev_action
                        would produce. (This reuse is the one piece of reasoning
                        carried over unmodified from your own handoff doc.)
    """
    assert mode in ("off", "onestep")
    num_agents = env.num_agents

    all_ep_rewards, all_ep_lengths, all_ep_successes = [], [], []

    obs, infos = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros([num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    episode_frame_count = np.zeros(num_agents, dtype=int)
    num_episodes_done = 0

    while num_episodes_done < num_episodes:
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

        if mode == "off":
            policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
            actions = policy_outputs["actions"]
            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)
            new_rnn_states = policy_outputs["new_rnn_states"]
        else:  # "onestep"
            values, best_action, imagined_states = plan_one_step(actor_critic, normalized_obs, rnn_states)
            actions = best_action
            new_rnn_states = torch.zeros_like(rnn_states)
            for i in range(num_agents):
                new_rnn_states[i] = imagined_states[int(best_action[i].item())][i]

        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions_to_step = preprocess_actions(env_info, actions)

        obs, rew, terminated, truncated, infos = env.step(actions_to_step)
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        dones = make_dones(terminated, truncated)
        infos = [{} for _ in range(num_agents)] if infos is None else infos

        if episode_reward is None:
            episode_reward = rew.float().clone()
        else:
            episode_reward += rew.float()

        rnn_states = new_rnn_states
        episode_frame_count += 1

        dones_np = dones.cpu().numpy()
        for agent_i, done_flag in enumerate(dones_np):
            if done_flag:
                ep_len = int(episode_frame_count[agent_i])
                all_ep_rewards.append(episode_reward[agent_i].item())
                all_ep_lengths.append(ep_len)
                agent_info = infos[agent_i] if isinstance(infos, (list, tuple)) else {}
                all_ep_successes.append(get_success(agent_info, ep_len, max_frames))

                rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                episode_reward[agent_i] = 0
                episode_frame_count[agent_i] = 0
                num_episodes_done += 1

    return {
        "mean_reward": float(np.mean(all_ep_rewards)),
        "std_reward": float(np.std(all_ep_rewards)),
        "mean_length": float(np.mean(all_ep_lengths)),
        "std_length": float(np.std(all_ep_lengths)),
        "success_rate": float(np.mean(all_ep_successes)),
        "episode_rewards": all_ep_rewards,
        "episode_lengths": all_ep_lengths,
        "episode_successes": all_ep_successes,
    }


def print_report(results_by_mode):
    print("\n" + "=" * 60)
    print("PLANNER vs BASELINE -- summary")
    print("=" * 60)
    header = f"{'mode':<10}{'mean reward':>14}{'mean length':>14}{'success rate':>15}"
    print(header)
    print("-" * len(header))
    for mode, r in results_by_mode.items():
        print(f"{mode:<10}{r['mean_reward']:>14.2f}{r['mean_length']:>14.1f}{r['success_rate']:>15.2%}")

    if "off" in results_by_mode and "onestep" in results_by_mode:
        off, onestep = results_by_mode["off"], results_by_mode["onestep"]
        print("-" * len(header))
        d_reward = onestep["mean_reward"] - off["mean_reward"]
        d_length = onestep["mean_length"] - off["mean_length"]
        d_success = onestep["success_rate"] - off["success_rate"]
        print(f"{'delta (onestep-off)':<10}{d_reward:>+14.2f}{d_length:>+14.1f}{d_success:>+15.2%}")
    print("=" * 60 + "\n")


def main():
    modes, cfg = parse_args()

    max_num_episodes = cfg.max_num_episodes if cfg.max_num_episodes is not None else 50
    max_num_frames = cfg.max_num_frames if cfg.max_num_frames is not None else 900

    cfg, actor_critic, env, env_info, device = build_model_and_env(cfg)

    results_by_mode = {}
    for mode in modes:
        print(f"\nRunning {max_num_episodes} episodes in mode={mode!r} "
              f"(max_frames={max_num_frames}) ...")
        results_by_mode[mode] = run_eval(
            cfg, actor_critic, env, env_info, device, mode,
            num_episodes=max_num_episodes,
            max_frames=max_num_frames,
        )

    print_report(results_by_mode)
    env.close()


if __name__ == "__main__":
    main()