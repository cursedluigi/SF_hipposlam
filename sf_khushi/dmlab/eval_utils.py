"""

Shared checkpoint/env loading for offline eval scripts 
replaced from the deleted planner_vs_baseline.py
so those scripts no longer depend on the planner code.

Usage from another script in sf_khushi/dmlab/:
    from eval_utils import build_model_and_env, parse_args
    cfg = parse_args()
    cfg, actor_critic, env, env_info, device = build_model_and_env(cfg)
    rnn_size = actor_critic.core.get_out_size()   # NOT get_rnn_size(cfg)
"""

import sys

import torch

from sf_khushi.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components

from sample_factory.algo.learning.learner import BaseLearner
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.attr_dict import AttrDict


def parse_args(argv=None):
    """Parse the usual Sample Factory / dmlab eval flags (--env, --experiment, --train_dir, ...)."""
    argv = sys.argv[1:] if argv is None else argv
    register_dmlab_components()
    return parse_dmlab_args(argv=argv, evaluation=True)


def make_env(cfg):
    """Mirrors enjoy.make_env(), without rendering."""
    return make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
    )


def load_state_dict(cfg, actor_critic, device):
    """Same as sample_factory.enjoy.load_state_dict, using BaseLearner."""
    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = BaseLearner.get_checkpoints(BaseLearner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = BaseLearner.load_checkpoint(checkpoints, device)
    if checkpoint_dict:
        actor_critic.load_state_dict(checkpoint_dict["model"])
    else:
        raise RuntimeError("Could not load checkpoint")


def build_model_and_env(cfg):
    """Mirrors the setup part of enjoy(): everything before its while-loop."""
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