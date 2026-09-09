"""
Login-node smoke test for plan_one_step.

Purpose: catch shape errors, missing-attribute errors, and wiring mistakes BEFORE
this ever touches enjoy_hipposlam.py or a real checkpoint. No SLURM, no training,
and -- deliberately -- no DeepMind Lab either.

This builds the model the same way sf_khushi/dmlab/train_hipposlam.py does (same
encoder/core/decoder factories, same cfg flags as your real
command_line/train_dmlab_1.sh launch script), but skips register_dmlab_envs(), which
is the one piece that actually requires deepmind_lab / the vis-mesa+sdl2 modules.
That means this script runs anywhere torch + torchvision are available -- doesn't
even need to be on the cluster, let alone in a SLURM job.

Run from the repo root (so the sf_khushi / sf_working_directories packages resolve):

    cd /home/fr/fr_kb470/SF_hipposlam   # or wherever your repo root is
    conda activate KhushiSF
    python test_plan_one_step.py
"""

import gymnasium as gym
import numpy as np
import torch

from plan_one_step import plan_one_step

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.algo.utils.model_context import global_model_factory
from sample_factory.model.actor_critic import create_actor_critic

from sf_khushi.dmlab.custom_params import add_hipposlam_env_args, hipposlam_override_defaults
from sf_khushi.dmlab.dmlab_params import add_dmlab_env_args, dmlab_override_defaults
from sf_khushi.dmlab.custom_encoder import make_hipposlam_encoder
from sf_khushi.dmlab.custom_core import make_hipposlam_core
from sf_khushi.dmlab.custom_decoder import make_hipposlam_decoder
from sf_khushi.dmlab.dmlab30 import DMLAB_INSTRUCTIONS


# Same flags as command_line/train_dmlab_1.sh, minus the SLURM/runner-only ones
# (--env, --experiment, --train_for_seconds, --num_workers, --with_pbt, --with_wandb,
#  --dmlab_use_level_cache, --dmlab_one_task_per_worker) that only matter once you're
# actually spinning up the env/runner, which this smoke test doesn't do.
TRAIN_DMLAB_1_FLAGS = [
    "--algo=APPO",
    "--use_rnn=True",
    "--rnn_type=gru",
    "--core_name=BypassSS",
    "--DG_name=linear_relu",
    "--Hippo_n_feature=16",
    "--Hippo_L=64",
    "--Hippo_R=8",
    "--rnn_size=1149",
    "--encoder_conv_architecture=layer2_resnet18",
    "--encoder_conv_mlp_layers=256",
    "--depth_sensor=True",
    "--DG_BN_intercept=2.43",
    "--decoder_mlp_layers", "128", "128",
    "--env_frameskip=8",
    "--device=cpu",
    "--seed=1111",
    "--use_prev_action=True",
    "--action_encoding=embedding",
    "--action_embed_dim=8",
]


def register_model_factories_only():
    """
    This is register_dmlab_components() from sf_khushi/dmlab/train_hipposlam.py, minus
    register_dmlab_envs() (which imports dmlab_env.py -> deepmind_lab). We don't need a
    real env to shape-check plan_one_step, only the encoder/core/decoder factories.
    """
    global_model_factory().register_encoder_factory(make_hipposlam_encoder)
    global_model_factory().register_model_core_factory(make_hipposlam_core)
    global_model_factory().register_decoder_factory(make_hipposlam_decoder)


def build_cfg(extra_flags=None):
    argv = list(TRAIN_DMLAB_1_FLAGS)
    if extra_flags:
        argv += extra_flags
    # env is required by parse_sf_args but unused since we never touch dmlab_env.py here
    argv = ["--env=dmlab_dummy"] + argv
    parser, cfg = parse_sf_args(argv, evaluation=True)
    add_hipposlam_env_args(parser)
    add_dmlab_env_args(parser)
    hipposlam_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def build_obs_space(cfg):
    """
    Matches DmlabGymEnv_custom.observation_space (dmlab_gym.py), transposed to
    channel-first the way Sample Factory's env wrappers would present it to the model.
    Note HipposlamEncoder.__init__ actually hardcodes its own 3-channel Box internally
    (custom_encoder.py ~line 599) regardless of what we pass here, so this obs_space is
    mostly a formality for create_actor_critic's bookkeeping -- but we match the real
    shapes anyway so nothing downstream (e.g. normalize_input_keys) trips on a mismatch.
    """
    num_channels = 4 if cfg.depth_sensor else 3
    return gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(low=0, high=255, shape=(num_channels, cfg.res_h, cfg.res_w), dtype=np.uint8),
            DMLAB_INSTRUCTIONS: gym.spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            "prev_action": gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.int32),
        }
    )


def build_dummy_obs(cfg, batch_size, device):
    num_channels = 4 if cfg.depth_sensor else 3
    obs = {
        "obs": torch.randint(
            0, 255, (batch_size, num_channels, cfg.res_h, cfg.res_w), dtype=torch.uint8
        ).float().to(device),
        DMLAB_INSTRUCTIONS: torch.ones((batch_size, 1), dtype=torch.int64).to(device),
        # placeholder -- plan_one_step overwrites this per candidate action anyway.
        "prev_action": torch.zeros((batch_size,), dtype=torch.int64).to(device),
    }
    return obs


def main():
    register_model_factories_only()
    cfg = build_cfg()

    device = "cpu"
    obs_space = build_obs_space(cfg)
    action_space = gym.spaces.Discrete(5)  # dmlab_reduced_action_set -> REDUCED_ACTION_SET has 5 entries

    model = create_actor_critic(cfg, obs_space, action_space)
    model.to(device)
    model.eval()

    print(f"encoder out size: {model.encoder.get_out_size()}")
    print(f"core out size (== rnn state size): {model.core.get_out_size()}")
    print(f"decoder out size: {model.decoder.get_out_size()}")

    B = 4
    obs = build_dummy_obs(cfg, batch_size=B, device=device)

    core_output_size = model.core.get_out_size()
    rnn_states = torch.zeros(B, core_output_size, device=device)

    values, best_action, imagined_states = plan_one_step(model, obs, rnn_states)

    print(f"\nvalues shape: {tuple(values.shape)}  (expected: ({B}, 5))")
    print(f"best_action shape: {tuple(best_action.shape)}  (expected: ({B},))")
    print(f"num imagined_states: {len(imagined_states)}  (expected: 5)")
    print(f"each imagined_state shape: {tuple(imagined_states[0].shape)}  (expected: ({B}, {core_output_size}))")
    print(f"\nvalues:\n{values}")
    print(f"best_action:\n{best_action}")

    assert values.shape == (B, len(imagined_states))
    assert best_action.shape == (B,)
    assert all(s.shape == (B, core_output_size) for s in imagined_states)
    print("\nAll shape assertions passed.")


if __name__ == "__main__":
    main()