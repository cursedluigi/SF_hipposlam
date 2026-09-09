"""
standalone one-step lookahead planner for HippoSLAM.

So plan_one_step calls real encoder -> core -> decoder -> critic_linear once per
candidate action, with a different prev_action fed in each time. No new numerics, no chance of
the standalone path silently drifting from the trained path.

"""

import torch


@torch.no_grad()
def plan_one_step(model, obs_dict, rnn_states, num_actions=None):
    """
    One-step lookahead: for each candidate action, "what if I had taken this action
    last step instead", run it through the real CA3 core, decode, and score with the real critic.

    Args:
        model: the ActorCriticSharedWeights instance (has .encoder, .core, .decoder, .critic_linear
               -- confirmed these are the exact attribute names Sample Factory uses when
               share_weights=True, which is your live config).
        obs_dict: dict of current-timestep observation tensors, exactly as the encoder expects
                  them (must include "prev_action" as a placeholder key -- it gets overwritten
                  per-candidate below, so its actual value doesn't matter).
                  Shapes are (B, ...) for every entry.
        rnn_states: Tensor (B, core_output_size) -- the CURRENT CA3 shift-register state,
                    i.e. what's in memory right now, before imagining the next step.
        num_actions: how many candidate actions to score. If None, read it off
                     model.encoder.action_encoder.num_actions (requires use_prev_action=True,
                     which is your live setup on this branch).

    Returns:
        values: Tensor (B, num_actions) -- critic's value estimate for each imagined action.
        best_action: Tensor (B,) -- argmax over actions.
        imagined_states: list of length num_actions, each a Tensor (B, core_output_size) --
                         the CA3 state you'd end up in after taking that action. Useful later
                         if you want to actually step the real env with the chosen action and
                         warm-start rnn_states with the matching imagined_states entry (though
                         in practice you'll just let the real env step and get the real state back).
    """
    device = next(model.parameters()).device
    B = rnn_states.shape[0]

    if num_actions is None:
        if not hasattr(model.encoder, "action_encoder"):
            raise ValueError(
                "model.encoder has no action_encoder -- this planner requires use_prev_action=True. "
                "Pass num_actions explicitly if you're testing against a differently-configured model."
            )
        num_actions = model.encoder.action_encoder.num_actions

    values = torch.zeros(B, num_actions, device=device)
    imagined_states = []

    for a in range(num_actions):
        # Shallow copy: same tensors, except we swap in the candidate action.
        # Doesn't touch the caller's obs_dict.
        imagined_obs = dict(obs_dict)
        imagined_obs["prev_action"] = torch.full((B,), a, dtype=torch.long, device=device)

        # Real encoder: visual+instruction features -> concat candidate action embedding ->
        # DG projection -> concat action-free bypass (x_pre_action). Nothing hand-rolled here.
        head_output = model.encoder(imagined_obs)  # (B, Hippo_n_feature + bypass_size)

        # Real core, plain-tensor branch: one shift-register roll + inject, from the CURRENT
        # rnn_states (unchanged across the loop -- we're branching hypothetically from the
        # same starting memory each time, not chaining actions).
        imagined_core_out, imagined_rnn_state = model.core(head_output, rnn_states)

        # Real decoder + real critic head.
        decoder_out = model.decoder(imagined_core_out)
        value = model.critic_linear(decoder_out).squeeze(-1)  # (B,)

        values[:, a] = value
        imagined_states.append(imagined_rnn_state)

    best_action = values.argmax(dim=1)
    return values, best_action, imagined_states