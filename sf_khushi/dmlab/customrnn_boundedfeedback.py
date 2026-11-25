from typing import Dict, Optional, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF, Tensor
from torch.nn import RNNBase
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

from sample_factory.utils.utils import log


class CustomRNN(RNNBase):

    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        rank=1,  # added the rank to allow low-rank adaptations
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None: ...

    @overload
    def __init__(self, *args, **kwargs) -> None: ...

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        if len(args) > 1:
            self.hidden_size = args[1]
        else:
            self.hidden_size = kwargs.get("hidden_size", 0)
        # Positional layout: (input_size, hidden_size, num_layers, nonlinearity, rank,
        #                    bias, batch_first, dropout, bidirectional, device, dtype)
        if len(args) > 3:
            self.nonlinearity = args[3]
        else:
            self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if len(args) > 4:
            self.rank = args[4]
        else:
            self.rank = kwargs.pop("rank", 1)
        # Action-conditioned feedback (keyword only). 0 = original shared feedback.
        self.n_actions = int(kwargs.pop("n_actions", 0))
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'.")

        if len(args) > 9:
            self.device = args[9]
        else:
            self.device = kwargs.get("device", None)
        if len(args) > 10:
            self.dtype = args[10]
        else:
            self.dtype = kwargs.get("dtype", None)
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        # Strip nonlinearity/rank (and positional device/dtype, which RNNBase expects
        # after proj_size) so the remaining args line up with RNNBase's signature.
        args = args[:3] + args[5:9]
        if self.device is not None:
            kwargs["device"] = self.device
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype

        # for layer in range(self.num_layers):
        # for direction in range(num_directions):

        super().__init__(mode, *args, **kwargs)

        # change the rnn constructor to include low rank adaptation
        # B
        self.lr_column = Parameter(torch.empty((self.hidden_size, self.rank), **factory_kwargs))
        # A
        self.lr_row = Parameter(torch.empty((self.rank, self.hidden_size), **factory_kwargs))

        self.reset_parameters()

        # Action-conditioned CA3 -> DG feedback:
        #     A(a) = lr_row + lr_row_action[a]
        # Zero-initialised, so training starts from exactly the shared feedback.
        # Created after reset_parameters() so it is not randomised.
        if self.n_actions > 0:
            self.lr_row_action = Parameter(torch.zeros((self.n_actions, self.rank, self.hidden_size), **factory_kwargs))
        else:
            self.lr_row_action = None

        # add to log
        self._feedback_forward_count = 0

        self.feedback_norm = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self._last_forward_summaries: Dict[str, float] = {}

        # Per-timestep finite/bound checks force a CPU-GPU sync every step.
        # Keep True while debugging; set False for full runs.
        self.debug_checks = True

        self._fixed_weights_set = False
        self._warned_unfixed = False

    def summaries(self) -> Dict[str, float]:
        return dict(self._last_forward_summaries)

    def _set_last_forward_summaries(
        self,
        state_abs_max,
        feedback_abs_max,
        empirical_gain,
    ) -> None:
        if len(state_abs_max) == 0:
            self._last_forward_summaries = {}
            return

        # Only record summaries for training (grad-enabled) forwards, so single-step
        # rollout inference under no_grad does not overwrite them.
        if not torch.is_grad_enabled():
            return

        state_per_t = torch.stack(state_abs_max)
        feedback_per_t = torch.stack(feedback_abs_max)

        self._last_forward_summaries = {
            "rnn/ca3_abs_max": float(state_per_t.max()),
            "rnn/ca3_abs_mean": float(state_per_t.mean()),
            "rnn/ca3_abs_last": float(state_per_t[-1]),
            "rnn/feedback_abs_max": float(feedback_per_t.max()),
            "rnn/feedback_abs_mean": float(feedback_per_t.mean()),
            "rnn/feedback_abs_last": float(feedback_per_t[-1]),
        }

        # State-norm ratio ||h_t|| / ||h_{t-1}|| (includes input drive, so it is not
        # a pure recurrence gain). Steps with no nonzero previous state are skipped.
        gain_per_t = torch.stack(empirical_gain)
        gain_per_t = gain_per_t[~torch.isnan(gain_per_t)]
        if gain_per_t.numel() > 0:
            self._last_forward_summaries.update(
                {
                    "rnn/state_norm_ratio_max": float(gain_per_t.max()),
                    "rnn/state_norm_ratio_mean": float(gain_per_t.mean()),
                    "rnn/state_norm_ratio_last": float(gain_per_t[-1]),
                }
            )

    @staticmethod
    def _state_norm_ratio(h_prev, h_new, eps: float = 1e-6):
        """
        Mean per-row ||h_new|| / ||h_prev|| over rows whose previous state is nonzero.
        Rows starting from a zero state (episode start / reset) are excluded, since
        the ratio is undefined there. Returns NaN if no row qualifies (filtered out
        in _set_last_forward_summaries); avoids a host sync per timestep.
        """
        prev_norm = h_prev.detach().norm(p=2, dim=-1)
        next_norm = h_new.detach().norm(p=2, dim=-1)
        valid = (prev_norm > eps).to(next_norm.dtype)
        ratio = next_norm / prev_norm.clamp_min(eps)
        return (ratio * valid).sum() / valid.sum()  # 0/0 -> NaN when no valid rows

    def set_fixed_weights(self, W_in, W_hh):
        """
        Configure fixed CA3 weights and constrained DG-feedback LoRA.

        Recurrence actually implemented in _bounded_feedback_step:
            fb_t = tanh( LayerNorm(h_{t-1}) @ normalize(W_feedback).T )   # DG space
            h_t  = phi( W_in @ (x_t + fb_t) + W_hh @ h_{t-1} )
        (direction-only W_feedback; not a linear low-rank W_hh update)

        With n_actions > 0, W_feedback depends on the previous action:
            W_feedback(a_{t-1}) = lr_row + lr_row_action[a_{t-1}]
        and the input carries one-hot(a_{t-1}) after the DG columns (see _split_input).

        lr_column = W_in          fixed
        lr_row    = W_feedback    trainable

        Requires rank == input_size, since lr_column holds W_in (hidden x input).
        Biases (if bias=True) are zeroed and frozen: the fixed CA3 has no bias.
        """
        if tuple(W_in.shape) != (self.hidden_size, self.input_size):
            raise ValueError(f"W_in must have shape {(self.hidden_size, self.input_size)}, got {tuple(W_in.shape)}")
        if tuple(W_hh.shape) != (self.hidden_size, self.hidden_size):
            raise ValueError(f"W_hh must have shape {(self.hidden_size, self.hidden_size)}, got {tuple(W_hh.shape)}")
        if self.rank != self.input_size:
            raise ValueError(
                f"set_fixed_weights requires rank == input_size (lr_column = W_in); "
                f"got rank={self.rank}, input_size={self.input_size}"
            )

        with torch.no_grad():  # disable gradient calculation for fixed weights
            self.weight_ih_l0.copy_(W_in)
            self.weight_hh_l0.copy_(W_hh)

            if self.bias:
                self.bias_ih_l0.zero_()
                self.bias_hh_l0.zero_()

            # Fixed DG -> CA3 map.
            self.lr_column.copy_(W_in)

            # Initial condition: no learned feedback yet.
            # self.lr_row.zero_() # comment out for only direction learning, since I normalize
            self.lr_row.normal_(mean=0.0, std=0.02)

        # Freeze fixed CA3 and DG -> CA3 structure.
        self.weight_ih_l0.requires_grad_(False)
        self.weight_hh_l0.requires_grad_(False)
        self.lr_column.requires_grad_(False)
        if self.bias:
            self.bias_ih_l0.requires_grad_(False)
            self.bias_hh_l0.requires_grad_(False)

        # Only learned matrix: CA3 -> DG feedback (plus per-action offsets if n_actions > 0,
        # starting at zero so the initial feedback equals the shared one).
        self.lr_row.requires_grad_(True)
        if self.lr_row_action is not None:
            with torch.no_grad():
                self.lr_row_action.zero_()
            self.lr_row_action.requires_grad_(True)

        self._fixed_weights_set = True

    def update_weights(self):
        #

        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tensor] = None) -> tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):
        self._feedback_forward_count += 1

        if not self._fixed_weights_set and not self._warned_unfixed:
            log.warning(
                "CustomRNN: set_fixed_weights() was not called; W_ih/W_hh are randomly "
                "initialised and trainable."
            )
            self._warned_unfixed = True

        if self.debug_checks:
            if not torch.isfinite(self.lr_row).all():
                raise RuntimeError("lr_row contains NaN or Inf")

            if not torch.isfinite(self.lr_column).all():
                raise RuntimeError("lr_column contains NaN or Inf")

            if self.lr_row_action is not None and not torch.isfinite(self.lr_row_action).all():
                raise RuntimeError("lr_row_action contains NaN or Inf")

        if isinstance(input, PackedSequence):
            if self.debug_checks and not torch.isfinite(input.data).all():
                raise RuntimeError("PackedSequence input contains NaN or Inf")

            return self._forward_bounded_feedback_packed(input, hx)

        if self.debug_checks and not torch.isfinite(input).all():
            raise RuntimeError("RNN tensor input contains NaN or Inf")

        return self._forward_bounded_feedback_tensor(input, hx)

    def _split_input(self, x_all):
        """
        With n_actions > 0 the caller appends one-hot(a_{t-1}) to the DG input:
            input[..., :input_size]  = DG input x_t
            input[..., input_size:]  = one-hot of the action taken at t-1 (the action that
                                       led to x_t). All-zero (e.g. episode start) = no action,
                                       which uses the shared feedback lr_row.
        """
        if self.n_actions == 0:
            return x_all, None
        return x_all[..., : self.input_size], x_all[..., self.input_size :]

    def _action_directions(self):
        """
        (n_actions + 1, rank, hidden): index 0 = shared lr_row (no action),
        index 1 + a = lr_row + lr_row_action[a]. Each matrix normalised to unit
        Frobenius norm, the same direction-only normalisation as the shared case.
        Computed once per forward, not per timestep.
        """
        mats = torch.cat([self.lr_row.unsqueeze(0), self.lr_row.unsqueeze(0) + self.lr_row_action], dim=0)
        norms = mats.flatten(1).norm(p=2, dim=1).clamp_min(1e-8)
        return mats / norms[:, None, None]

    def _bounded_feedback_step(
        self,
        x_t,
        h_prev,
        W_in,
        W_hh_base,
        a_prev=None,
        action_directions=None,
    ):
        h_context = F.layer_norm(
            h_prev,
            normalized_shape=(self.hidden_size,),
        )

        if a_prev is None:
            direction = F.normalize(
                self.lr_row,
                p=2.0,
                dim=None,
                eps=1e-8,
            )

            raw_feedback = h_context @ direction.T
        else:
            # Row index into action_directions: 0 if no action given, else 1 + argmax.
            has_action = a_prev.sum(dim=-1) > 0.5
            idx = torch.where(has_action, a_prev.argmax(dim=-1) + 1, torch.zeros_like(has_action, dtype=torch.long))
            # Feedback under every action, then pick each row's own: (B, n_actions + 1, rank).
            raw_all = torch.einsum("bh,kdh->bkd", h_context, action_directions)
            raw_feedback = raw_all.gather(1, idx[:, None, None].expand(-1, 1, raw_all.shape[-1])).squeeze(1)

        feedback = torch.tanh(raw_feedback)

        effective_dg_input = x_t + feedback

        pre_activation = effective_dg_input @ W_in.T + h_prev @ W_hh_base.T

        if self.bias:
            pre_activation = pre_activation + self.bias_ih_l0 + self.bias_hh_l0

        if self.nonlinearity == "relu":
            h_new = F.relu(pre_activation)
        else:
            h_new = torch.tanh(pre_activation)

        return h_new, feedback, raw_feedback

    def _forward_bounded_feedback_tensor(
        self,
        input: Tensor,
        hx: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """
        Explicit bounded-feedback recurrence for non-packed RNN input.
        """
        if self.num_layers != 1:
            raise NotImplementedError("Bounded-feedback recurrence currently supports num_layers=1 only")

        if self.bidirectional:
            raise NotImplementedError("Bounded-feedback recurrence does not support bidirectional=True")

        if self.dropout != 0.0:
            raise NotImplementedError("Bounded-feedback recurrence does not support nonzero dropout")

        if input.dim() not in (2, 3):
            raise ValueError(f"Expected a 2D or 3D tensor, got {input.dim()}D tensor")

        original_input_was_unbatched = input.dim() == 2

        # Match standard torch.nn.RNN behavior for unbatched sequence input.
        if original_input_was_unbatched:
            batch_dim = 0 if self.batch_first else 1
            input = input.unsqueeze(batch_dim)

            if hx is not None:
                if hx.dim() != 2:
                    raise RuntimeError("For unbatched 2D input, hx must be 2D; " f"got hx.dim()={hx.dim()}")
                hx = hx.unsqueeze(1)

        # Work internally as [B, T, input_size].
        if self.batch_first:
            input_batched = input
        else:
            input_batched = input.transpose(0, 1)

        B, T, input_size = input_batched.shape

        if input_size != self.input_size + self.n_actions:
            raise RuntimeError(
                f"Expected input_size={self.input_size} + n_actions={self.n_actions}, got {input_size}"
            )

        if hx is None:
            h = torch.zeros(
                B,
                self.hidden_size,
                dtype=input_batched.dtype,
                device=input_batched.device,
            )
        else:
            if hx.dim() != 3:
                raise RuntimeError(f"Expected hx to be 3D after batching, got hx.dim()={hx.dim()}")

            if hx.shape != (1, B, self.hidden_size):
                raise RuntimeError(
                    "Invalid hx shape for bounded-feedback RNN: "
                    f"expected {(1, B, self.hidden_size)}, got {tuple(hx.shape)}"
                )

            h = hx[0]

        W_in = self.weight_ih_l0
        W_hh_base = self.weight_hh_l0

        if self.debug_checks:
            if not torch.isfinite(h).all():
                raise RuntimeError("Initial hidden state contains NaN or Inf")

            if not torch.isfinite(W_in).all():
                raise RuntimeError("W_in contains NaN or Inf")

            if not torch.isfinite(W_hh_base).all():
                raise RuntimeError("W_hh_base contains NaN or Inf")

        action_directions = self._action_directions() if self.n_actions > 0 else None

        outputs = []

        # One scalar Tensor per timestep.
        state_abs_max = []
        feedback_abs_max = []
        empirical_gain = []

        for t in range(T):
            x_t, a_prev = self._split_input(input_batched[:, t, :])
            h_prev = h

            h_new, feedback, raw_feedback = self._bounded_feedback_step(
                x_t=x_t,
                h_prev=h,
                W_in=W_in,
                W_hh_base=W_hh_base,
                a_prev=a_prev,
                action_directions=action_directions,
            )

            # tanh feedback must stay in [-1, 1].
            feedback_max = feedback.detach().abs().max()

            # A tanh CA3 activation must also remain in [-1, 1].
            h_new_max = h_new.detach().abs().max()

            if self.debug_checks:
                if not torch.isfinite(raw_feedback).all():
                    raise RuntimeError(f"Raw feedback contains NaN/Inf at timestep={t}")

                if not torch.isfinite(feedback).all():
                    raise RuntimeError(f"Bounded feedback contains NaN/Inf at timestep={t}")

                if not torch.isfinite(h_new).all():
                    raise RuntimeError(
                        f"CA3 hidden state contains NaN/Inf at timestep={t}; "
                        f"previous_h_abs_max={h.detach().abs().max().item():.6g}, "
                        f"feedback_abs_max={feedback.detach().abs().max().item():.6g}"
                    )

                if feedback_max.item() > 1.00001:
                    raise RuntimeError(
                        f"Bounded feedback exceeds tanh bound at timestep={t}: " f"max_abs={feedback_max.item():.6g}"
                    )

                if self.nonlinearity == "tanh" and h_new_max.item() > 1.00001:
                    raise RuntimeError(
                        f"Tanh CA3 state exceeds tanh bound at timestep={t}: " f"max_abs={h_new_max.item():.6g}"
                    )

            h = h_new
            outputs.append(h)

            state_abs_max.append(h_new_max)
            feedback_abs_max.append(feedback_max)

            empirical_gain.append(self._state_norm_ratio(h_prev, h_new))

        self._set_last_forward_summaries(
            state_abs_max=state_abs_max,
            feedback_abs_max=feedback_abs_max,
            empirical_gain=empirical_gain,
        )

        output_batched = torch.stack(outputs, dim=1)
        hidden = h.unsqueeze(0)

        if self.debug_checks:
            if not torch.isfinite(output_batched).all():
                raise RuntimeError("Bounded-feedback RNN output contains NaN or Inf")

            if not torch.isfinite(hidden).all():
                raise RuntimeError("Bounded-feedback final hidden state contains NaN or Inf")

        if self._feedback_forward_count % 50 == 0:
            with torch.no_grad():
                state_per_t = torch.stack(state_abs_max)
                feedback_per_t = torch.stack(feedback_abs_max)

                log.debug(
                    "Bounded-feedback tensor summary: "
                    f"batch={B}, timesteps={T}, "
                    f"ca3_abs_max={state_per_t.max().item():.6g}, "
                    f"feedback_abs_max={feedback_per_t.max().item():.6g}, "
                    f"final_hidden_abs_max={hidden.detach().abs().max().item():.6g}"
                )

                log.debug(
                    "Bounded-feedback tensor timestep maxima: "
                    f"ca3={state_per_t.cpu().tolist()}, "
                    f"feedback={feedback_per_t.cpu().tolist()}"
                )

        # Return to the input layout expected by torch.nn.RNN.
        if self.batch_first:
            output = output_batched
        else:
            output = output_batched.transpose(0, 1)

        if original_input_was_unbatched:
            batch_dim = 0 if self.batch_first else 1
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        return output, hidden

    def _forward_bounded_feedback_packed(
        self,
        packed_input: PackedSequence,
        hx: Optional[Tensor],
    ) -> tuple[PackedSequence, Tensor]:
        """
        Explicit bounded-feedback recurrence for PackedSequence input.

        Supports:
            - num_layers == 1
            - bidirectional == False
            - dropout == 0
            - tanh or relu
        """
        if self.num_layers != 1:
            raise NotImplementedError("Bounded-feedback recurrence currently supports num_layers=1 only")

        if self.bidirectional:
            raise NotImplementedError("Bounded-feedback recurrence does not support bidirectional=True")

        if self.dropout != 0.0:
            raise NotImplementedError("Bounded-feedback recurrence does not support nonzero dropout")

        data = packed_input.data
        batch_sizes = packed_input.batch_sizes
        sorted_indices = packed_input.sorted_indices
        unsorted_indices = packed_input.unsorted_indices

        if data.dim() != 2:
            raise RuntimeError(f"PackedSequence data must be 2D, got shape={tuple(data.shape)}")

        if data.shape[1] != self.input_size + self.n_actions:
            raise RuntimeError(
                f"Expected packed input_size={self.input_size} + n_actions={self.n_actions}, " f"got {data.shape[1]}"
            )

        if self.debug_checks and not torch.isfinite(data).all():
            raise RuntimeError("PackedSequence input data contains NaN or Inf")

        max_batch_size = int(batch_sizes[0].item())

        if hx is None:
            h = torch.zeros(
                max_batch_size,
                self.hidden_size,
                dtype=data.dtype,
                device=data.device,
            )
        else:
            if hx.dim() != 3:
                raise RuntimeError(f"Packed hx must be 3D, got hx.dim()={hx.dim()}")

            # Reorder initial hidden states to the PackedSequence sorted order.
            hx = self.permute_hidden(hx, sorted_indices)

            if hx.shape != (1, max_batch_size, self.hidden_size):
                raise RuntimeError(
                    "Invalid packed hx shape: "
                    f"expected {(1, max_batch_size, self.hidden_size)}, "
                    f"got {tuple(hx.shape)}"
                )

            h = hx[0]

        W_in = self.weight_ih_l0
        W_hh_base = self.weight_hh_l0

        if self.debug_checks:
            if not torch.isfinite(h).all():
                raise RuntimeError("Initial packed hidden state contains NaN or Inf")

            if not torch.isfinite(W_in).all():
                raise RuntimeError("W_in contains NaN or Inf")

            if not torch.isfinite(W_hh_base).all():
                raise RuntimeError("W_hh_base contains NaN or Inf")

        action_directions = self._action_directions() if self.n_actions > 0 else None

        outputs = []

        # One scalar Tensor per packed timestep.
        state_abs_max = []
        feedback_abs_max = []
        empirical_gain = []

        offset = 0

        for t, batch_size_tensor in enumerate(batch_sizes):
            active_batch_size = int(batch_size_tensor.item())

            # The first active_batch_size rows are the active sorted sequences.
            x_t, a_prev = self._split_input(data[offset : offset + active_batch_size])
            h_active = h[:active_batch_size]
            h_prev = h_active

            h_new, feedback, raw_feedback = self._bounded_feedback_step(
                x_t=x_t,
                h_prev=h_active,
                W_in=W_in,
                W_hh_base=W_hh_base,
                a_prev=a_prev,
                action_directions=action_directions,
            )

            feedback_max = feedback.detach().abs().max()

            h_new_max = h_new.detach().abs().max()

            if self.debug_checks:
                if not torch.isfinite(raw_feedback).all():
                    raise RuntimeError(f"Packed raw feedback contains NaN/Inf at timestep={t}")

                if not torch.isfinite(feedback).all():
                    raise RuntimeError(f"Packed bounded feedback contains NaN/Inf at timestep={t}")

                if not torch.isfinite(h_new).all():
                    raise RuntimeError(
                        f"Packed CA3 state contains NaN/Inf at timestep={t}; "
                        f"previous_h_abs_max={h_active.detach().abs().max().item():.6g}, "
                        f"feedback_abs_max={feedback.detach().abs().max().item():.6g}"
                    )

                if feedback_max.item() > 1.00001:
                    raise RuntimeError(
                        f"Packed bounded feedback exceeds tanh bound at timestep={t}: "
                        f"max_abs={feedback_max.item():.6g}"
                    )

                if self.nonlinearity == "tanh" and h_new_max.item() > 1.00001:
                    raise RuntimeError(
                        f"Packed tanh CA3 state exceeds tanh bound at timestep={t}: "
                        f"max_abs={h_new_max.item():.6g}"
                    )

            # Only active sequence states are updated. Inactive trailing rows retain
            # their final valid state for the final hidden-state output.
            h = h.clone()
            h[:active_batch_size] = h_new

            outputs.append(h_new)

            state_abs_max.append(h_new_max)
            feedback_abs_max.append(feedback_max)

            empirical_gain.append(self._state_norm_ratio(h_prev, h_new))

            offset += active_batch_size

        self._set_last_forward_summaries(
            state_abs_max=state_abs_max,
            feedback_abs_max=feedback_abs_max,
            empirical_gain=empirical_gain,
        )

        if offset != data.shape[0]:
            raise RuntimeError(
                f"Packed data traversal mismatch: consumed {offset}, " f"but data has {data.shape[0]} rows"
            )

        output_data = torch.cat(outputs, dim=0)

        if self.debug_checks and not torch.isfinite(output_data).all():
            raise RuntimeError("Packed bounded-feedback output contains NaN or Inf")

        hidden_sorted = h.unsqueeze(0)

        if self.debug_checks and not torch.isfinite(hidden_sorted).all():
            raise RuntimeError("Packed bounded-feedback final hidden state contains NaN or Inf")

        # Restore original batch order for the caller.
        hidden = self.permute_hidden(hidden_sorted, unsorted_indices)

        output_packed = PackedSequence(
            output_data,
            batch_sizes,
            sorted_indices,
            unsorted_indices,
        )

        if self._feedback_forward_count % 50 == 0:
            with torch.no_grad():
                state_per_t = torch.stack(state_abs_max)
                feedback_per_t = torch.stack(feedback_abs_max)

                log.debug(
                    "Bounded-feedback packed summary: "
                    f"timesteps={len(batch_sizes)}, "
                    f"max_batch={max_batch_size}, "
                    f"ca3_abs_max={state_per_t.max().item():.6g}, "
                    f"feedback_abs_max={feedback_per_t.max().item():.6g}, "
                    f"final_hidden_abs_max={hidden.detach().abs().max().item():.6g}"
                )

                log.debug(
                    "Bounded-feedback packed timestep maxima: "
                    f"ca3={state_per_t.cpu().tolist()}, "
                    f"feedback={feedback_per_t.cpu().tolist()}"
                )

        return output_packed, hidden