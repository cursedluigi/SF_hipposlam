import argparse
import json
import time
 
import numpy as np
 
N_FEATURES = 16                    # number of DG features 
SEQ_LEN = 64 + 8 - 1               # 71 slots per sequence
N_CORE = N_FEATURES * SEQ_LEN      # 1136. bypass 13dim dropped
 
 
def load_transitions(path, heldout_frac=0.2, seed=0):
#load exp transition npz data
    data = np.load(path)
 
    # A row is a usable transition (t -> t+1) only if it is not the last step of an episode
    usable_rows = np.nonzero(~data["last"])[0]
 
    # Pick ~20% of the episodes at random to hold out.
    all_episodes = np.unique(data["episode_id"])
    n_heldout = max(1, int(round(heldout_frac * len(all_episodes))))
    rng = np.random.RandomState(seed)
    heldout_episodes = set(rng.choice(all_episodes, size=n_heldout, replace=False).tolist())
 
    row_is_heldout = np.isin(data["episode_id"][usable_rows], list(heldout_episodes))
 
    has_prerelu = "dg_prerelu" in data.files     # dumps from the new build_transition_dataset.py
 
    def build_split(rows):
        split = {
            "x":          data["x"][rows],          # CA3 state at t
            "action":     data["action"][rows],     # action taken at t (the EXECUTED one)
            "x_next":     data["x"][rows + 1],      # CA3 state at t+1
            "dg_next":    data["dg"][rows + 1],     # DG output at t+1  --> our prediction
            "reward":     data["reward"][rows],
            "episode_id": data["episode_id"][rows],
        }
        if has_prerelu:
            # DG before the ReLU at t+1: BN(z) - intercept. ReLU of it == dg_next.
            split["dg_prerelu_next"] = data["dg_prerelu"][rows + 1]
            # was the action at t an epsilon-random one?
            split["is_random"] = data["is_random"][rows]
        else:
            split["is_random"] = np.zeros(len(rows), dtype=bool)
        return split
 
    train = build_split(usable_rows[~row_is_heldout])
    heldout = build_split(usable_rows[row_is_heldout])
    return train, heldout
 
 
def prepare(split, binary=False, prerelu=False):
    """Turn a split into the arrays the model uses: x, action, y, episode ids.
 
    prerelu=True: y is the DG value BEFORE the ReLU (BN(z) - intercept), so a plain
    linear prediction fits it and the delta rule always has a gradient. A feature
    fires exactly when y > 0, so AUROC / "live" below work unchanged.
    """
    x = split["x"][:, :N_CORE].astype(np.float64)    # keep only the CA3 core
    if prerelu:
        assert "dg_prerelu_next" in split, "old dump: re-collect with the new build_transition_dataset.py"
        assert not binary, "--binary_targets and --prerelu don't go together"
        y = split["dg_prerelu_next"].astype(np.float64)
    else:
        y = split["dg_next"].astype(np.float64)
    if binary:
        # Predict "did this feature fire?" (0/1) instead of how strongly.
        y = (y > 0).astype(np.float64)
    actions = split["action"].astype(np.int64)
    return x, actions, y, split["episode_id"]
 
 
class ACCA3Predictor:
    # pre   = L . R(a) . x + b
    # y_hat = pre            (linear;   use with --prerelu targets)
    # y_hat = relu(pre)      (output_relu=True; the earlier ReLU + learnt-threshold variant)
    # learned online w/ delta rule
 
    def __init__(self, n_in, n_out, rank, n_actions, action_conditioned=True,
                 #learning rates, 5%
                 eta_L=0.05, eta_R=0.05, normalize=True, eps=1e-6,
                 init_scale=1.0, seed=0, output_relu=False, use_bias=True):
        self.action_conditioned = action_conditioned
        self.eta_L = eta_L
        self.eta_R = eta_R
        self.normalize = normalize
        self.eps = eps
        self.output_relu = output_relu
        self.use_bias = use_bias
 
        # One random starting matrix R0, scaled so z has a sensible size.
        rng = np.random.RandomState(seed)
        R0 = rng.randn(rank, n_in) * init_scale / np.sqrt(n_in)
 
        # "action" variant: one copy of R0 per action.
        # "shared" variant: just one R that every action uses.
        if action_conditioned:
            n_R = n_actions
        else:
            n_R = 1
        self.R = np.stack([R0.copy() for _ in range(n_R)])   # (n_R, rank, n_in)
 
        # L starts at zero -> the first prediction is just b for every input
        #therefore, R learns when L is nonzero
        self.L = np.zeros((n_out, rank))
 
        # per-feature offset. Set it with init_bias(); stays 0 if use_bias=False.
        # It is learned together with L, as one extra input fixed at 1: L_aug = [L | b],
        # z_aug = [z, 1], and the normalized step uses ‖z_aug‖² = z·z + 1. Updating b with its
        # own separate rate lets L and b drift against each other (z has a large mean
        # component, so L·z and b nearly cancel) -- that diverged in testing.
        self.b = np.zeros(n_out)
 
    def init_bias(self, y_train):
        # Start b at the training mean of the target: with L = 0 the model is then
        # exactly the rate baseline, and L, R only have to learn the deviations.
        if self.use_bias:
            self.b = y_train.mean(axis=0).astype(np.float64).copy()
 
    def which_R(self, action):
        #shared model always uses R number 0
        if self.action_conditioned:
            return int(action)
        return 0
 
    def _out(self, pre):
        return np.maximum(pre, 0.0) if self.output_relu else pre
 
    def predict(self, x, actions):
        #Predict for a whole batch. x: (B, n_in), actions: (B,) -> (B, n_out)
        # B-> batch size(transitions), n_in-> CA3 core dim, n_out-> DG dim
        y_hat = np.zeros((len(x), self.L.shape[0]))
        actions = np.asarray(actions)
 
        #shared model: all actions use the same R; no action dependence
        if not self.action_conditioned:
            z = x @ self.R[0].T                      # (B, rank)
            return self._out(z @ self.L.T + self.b)  # (B, n_out)
 
        # Action model: handle each action separately with its own R.
        for k in range(len(self.R)):    # k = action index
            rows = actions == k         #which rows of the batch have this action
            if rows.any():
                z = x[rows] @ self.R[k].T            # (b, rank)
                y_hat[rows] = z @ self.L.T + self.b  # (b, n_out)
        return self._out(y_hat)
 
    def update(self, x, action, y):
        #One delta-rule step on ONE transition. Returns the error before the step.
        k = self.which_R(action)
 
        # --- forward pass ---
        z = self.R[k] @ x                    # bottleneck code, (rank,)
        pre = self.L @ z + self.b
        y_hat = self._out(pre)
        error = y - y_hat                    # NOT gated by the ReLU derivative
 
        # --- how much blame does each bottleneck unit get? ---
        # Must use L as it is NOW, before we update it a few lines below.
        error_for_R = self.L.T @ error   # (rank,)
 
        # --- step sizes ---
        bias_in = 1.0 if self.use_bias else 0.0          # the constant input that feeds b
        if self.normalize:
            step_L = self.eta_L / (z @ z + bias_in + self.eps)
            step_R = self.eta_R / ((x @ x) * (np.sum(self.L ** 2) + self.eps) + self.eps)
        else:
            step_L = self.eta_L
            step_R = self.eta_R
 
        # --- the actual learning ---
        self.L += step_L * np.outer(error, z)
        self.R[k] += step_R * np.outer(error_for_R, x)   # only the R of the action taken
        if self.use_bias:
            self.b += step_L * error                      # same step as L: b is L's column for input 1
        return error
 
    def update_bias_only(self, y):
        # Used when CA3 is silent (x = 0): then pre = b, and only b can learn.
        if self.use_bias:
            error = y - self._out(self.b)
            step = self.eta_L / (1.0 + self.eps) if self.normalize else self.eta_L
            self.b += step * error
 
 
def train(model, x, actions, y, epochs=1, eval_fn=None, eval_every=0,
          shuffle=False, seed=0):
    """Feed the transitions to the model one by one.
 
    By default rows are used in their stored order (episode by episode), like
    an agent experiencing them. Returns a learning curve: list of
    (step, held-out score) pairs, taken every `eval_every` steps.
    """
    curve = []
    rng = np.random.RandomState(seed)
    n_rows = len(x)
    step = 0
 
    for _ in range(epochs):
        if shuffle:
            order = rng.permutation(n_rows)
        else:
            order = np.arange(n_rows)
 
        for i in order:
            step += 1
 
            # If CA3 is completely silent, z = 0 and L, R can't change
            # anything useful, so only the bias learns (the step still counts).
            if not x[i].any():
                model.update_bias_only(y[i])
                continue
 
            model.update(x[i], actions[i], y[i])
 
            if eval_fn is not None and eval_every and step % eval_every == 0:
                curve.append((step, eval_fn(model)))
    return curve
 
#store predictions 
def auroc(scores, labels):
    """Area under the ROC curve, computed from ranks (Mann-Whitney U).
 
    Meaning: pick one step where the feature fired and one where it didn't.
    AUROC = probability that the model gave the "fired" step the higher score.
    0.5 = no better than chance, 1.0 = perfect ranking.
    Returns nan if all labels are the same (then it's undefined).
    """
    is_pos = labels > 0
    is_neg = ~is_pos
    n_pos, n_neg = is_pos.sum(), is_neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
 
    # Give every score its rank (1 = smallest). Tied scores share the average rank.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1   # average rank of the tie group
        i = j + 1
 
    # Sum of positive ranks, minus the smallest it could possibly be, divided by
    # the number of (positive, negative) pairs.
    U = ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)
 
 
def score(y_hat, y, live, mse_rate=None):
    """MSE and AUROC, averaged over the 'live' features only.
 
    live = features that fire at least once in training. A feature that never
    fires can't be learned, and including it would just dilute the numbers.
    """
    mse_per_feature = ((y - y_hat) ** 2).mean(axis=0)
 
    result = {
        "mse_live": float(mse_per_feature[live].mean()),
        "mse_per_feature": mse_per_feature.tolist(),
    }
 
    if live.any():
        aurocs = [auroc(y_hat[:, f], y[:, f]) for f in np.nonzero(live)[0]]
        result["auroc_live"] = float(np.nanmean(aurocs))   # nan = feature never fired in held-out
    else:
        result["auroc_live"] = float("nan")
 
    if mse_rate is not None:
        # Skill: 0 = as good as just predicting the firing rate, 1 = perfect,
        # negative = worse than the rate baseline.
        result["skill_vs_rate"] = float(1 - result["mse_live"] / mse_rate)
    return result
 
 
def paired_bootstrap(y_hat_a, y_hat_b, y, episode_ids, live, n_boot=2000, seed=0):
    """Is model a really better than model b, or is it noise?
 
    We compute mse(a) - mse(b). Negative = a is better.
    For the uncertainty we resample whole held-out EPISODES (with replacement)
    2000 times, because steps inside one episode are strongly correlated and
    are not independent samples. "Paired" = both models are scored on the
    same resampled episodes each time.
    Returns the observed difference and its 95% interval.
    """
    # squared error per row, averaged over live features
    sq_err_a = ((y - y_hat_a) ** 2)[:, live].mean(axis=1)
    sq_err_b = ((y - y_hat_b) ** 2)[:, live].mean(axis=1)
 
    # list of row indices for each held-out episode
    episodes = np.unique(episode_ids)
    rows_of_episode = [np.nonzero(episode_ids == e)[0] for e in episodes]
 
    rng = np.random.RandomState(seed)
    boot_diffs = []
    for _ in range(n_boot):
        picked_episodes = rng.randint(0, len(rows_of_episode), len(rows_of_episode))
        rows = np.concatenate([rows_of_episode[j] for j in picked_episodes])
        boot_diffs.append(sq_err_a[rows].mean() - sq_err_b[rows].mean())
    boot_diffs = np.array(boot_diffs)
 
    return {
        "diff": float(sq_err_a.mean() - sq_err_b.mean()),
        "ci95": [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))],
    }
 
 
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Train the action-conditioned CA3 predictor with the delta rule.")
    p.add_argument("data", help="Phase 1 .npz from build_transition_dataset.py")
    p.add_argument("--ranks", type=int, nargs="+", default=[4])
    p.add_argument("--eta_L", type=float, default=0.05)
    p.add_argument("--eta_R", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=1, help="1 = single online pass")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--shuffle", action="store_true", help="shuffle rows (default: episode order)")
    p.add_argument("--binary_targets", action="store_true", help="predict fired/not instead of DG magnitude")
    p.add_argument("--prerelu", action="store_true",
                   help="target = DG before the ReLU (BN(z) - intercept); linear model + bias")
    p.add_argument("--output_relu", action="store_true",
                   help="y_hat = relu(L R(a) x + b): the earlier ReLU + learnt-threshold variant")
    p.add_argument("--n_actions", type=int, default=5)
    p.add_argument("--heldout_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="init seed for R0")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--out", default="ac_ca3predictor_results.json")
    args = p.parse_args(argv)
 
    # ---- data ----
    train_split, heldout_split = load_transitions(args.data, args.heldout_frac, args.split_seed)
    x_train, a_train, y_train, _ = prepare(train_split, args.binary_targets, args.prerelu)
    x_test, a_test, y_test, ep_test = prepare(heldout_split, args.binary_targets, args.prerelu)
    dg_test = heldout_split["dg_next"].astype(np.float64)   # post-ReLU DG, for the post-ReLU MSE
 
    # Train on everything (incl. epsilon-random actions), evaluate on ON-POLICY held-out
    # transitions only: those are the (x, a) pairs the imagination will be asked about.
    on_policy = ~heldout_split["is_random"]
    x_test, a_test, y_test, ep_test, dg_test = (
        x_test[on_policy], a_test[on_policy], y_test[on_policy], ep_test[on_policy], dg_test[on_policy])
    use_bias = args.prerelu or args.output_relu
 
    live = (y_train > 0).any(axis=0)   # which DG features ever fire in training
 
    print(f"train {len(x_train)} transitions / {len(np.unique(train_split['episode_id']))} episodes, "
          f"held-out {len(x_test)} / {len(np.unique(ep_test))}")
    print(f"live features (fire in train): {np.nonzero(live)[0].tolist()}  ({live.sum()}/16)")
    print("action counts (train):", np.bincount(a_train, minlength=args.n_actions).tolist())
    print(f"target: {'pre-ReLU DG' if args.prerelu else 'DG'}, output: {'relu' if args.output_relu else 'linear'}, "
          f"bias: {use_bias}, held-out on-policy rows: {int(on_policy.sum())}/{len(on_policy)}")
 
    results = {
        "args": vars(args),
        "live_features": np.nonzero(live)[0].tolist(),
        "heldout_episodes": np.unique(ep_test).tolist(),
        "runs": {},
    }
 
    firing_rate = y_train.mean(axis=0)                       # (16,)
    y_hat_rate = np.broadcast_to(firing_rate, y_test.shape)  # same prediction every step
    mse_rate = ((y_test - y_hat_rate) ** 2).mean(axis=0)[live].mean()
 
    results["runs"]["zero"] = score(np.zeros_like(y_test), y_test, live, mse_rate)
    results["runs"]["rate"] = score(y_hat_rate, y_test, live, mse_rate)
 
    #the learned models: shared vs action, at each rank
    def heldout_mse(model):
        return score(model.predict(x_test, a_test), y_test, live)["mse_live"]
 
    predictions = {}
    for rank in args.ranks:
        for variant in ("shared", "action"):
            model = ACCA3Predictor(
                n_in=x_train.shape[1], n_out=N_FEATURES, rank=rank, n_actions=args.n_actions,
                action_conditioned=(variant == "action"),
                eta_L=args.eta_L, eta_R=args.eta_R,
                normalize=not args.no_normalize, seed=args.seed,
                output_relu=args.output_relu, use_bias=use_bias,
            )
            model.init_bias(y_train)
 
            t0 = time.time()
            curve = train(model, x_train, a_train, y_train, args.epochs,
                          eval_fn=heldout_mse, eval_every=args.eval_every,
                          shuffle=args.shuffle, seed=args.seed)
            y_hat = model.predict(x_test, a_test)
 
            if not np.isfinite(y_hat).all():
                print(f"  {variant} r={rank}: DIVERGED (non-finite predictions) -- lower eta")
 
            name = f"{variant}_r{rank}"
            predictions[name] = y_hat
            results["runs"][name] = {
                **score(y_hat, y_test, live, mse_rate),
                # MSE of the DG that would actually be injected: relu(prediction) vs real DG
                "mse_dg_post_relu_live": float(((np.maximum(y_hat, 0) - dg_test) ** 2)[:, live].mean()),
                "curve": curve,
                "seconds": round(time.time() - t0, 1),
            }
 
        # Gate 1 at this rank: action vs shared
        results["runs"][f"gate1_r{rank}"] = paired_bootstrap(
            predictions[f"action_r{rank}"], predictions[f"shared_r{rank}"],
            y_test, ep_test, live)
 
    # ---- print a summary table ----
    print(f"\n{'variant':<12}{'MSE (live)':>12}{'skill vs rate':>15}{'AUROC (live)':>14}")
    for name, r in results["runs"].items():
        if name.startswith("gate1"):
            continue
        print(f"{name:<12}{r['mse_live']:>12.4g}"
              f"{r.get('skill_vs_rate', float('nan')):>15.3f}{r['auroc_live']:>14.3f}")
 
    print("\nGate 1 (action - shared, MSE on live features)")
    for rank in args.ranks:
        g = results["runs"][f"gate1_r{rank}"]
        low, high = g["ci95"]
        if high < 0:
            verdict = "action better"        # whole interval below 0
        elif low > 0:
            verdict = "shared better"        # whole interval above 0
        else:
            verdict = "no clear difference"  # interval contains 0
        print(f"  r={rank:<3} diff {g['diff']:+.4g}   95% CI [{low:+.4g}, {high:+.4g}]   {verdict}")
 
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved to {args.out}")
 
 
if __name__ == "__main__":
    main()
 
 