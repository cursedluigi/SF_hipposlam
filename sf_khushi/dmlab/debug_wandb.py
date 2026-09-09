import wandb
import pandas as pd

api = wandb.Api(timeout=60)

ENTITY = "xiaoxionglin-bernstein-center-freiburg"
PROJECT = "sample_factory"

CORRECT_RUN_NAMES = [
    "prevaction_w_policies_fixed_s4444_20260819_155532_303960",
    "prevaction_w_policies_fixed_s3333_20260819_155532_313821",
    "prevaction_w_policies_fixed_s2222_20260819_155532_318036",
    "prevaction_w_policies_fixed_s1111_20260819_155532_317971",
    "prevaction_w_policies_fixed_s5555_20260819_155532_350658"
]

want_cols = [
    "_step",
    "0/train/env_steps", "1/train/env_steps", "2/train/env_steps", "3/train/env_steps",
    "0/reward/reward", "1/reward/reward", "2/reward/reward", "3/reward/reward",
    "0/policy_stats/avg_true_objective", "1/policy_stats/avg_true_objective",
    "2/policy_stats/avg_true_objective", "3/policy_stats/avg_true_objective",
    "0/zz_pbt/cfg_learning_rate", "1/zz_pbt/cfg_learning_rate",
    "2/zz_pbt/cfg_learning_rate", "3/zz_pbt/cfg_learning_rate",
]

for name in CORRECT_RUN_NAMES:
    r = api.run(f"{ENTITY}/{PROJECT}/{name}")
    print(f"Pulling {r.name} (state: {r.state}) ...")

    # No keys filter this time — pull everything, then subset
    hist = r.history(samples=2000)

    present_cols = [c for c in want_cols if c in hist.columns]
    missing_cols = [c for c in want_cols if c not in hist.columns]
    if missing_cols:
        print(f"  WARNING missing columns: {missing_cols}")

    hist = hist[present_cols].copy()
    hist["run_name"] = r.name
    hist["state"] = r.state
    hist["seed"] = r.config.get("seed", "unknown")

    outpath = f"{r.name}_history.csv"
    hist.to_csv(outpath, index=False)
    print(f"  Saved {outpath} ({len(hist)} rows)")