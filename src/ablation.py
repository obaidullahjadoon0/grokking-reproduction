"""
Runs the extension ablation sweep — this is what turns "I reproduced a
paper" into "I reproduced and extended a paper."

Sweeps: weight decay, optimizer, operation type. Each combination gets its
own run_id in the SQLite DB, all queryable/plottable afterward with
src/analyze.py --plot ablation_comparison.

Usage:
    python src/ablation.py --config configs/config.yaml
    python src/ablation.py --config configs/config.yaml --quick   # fewer epochs, for a fast smoke test
"""

import argparse
import copy
import yaml

from train import train, load_config


# Edit this to change what gets swept. Keep it small at first -- each entry
# trains an entire model, so 3 weight decays x 3 optimizers x 2 ops = 18 runs.
SWEEP = {
    "weight_decay": [0.0, 0.1, 1.0],
    "optimizer": ["adamw", "adam", "sgd"],
    "operation": ["add", "mul"],
}


def run_sweep(cfg: dict, quick: bool = False):
    if quick:
        cfg = copy.deepcopy(cfg)
        cfg["epochs"] = 500  # fast smoke test, won't necessarily show full grokking

    results = []

    print("=== Sweep 1: weight decay (optimizer=adamw, operation=add fixed) ===")
    for wd in SWEEP["weight_decay"]:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["weight_decay"] = wd
        run_cfg["optimizer"] = "adamw"
        run_cfg["operation"] = "add"
        run_name = f"wd_{wd}"
        run_id = train(run_cfg, run_name=run_name)
        results.append((run_name, run_id))

    print("=== Sweep 2: optimizer (weight_decay=1.0, operation=add fixed) ===")
    for opt in SWEEP["optimizer"]:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["weight_decay"] = 1.0
        run_cfg["optimizer"] = opt
        run_cfg["operation"] = "add"
        run_name = f"opt_{opt}"
        run_id = train(run_cfg, run_name=run_name)
        results.append((run_name, run_id))

    print("=== Sweep 3: operation type (weight_decay=1.0, optimizer=adamw fixed) ===")
    for op in SWEEP["operation"]:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["weight_decay"] = 1.0
        run_cfg["optimizer"] = "adamw"
        run_cfg["operation"] = op
        run_name = f"op_{op}"
        run_id = train(run_cfg, run_name=run_name)
        results.append((run_name, run_id))

    print("\n=== Sweep complete ===")
    for run_name, run_id in results:
        print(f"  {run_name:15s} -> run_id={run_id}")
    print("\nNext: python src/analyze.py --plot ablation_comparison")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--quick", action="store_true", help="fast smoke test with fewer epochs")
    args = parser.parse_args()

    import os
    os.makedirs("results", exist_ok=True)
    cfg = load_config(args.config)
    run_sweep(cfg, quick=args.quick)
