"""
Plots training curves, visualizes learned embeddings (the interpretability
payoff), and compares ablation runs. Reads everything from the SQLite DB
that train.py/ablation.py wrote to.

Usage:
    python src/analyze.py --run_id latest --plot curves
    python src/analyze.py --run_id latest --plot embeddings
    python src/analyze.py --plot ablation_comparison
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import db as dbmod
from model import GrokTransformer
from data import build_dataset

DB_PATH = "db/runs.sqlite"


def plot_curves(run_id: str):
    if run_id == "latest":
        run_id = dbmod.get_latest_run_id(DB_PATH)
    epochs, train_loss, train_acc, val_loss, val_acc = dbmod.get_metrics(DB_PATH, run_id)
    cfg = dbmod.get_run_config(DB_PATH, run_id)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_acc, label="train acc")
    axes[0].plot(epochs, val_acc, label="val acc")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title(f"Accuracy — run {run_id} (wd={cfg.get('weight_decay')}, op={cfg.get('optimizer')})")
    axes[0].legend()
    axes[0].set_xscale("log")

    axes[1].plot(epochs, train_loss, label="train loss")
    axes[1].plot(epochs, val_loss, label="val loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    plt.tight_layout()
    out_path = f"results/curves_{run_id}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # tell the user whether grokking actually happened in this run
    if max(val_acc) > 0.9 and train_acc[-1] > 0.99:
        grok_epoch = next((e for e, a in zip(epochs, val_acc) if a > 0.9), None)
        print(f"Grokking detected: val accuracy crossed 90% at epoch {grok_epoch}.")
    else:
        print("Grokking not clearly detected in this run yet — it may need more epochs, "
              "or these hyperparameters (esp. weight_decay) may not support it. Try increasing "
              "epochs in configs/config.yaml, or check the ablation sweep for comparison.")


def plot_embeddings(run_id: str):
    if run_id == "latest":
        run_id = dbmod.get_latest_run_id(DB_PATH)
    cfg = dbmod.get_run_config(DB_PATH, run_id)

    train_x, train_y, val_x, val_y, vocab_size = build_dataset(
        p=cfg["p"], operation=cfg["operation"], train_fraction=cfg["train_fraction"], seed=cfg["seed"]
    )
    model = GrokTransformer(
        vocab_size=vocab_size, seq_len=4, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
    )
    state_path = f"results/model_{run_id}.pt"
    if not os.path.exists(state_path):
        print(f"No saved weights found at {state_path}. Did training finish for this run?")
        return
    model.load_state_dict(torch.load(state_path, map_location="cpu"))

    embeddings = model.get_number_embeddings(cfg["p"])  # (p, d_model)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c=range(cfg["p"]), cmap="hsv", s=40)
    for i in range(cfg["p"]):
        if i % 5 == 0:  # label every 5th number to keep it readable
            ax.annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=8)
    ax.set_title(f"Number embeddings (PCA) — run {run_id}\n"
                 f"A circular/ring layout here is the signature of a learned modular circuit")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    out_path = f"results/embeddings_{run_id}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print("If you see a roughly circular arrangement of numbers 0..p-1, the model has learned "
          "a modular/rotational representation -- this is the mechanistic signature described "
          "in the interpretability follow-up work on grokking.")


def plot_ablation_comparison():
    runs = dbmod.list_all_runs(DB_PATH)
    if not runs:
        print("No runs found. Run src/ablation.py first.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, run_name, _ in runs:
        epochs, _, _, _, val_acc = dbmod.get_metrics(DB_PATH, run_id)
        if epochs:
            ax.plot(epochs, val_acc, label=run_name)

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Ablation comparison: when/whether grokking happens")
    ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)

    out_path = "results/ablation_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="latest")
    parser.add_argument("--plot", type=str, required=True,
                         choices=["curves", "embeddings", "ablation_comparison"])
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.plot == "curves":
        plot_curves(args.run_id)
    elif args.plot == "embeddings":
        plot_embeddings(args.run_id)
    elif args.plot == "ablation_comparison":
        plot_ablation_comparison()
