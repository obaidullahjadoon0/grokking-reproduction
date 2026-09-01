"""
Trains the GrokTransformer on the modular arithmetic task and logs
every metric to SQLite. This is the "reproduce the paper" script.

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --run_name my_experiment --weight_decay 0.0
"""

import argparse
import time
import uuid
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from data import build_dataset
from model import GrokTransformer
import db as dbmod


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_optimizer(model, cfg):
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]))
    elif cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    elif cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {cfg['optimizer']}")


def evaluate(model, inputs, targets, device, batch_size=2048):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            x = inputs[i:i + batch_size].to(device)
            y = targets[i:i + batch_size].to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += len(y)
    model.train()
    return total_loss / total_n, total_correct / total_n


def train(cfg: dict, run_name: str = None):
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")

    train_x, train_y, val_x, val_y, vocab_size = build_dataset(
        p=cfg["p"], operation=cfg["operation"], train_fraction=cfg["train_fraction"], seed=cfg["seed"]
    )

    model = GrokTransformer(
        vocab_size=vocab_size, seq_len=4, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
    ).to(device)

    optimizer = make_optimizer(model, cfg)

    run_id = str(uuid.uuid4())[:8]
    run_name = run_name or cfg.get("run_name", "run")
    dbmod.create_run(cfg["db_path"], run_id, run_name, cfg)
    print(f"Run ID: {run_id}  ({run_name})")

    n_train = len(train_x)
    batch_size = min(cfg["batch_size"], n_train)

    pbar = tqdm(range(1, cfg["epochs"] + 1))
    for epoch in pbar:
        # shuffle each epoch
        perm = torch.randperm(n_train)
        train_x_shuf = train_x[perm]
        train_y_shuf = train_y[perm]

        epoch_loss, epoch_correct = 0.0, 0
        for i in range(0, n_train, batch_size):
            x = train_x_shuf[i:i + batch_size].to(device)
            y = train_y_shuf[i:i + batch_size].to(device)

            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            epoch_correct += (logits.argmax(dim=-1) == y).sum().item()

        if epoch % cfg["log_every"] == 0 or epoch == 1 or epoch == cfg["epochs"]:
            train_loss = epoch_loss / n_train
            train_acc = epoch_correct / n_train
            val_loss, val_acc = evaluate(model, val_x, val_y, device)
            dbmod.log_metrics(cfg["db_path"], run_id, epoch, train_loss, train_acc, val_loss, val_acc)
            pbar.set_description(
                f"epoch {epoch} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}"
            )

    # save final model weights for the interpretability analysis step
    torch.save(model.state_dict(), f"results/model_{run_id}.pt")
    print(f"\nDone. Run ID = {run_id}. Model saved to results/model_{run_id}.pt")
    print(f"Next: python src/analyze.py --run_id {run_id} --plot curves")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    # allow quick CLI overrides for common ablation knobs
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--operation", type=str, default=None)
    parser.add_argument("--train_fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.optimizer is not None:
        cfg["optimizer"] = args.optimizer
    if args.operation is not None:
        cfg["operation"] = args.operation
    if args.train_fraction is not None:
        cfg["train_fraction"] = args.train_fraction
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    import os
    os.makedirs("results", exist_ok=True)
    train(cfg, run_name=args.run_name)
