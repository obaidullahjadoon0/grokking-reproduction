"""
Generates the modular arithmetic dataset used in the grokking paper.

Each example is a sequence of 4 tokens: [a, op, b, =] and the model must
predict the 5th token: c = (a OP b) mod p.

Vocabulary layout:
  tokens 0 .. p-1      -> the numbers 0..p-1
  token  p              -> '+' (or the active operator)
  token  p+1             -> '=' 
  (we reuse one "operator" slot regardless of which op is configured,
   since each run only uses one operation)
"""

import itertools
import random
import torch


def op_fn(a: int, b: int, p: int, operation: str) -> int:
    if operation == "add":
        return (a + b) % p
    elif operation == "sub":
        return (a - b) % p
    elif operation == "mul":
        return (a * b) % p
    else:
        raise ValueError(f"Unknown operation: {operation}")


def build_dataset(p: int, operation: str, train_fraction: float, seed: int = 0):
    """
    Returns train_inputs, train_targets, val_inputs, val_targets as LongTensors.
    inputs shape: (N, 4) = [a, op_token, b, eq_token]
    targets shape: (N,)   = c
    """
    op_token = p
    eq_token = p + 1
    vocab_size = p + 2

    rng = random.Random(seed)
    all_pairs = list(itertools.product(range(p), range(p)))
    rng.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_fraction)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    def make_tensors(pairs):
        inputs = torch.zeros(len(pairs), 4, dtype=torch.long)
        targets = torch.zeros(len(pairs), dtype=torch.long)
        for i, (a, b) in enumerate(pairs):
            c = op_fn(a, b, p, operation)
            inputs[i] = torch.tensor([a, op_token, b, eq_token])
            targets[i] = c
        return inputs, targets

    train_inputs, train_targets = make_tensors(train_pairs)
    val_inputs, val_targets = make_tensors(val_pairs)

    return train_inputs, train_targets, val_inputs, val_targets, vocab_size


if __name__ == "__main__":
    # quick sanity check
    tr_x, tr_y, va_x, va_y, vocab = build_dataset(p=97, operation="add", train_fraction=0.3)
    print("vocab size:", vocab)
    print("train examples:", tr_x.shape[0], "val examples:", va_x.shape[0])
    print("sample:", tr_x[0].tolist(), "-> target", tr_y[0].item())
