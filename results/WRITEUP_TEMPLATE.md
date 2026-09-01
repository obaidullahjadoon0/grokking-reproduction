# Grokking: My Reproduction & Extension

## What I reproduced
I reimplemented the core setup from Power et al.'s "Grokking" paper: a small
decoder-only transformer trained on modular addition (`a + b mod 97`), built
entirely from scratch in PyTorch (no `nn.TransformerEncoder`).

**Result:** Grokking reproduced successfully. Training accuracy hit 100%
almost immediately, while validation accuracy stayed near 0% for hundreds of
epochs — then jumped sharply, crossing 90% at epoch 450 and eventually
reaching 100%. This is the exact delayed-generalization pattern the paper
describes: the model looks fully "done" from the training loss alone, long
before it can actually generalize.

![Training curves](results/curves_31dfb7ca.png)

## What I found inside the model
[Describe what you actually see in embeddings_31dfb7ca.png here — e.g.:
"Projecting the model's learned number embeddings (0–96) down to 2D with
PCA shows them arranged in a rough ring, in numerical order around the
circle. This matches the mechanistic interpretability finding that grokked
models learn a Fourier/rotational representation of the numbers — the
network effectively discovers that modular addition is easier to compute
if the numbers are laid out like a clock."]

![Embeddings](results/embeddings_31dfb7ca.png)

## My extension
Beyond reproducing the base result, I ran an ablation sweep across weight
decay, optimizer choice, and operation type, to test: **is weight decay
alone sufficient for grokking, or does the specific optimizer mechanism
matter too?**

**Findings** (full 5000-epoch runs, `results/ablation_comparison.png`):

- **Weight decay is essential, and the amount matters.** With `weight_decay=0`,
  the model never grokked (val accuracy stuck at ~3%) even after 5000 epochs.
  With `weight_decay=1.0`, it grokked fully to 100%. An intermediate value
  (`0.1`) produced a partial, much slower grok — val accuracy was still
  climbing (~32%) at epoch 5000, suggesting it would eventually grok given
  more time, just far more slowly.

- **AdamW and Adam are not interchangeable, even at the same weight decay
  value.** AdamW (which decouples weight decay from the gradient update)
  grokked fully. Plain Adam, using the same nominal weight decay setting,
  never grokked (~3%). This suggests grokking depends specifically on *how*
  weight decay is applied, not just that some weight decay exists —
  something the original paper doesn't isolate.

- **SGD failed outright** — it couldn't even fit the training set within
  5000 epochs (train accuracy stuck at ~1.5%), let alone generalize. Whatever
  is happening mechanistically seems to require an adaptive optimizer to
  navigate the loss landscape at all.

- **Grokking isn't specific to addition.** Multiplication grokked just as
  reliably as addition (~100%), so the effect generalizes across at least
  two algebraic operations.

![Ablation comparison](results/ablation_comparison.png)

## What I'd try next
- Sweep intermediate weight decay values (0.2–0.7) to find where the
  transition from "no grok" to "full grok" actually happens.
- Let the `weight_decay=0.1` and plain-`Adam` runs continue well past 5000
  epochs to see if they eventually grok too, just much later — or whether
  they're stuck in a genuinely different regime.
- Test whether a plain MLP (no attention) can grok on the same task, to see
  how much of this is transformer-specific vs. a general property of
  overparameterized networks trained with weight decay.

## Tools used
Python, PyTorch, SQLite, Matplotlib, scikit-learn — no paid services, ran
entirely on CPU.
