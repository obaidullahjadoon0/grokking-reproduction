# Grokking: Reproducing & Extending "Generalization Beyond Overfitting"

**Status:** template ready to run · **Cost:** $0 (CPU-friendly, no paid APIs, no cloud bills)

## 1. The paper

Power et al., *"Grokking: Generalization Beyond Overfitting on Small Algorithmic
Datasets"* (2021) found something strange: train a small transformer on a
simple modular arithmetic task (e.g. `a + b mod p`), and it will first
**memorize** the training set (100% train accuracy, ~0% validation accuracy)
and then — often *thousands of steps later*, long after training loss looks
"done" — validation accuracy suddenly jumps to ~100% too. They called this
delayed generalization **"grokking."**

Follow-up mechanistic interpretability work (notably Neel Nanda's) showed
*why*: the network is secretly learning a Fourier/circular representation of
the numbers mod p, and once that circuit fully forms, generalization appears
almost overnight.

This is a great portfolio project because it's small, it's a real
unsolved-feeling mystery in deep learning, and it rewards you for actually
looking inside the model instead of just reporting a final accuracy number.

## 2. What this repo does

1. **Reproduces** the core result: train a tiny decoder-only transformer on
   modular addition, and watch it grok.
2. **Logs everything** to a local SQLite database (no cloud dashboard
   required) so you can query/plot runs later.
3. **Extends** the paper with an ablation sweep across:
   - weight decay (the ingredient the paper found critical)
   - optimizer choice (AdamW vs. plain Adam vs. SGD)
   - operation type (addition vs. multiplication vs. subtraction)
   - training data fraction (how much data is needed before grokking is even possible)
4. **Visualizes the learned circuit**: PCA/2D projection of the model's
   number embeddings, which — if grokking happened — should show the
   circular structure the interpretability literature predicts.

## 3. Project structure

```
grokking-project/
├── README.md                <- you are here
├── requirements.txt
├── configs/
│   └── config.yaml           <- all hyperparameters, edit this first
├── src/
│   ├── data.py                <- generates the modular arithmetic dataset
│   ├── model.py                <- the transformer, written from scratch
│   ├── train.py                 <- training loop + SQLite logging
│   ├── ablation.py               <- runs the extension sweep
│   ├── analyze.py                 <- plots + embedding visualization
│   └── db.py                       <- tiny SQLite helper
├── db/
│   └── runs.sqlite             <- created automatically on first run
├── results/                    <- plots get saved here
└── notebooks/
    └── explore.ipynb           <- optional interactive exploration
```

## 4. How to run it (all free, all local)

```bash
cd grokking-project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Reproduce the core grokking result (~5-15 min on CPU, ~1-2 min on GPU)
python src/train.py --config configs/config.yaml

# 2. Look at the curves — you want to see the val-accuracy "grok" jump
python src/analyze.py --run_id latest --plot curves

# 3. Look inside the model — circular embedding structure = "it worked"
python src/analyze.py --run_id latest --plot embeddings

# 4. Run the extension ablation sweep (this is YOUR contribution)
python src/ablation.py --config configs/config.yaml

# 5. Compare all ablation runs
python src/analyze.py --plot ablation_comparison
```

Everything (data generation, training, logging, plotting) runs on CPU in
minutes because the model and dataset are intentionally tiny (that's the
whole point of the paper — the effect shows up even in toy settings). If you
have any GPU (even a free Google Colab T4), just set `device: cuda` in the
config and it'll be faster.

## 5. What "done" looks like

- A `results/curves_<run_id>.png` showing the classic grokking shape: train
  accuracy hits ~100% early, val accuracy stays near 0% for a long plateau,
  then jumps to ~100%.
- A `results/embeddings_<run_id>.png` showing numbers arranged in a rough
  circle/ring in embedding space (this is the "aha" visual for your README
  and LinkedIn post).
- A `results/ablation_comparison.png` showing how weight decay / optimizer /
  operation change *when* (or *whether*) grokking happens.
- A one-paragraph writeup (template in `results/WRITEUP_TEMPLATE.md`) of what
  you found in your ablation that the original paper didn't test.

## 6. Suggested extension ideas (pick one or more, this is where it becomes "yours")

- Does grokking happen for multiplication / subtraction as reliably as addition?
- What's the minimum training-data fraction where grokking still occurs?
- Does removing weight decay entirely prevent grokking completely, or just delay it further?
- Can you predict *when* grokking will happen from early training dynamics (e.g. gradient norm patterns)?
- Try a non-transformer architecture (small MLP) on the same task — does grokking still happen?

## 7. Why this is a strong portfolio piece

- It's not another Kaggle notebook — it's a reproduction of a real, cited
  research finding, with your own novel ablation on top.
- The interpretability angle (visualizing the circular embedding) gives you
  a genuinely interesting image to put in a LinkedIn post — "I made a neural
  network learn to do modular arithmetic in a circle" is a strong hook.
- It demonstrates: PyTorch from-scratch model building, experiment tracking
  discipline (SQL logging), scientific ablation methodology, and the ability
  to read and engage critically with a paper — exactly what research-adjacent
  interviewers probe for.

## 8. Tools used (all free)

| Tool | Purpose | Cost |
|---|---|---|
| Python + PyTorch | model + training | free, open source |
| SQLite (via Python's built-in `sqlite3`) | experiment logging | free, no server needed |
| Matplotlib | plots | free |
| NumPy / scikit-learn (PCA) | embedding analysis | free |
| Google Colab (optional) | free GPU if you want faster training | free tier |
| GitHub | hosting your repo publicly | free |
| GitHub Pages / Streamlit Community Cloud (optional) | free demo hosting if you want a live page | free tier |

No API keys, no paid experiment-tracking service, nothing that requires a
credit card.
