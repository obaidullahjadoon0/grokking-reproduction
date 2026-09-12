# Grokking: Reproducing & Extending "Generalization Beyond Overfitting"

**Status:** Reproduced and extended — grokking confirmed, ablation complete · **Cost:** $0 (CPU-only, no paid APIs, no cloud bills)

## 1. The paper

Power, Burda, Edwards, Babuschkin, and Misra's *"Grokking: Generalization
Beyond Overfitting on Small Algorithmic Datasets"* (2021) found something
strange: train a small transformer on a simple modular arithmetic task
(e.g. `a + b mod p`), and it will first **memorize** the training set (100%
train accuracy, ~0% validation accuracy) and then — often *thousands of
steps later*, long after training loss looks "done" — validation accuracy
suddenly jumps to ~100% too. They called this delayed generalization
**"grokking."**

Follow-up mechanistic interpretability work (notably Neel Nanda's) showed
*why*: the network is secretly learning a Fourier/circular representation of
the numbers mod p, and once that circuit fully forms, generalization appears
almost overnight.

## 2. What I did

I built a decoder-only transformer entirely from scratch in PyTorch (no
`nn.TransformerEncoder`), trained it on modular addition, and reproduced
the grokking effect directly:

- Training accuracy reached 100% almost immediately.
- Validation accuracy stayed near 0% for hundreds of epochs.
- Validation accuracy then jumped sharply, crossing 90% at **epoch 450**
  and eventually reaching 100%.

![Training curves showing the grokking jump](results/curves_31dfb7ca.png)

I then looked inside the trained model by projecting its learned number
embeddings (0–96) down to 2D with PCA:

![Learned number embeddings](results/embeddings_31dfb7ca.png)

*(If the embeddings show a roughly circular/ring layout of the numbers in
order, that's the mechanistic signature described in the interpretability
literature — the network has learned a rotational representation of
modular arithmetic. Update this caption with what your specific plot shows.)*

## 3. My extension: does the optimizer mechanism matter, not just weight decay?

The original paper identifies weight decay as critical to grokking, but
doesn't isolate *how* different optimizers apply it. I ran a controlled
ablation across weight decay values, optimizer choice, and operation type
(all runs: 5000 epochs, single seed, single CPU):

![Ablation comparison across weight decay, optimizer, and operation](results/ablation_comparison.png)

**Findings:**

- **Weight decay amount matters a lot.** `weight_decay=0` never grokked
  (val accuracy stuck around 3%). `weight_decay=1.0` grokked fully (100%).
  An intermediate value (`0.1`) produced a much slower, partial grok — still
  climbing (~32%) at epoch 5000 rather than plateaued.
- **AdamW and Adam are not interchangeable, even at identical nominal
  weight decay.** AdamW (which decouples weight decay from the gradient
  update) grokked fully. Plain Adam, same weight decay setting, never
  grokked (~3%). This suggests grokking depends specifically on *how*
  weight decay is applied to the update rule, not just that some weight
  decay exists.
- **SGD failed outright** — it couldn't even fit the training set within
  5000 epochs (train accuracy ~1.5%), let alone generalize.
- **Grokking isn't addition-specific.** Multiplication grokked just as
  reliably as addition (~100%).

## 4. What I'd try next

- Sweep intermediate weight decay values (0.2–0.7) to find where the
  transition from "no grok" to "full grok" happens.
- Let the `weight_decay=0.1` and plain-Adam runs continue well past 5000
  epochs — do they eventually grok too, just later, or are they in a
  genuinely different regime?
- Test whether a plain MLP (no attention) groks on the same task, to see
  how much of this effect is transformer-specific.

## 5. Project structure

```
grokking-project/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── ablation.py
│   ├── analyze.py
│   └── db.py
├── db/
│   └── runs.sqlite
└── results/
    ├── curves_*.png
    ├── embeddings_*.png
    ├── ablation_comparison.png
    └── model_*.pt
```

## 6. How to run it (all free, all local)

```bash
cd grokking-project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/train.py --config configs/config.yaml
python src/analyze.py --run_id latest --plot curves
python src/analyze.py --run_id latest --plot embeddings
python src/ablation.py --config configs/config.yaml
python src/analyze.py --plot ablation_comparison
```

Runs entirely on CPU — no GPU required, though setting `device: cuda` in
the config will speed things up if you have one available.

## 7. Tools used (all free)

| Tool | Purpose | Cost |
|---|---|---|
| Python + PyTorch | model + training | free, open source |
| SQLite | experiment logging | free, no server needed |
| Matplotlib | plots | free |
| scikit-learn (PCA) | embedding analysis | free |
| GitHub | hosting | free |

No API keys, no paid experiment-tracking service.