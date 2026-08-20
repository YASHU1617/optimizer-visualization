# Optimizer Visualizer: From SGD to AdamW

Interactive Streamlit app implementing SGD, Momentum, NAG, AdaGrad, RMSProp,
Adam, and AdamW entirely from scratch in NumPy, plus a visual playground
(Part A) and a real MLP training dashboard on the Breast Cancer Wisconsin
dataset (Part B).

## Files
- `optimizers.py` — the 7 optimizer classes (pure math, no plotting/UI).
- `surfaces.py` — the 4 elongated-bowl loss surfaces (L1–L4).
- `mlp.py` — hand-written MLP: forward pass, BCE loss, manual backprop.
- `app.py` — Streamlit UI (Part A + Part B), the only file that imports
  `streamlit`, `matplotlib`, etc.

## Run locally
```bash
pip install streamlit numpy pandas matplotlib scikit-learn
streamlit run app.py
```

## Run on Streamlit Community Cloud (share.streamlit.io)
1. Push this folder to a GitHub repo (needs at minimum `app.py`,
   `optimizers.py`, `surfaces.py`, `mlp.py`).
2. Add a `requirements.txt` (included below) so the cloud build installs
   the right packages.
3. On share.streamlit.io, click **Create app** → **Deploy a public app from
   a GitHub repo** → point it at the repo and set **Main file path** to
   `app.py`.

`requirements.txt`:
```
streamlit
numpy
pandas
matplotlib
scikit-learn
```

## Default hyperparameters (also shown in-app)
- Max iterations (Part A): 500
- Learning rate: 0.01
- beta (Momentum/RMSProp): 0.9
- beta1, beta2 (Adam/AdamW): 0.9, 0.999
- lambda (AdamW weight decay): 1e-3
- eps (numerical stability): 1e-8
- Start point (Part A): (x0, y0) = (8, 8)
- MLP: Input → Dense(16)+ReLU → Dense(8)+ReLU → Dense(1)+Sigmoid,
  binary cross-entropy loss, He-style weight init.

## Notes for your write-up
- With the default beta=0.9, lr=0.01 on the L2 (k=50) surface, NAG and
  Momentum sit right at the edge of the stable region for this curvature —
  try nudging beta down or lr down in the app and watch the y-trajectory
  go from diverging to converging. This is a genuine, useful example for
  reflection question A7-Q8 / A6 (the LR/beta sensitivity explorer), not a
  bug.
- Convergence epoch in the Part B comparison table is defined as the first
  epoch at which test loss is within 1% of its final value, computed
  automatically from the recorded history.
