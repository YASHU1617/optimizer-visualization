"""
mlp.py
-------
A tiny Multi-Layer Perceptron implemented entirely by hand:
Input -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid
Binary cross-entropy loss. Forward pass, loss, and full backprop are all
written out explicitly -- no autograd, no framework layers.

Each weight matrix / bias vector is treated as its own "params" tensor and
gets its own Optimizer instance from optimizers.py, so Part B reuses the
*exact same* optimizer classes as Part A.
"""

import numpy as np


def init_params(input_dim, seed=42):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, 16)),
        "b1": np.zeros(16),
        "W2": rng.normal(0, np.sqrt(2.0 / 16), (16, 8)),
        "b2": np.zeros(8),
        "W3": rng.normal(0, np.sqrt(2.0 / 8), (8, 1)),
        "b3": np.zeros(1),
    }
    return params


def relu(z):
    return np.maximum(0.0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    a3 = sigmoid(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3}
    return a3, cache


def bce_loss(y_pred, y_true, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_true = y_true.reshape(-1, 1)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def backward(params, cache, y_true):
    """Full manual backprop through the 3-layer network."""
    X, z1, a1, z2, a2, a3 = (
        cache["X"], cache["z1"], cache["a1"],
        cache["z2"], cache["a2"], cache["a3"],
    )
    m = X.shape[0]
    y_true = y_true.reshape(-1, 1)

    # dL/dz3 for BCE + sigmoid simplifies to (a3 - y)
    dz3 = (a3 - y_true) / m
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * relu_deriv(z2)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * relu_deriv(z1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def accuracy(y_pred, y_true, threshold=0.5):
    preds = (y_pred.flatten() >= threshold).astype(int)
    return float(np.mean(preds == y_true.flatten()))
