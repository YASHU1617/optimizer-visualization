"""
optimizers.py
--------------
From-scratch implementations of SGD, Momentum, NAG, AdaGrad, RMSProp,
Adam, and AdamW. Pure NumPy math only -- NO plotting, NO UI code here.

Every optimizer exposes the same interface:

    opt = SGD(lr=0.01)
    lookahead_params = opt.get_lookahead(params)   # used only by NAG
    grad = compute_gradient(lookahead_params)       # caller computes this
    params = opt.step(params, grad)                 # returns updated params
    opt.reset()                                      # clears internal state

`params` and `grad` are plain NumPy arrays of any shape (a 2-vector [x, y]
for the Part-A playground, or a weight matrix / bias vector for the Part-B
neural network). Each optimizer instance owns exactly one parameter
tensor's worth of state (velocity, moving averages, timestep) so that
reusing the same classes for many weight matrices just means creating
one optimizer instance per matrix.
"""

import numpy as np


class Optimizer:
    """Base class. Subclasses must implement step()."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def get_lookahead(self, params):
        """Point at which the gradient should be evaluated.
        Only NAG overrides this; everyone else just uses `params`."""
        return params

    def step(self, params, grad):
        raise NotImplementedError

    def reset(self):
        """Clear internal state so the optimizer can be reused from t=0."""
        pass

    def effective_lr(self):
        """Per-parameter effective learning rate, for optimizers that
        adapt it (AdaGrad/RMSProp/Adam/AdamW). Returns a scalar `self.lr`
        for optimizers with no adaptive scaling."""
        return self.lr


class SGD(Optimizer):
    """theta_{t+1} = theta_t - eta * g_t"""

    def step(self, params, grad):
        return params - self.lr * grad


class Momentum(Optimizer):
    """v_t = beta*v_{t-1} + (1-beta)*g_t
       theta_{t+1} = theta_t - eta*v_t"""

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def step(self, params, grad):
        if self.v is None:
            self.v = np.zeros_like(params, dtype=float)
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return params - self.lr * self.v

    def reset(self):
        self.v = None


class NAG(Optimizer):
    """Nesterov Accelerated Gradient.
       Gradient is evaluated at the *look-ahead* point theta_t - beta*v_{t-1},
       not at theta_t itself. The caller MUST call get_lookahead() first,
       compute the gradient there, then pass that gradient into step()."""

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def get_lookahead(self, params):
        if self.v is None:
            self.v = np.zeros_like(params, dtype=float)
        return params - self.beta * self.v

    def step(self, params, grad):
        if self.v is None:
            self.v = np.zeros_like(params, dtype=float)
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return params - self.lr * self.v

    def reset(self):
        self.v = None


class AdaGrad(Optimizer):
    """G_t = G_{t-1} + g_t^2
       theta_{t+1} = theta_t - eta*g_t / sqrt(G_t + eps)"""

    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.G = None

    def step(self, params, grad):
        if self.G is None:
            self.G = np.zeros_like(params, dtype=float)
        self.G = self.G + grad ** 2
        return params - self.lr * grad / (np.sqrt(self.G) + self.eps)

    def reset(self):
        self.G = None

    def effective_lr(self):
        if self.G is None:
            return self.lr
        return self.lr / (np.sqrt(self.G) + self.eps)


class RMSProp(Optimizer):
    """v_t = beta*v_{t-1} + (1-beta)*g_t^2
       theta_{t+1} = theta_t - eta*g_t / sqrt(v_t + eps)"""

    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.v = None

    def step(self, params, grad):
        if self.v is None:
            self.v = np.zeros_like(params, dtype=float)
        self.v = self.beta * self.v + (1 - self.beta) * grad ** 2
        return params - self.lr * grad / (np.sqrt(self.v) + self.eps)

    def reset(self):
        self.v = None

    def effective_lr(self):
        if self.v is None:
            return self.lr
        return self.lr / (np.sqrt(self.v) + self.eps)


class Adam(Optimizer):
    """m_t, v_t moving averages, bias-corrected to m_hat, v_hat.
       theta_{t+1} = theta_t - eta * m_hat / (sqrt(v_hat) + eps)"""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grad):
        if self.m is None:
            self.m = np.zeros_like(params, dtype=float)
            self.v = np.zeros_like(params, dtype=float)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def effective_lr(self):
        if self.v is None:
            return self.lr
        t = max(self.t, 1)
        v_hat = self.v / (1 - self.beta2 ** t)
        return self.lr / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Identical to Adam but with DECOUPLED weight decay: the lambda*theta
       term is added directly to the parameter update, not folded into the
       gradient before the moment estimates are computed. This is the key
       difference from "Adam + L2 regularization", where lambda*theta would
       instead be added to g_t and would therefore get divided by
       sqrt(v_hat) along with the rest of the gradient -- distorting the
       effective decay rate for parameters with large/small gradients."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=1e-3):
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def step(self, params, grad):
        if self.m is None:
            self.m = np.zeros_like(params, dtype=float)
            self.v = np.zeros_like(params, dtype=float)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params
        return params - self.lr * update


# Registry used by the UI layer. Keeping this here (not in app.py) is what
# lets Part A and Part B share the exact same optimizer objects/config.
OPTIMIZER_REGISTRY = {
    "SGD": SGD,
    "Momentum": Momentum,
    "NAG": NAG,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
    "Adam": Adam,
    "AdamW": AdamW,
}

# Fixed color per optimizer, reused identically across every plot in the app.
OPTIMIZER_COLORS = {
    "SGD": "#e41a1c",
    "Momentum": "#377eb8",
    "NAG": "#4daf4a",
    "AdaGrad": "#984ea3",
    "RMSProp": "#ff7f00",
    "Adam": "#a65628",
    "AdamW": "#17becf",
}


def make_optimizer(name, lr, beta, beta1, beta2, weight_decay):
    """Factory: build a fresh optimizer instance from the current UI
    hyperparameter values. Each call returns a brand-new, zero-state
    instance -- callers must not share instances across separate runs."""
    cls = OPTIMIZER_REGISTRY[name]
    if name == "SGD":
        return cls(lr=lr)
    if name in ("Momentum", "NAG"):
        return cls(lr=lr, beta=beta)
    if name == "AdaGrad":
        return cls(lr=lr)
    if name == "RMSProp":
        return cls(lr=lr, beta=beta)
    if name == "Adam":
        return cls(lr=lr, beta1=beta1, beta2=beta2)
    if name == "AdamW":
        return cls(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
