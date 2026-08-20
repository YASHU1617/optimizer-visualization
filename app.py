"""
app.py
-------
Streamlit entry point. Two tabs:
  Part A - 2D loss-surface optimizer playground (animated)
  Part B - Same optimizers training a real MLP on Breast Cancer data

Run with:
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from optimizers import make_optimizer, OPTIMIZER_COLORS
import surfaces
import mlp

st.set_page_config(page_title="Optimizer Visualizer: SGD to AdamW", layout="wide")

OPT_NAMES = ["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"]

EXPLANATIONS = {
    "SGD": "Plain SGD always steps directly along the current gradient, so on "
           "an elongated bowl it repeatedly overshoots across the steep "
           "direction while crawling along the shallow one -- the classic zig-zag.",
    "Momentum": "Momentum averages recent gradients into a velocity vector. "
                "Components that keep flipping sign (the steep direction) "
                "partially cancel out, while components that stay consistent "
                "(the shallow direction) build up speed.",
    "NAG": "NAG evaluates the gradient at a look-ahead point (theta - beta*v) "
           "instead of at theta itself. This lets it 'see' whether the momentum "
           "step is about to overshoot the minimum and correct earlier, "
           "reducing overshoot compared to plain Momentum.",
    "AdaGrad": "AdaGrad divides each parameter's learning rate by the square "
               "root of its accumulated squared gradients. Parameters that "
               "have historically had large gradients get a smaller effective "
               "step, and vice versa -- hence 'different effective learning "
               "rates per parameter'.",
    "RMSProp": "AdaGrad's accumulator only grows, so its effective learning "
               "rate keeps shrinking toward zero. RMSProp replaces the running "
               "sum with an exponential moving average, so old gradients decay "
               "away and the effective learning rate can stay stable or even "
               "grow back if gradients shrink.",
    "Adam": "Adam combines Momentum's first-moment averaging (m_t) with "
            "RMSProp's second-moment scaling (v_t), plus bias correction so "
            "early steps (when m_t and v_t are still near zero) aren't "
            "artificially shrunk.",
    "AdamW": "AdamW decouples weight decay from the gradient-based update: "
             "lambda*theta is subtracted directly, instead of being added to "
             "the gradient and then divided by sqrt(v_hat) like L2 "
             "regularization folded into Adam would do. This keeps the decay "
             "rate consistent regardless of a parameter's gradient scale.",
}


def styled_fig():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    return fig, ax


# --------------------------------------------------------------------------
# PART A: 2D Playground
# --------------------------------------------------------------------------
def part_a():
    st.header("Part A -- Optimizer Playground on a 2D Loss Surface")

    with st.expander("How to use this tool", expanded=False):
        st.markdown(
            "- Pick a loss surface and one or more optimizers.\n"
            "- Tune learning rate / beta / beta1 / beta2 / lambda with the sliders.\n"
            "- Press **Play** to animate the trajectories step by step, or **Step** "
            "to advance one iteration at a time, or **Reset** to start over.\n"
            "- The contour view (left) shows the path in (x, y) space; the loss "
            "curve (right) shows L(theta_t) vs. iteration, in the same colors."
        )

    left, right = st.columns([1, 2])

    with left:
        surface_name = st.selectbox("Loss surface", list(surfaces.SURFACES.keys()),
                                     index=1)
        k = surfaces.SURFACES[surface_name]

        chosen = st.multiselect("Optimizers to overlay", OPT_NAMES,
                                 default=["SGD", "Momentum", "Adam"])

        lr = st.slider("Learning rate (eta)", min_value=0.0001, max_value=0.5,
                        value=0.01, step=0.0001, format="%.4f")
        if lr <= 0:
            st.error("Learning rate must be positive -- using 0.0001 instead.")
            lr = 0.0001

        beta = st.slider("beta (Momentum / RMSProp)", 0.0, 0.999, 0.9, 0.001)
        beta1 = st.slider("beta1 (Adam / AdamW)", 0.0, 0.999, 0.9, 0.001)
        beta2 = st.slider("beta2 (Adam / AdamW)", 0.0, 0.9999, 0.999, 0.0001)
        wd = st.slider("lambda -- AdamW weight decay", 0.0, 0.05, 0.001, 0.0001,
                        format="%.4f")

        x0 = st.number_input("x0 (start)", value=8.0)
        y0 = st.number_input("y0 (start)", value=8.0)

        max_iter = 500
        speed = st.slider("Playback speed (steps/sec)", 1, 60, 15)

        c1, c2, c3, c4 = st.columns(4)
        play = c1.button("Play")
        pause = c2.button("Pause")
        step_btn = c3.button("Step")
        reset_btn = c4.button("Reset")

        for name in chosen:
            with st.expander(f"Why does {name} behave this way?"):
                st.write(EXPLANATIONS[name])

    # --- state -------------------------------------------------------
    if "a_index" not in st.session_state:
        st.session_state.a_index = 0
    if "a_playing" not in st.session_state:
        st.session_state.a_playing = False

    if reset_btn:
        st.session_state.a_index = 0
        st.session_state.a_playing = False
    if play:
        st.session_state.a_playing = True
    if pause:
        st.session_state.a_playing = False
    if step_btn:
        st.session_state.a_index = min(st.session_state.a_index + 1, max_iter - 1)
        st.session_state.a_playing = False

    # --- compute full trajectories (cheap: <=7 optimizers x 500 steps) --
    trajectories, losses = {}, {}
    for name in chosen:
        opt = make_optimizer(name, lr=lr, beta=beta, beta1=beta1, beta2=beta2,
                              weight_decay=wd)
        params = np.array([x0, y0], dtype=float)
        traj = [params.copy()]
        loss_hist = [surfaces.loss(params, k)]
        for _ in range(max_iter - 1):
            lookahead = opt.get_lookahead(params)
            g = surfaces.grad(lookahead, k)
            params = opt.step(params, g)
            traj.append(params.copy())
            loss_hist.append(surfaces.loss(params, k))
        trajectories[name] = np.array(traj)
        losses[name] = np.array(loss_hist)

    idx = st.session_state.a_index

    with right:
        col1, col2 = st.columns(2)

        # View 1: contour map
        with col1:
            fig, ax = styled_fig()
            xs = np.linspace(-10, 10, 200)
            ys = np.linspace(-10, 10, 200)
            X, Y = np.meshgrid(xs, ys)
            Z = X ** 2 + k * Y ** 2
            ax.contourf(X, Y, Z, levels=30, cmap="Blues")
            ax.plot(0, 0, marker="*", color="gold", markersize=16,
                     markeredgecolor="black", label="Global min", zorder=5)
            for name in chosen:
                pts = trajectories[name][: idx + 1]
                ax.plot(pts[:, 0], pts[:, 1], color=OPTIMIZER_COLORS[name],
                         linewidth=1.8, label=name)
                ax.scatter(pts[-1, 0], pts[-1, 1], color=OPTIMIZER_COLORS[name],
                            s=45, zorder=6, edgecolor="black")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title(f"Trajectories on {surface_name} (iter {idx})")
            ax.legend(loc="upper right", fontsize=7)
            st.pyplot(fig)
            plt.close(fig)

        # View 2: loss curve
        with col2:
            fig2, ax2 = styled_fig()
            for name in chosen:
                ax2.plot(losses[name][: idx + 1], color=OPTIMIZER_COLORS[name],
                          label=name, linewidth=1.8)
            ax2.set_xlabel("iteration"); ax2.set_ylabel("L(theta_t)")
            ax2.set_yscale("log")
            ax2.set_title("Loss vs. iteration")
            ax2.legend(loc="upper right", fontsize=7)
            st.pyplot(fig2)
            plt.close(fig2)

    st.caption(
        f"Condition number of this bowl's Hessian = {surfaces.condition_number(k):.0f}. "
        "Higher condition number -> narrower bowl -> worse zig-zagging for plain SGD."
    )

    # --- drive the animation ------------------------------------------
    if st.session_state.a_playing and idx < max_iter - 1:
        time.sleep(1.0 / speed)
        st.session_state.a_index += 1
        st.rerun()
    elif st.session_state.a_playing and idx >= max_iter - 1:
        st.session_state.a_playing = False


# --------------------------------------------------------------------------
# PART B: Neural network dashboard
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, data.feature_names


def part_b():
    st.header("Part B -- Training a Real MLP with the Same Optimizers")

    X_train, X_test, y_train, y_test, feat_names = load_data()
    st.write(
        f"**Dataset:** Breast Cancer Wisconsin -- "
        f"{X_train.shape[0]} train samples, {X_test.shape[0]} test samples, "
        f"{X_train.shape[1]} features."
    )

    with st.expander("How to use this tool", expanded=False):
        st.markdown(
            "- Choose one or more optimizers, set hyperparameters, epochs and "
            "batch size, then press **Train**.\n"
            "- Charts update every epoch as training runs.\n"
            "- A comparison table is generated automatically once every "
            "selected optimizer has finished training."
        )

    c1, c2 = st.columns([1, 2])
    with c1:
        chosen = st.multiselect("Optimizers", OPT_NAMES,
                                 default=["SGD", "Momentum", "Adam", "AdamW"],
                                 key="b_opts")
        lr = st.slider("Learning rate (eta)", 0.0001, 0.5, 0.01, 0.0001,
                        format="%.4f", key="b_lr")
        if lr <= 0:
            st.error("Learning rate must be positive -- using 0.0001 instead.")
            lr = 0.0001
        beta = st.slider("beta", 0.0, 0.999, 0.9, 0.001, key="b_beta")
        beta1 = st.slider("beta1", 0.0, 0.999, 0.9, 0.001, key="b_beta1")
        beta2 = st.slider("beta2", 0.0, 0.9999, 0.999, 0.0001, key="b_beta2")
        wd = st.slider("lambda (AdamW)", 0.0, 0.05, 0.001, 0.0001, key="b_wd")
        epochs = st.slider("Epochs", 5, 300, 60, key="b_epochs")
        batch_size = st.select_slider("Batch size", [8, 16, 32, 64, 128], 32,
                                       key="b_bs")
        animate = st.checkbox("Animate training (slower, visibly per-epoch)",
                               value=False)
        train_btn = st.button("Train", type="primary")

    if not chosen:
        st.info("Select at least one optimizer to train.")
        return

    if train_btn:
        history = {name: {"train_loss": [], "test_loss": [], "train_acc": [],
                           "test_acc": [], "eff_lr": []} for name in chosen}

        chart_loss = st.empty()
        chart_acc = st.empty()
        chart_efflr = st.empty()
        progress = st.progress(0.0)

        n = X_train.shape[0]
        rng = np.random.default_rng(0)

        for name in chosen:
            params = mlp.init_params(X_train.shape[1], seed=42)
            optims = {k_: make_optimizer(name, lr=lr, beta=beta, beta1=beta1,
                                          beta2=beta2, weight_decay=wd)
                      for k_ in params}

            for epoch in range(epochs):
                order = rng.permutation(n)
                for start in range(0, n, batch_size):
                    idx = order[start:start + batch_size]
                    Xb, yb = X_train[idx], y_train[idx]
                    y_pred, cache = mlp.forward(params, Xb)
                    grads = mlp.backward(params, cache, yb)
                    for k_ in params:
                        params[k_] = optims[k_].step(params[k_], grads[k_])

                train_pred, _ = mlp.forward(params, X_train)
                test_pred, _ = mlp.forward(params, X_test)
                history[name]["train_loss"].append(mlp.bce_loss(train_pred, y_train))
                history[name]["test_loss"].append(mlp.bce_loss(test_pred, y_test))
                history[name]["train_acc"].append(mlp.accuracy(train_pred, y_train))
                history[name]["test_acc"].append(mlp.accuracy(test_pred, y_test))
                history[name]["eff_lr"].append(float(np.mean(optims["W1"].effective_lr())))

                if animate or epoch == epochs - 1:
                    _draw_live_charts(chart_loss, chart_acc, chart_efflr, history)
                    if animate:
                        time.sleep(0.03)

            progress.progress((chosen.index(name) + 1) / len(chosen))

        st.session_state.b_history = history
        _draw_live_charts(chart_loss, chart_acc, chart_efflr, history)

    if "b_history" in st.session_state:
        _comparison_table(st.session_state.b_history)


def _draw_live_charts(chart_loss, chart_acc, chart_efflr, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in history.items():
        if not h["train_loss"]:
            continue
        c = OPTIMIZER_COLORS[name]
        ax1.plot(h["train_loss"], color=c, linestyle="-", label=f"{name} train")
        ax1.plot(h["test_loss"], color=c, linestyle="--", label=f"{name} test")
        ax2.plot(h["test_acc"], color=c, label=name)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.set_title("Train/Test loss")
    ax1.legend(fontsize=6)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("Test accuracy")
    ax2.legend(fontsize=7)
    chart_loss.pyplot(fig)
    plt.close(fig)

    fig2, ax3 = plt.subplots(figsize=(11, 3))
    any_adaptive = False
    for name, h in history.items():
        if name in ("AdaGrad", "RMSProp", "Adam", "AdamW") and h["eff_lr"]:
            any_adaptive = True
            ax3.plot(h["eff_lr"], color=OPTIMIZER_COLORS[name], label=name)
    ax3.set_xlabel("epoch"); ax3.set_ylabel("effective LR (mean, W1)")
    ax3.set_title("Effective learning rate over training (adaptive optimizers)")
    if any_adaptive:
        ax3.legend(fontsize=7)
    chart_efflr.pyplot(fig2)
    plt.close(fig2)


def _comparison_table(history):
    st.subheader("Comparison table (auto-computed)")
    rows = []
    for name, h in history.items():
        if not h["train_loss"]:
            continue
        final_test_loss = h["test_loss"][-1]
        threshold = abs(final_test_loss) * 0.01
        conv_epoch = next(
            (i for i, v in enumerate(h["test_loss"])
             if abs(v - final_test_loss) <= threshold),
            len(h["test_loss"]) - 1,
        )
        rows.append({
            "Optimizer": name,
            "Final Train Loss": round(h["train_loss"][-1], 4),
            "Final Test Loss": round(final_test_loss, 4),
            "Train Acc.": round(h["train_acc"][-1], 4),
            "Test Acc.": round(h["test_acc"][-1], 4),
            "Convergence Epoch": conv_epoch + 1,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# --------------------------------------------------------------------------
def main():
    st.title("Optimizer Visualizer: From SGD to AdamW")
    tab_a, tab_b = st.tabs(["Part A -- 2D Playground", "Part B -- Neural Network"])
    with tab_a:
        part_a()
    with tab_b:
        part_b()


if __name__ == "__main__":
    main()
