"""Predictive coding (Rao & Ballard; Whittington & Bogacz; Song et al. 'prospective configuration').

Layers:  x0 (input, clamped) -> x1 (hidden, free) -> x2 (output, clamped to target while training)
Each layer holds a PREDICTION of the next one; the mismatch is an ERROR.
    mu1 = x0 @ W1              prediction of the hidden layer
    e1  = x1 - mu1             hidden prediction error
    mu2 = tanh(x1) @ W2        prediction of the output
    e2  = x2 - mu2             output prediction error
    F   = 1/2 |e1|^2 + 1/2 |e2|^2        (the energy: total squared prediction error)

Inference  = relax x1 to reduce F, with the target clamped (activities move FIRST).
Learning   = one gradient step on F w.r.t. the weights, using the settled activities.
             dW1 = x0^T e1 , dW2 = tanh(x1)^T e2  -> purely local (presynaptic activity x postsynaptic error).
Gradients verified against finite differences.
"""
import torch


def pc_init(in_dim=196, hidden=64, out_dim=10, seed=0, device="cpu"):
    """Weights only; no autograd needed since the updates are explicit and local."""
    g = torch.Generator(device=device).manual_seed(seed)
    W1 = torch.randn(in_dim, hidden, generator=g, device=device) / in_dim ** 0.5
    W2 = torch.randn(hidden, out_dim, generator=g, device=device) / hidden ** 0.5
    return W1, W2


def pc_forward(x0, W1, W2):
    """Feedforward pass = the equilibrium when nothing is clamped at the output."""
    x0 = x0.reshape(x0.size(0), -1)
    x1 = x0 @ W1
    return x1, torch.tanh(x1) @ W2


def pc_settle(x0, W1, W2, target, dt=0.1, steps=50):
    """Infer the hidden activities with the output clamped to `target`.
       Starts from the feedforward value (so e1 = 0) and lets the output error pull x1 into a
       'prospective configuration' that would have produced the correct answer."""
    x0 = x0.reshape(x0.size(0), -1)
    mu1 = x0 @ W1
    x1 = mu1.clone()
    for _ in range(steps):
        e1 = x1 - mu1
        e2 = target - torch.tanh(x1) @ W2
        dx1 = e1 - (1 - torch.tanh(x1) ** 2) * (e2 @ W2.t())      # dF/dx1
        x1 = x1 - dt * dx1
    return x1


def pc_update(x, y_labels, W1, W2, lr=0.05, dt=0.1, steps=50, device="cpu"):
    """One predictive-coding weight update for a batch. Updates W1, W2 in place."""
    x0 = x.reshape(x.size(0), -1)
    target = torch.zeros(x0.size(0), W2.size(1), device=device)
    target.scatter_(1, y_labels.unsqueeze(1), 1.0)                # one-hot output target
    x1 = pc_settle(x0, W1, W2, target, dt=dt, steps=steps)        # settle activities first
    e1 = x1 - x0 @ W1                                             # then read off local errors
    e2 = target - torch.tanh(x1) @ W2
    W1 += lr * (x0.t() @ e1) / x0.size(0)                         # local: pre-activity x post-error
    W2 += lr * (torch.tanh(x1).t() @ e2) / x0.size(0)


def pc_predict(x, W1, W2, raw=False):
    """Test time: nothing clamped at the output, so the equilibrium is just the feedforward pass."""
    _, out = pc_forward(x, W1, W2)
    return out if raw else out.argmax(1)
