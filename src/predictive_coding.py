"""Predictive coding / prospective configuration -- MULTI-LAYER.

    x0 (input, clamped) -> x1 ... x_{L-1} (hidden, FREE) -> output (clamped to target)

    mu_1 = x0 @ W1                     prediction of hidden layer 1
    mu_l = f(x_{l-1}) @ W_l            prediction of hidden layer l   (l >= 2)
    e_l  = x_l - mu_l                  prediction error at layer l
    mu_L = f(x_{L-1}) @ W_L            prediction of the output
    e_L  = -dL/dmu_L                   output error (from model.output_error, so the OUTPUT
                                       STRUCTURE is whatever the experiment specified)
    F    = 1/2 sum_l |e_l|^2

Inference:  dF/dx_l = e_l - f'(x_l) * (e_{l+1} @ W_{l+1}^T)     relax every hidden layer.
Learning:   dW_1 = x0^T e_1 ,   dW_l = f(x_{l-1})^T e_l         purely local.

Each x_l is initialised to mu_l, so after settling e_l IS the displacement of that layer
from its feedforward value -- the quantity hypothesis H1 concerns.
"""
import torch
from .model import (Arch, Objective, UNIFIED_ARCH, UNIFIED_OBJ, init_params, flatten,
                    forward, hidden_code, make_target, active_vector, output_error,
                    batch_scale)


def pc_init(in_dim=196, hidden=64, out_dim=10, seed=0, device="cpu", arch=None):
    """Always returns a Params. arch=None builds the old one-hidden-layer tanh default.

    Earlier versions returned a bare (W1, W2) tuple in that case, which pc_update can no
    longer consume now that it takes a Params -- dead code that would have failed
    confusingly. Every experiment goes through build_method, so nothing depended on it."""
    if arch is None:
        arch = Arch(in_dim=in_dim, hidden=hidden, out_dim=out_dim, act="tanh", bias=False)
    return init_params(arch, seed=seed, device=device)


def pc_settle(x0, p, arch, obj, target, active_vec=None, dt=0.1, steps=50, trace=False,
              x_lr_discount=1.0, x_lr_amplifier=1.0):
    """Relax every hidden layer with the output clamped. steps=0 returns the feedforward state.

    NOTE: steps=0 is NOT backprop -- every e_l is zero by construction, so W1 does not move
    at all. The small-step limit approaches backprop; zero steps does not. Verified in
    tests/test_numpy_mirror.py.
    """
    x0 = flatten(x0)
    L = arch.n_weights
    # feedforward initialisation: x_l = mu_l, so every hidden error starts at zero
    mus, xs, a = [], [], x0
    for i in range(L - 1):
        mu = a @ p.Ws[i] if p.bs[i] is None else a @ p.Ws[i] + p.bs[i]
        mus.append(mu)
        xs.append(mu.clone())
        a = arch.f(mu)
    # Song & Bogacz set optimizer_x = SGD(lr=0.1), x_lr_discount=0.9, x_lr_amplifier=1.0.
    # That is a backtracking rule on the INFERENCE step size: multiply it by the discount
    # whenever the energy fails to fall, leave it alone when it does (amplifier 1.0 = off).
    # Defaults here are 1.0/1.0, i.e. the plain fixed-step relaxation we used before.
    backtracking = (x_lr_discount != 1.0) or (x_lr_amplifier != 1.0)
    dev = p.Ws[0].device
    lr_x = torch.tensor(float(dt), device=dev) if backtracking else dt
    amp_t = torch.tensor(float(x_lr_amplifier), device=dev)
    disc_t = torch.tensor(float(x_lr_discount), device=dev)
    disp, last_E = [], None
    for _ in range(steps):
        # top-down: output error first, then propagate the correction downward
        top = arch.f(xs[-1])
        mu_out = top @ p.Ws[L - 1] if p.bs[L - 1] is None else top @ p.Ws[L - 1] + p.bs[L - 1]
        e_next = output_error(mu_out, target, obj, active_vec)
        W_next = p.Ws[L - 1]
        if backtracking:                       # skip the energy sum when it is not needed
            # e_next is -dL/dout, which equals the residual (target - out) ONLY under squared
            # error, so 0.5*e^2 is the output energy only there. Under ce or hinge this would
            # backtrack on a quantity that is not the energy, and silently.
            if obj.loss != "mse":
                raise ValueError(
                    f"step-size backtracking assumes squared error, got loss={obj.loss!r}. "
                    "Set x_lr_discount=1.0 to disable it, or use loss='mse'."
                )
            # kept as a 0-dim TENSOR. float() here forces a device-to-host sync on every
            # settling step -- 64 per weight update, ~61,000 per run -- each stalling the GPU
            # for arithmetic that costs almost nothing. This is the difference between Colab
            # being fast and Colab being pointless.
            E = 0.5 * ((e_next ** 2).sum()
                       + sum(((xs[l] - mus[l]) ** 2).sum() for l in range(L - 1)))
            if last_E is not None:
                lr_x = lr_x * torch.where(E < last_E, amp_t, disc_t)
            last_E = E
        new = [None] * (L - 1)
        for l in range(L - 2, -1, -1):
            e_l = xs[l] - mus[l]
            new[l] = xs[l] - lr_x * (e_l - arch.df(xs[l]) * (e_next @ W_next.t()))
            e_next, W_next = e_l, p.Ws[l]
        xs = new
        # predictions of layers 2..L-1 depend on the layer below, so refresh them
        for l in range(1, L - 1):
            a = arch.f(xs[l - 1])
            mus[l] = a @ p.Ws[l] if p.bs[l] is None else a @ p.Ws[l] + p.bs[l]
        if trace:
            disp.append(float(sum((x - m).abs().mean() for x, m in zip(xs, mus)) / len(xs)))
    return (xs, mus, disp) if trace else (xs, mus)


def pc_update(x, y_labels, p, arch=UNIFIED_ARCH, obj=UNIFIED_OBJ, lr=0.05, dt=0.1,
              steps=50, active=None, device="cpu", return_delta=False, freeze=(),
              opt=None, x_lr_discount=1.0, x_lr_amplifier=1.0):
    """One predictive-coding weight update for a batch. Updates p in place."""
    x0 = flatten(x)
    n, L = x0.size(0), arch.n_weights
    scale = batch_scale(obj, n)
    target = make_target(y_labels, arch, obj, device=device)
    av = active_vector(active, arch, device=device)
    xs, mus = pc_settle(x0, p, arch, obj, target, av, dt=dt, steps=steps,
                        x_lr_discount=x_lr_discount, x_lr_amplifier=x_lr_amplifier)

    acts = [x0] + [arch.f(x) for x in xs]                    # presynaptic activity per weight
    errs = [xs[l] - mus[l] for l in range(L - 1)]            # hidden errors
    top = acts[-1]
    mu_out = top @ p.Ws[L - 1] if p.bs[L - 1] is None else top @ p.Ws[L - 1] + p.bs[L - 1]
    errs.append(output_error(mu_out, target, obj, av))       # output error

    if opt is None:                                   # raw local update, lr applied directly
        for i in range(L):
            if f"W{i + 1}" in freeze:
                continue
            p.Ws[i] += lr * (acts[i].t() @ errs[i]) / scale
            if p.bs[i] is not None and f"b{i + 1}" not in freeze:
                p.bs[i] += lr * errs[i].sum(0) / scale
    else:                                             # identical local gradients, via torch
        opt.zero_grad()
        for i in range(L):
            frozen = f"W{i + 1}" in freeze
            p.Ws[i].grad = torch.zeros_like(p.Ws[i]) if frozen else -(acts[i].t() @ errs[i]) / n
            if p.bs[i] is not None:
                p.bs[i].grad = torch.zeros_like(p.bs[i]) if frozen else -errs[i].mean(0)
        opt.step()
    if return_delta:
        return dict(displacement=float(sum(e.abs().mean() for e in errs[:-1]) / max(1, L - 1)))


def pc_predict(x, p, arch=UNIFIED_ARCH, raw=False):
    """Test time: nothing clamped, so the equilibrium IS the feedforward pass."""
    _, out = forward(x, p, arch)
    return out if raw else out.argmax(1)


def pc_features(x, p, arch=UNIFIED_ARCH):
    return hidden_code(x, p, arch)
