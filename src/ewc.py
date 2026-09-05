"""Elastic Weight Consolidation [R4] Kirkpatrick et al. (2017), PNAS.

Deliberately separate from methods.py / predictive_coding.py rather than threaded into them:
duplicating the two update loops costs a little repetition and buys zero risk to the four rules
already in use.

    F_i           diagonal empirical Fisher: mean over examples of (dL/dtheta_i)^2, using the
                  SAME loss (model.loss_value) every rule already trains against -- squared error
                  read as a Gaussian log-likelihood, so this is a property of the loss surface at
                  the current weights, not of whichever rule produced them. Uses the true label
                  (the common "empirical Fisher" approximation), not a label sampled from the
                  model's own output distribution.
    anchor        weight snapshot at the moment the penalty is switched on (task 1's end).
    penalty       (lam/2) * sum_i F_i * (theta_i - anchor_i)^2, added to whatever objective the
                  base rule already trains with.

Only ONE previous task is ever anchored against here (this project runs two tasks total), so
there is no running accumulation across more than one boundary -- that is what the original
method needs for 3+ tasks and this does not.

`ewc_state` is a plain mutable dict, filled by `set_anchor` at the task-1/task-2 boundary and
empty beforehand -- the same mutable-holder-via-closure pattern `handle["freeze"]` already uses,
so a script wires it up with an `on_task_end` hook exactly as 130/131 already do for freezing and
NCM prototypes. Empty state = penalty is a no-op, so backprop_ewc/pc_ewc are numerically
IDENTICAL to backprop/pc during task 1.
"""
import torch

from .model import (forward, make_target, active_vector, output_error, loss_value,
                    batch_scale, hidden_code, init_params, replace)
from .methods import make_optimizer, _apply_freeze, _publish, _spec


def fisher_information(p, arch, obj, x, y, device="cpu"):
    """Diagonal empirical Fisher over the examples in (x, y), one example at a time -- a
    per-batch mean-gradient-squared would underestimate the true diagonal, per-example and
    averaged after squaring is the standard (Kirkpatrick et al.) approximation.

    Returns {name: tensor}, same shapes as p.named(), for every non-None tensor in p.
    """
    named = {k: v for k, v in p.named().items() if v is not None}
    prior_rg = {k: v.requires_grad for k, v in named.items()}
    for v in named.values():
        v.requires_grad_(True)

    fisher = {k: torch.zeros_like(v) for k, v in named.items()}
    keys, tensors = list(named.keys()), list(named.values())
    n = x.size(0)
    for i in range(n):
        xi, yi = x[i:i + 1], y[i:i + 1]
        target = make_target(yi, arch, obj, device=device)
        _, out = forward(xi, p, arch)
        loss = loss_value(out, target, obj)
        grads = torch.autograd.grad(loss, tensors, retain_graph=False, allow_unused=True)
        for k, g in zip(keys, grads):
            if g is not None:
                fisher[k] += g.detach() ** 2
    for k in fisher:
        fisher[k] /= max(n, 1)
    for k, v in named.items():
        v.requires_grad_(prior_rg[k])
    return fisher


def set_anchor(ewc_state, p, importance, lam):
    """Populate `ewc_state` in place: from this call on, the penalty is active."""
    named = {k: v for k, v in p.named().items() if v is not None}
    ewc_state["importance"] = {k: v.detach().clone() for k, v in importance.items()}
    ewc_state["anchor"] = {k: v.detach().clone() for k, v in named.items()}
    ewc_state["lam"] = float(lam)


def _penalty_grad(name, w, ewc_state):
    """lam * F_i * (w - anchor_i) for one named tensor, or None if EWC is inactive / this
    tensor has no recorded importance (e.g. a bias the base rule leaves as None elsewhere)."""
    if not ewc_state or "importance" not in ewc_state:
        return None
    F = ewc_state["importance"].get(name)
    a = ewc_state["anchor"].get(name)
    if F is None or a is None:
        return None
    return ewc_state["lam"] * F * (w.detach() - a)


def add_ewc_grad(p, ewc_state):
    """Backprop path: add the penalty gradient onto .grad, in place, before opt.step(). A
    no-op while ewc_state is empty, so backprop_ewc reproduces plain backprop exactly."""
    if not ewc_state:
        return
    for name, w in p.named().items():
        if w is None or w.grad is None:
            continue
        g = _penalty_grad(name, w, ewc_state)
        if g is not None:
            w.grad = w.grad + g


def pc_update_ewc(x, y_labels, p, arch, obj, lr, dt, steps, ewc_state, active=None,
                  device="cpu", freeze=(), return_delta=False):
    """PC's raw (opt=None) update loop, duplicated from predictive_coding.pc_update, with the
    EWC penalty subtracted as an extra gradient-descent term on the same raw update. Only the
    raw path is duplicated -- optimizer=='sgd' is this project's default everywhere PC is used,
    and the torch-optimizer branch was for parity experiments this rule doesn't need here."""
    from .predictive_coding import pc_settle

    x0 = x.reshape(x.size(0), -1)
    n, L = x0.size(0), arch.n_weights
    scale = batch_scale(obj, n)
    target = make_target(y_labels, arch, obj, device=device)
    av = active_vector(active, arch, device=device)
    xs, mus = pc_settle(x0, p, arch, obj, target, av, dt=dt, steps=steps)

    acts = [x0] + [arch.f(x) for x in xs]
    errs = [xs[l] - mus[l] for l in range(L - 1)]
    top = acts[-1]
    mu_out = top @ p.Ws[L - 1] if p.bs[L - 1] is None else top @ p.Ws[L - 1] + p.bs[L - 1]
    errs.append(output_error(mu_out, target, obj, av))

    for i in range(L):
        wname, bname = f"W{i + 1}", f"b{i + 1}"
        if wname not in freeze:
            dW = lr * (acts[i].t() @ errs[i]) / scale
            g = _penalty_grad(wname, p.Ws[i], ewc_state)
            if g is not None:
                dW = dW - lr * g
            p.Ws[i] += dW
        if p.bs[i] is not None and bname not in freeze:
            db = lr * errs[i].sum(0) / scale
            g = _penalty_grad(bname, p.bs[i], ewc_state)
            if g is not None:
                db = db - lr * g
            p.bs[i] += db
    if return_delta:
        return dict(displacement=float(sum(e.abs().mean() for e in errs[:-1]) / max(1, L - 1)))


# ------------------------------------------------------------------ builders (duplicate the
# relevant make_backprop / make_pc bodies; see module docstring for why)

def make_backprop_ewc(in_dim=196, hidden=64, out_dim=10, lr=0.05, optimizer="sgd", seed=0,
                      device="cpu", arch=None, obj=None, handle=None, **_):
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    freeze, ewc_state = set(), {}

    def train_step(x, y, active=None):
        n = x.size(0)
        target = make_target(y, arch, obj, device=device)
        av = active_vector(active, arch, device=device)
        _, out = forward(x, p, arch)
        e = output_error(out, target, obj, av)
        opt.zero_grad()
        out.backward(-e / batch_scale(obj, n))
        _apply_freeze(p, freeze)
        add_ewc_grad(p, ewc_state)
        opt.step()

    def predict(x, raw=False):
        with torch.no_grad():
            _, out = forward(x, p, arch)
        return out if raw else out.argmax(1)

    def features(x):
        with torch.no_grad():
            return hidden_code(x, p, arch)

    _publish(handle, p, arch, obj, features, freeze=freeze)
    if handle is not None:
        handle["ewc"] = ewc_state
    return train_step, predict


def make_pc_ewc(in_dim=196, hidden=64, out_dim=10, lr=0.05, dt=0.1, steps=50, seed=0,
               device="cpu", arch=None, obj=None, handle=None, **_):
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device)
    diag, freeze, ewc_state = {}, set(), {}

    def train_step(x, y, active=None):
        d = pc_update_ewc(x, y, p, arch, obj, lr, dt, steps, ewc_state, active=active,
                          device=device, freeze=freeze, return_delta=True)
        diag["displacement"] = d["displacement"]

    def predict(x, raw=False):
        from .predictive_coding import pc_predict
        return pc_predict(x, p, arch, raw=raw)

    def features(x):
        return hidden_code(x, p, arch)

    _publish(handle, p, arch, obj, features, diag=diag, freeze=freeze)
    if handle is not None:
        handle["ewc"] = ewc_state
    return train_step, predict


# ------------------------------------------------------------------ Synaptic Intelligence
# [R12] Zenke, Poole & Ganguli (2017), ICML. Same penalty machinery as EWC above (set_anchor /
# _penalty_grad / add_ewc_grad / pc_update_ewc are reused unchanged) -- the only thing that
# differs is HOW `importance` is computed: not a one-shot Fisher snapshot, but a running
# path-integral accumulated over every update of task 1, finalised into an importance dict at
# the task boundary by `finalize_si`, then handed to the exact same set_anchor EWC uses.
#
# Zenke's definition: Omega_i = sum over every update of -g_i(t) * dw_i(t), where g_i is the
# gradient the update actually used and dw_i is the change it actually made -- i.e. credit for
# how much that step's movement reduced the loss, accumulated over the WHOLE trajectory. Unlike
# Fisher (a property of the final weights alone, independent of the rule), this depends on the
# actual path taken -- backprop and PC will generally disagree here even from the same start and
# end point, because their per-step directions differ. That is expected, not a bug.

def finalize_si(ewc_state, omega_state, p, lam, xi=0.001):
    """Omega -> normalised importance c_i = Omega_i / ((w_i(T)-w_i(0))^2 + xi), then hand it to
    set_anchor -- from this point on SI and EWC are applied identically."""
    named = {k: v for k, v in p.named().items() if v is not None}
    importance = {}
    for k, v in named.items():
        dw_total = v.detach() - omega_state["w0"][k]
        importance[k] = omega_state["omega"][k] / (dw_total ** 2 + xi)
    set_anchor(ewc_state, p, importance, lam)


def make_backprop_si(in_dim=196, hidden=64, out_dim=10, lr=0.05, optimizer="sgd", seed=0,
                     device="cpu", arch=None, obj=None, handle=None, **_):
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    freeze, ewc_state = set(), {}
    named = {k: v for k, v in p.named().items() if v is not None}
    omega_state = {"omega": {k: torch.zeros_like(v) for k, v in named.items()},
                  "w0": {k: v.detach().clone() for k, v in named.items()}}

    def train_step(x, y, active=None):
        n = x.size(0)
        target = make_target(y, arch, obj, device=device)
        av = active_vector(active, arch, device=device)
        _, out = forward(x, p, arch)
        e = output_error(out, target, obj, av)
        opt.zero_grad()
        out.backward(-e / batch_scale(obj, n))
        _apply_freeze(p, freeze)
        add_ewc_grad(p, ewc_state)      # no-op until finalize_si has run (task 1 = accumulate only)
        grad_before = {k: (v.grad.detach().clone() if v.grad is not None else None)
                      for k, v in named.items()}
        w_before = {k: v.detach().clone() for k, v in named.items()}
        opt.step()
        for k, v in named.items():
            if grad_before[k] is not None:
                omega_state["omega"][k] += -grad_before[k] * (v.detach() - w_before[k])

    def predict(x, raw=False):
        with torch.no_grad():
            _, out = forward(x, p, arch)
        return out if raw else out.argmax(1)

    def features(x):
        with torch.no_grad():
            return hidden_code(x, p, arch)

    _publish(handle, p, arch, obj, features, freeze=freeze)
    if handle is not None:
        handle["ewc"] = ewc_state
        handle["si_omega"] = omega_state
    return train_step, predict


def pc_update_si(x, y_labels, p, arch, obj, lr, dt, steps, ewc_state, omega_state, active=None,
                 device="cpu", freeze=(), return_delta=False):
    """pc_update_ewc, plus SI's Omega accumulation. PC's raw update has no explicit gradient to
    record: dW_i = lr*(acts^T err)_i IS the actual step, so treating it the way plain SGD relates
    step to gradient (dw = -lr*g, i.e. g = -dW/lr) gives Omega_i += -g_i*dW_i = dW_i^2/lr --
    algebraically simpler than reconstructing g explicitly, so that is what is accumulated."""
    from .predictive_coding import pc_settle

    x0 = x.reshape(x.size(0), -1)
    n, L = x0.size(0), arch.n_weights
    scale = batch_scale(obj, n)
    target = make_target(y_labels, arch, obj, device=device)
    av = active_vector(active, arch, device=device)
    xs, mus = pc_settle(x0, p, arch, obj, target, av, dt=dt, steps=steps)

    acts = [x0] + [arch.f(x) for x in xs]
    errs = [xs[l] - mus[l] for l in range(L - 1)]
    top = acts[-1]
    mu_out = top @ p.Ws[L - 1] if p.bs[L - 1] is None else top @ p.Ws[L - 1] + p.bs[L - 1]
    errs.append(output_error(mu_out, target, obj, av))

    for i in range(L):
        wname, bname = f"W{i + 1}", f"b{i + 1}"
        if wname not in freeze:
            dW = lr * (acts[i].t() @ errs[i]) / scale
            g = _penalty_grad(wname, p.Ws[i], ewc_state)
            if g is not None:
                dW = dW - lr * g
            p.Ws[i] += dW
            omega_state["omega"][wname] += dW.detach() ** 2 / lr
        if p.bs[i] is not None and bname not in freeze:
            db = lr * errs[i].sum(0) / scale
            g = _penalty_grad(bname, p.bs[i], ewc_state)
            if g is not None:
                db = db - lr * g
            p.bs[i] += db
            omega_state["omega"][bname] += db.detach() ** 2 / lr
    if return_delta:
        return dict(displacement=float(sum(e.abs().mean() for e in errs[:-1]) / max(1, L - 1)))


def make_pc_si(in_dim=196, hidden=64, out_dim=10, lr=0.05, dt=0.1, steps=50, seed=0,
              device="cpu", arch=None, obj=None, handle=None, **_):
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device)
    diag, freeze, ewc_state = {}, set(), {}
    named = {k: v for k, v in p.named().items() if v is not None}
    omega_state = {"omega": {k: torch.zeros_like(v) for k, v in named.items()},
                  "w0": {k: v.detach().clone() for k, v in named.items()}}

    def train_step(x, y, active=None):
        d = pc_update_si(x, y, p, arch, obj, lr, dt, steps, ewc_state, omega_state, active=active,
                         device=device, freeze=freeze, return_delta=True)
        diag["displacement"] = d["displacement"]

    def predict(x, raw=False):
        from .predictive_coding import pc_predict
        return pc_predict(x, p, arch, raw=raw)

    def features(x):
        return hidden_code(x, p, arch)

    _publish(handle, p, arch, obj, features, diag=diag, freeze=freeze)
    if handle is not None:
        handle["ewc"] = ewc_state
        handle["si_omega"] = omega_state
    return train_step, predict
