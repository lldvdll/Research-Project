"""Equilibrium Propagation (Scellier & Bengio 2017) -- MULTI-LAYER.

Hopfield-style energy over the free state s_1 ... s_L (s_L is the output):

    E = sum_l 1/2|s_l|^2  -  s_1.(x W_1)  -  sum_{l>=2} s_l.(f(s_{l-1}) W_l)

Two settlings per update: a free phase, then a nudged phase warm-started from it in which
the output is pulled toward the target by beta. Learning:

    dW = (grad_nudged - grad_free) / beta

a finite-difference estimate of the cost gradient, exact as beta -> 0 (Millidge et al. 2022).
Verified numerically for both tanh and sigmoid in tests/test_numpy_mirror.py; sigmoid gives
the same convergence rate but max|f'| = 0.25 against tanh's 1.0, so it needs a proportionally
LARGER learning rate.

beta is NOT a learning rate. It is a finite-difference step size in the denominator, and its
bias grows with its size (Laborieux et al. 2021): measured rel. error ~42% at beta=0.3,
~27% at 0.1, ~1.6% at 0.005.
"""
import torch
from .model import (Arch, Objective, UNIFIED_ARCH, UNIFIED_OBJ, init_params, flatten,
                    make_target, active_vector, output_error, batch_scale)


def eqprop_init(in_dim=196, hidden=64, out_dim=10, seed=0, device="cpu", arch=None):
    """Always returns a Params with grads enabled. See the note in pc_init."""
    if arch is None:
        arch = Arch(in_dim=in_dim, hidden=hidden, out_dim=out_dim, act="tanh", bias=False)
    return init_params(arch, seed=seed, device=device).requires_grad_(True)


def eqprop_energy(x, states, p, arch):
    """states = [s_1, ..., s_L]; s_L is the output layer."""
    E = sum(0.5 * (s ** 2).sum() for s in states)
    prev = x
    for i, s in enumerate(states):
        drive = prev @ p.Ws[i] if p.bs[i] is None else prev @ p.Ws[i] + p.bs[i]
        E = E - (s * drive).sum()
        prev = arch.f(s)
    return E


def eqprop_settle(x, p, arch, obj=None, target=None, active_vec=None, beta=0.0, dt=0.3,
                  max_steps=800, settle_tol=1e-4, init=None,
                  device="cpu", return_steps=False):
    """Relax all layers until the state stops moving: until one step displaces it by less than
    `settle_tol` of its own norm. The standard relative-residual stopping rule.

    A test on the STATE, not on the rate at which movement is improving. The previous rule
    stopped once per-step movement had failed to improve by min_delta for `settle_patience`
    consecutive steps, and experiment 50 measured both of its failure modes:

      it fires on a temporary plateau -- at initialisation, seed 2 needed 529 steps to reach
      equilibrium and the rule stopped at 178, so the weight update was computed from a state
      a third of the way there;
      and it overshoots everywhere else by 3-4x, because movement goes on decaying long after
      the state has arrived.

    A relative displacement test does neither: it is small only once the state has settled.

    CHOOSING settle_tol, AND WHEN IT MUST BE RE-DERIVED. Near the fixed point the error decays
    geometrically, so stopping at move <= tol*|state| leaves a distance of roughly tol/(dt*L)
    from equilibrium, where L is the slowest curvature of the energy. The tolerance is
    therefore NOT dimensionless in dt and not a universal constant: 1e-4 is calibrated for
    dt=0.3 at the protocol's architecture, where experiment 50 measures it stopping at worst
    0.57% from the settled state -- against the 2% that experiment calls settled, so a 3.5x
    safety margin. Change dt, the width or the depth and re-run experiment 50, which re-derives
    it. Calibrate on that DISTANCE, never on the step count: step counts are discretised and
    the distance falls steeply near convergence, so a step-count test is over-sensitive
    exactly at the boundary where the choice is made.

    `return_steps` reports whether it converged or hit max_steps -- log it as a validity gate.
    """
    with torch.enable_grad():
        x = flatten(x)
        w = arch.widths
        if init is None:
            states = [torch.zeros(x.size(0), w[i + 1], device=device).requires_grad_(True)
                      for i in range(arch.n_weights)]
        else:
            states = [s.clone().requires_grad_(True) for s in init]
        used = 0
        for step in range(max_steps):
            used = step + 1
            grads = torch.autograd.grad(eqprop_energy(x, states, p, arch), states)
            grads = list(grads)
            if target is not None and beta:
                grads[-1] = grads[-1] - beta * output_error(states[-1], target, obj, active_vec)
            move = (dt * sum(g.pow(2).sum() for g in grads).sqrt()).item()
            for s, g in zip(states, grads):
                s.data -= dt * g
            if move <= settle_tol * sum(s.pow(2).sum() for s in states).sqrt().item():
                break
    out = [s.detach() for s in states]
    return (out, used, used >= max_steps) if return_steps else out


def eqprop_update(x, y_labels, p, opt, arch=UNIFIED_ARCH, obj=UNIFIED_OBJ, beta=0.3, dt=0.3,
                  max_steps=800, settle_tol=1e-4, active=None, gate_frac=None,
                  device="cpu", return_delta=False, freeze=()):
    """One EqProp weight update. `gate_frac` (advisor point 4) updates only the hidden nodes
       that move MOST under the nudge, in the TOP hidden layer, and freezes the rest."""
    x = flatten(x)
    n = x.size(0)
    target = make_target(y_labels, arch, obj, device=device)
    av = active_vector(active, arch, device=device)

    free = eqprop_settle(x, p, arch, dt=dt, max_steps=max_steps,
                         settle_tol=settle_tol, device=device)
    nudged = eqprop_settle(x, p, arch, obj=obj, target=target, active_vec=av, beta=beta,
                           dt=dt, max_steps=max_steps, settle_tol=settle_tol,
                           init=free, device=device)

    tensors = p.tensors()
    g_f = torch.autograd.grad(eqprop_energy(x, free, p, arch), tensors)
    g_n = torch.autograd.grad(eqprop_energy(x, nudged, p, arch), tensors)
    grads = [(gn - gf) / (beta * batch_scale(obj, n)) for gn, gf in zip(g_n, g_f)]

    name_of = {id(t): nm for nm, t in p.named().items() if t is not None}
    if gate_frac is not None and arch.n_hidden >= 1:
        li = arch.n_hidden - 1                                   # top hidden layer
        shift = (nudged[li] - free[li]).abs().mean(0)
        k = max(1, int(gate_frac * shift.numel()))
        mask = torch.zeros_like(shift)
        mask[torch.topk(shift, k).indices] = 1.0
        for j, t in enumerate(tensors):
            nm = name_of.get(id(t), "")
            if nm == f"W{li + 1}":
                grads[j] = grads[j] * mask.unsqueeze(0)          # columns  [in, hidden]
            elif nm == f"W{li + 2}":
                grads[j] = grads[j] * mask.unsqueeze(1)          # rows     [hidden, out]
            elif nm == f"b{li + 1}":
                grads[j] = grads[j] * mask

    opt.zero_grad()
    for t, g in zip(tensors, grads):
        t.grad = torch.zeros_like(g) if name_of.get(id(t)) in freeze else g
    opt.step()
    if return_delta:
        sat = float(arch.f(free[arch.n_hidden - 1]).abs().gt(0.95).float().mean()) \
            if arch.n_hidden >= 1 else 0.0
        return dict(shift=float(sum((n_ - f_).abs().mean()
                                    for n_, f_ in zip(nudged[:-1], free[:-1]))),
                    saturation=sat)


def eqprop_predict(x, p, arch=UNIFIED_ARCH, dt=0.3, max_steps=800, settle_tol=1e-4,
                   device="cpu", raw=False):
    states = eqprop_settle(x, p, arch, dt=dt, max_steps=max_steps,
                           settle_tol=settle_tol, device=device)
    return states[-1] if raw else states[-1].argmax(1)


def eqprop_features(x, p, arch=UNIFIED_ARCH, dt=0.3, max_steps=800, settle_tol=1e-4,
                    device="cpu"):
    states = eqprop_settle(x, p, arch, dt=dt, max_steps=max_steps,
                           settle_tol=settle_tol, device=device)
    return arch.f(states[arch.n_hidden - 1])


def eqprop_free_output_scale(x, p, arch=UNIFIED_ARCH, **kw):
    """Safety gate: how large are the free-phase output units? If they settle far below the
       target magnitude, the loss will drive large weights and saturate the network."""
    states = eqprop_settle(x, p, arch, **kw)
    y = states[-1]
    return dict(mean_abs=y.abs().mean().item(), max_abs=y.abs().max().item())
