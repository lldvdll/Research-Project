"""k-winner-take-all lateral competition, simplified from Ororbia et al. (2022) "Lifelong Neural
Predictive Coding" (the Sequential Neural Coding Network's core competition mechanism), with the
task-context memory/retrieval dropped -- see the design discussion this was agreed against:
using a task descriptor at test time would make this Task-IL, not compatible with the
Domain-IL/Class-IL protocol this project studies. What is kept is pure, task-agnostic
per-example competition: only the k most active hidden units transmit to the next layer.

Deliberately separate from methods.py / model.py / predictive_coding.py, same reasoning as
ewc.py: this needs to change the FORWARD pass (which layer between input and output nothing
else in this project needs to touch), so it gets its own small forward variant rather than
threading a masking option through the shared one. ONE HIDDEN LAYER ONLY -- this project's depth
is fixed at 1 for the whole report series, and asserting that here means a future depth
experiment fails loudly instead of silently masking the wrong layer.

    kwta(a, k)        zero every activation except the k largest, per example (last dim).
    forward_kwta      model.forward's 1-hidden-layer case, with kwta applied to the hidden
                      layer's POST-activation code -- competition is over what transmits
                      downstream, not over the raw pre-activation drive.

PC's own settling (predictive_coding.pc_settle) is reused UNCHANGED: competition is applied only
to the settled hidden state, after relaxation finishes, before it is used for the output
error/W2 update -- not inside the relaxation loop itself. W1's own update (which depends only on
the input and the hidden layer's error, not on what the hidden layer transmits onward) is
therefore unaffected by k, exactly as backprop's W1 update is.
"""
import torch

from .model import make_target, active_vector, output_error, loss_value, batch_scale, replace, init_params
from .methods import make_optimizer, _apply_freeze, _publish, _spec


def kwta(a, k):
    """Zero all but the k largest activations per example. k >= width is a no-op (k-WTA off)."""
    if k >= a.size(-1):
        return a
    kth = torch.topk(a, k, dim=-1).values[:, -1:]
    return torch.where(a >= kth, a, torch.zeros_like(a))


def forward_kwta(x, p, arch, k):
    """model.forward's one-hidden-layer case, with k-WTA on the hidden layer's transmitted
    (post-activation) code. Returns (z1, out) -- z1 is the RAW pre-activation hidden state, same
    convention as model.forward, so probes expecting it (e.g. hidden_pre-style diagnostics)
    still see the undistorted drive; only what reaches the output layer is competed over."""
    assert arch.n_weights == 2, "k-WTA is implemented for one hidden layer only (this series' depth)"
    a = x.reshape(x.size(0), -1)
    z1 = a @ p.Ws[0] if p.bs[0] is None else a @ p.Ws[0] + p.bs[0]
    h = kwta(arch.f(z1), k)
    out = h @ p.Ws[1] if p.bs[1] is None else h @ p.Ws[1] + p.bs[1]
    return z1, out


def features_kwta(x, p, arch, k):
    """The competed (post-kwta) hidden code -- what NCM-style probes should see if they want to
    know what actually reaches the output, not the uncompeted drive."""
    a = x.reshape(x.size(0), -1)
    z1 = a @ p.Ws[0] if p.bs[0] is None else a @ p.Ws[0] + p.bs[0]
    return kwta(arch.f(z1), k)


def make_backprop_kwta(in_dim=196, hidden=64, out_dim=10, lr=0.05, k=None, optimizer="sgd",
                       seed=0, device="cpu", arch=None, obj=None, handle=None, **_):
    """k defaults to `hidden` (=off, keeps every unit) if not given -- an experiment sweeping k
    should always pass it explicitly."""
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    k = arch.hidden if k is None else k
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    freeze = set()

    def train_step(x, y, active=None):
        n = x.size(0)
        target = make_target(y, arch, obj, device=device)
        av = active_vector(active, arch, device=device)
        _, out = forward_kwta(x, p, arch, k)
        e = output_error(out, target, obj, av)
        opt.zero_grad()
        out.backward(-e / batch_scale(obj, n))
        _apply_freeze(p, freeze)
        opt.step()

    def predict(x, raw=False):
        with torch.no_grad():
            _, out = forward_kwta(x, p, arch, k)
        return out if raw else out.argmax(1)

    def features(x):
        with torch.no_grad():
            return features_kwta(x, p, arch, k)

    _publish(handle, p, arch, obj, features, freeze=freeze)
    return train_step, predict


def pc_update_kwta(x, y_labels, p, arch, obj, lr, dt, steps, k, active=None, device="cpu",
                   freeze=(), return_delta=False):
    """pc_update's raw loop, with k-WTA applied to the SETTLED hidden state before it is used
    for the output error / W2's update. pc_settle itself (the relaxation) is reused unchanged."""
    from .predictive_coding import pc_settle

    x0 = x.reshape(x.size(0), -1)
    n, L = x0.size(0), arch.n_weights
    assert L == 2, "k-WTA is implemented for one hidden layer only (this series' depth)"
    scale = batch_scale(obj, n)
    target = make_target(y_labels, arch, obj, device=device)
    av = active_vector(active, arch, device=device)
    xs, mus = pc_settle(x0, p, arch, obj, target, av, dt=dt, steps=steps)

    acts = [x0, kwta(arch.f(xs[0]), k)]                # competition on what transmits to W2
    errs = [xs[0] - mus[0]]                             # W1's own error is unaffected by k
    mu_out = acts[1] @ p.Ws[1] if p.bs[1] is None else acts[1] @ p.Ws[1] + p.bs[1]
    errs.append(output_error(mu_out, target, obj, av))

    for i in range(L):
        wname, bname = f"W{i + 1}", f"b{i + 1}"
        if wname not in freeze:
            p.Ws[i] += lr * (acts[i].t() @ errs[i]) / scale
        if p.bs[i] is not None and bname not in freeze:
            p.bs[i] += lr * errs[i].sum(0) / scale
    if return_delta:
        return dict(displacement=float((xs[0] - mus[0]).abs().mean()))


def make_pc_kwta(in_dim=196, hidden=64, out_dim=10, lr=0.05, dt=0.1, steps=50, k=None, seed=0,
                 device="cpu", arch=None, obj=None, handle=None, **_):
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    k = arch.hidden if k is None else k
    p = init_params(arch, seed=seed, device=device)
    diag, freeze = {}, set()

    def train_step(x, y, active=None):
        d = pc_update_kwta(x, y, p, arch, obj, lr, dt, steps, k, active=active, device=device,
                           freeze=freeze, return_delta=True)
        diag["displacement"] = d["displacement"]

    def predict(x, raw=False):
        with torch.no_grad():
            _, out = forward_kwta(x, p, arch, k)
        return out if raw else out.argmax(1)

    def features(x):
        with torch.no_grad():
            return features_kwta(x, p, arch, k)

    _publish(handle, p, arch, obj, features, diag=diag, freeze=freeze)
    return train_step, predict
