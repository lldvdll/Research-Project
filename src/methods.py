"""Method builders. Each returns (train_step, predict).

    train_step(x, y, active=None)   one update. `active` = the class indices present in the
                                    current task; used only when obj.mask is True.
    predict(x, raw=False)           class indices, or raw pre-argmax outputs.

Optional `handle` dict is filled with:
    handle["params"]    -> the Params object          (weight-space analysis)
    handle["features"]  -> fn(x) -> hidden code       (NCM / CKA probes)
    handle["arch"], handle["obj"], handle["diag"]

CONTROLLING THE COMPARISON
--------------------------
Pass `arch=` and `obj=` and EVERY method gets that architecture and that output structure:

    from src.model import UNIFIED_ARCH, UNIFIED_OBJ
    build_method("pc", arch=UNIFIED_ARCH, obj=UNIFIED_OBJ, ...)

Pass neither and each method falls back to model.LEGACY_SPEC -- i.e. what it used before
unification (backprop: ReLU + cross-entropy + biases; pc: tanh + squared error; eqprop:
tanh + hinge with +-1 targets). That default exists ONLY for backward compatibility with
experiments 11-15. New experiments should always pass both explicitly.
"""
import torch

from .model import (Arch, Objective, Params, UNIFIED_ARCH, UNIFIED_OBJ, LEGACY_SPEC,
                    init_params, flatten, forward, hidden_code, make_target, active_vector,
                    output_error, loss_value, batch_scale, resolve_freeze, replace)
from .predictive_coding import pc_update, pc_predict, pc_settle
from .eqprop import eqprop_update, eqprop_settle


def make_optimizer(tensors, name="sgd", lr=0.05, momentum=0.9):
    """One optimiser factory for every rule, so 'which optimiser' becomes an experimental
       variable rather than a hard-coded asymmetry. Song & Bogacz's library permits any torch
       optimiser for the weights; ours previously forced plain SGD everywhere."""
    if name == "sgd":
        return torch.optim.SGD(tensors, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(tensors, lr=lr, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(tensors, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")


def _spec(name, arch, obj):
    """Resolve (arch, obj), falling back to the pre-unification defaults for this method."""
    la, lo = LEGACY_SPEC.get(name.replace("_gated", "").replace("_replay", ""),
                             (UNIFIED_ARCH, UNIFIED_OBJ))
    return (arch or la), (obj or lo)


def _apply_freeze(p, freeze):
    """Zero the gradient of any parameter selected by `freeze` (a mutable set of specs), so an
       experiment can freeze part of the network mid-run via handle["freeze"].add("W1").
       Multi-layer aware: see model.resolve_freeze for the accepted specs."""
    if not freeze:
        return
    wi, bi = resolve_freeze(freeze, len(p.W))
    for i in wi:
        if p.W[i].grad is not None:
            p.W[i].grad.zero_()
    if p.b is not None:
        for i in (bi | wi):
            if i < len(p.b) and p.b[i].grad is not None:
                p.b[i].grad.zero_()


def _publish(handle, p, arch, obj, features, diag=None, freeze=None):
    if handle is not None:
        handle["freeze"] = freeze if freeze is not None else set()
        handle["params"] = p
        handle["arch"] = arch
        handle["obj"] = obj
        handle["features"] = features
        handle["diag"] = diag if diag is not None else {}


# ------------------------------------------------------------------ backprop
def make_backprop(in_dim=196, hidden=64, out_dim=10, lr=0.05, optimizer="sgd", seed=0,
                  device="cpu", arch=None, obj=None, handle=None, **_):
    arch, obj = _spec("backprop", arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    freeze = set()

    def train_step(x, y, active=None):
        n = x.size(0)
        target = make_target(y, arch, obj, device=device)
        av = active_vector(active, arch, device=device)
        _, out = forward(x, p, arch)
        e = output_error(out, target, obj, av)
        opt.zero_grad()
        out.backward(-e / batch_scale(obj, n))          # dL/dout = -e  (identical signal to PC's e2)
        _apply_freeze(p, freeze)
        opt.step()

    def predict(x, raw=False):
        with torch.no_grad():
            _, out = forward(x, p, arch)
        return out if raw else out.argmax(1)

    def features(x):
        with torch.no_grad():
            return hidden_code(x, p, arch)

    _publish(handle, p, arch, obj, features, freeze=freeze)
    return train_step, predict


# ------------------------------------------------------------------ replay
def make_replay(train_data, class_idx, in_dim=196, hidden=64, out_dim=10, lr=0.05,
                optimizer="sgd",
                per_class=20, replay_frac=None, buffer_seed=None, seed=0, device="cpu",
                arch=None, obj=None, handle=None, **_):
    """Backprop + a stored-example buffer. NOT a learning rule -- a reference upper bound.

    replay_frac = None  : legacy behaviour, replay is APPENDED (batch grows to 2x, the model
                          sees strictly more data per step -- a confound).
    replay_frac = 0.5   : batch size held FIXED; half the batch is replay, so the model sees
                          FEWER new examples per step. Conservative; prefer this.
    buffer_seed         : dedicated RNG for which examples are stored and sampled, so buffer
                          composition is its own variance source (defaults to `seed`).
    """
    arch, obj = _spec("replay", arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    g = torch.Generator(device="cpu").manual_seed(int(seed if buffer_seed is None else buffer_seed))
    mem_x, mem_y, seen, freeze = [], [], set(), set()

    def _store(c):
        pool = class_idx[c]
        pick = pool[torch.randperm(len(pool), generator=g)[:per_class]]
        mem_x.append(torch.stack([train_data[i][0] for i in pick.tolist()]).to(device).reshape(len(pick), -1))
        mem_y.append(torch.full((len(pick),), c, device=device))

    def train_step(x, y, active=None):
        for c in y.unique().tolist():
            if c not in seen:
                seen.add(c)
                _store(c)
        xf = flatten(x)
        if mem_x:
            rx, ry = torch.cat(mem_x), torch.cat(mem_y)
            k = xf.size(0) if replay_frac is None else int(round(replay_frac * xf.size(0)))
            k = min(k, len(ry))
            if k > 0:
                s = torch.randperm(len(ry), generator=g)[:k]
                keep = xf.size(0) if replay_frac is None else xf.size(0) - k
                xf = torch.cat([xf[:keep], rx[s]])
                y = torch.cat([y[:keep], ry[s]])
        n = xf.size(0)
        target = make_target(y, arch, obj, device=device)
        # replayed classes are legitimately present in this batch, so they are ACTIVE
        av = active_vector(None if active is None else sorted(set(active) | seen), arch, device=device)
        _, out = forward(xf, p, arch)
        e = output_error(out, target, obj, av)
        opt.zero_grad()
        out.backward(-e / batch_scale(obj, n))
        _apply_freeze(p, freeze)
        opt.step()

    def predict(x, raw=False):
        with torch.no_grad():
            _, out = forward(x, p, arch)
        return out if raw else out.argmax(1)

    def features(x):
        with torch.no_grad():
            return hidden_code(x, p, arch)

    _publish(handle, p, arch, obj, features, diag={"buffer_classes": seen}, freeze=freeze)
    return train_step, predict


# ------------------------------------------------------------------ predictive coding
def make_pc(in_dim=196, hidden=64, out_dim=10, lr=0.05, dt=0.1, steps=50, optimizer="sgd",
            seed=0, device="cpu", arch=None, obj=None, handle=None, **_):
    """steps=0 disables relaxation entirely -> the 'PC without prospective configuration'
       control that should collapse onto backprop (experiment 23)."""
    arch, obj = _spec("pc", arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device)
    # PC computes its updates locally and explicitly. Routing them through a torch optimiser
    # lets momentum/Adam apply to PC exactly as to backprop, so the optimiser is a controlled
    # variable. optimizer="sgd" reproduces the previous raw `W += lr * dW` behaviour exactly.
    opt = None
    if optimizer != "sgd":
        p.requires_grad_(True)
        opt = make_optimizer(p.tensors(), optimizer, lr)
    diag, freeze = {}, set()

    def train_step(x, y, active=None):
        d = pc_update(x, y, p, arch=arch, obj=obj, lr=lr, dt=dt, steps=steps, active=active,
                      device=device, return_delta=True, freeze=freeze, opt=opt)
        diag["displacement"] = d["displacement"]

    def predict(x, raw=False):
        return pc_predict(x, p, arch, raw=raw)

    def features(x):
        return hidden_code(x, p, arch)

    _publish(handle, p, arch, obj, features, diag=diag, freeze=freeze)
    return train_step, predict


# ------------------------------------------------------------------ equilibrium propagation
def make_eqprop(in_dim=196, hidden=64, out_dim=10, lr=0.005, beta=0.3, dt=0.3, max_steps=500,
                settle_patience=30, gate_frac=None, optimizer="sgd", seed=0, device="cpu",
                arch=None, obj=None, handle=None, **_):
    arch, obj = _spec("eqprop", arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    diag, freeze = {}, set()

    def train_step(x, y, active=None):
        d = eqprop_update(x, y, p, opt, arch=arch, obj=obj, beta=beta, dt=dt,
                          max_steps=max_steps, settle_patience=settle_patience,
                          active=active, gate_frac=gate_frac, device=device,
                          return_delta=True, freeze=freeze)
        diag.update(d)

    def predict(x, raw=False):
        states = eqprop_settle(x, p, arch, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)
        return states[-1] if raw else states[-1].argmax(1)

    def features(x):
        from .eqprop import eqprop_features
        return eqprop_features(x, p, arch, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)

    _publish(handle, p, arch, obj, features, diag=diag, freeze=freeze)
    return train_step, predict


def make_eqprop_gated(gate_frac=0.3, **kw):
    return make_eqprop(gate_frac=gate_frac, **kw)


# ------------------------------------------------------------------ dispatch
METHOD_DEFAULTS = {
    "backprop":     dict(lr=0.05),
    "replay":       dict(lr=0.05, per_class=20),
    "eqprop":       dict(lr=0.005, beta=0.3, dt=0.3, max_steps=500, settle_patience=30),
    "pc":           dict(lr=0.05, dt=0.1, steps=50),
    "eqprop_gated": dict(lr=0.005, beta=0.3, dt=0.3, max_steps=500, settle_patience=30,
                         gate_frac=0.3),
}

_NEEDS_DATA = {"replay"}
_BUILDERS = {"backprop": make_backprop, "replay": make_replay, "eqprop": make_eqprop,
             "pc": make_pc, "eqprop_gated": make_eqprop_gated}


def build_method(name, in_dim=196, hidden=64, out_dim=10, seed=0, device="cpu",
                 train_data=None, class_idx=None, handle=None,
                 arch=None, obj=None, **overrides):
    """(train_step, predict) for `name`.

    Defaults come from METHOD_DEFAULTS; **overrides replaces any of them. Unknown keys are
    swallowed by each builder's **_, so prefer explicit arguments.
    Pass arch= and obj= to force every method onto the same specification.
    """
    if name not in _BUILDERS:
        raise ValueError(f"unknown method {name!r}; known: {sorted(_BUILDERS)}")
    kw = dict(METHOD_DEFAULTS[name])
    kw.update(overrides)
    kw.update(in_dim=in_dim, hidden=hidden, out_dim=out_dim, seed=seed, device=device,
              handle=handle, arch=arch, obj=obj)
    if name in _NEEDS_DATA:
        if train_data is None or class_idx is None:
            raise ValueError(f"{name} needs train_data and class_idx")
        return _BUILDERS[name](train_data, class_idx, **kw)
    return _BUILDERS[name](**kw)
