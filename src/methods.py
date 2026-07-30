"""Methods: each builder sets up a model and returns (train_step, predict) closures.
   Add a method = add a make_* function, then register it in METHOD_DEFAULTS / build_method.

   predict(x, raw=False) -> class indices, or raw pre-argmax outputs when raw=True.
   handle=<dict>         -> optional; if given, handle["params"] is set to the model's parameter
                            tensors. Needed for weight-space analysis (path length, update
                            direction). Purely additive: omit it and nothing changes.
"""
import torch
import torch.nn as nn
from .eqprop import eqprop_init, eqprop_update, eqprop_settle
from .predictive_coding import pc_init, pc_update, pc_predict


def make_mlp(in_dim=196, hidden=64, out_dim=10):
    return nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def make_backprop(in_dim=196, hidden=64, lr=0.1, seed=0, device="cpu", handle=None):
    torch.manual_seed(seed)
    model = make_mlp(in_dim, hidden).to(device)
    opt, lf = torch.optim.SGD(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    if handle is not None:
        handle["params"] = list(model.parameters())

    def train_step(x, y):
        opt.zero_grad(); lf(model(x), y).backward(); opt.step()

    def predict(x, raw=False):
        with torch.no_grad():
            out = model(x)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_replay(train_data, class_idx, in_dim=196, hidden=64, lr=0.1, per_class=20,
                seed=0, device="cpu", handle=None):
    """Experience-replay control: stores `per_class` examples the first time each class is seen,
       mixes an equal-sized replay sample into every batch."""
    torch.manual_seed(seed)
    model = make_mlp(in_dim, hidden).to(device)
    opt, lf = torch.optim.SGD(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    mem_x, mem_y, seen = [], [], set()
    if handle is not None:
        handle["params"] = list(model.parameters())

    def train_step(x, y):
        for c in y.unique().tolist():
            if c not in seen:
                seen.add(c)
                sel = class_idx[c][:per_class]
                mem_x.append(torch.stack([train_data[i][0] for i in sel]).to(device))
                mem_y.append(torch.full((len(sel),), c).to(device))
        if mem_x:
            rx, ry = torch.cat(mem_x), torch.cat(mem_y)
            s = torch.randperm(len(ry))[:x.size(0)]
            x, y = torch.cat([x, rx[s]]), torch.cat([y, ry[s]])
        opt.zero_grad(); lf(model(x), y).backward(); opt.step()

    def predict(x, raw=False):
        with torch.no_grad():
            out = model(x)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_eqprop(in_dim=196, hidden=64, lr=0.03, beta=0.3, dt=0.3, max_steps=500,
                settle_patience=30, seed=0, device="cpu", handle=None):
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)
    if handle is not None:
        handle["params"] = [W1, W2]

    def train_step(x, y):
        eqprop_update(x, y, W1, W2, opt, beta=beta, dt=dt, max_steps=max_steps,
                      settle_patience=settle_patience, device=device)

    def predict(x, raw=False):
        _, out = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_pc(in_dim=196, hidden=64, lr=0.05, dt=0.1, steps=50, seed=0, device="cpu", handle=None):
    """Predictive coding: settle the hidden activities toward the target, then update weights locally."""
    W1, W2 = pc_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    if handle is not None:
        handle["params"] = [W1, W2]

    def train_step(x, y):
        pc_update(x, y, W1, W2, lr=lr, dt=dt, steps=steps, device=device)

    def predict(x, raw=False):
        return pc_predict(x, W1, W2, raw=raw)

    return train_step, predict


def make_eqprop_gated(in_dim=196, hidden=64, lr=0.03, beta=0.3, dt=0.3, max_steps=500,
                      settle_patience=30, gate_frac=0.3, seed=0, device="cpu", handle=None):
    from .eqprop import eqprop_update_gated
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)
    if handle is not None:
        handle["params"] = [W1, W2]

    def train_step(x, y):
        eqprop_update_gated(x, y, W1, W2, opt, beta=beta, dt=dt, max_steps=max_steps,
                            settle_patience=settle_patience, gate_frac=gate_frac, device=device)

    def predict(x, raw=False):
        _, out = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_eqprop_replay(train_data, class_idx, in_dim=196, hidden=64, lr=0.03, beta=0.3, dt=0.3,
                       max_steps=500, settle_patience=30, per_class=20, seed=0, device="cpu",
                       handle=None):
    """EqProp with a stored real-example replay buffer mixed into each batch."""
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)
    mem_x, mem_y, seen = [], [], set()
    if handle is not None:
        handle["params"] = [W1, W2]

    def train_step(x, y):
        for c in y.unique().tolist():
            if c not in seen:
                seen.add(c)
                sel = class_idx[c][:per_class]
                mem_x.append(torch.stack([train_data[i][0] for i in sel]).to(device).reshape(len(sel), -1))
                mem_y.append(torch.full((len(sel),), c, device=device))
        xf = x.reshape(x.size(0), -1)
        if mem_x:
            rx, ry = torch.cat(mem_x), torch.cat(mem_y)
            s = torch.randperm(len(ry))[:x.size(0)]
            xf, y = torch.cat([xf, rx[s]]), torch.cat([y, ry[s]])
        eqprop_update(xf, y, W1, W2, opt, beta=beta, dt=dt, max_steps=max_steps,
                      settle_patience=settle_patience, device=device)

    def predict(x, raw=False):
        _, out = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_eqprop_synthetic(in_dim=196, hidden=64, lr=0.03, beta=0.3, dt=0.3, max_steps=500,
                          settle_patience=30, n_synth=20, gen_steps=200, seed=0, device="cpu",
                          handle=None):
    """EqProp with GENERATIVE replay: at each new class, regenerate synthetic examples of the
       already-learned classes from the model itself and mix them in."""
    from .eqprop import eqprop_generate
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)
    seen, synth_x, synth_y = [], [], []
    if handle is not None:
        handle["params"] = [W1, W2]

    def train_step(x, y):
        for c in y.unique().tolist():
            if c not in seen:
                if seen:
                    synth_x.clear(); synth_y.clear()
                    for pc_ in seen:
                        synth_x.append(eqprop_generate(W1, W2, pc_, n_synth, gen_steps=gen_steps, device=device))
                        synth_y.append(torch.full((n_synth,), pc_, device=device))
                seen.append(c)
        xf = x.reshape(x.size(0), -1)
        if synth_x:
            sx, sy = torch.cat(synth_x), torch.cat(synth_y)
            s = torch.randperm(len(sy))[:x.size(0)]
            xf, y = torch.cat([xf, sx[s]]), torch.cat([y, sy[s]])
        eqprop_update(xf, y, W1, W2, opt, beta=beta, dt=dt, max_steps=max_steps,
                      settle_patience=settle_patience, device=device)

    def predict(x, raw=False):
        _, out = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)
        return out if raw else out.argmax(1)

    return train_step, predict


# --------------------------------------------------------------------------------------
# Single dispatch point, so an experiment can set `hidden` (and any hyperparameter) for
# EVERY method uniformly instead of repeating a per-method build function in each script.
# --------------------------------------------------------------------------------------
METHOD_DEFAULTS = {
    "backprop":  dict(lr=0.05),
    "replay":    dict(lr=0.05, per_class=20),
    "eqprop":    dict(lr=0.005, beta=0.3, dt=0.3, max_steps=500, settle_patience=30),
    "pc":        dict(lr=0.05, dt=0.1, steps=50),
    "eqprop_gated":     dict(lr=0.005, beta=0.3, dt=0.3, max_steps=500, settle_patience=30, gate_frac=0.3),
    "eqprop_replay":    dict(lr=0.005, beta=0.3, dt=0.3, max_steps=500, settle_patience=30, per_class=20),
    "eqprop_synthetic": dict(lr=0.005, beta=0.3, dt=0.3, max_steps=500, settle_patience=30,
                             n_synth=20, gen_steps=200),
}

_NEEDS_DATA = {"replay", "eqprop_replay"}       # builders that need the training set + class index

_BUILDERS = {
    "backprop": make_backprop, "replay": make_replay, "eqprop": make_eqprop, "pc": make_pc,
    "eqprop_gated": make_eqprop_gated, "eqprop_replay": make_eqprop_replay,
    "eqprop_synthetic": make_eqprop_synthetic,
}


def build_method(name, in_dim=196, hidden=64, seed=0, device="cpu",
                 train_data=None, class_idx=None, handle=None, **overrides):
    """(train_step, predict) for `name`, with `hidden` applied uniformly.

    Defaults come from METHOD_DEFAULTS; **overrides replaces any of them, so an experiment can
    do build_method("eqprop", hidden=64, lr=0.05, beta=1.0). Unknown keys raise TypeError, which
    is what you want -- a typo'd hyperparameter should fail loudly, not be silently ignored.
    """
    if name not in _BUILDERS:
        raise ValueError(f"unknown method {name!r}; known: {sorted(_BUILDERS)}")
    kw = dict(METHOD_DEFAULTS[name]); kw.update(overrides)
    kw.update(in_dim=in_dim, hidden=hidden, seed=seed, device=device, handle=handle)
    if name in _NEEDS_DATA:
        if train_data is None or class_idx is None:
            raise ValueError(f"{name} needs train_data and class_idx")
        return _BUILDERS[name](train_data, class_idx, **kw)
    return _BUILDERS[name](**kw)
