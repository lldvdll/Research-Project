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
THE DEFAULT IS THE PROTOCOL. Every method gets model.UNIFIED_ARCH and model.UNIFIED_OBJ --
the same architecture and the same output structure -- unless you say otherwise. Only the
learning rule differs, which is what makes a learning-rule comparison a learning-rule
comparison. Passing `arch=`/`obj=` explicitly is still preferred in new work, because it puts
the specification in the script where a reader will look for it.

To reproduce a script written before unification, ask for its old specification BY NAME:

    make_backprop(..., **legacy("backprop"))     # ReLU + cross-entropy, as experiments 01-15

Nothing falls back to model.LEGACY_SPEC silently. Under the legacy specification each rule got
a different nonlinearity and a different loss, so any difference between rules confounds the
rule with its output structure -- that is the confound the unified protocol exists to remove.
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


def _spec(arch, obj):
    """Resolve (arch, obj). The default is the unified protocol, the same for every rule.

    Deliberately takes no method name: there is no way for the specification to depend on which
    rule is being built, which is the property that makes the comparison controlled."""
    return (arch or UNIFIED_ARCH), (obj or UNIFIED_OBJ)


def legacy(name):
    """The (arch, obj) this method used BEFORE unification, as keyword arguments.

    Experiments 01-15 were written against these and now pass them explicitly, so changing the
    library default cannot silently change what an old script means:

        make_eqprop(..., **legacy("eqprop"))     # tanh, no bias, hinge with +-1 targets

    Do not use in new work. Under these specifications backprop trains with ReLU and
    cross-entropy while PC trains with tanh and squared error, so the rules are not comparable
    to each other -- see model.LEGACY_SPEC."""
    a, o = LEGACY_SPEC[name.replace("_gated", "").replace("_replay", "")]
    return dict(arch=a, obj=o)


def _apply_freeze(p, freeze):
    """Zero the gradient of every parameter NAMED in `freeze`, so an experiment can hold part of
       the network still mid-run via handle["freeze"].add("W1").

       Names are exact and 1-indexed: W1, b1, W2, b2, ... Freezing "W1" leaves b1 learning; to
       hold a whole layer still, name both. That is the same convention pc_update and
       eqprop_update already use, so one freeze set means the same thing under all four rules --
       which is what makes a freezing experiment a comparison rather than a confound.
       Unknown names are ignored (see model.resolve_freeze), so a set written for a 2-layer net
       is harmless on a deeper one."""
    if not freeze:
        return
    for t in resolve_freeze(p, freeze):
        if t.grad is not None:
            t.grad.zero_()


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
    arch, obj = _spec(arch, obj)
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
def make_replay(train_data=None, class_idx=None, in_dim=196, hidden=64, out_dim=10, lr=0.05,
                optimizer="sgd",
                per_class=20, replay_frac=None, buffer_seed=None, seed=0, device="cpu",
                arch=None, obj=None, handle=None, **_):
    """Backprop + a stored-example buffer. NOT a learning rule -- a reference upper bound.

    THE BUFFER STORES FROM THE STREAM -- a per-label reservoir of `per_class` examples each, and
    the dataset is never consulted. Two reasons:

      * Domain-IL. `y` is the output UNIT an example was trained on, not its class: under a
        label_map, classes 5-9 arrive as units 0-4. Keying the buffer by class id therefore
        mis-keys task 2, and any 'first per_class wins' rule never stores it at all, because
        units 0-4 filled up during task 1. A reservoir fixes both -- see _store.
      * It is the honest control. The earlier version fetched `per_class` images of a class
        from the full training pool the instant it first saw ONE example of it -- i.e. it drew
        on data the network had not been shown yet.

    replay_frac = None  : legacy behaviour, replay is APPENDED (batch grows to 2x, the model
                          sees strictly more data per step -- a confound).
    replay_frac = 0.5   : batch size held FIXED; half the batch is replay, so the model sees
                          FEWER new examples per step. Conservative; prefer this.
    buffer_seed         : dedicated RNG for WHICH stored examples are replayed each step, so
                          buffer sampling is its own variance source (defaults to `seed`).
    train_data, class_idx : accepted and ignored, so scripts 01-15 still call this unchanged.
                          The buffer no longer needs the dataset.
    """
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    g = torch.Generator(device="cpu").manual_seed(int(seed if buffer_seed is None else buffer_seed))
    mem_x, mem_y, freeze = [], [], set()
    slots, counts = {}, {}          # label -> its row indices in mem_x ; label -> examples seen

    def _store(xf, y):
        """Per-label reservoir: `per_class` slots per label, holding a uniform sample of every
        example of that label seen SO FAR (Algorithm R, one reservoir per label).

        Straight 'keep the first per_class' would not do. Under Domain-IL classes 5-9 arrive as
        units 0-4, whose slots are already full from task 1, so task 2 would never enter the
        buffer at all. With a reservoir, class 5 competes with class 0 for unit 0's slots and
        the buffer ends up holding both -- with no need to know where the task boundary was,
        which is the point: `y` and `active` are identical across tasks under Domain-IL, so
        train_step cannot detect a boundary even in principle.

        Buffer size is fixed at per_class * n_labels and balanced across labels by construction.
        """
        u = torch.rand(y.size(0), generator=g)          # one RNG call per step, not per example
        for i, c in enumerate(y.tolist()):
            n = counts.get(c, 0)
            counts[c] = n + 1
            row = slots.setdefault(c, [])
            if len(row) < per_class:
                row.append(len(mem_x))
                mem_x.append(xf[i].detach().clone())    # clone: xf is a view onto the batch
                mem_y.append(c)
            else:
                j = int(u[i] * (n + 1))                 # uniform on 0..n
                if j < per_class:
                    mem_x[row[j]] = xf[i].detach().clone()

    def train_step(x, y, active=None):
        xf = flatten(x)
        _store(xf, y)
        if mem_x:
            rx, ry = torch.stack(mem_x), torch.tensor(mem_y, device=device)
            k = xf.size(0) if replay_frac is None else int(round(replay_frac * xf.size(0)))
            k = min(k, len(ry))
            if k > 0:
                s = torch.randperm(len(ry), generator=g)[:k]
                keep = xf.size(0) if replay_frac is None else xf.size(0) - k
                xf = torch.cat([xf[:keep], rx[s]])
                y = torch.cat([y[:keep], ry[s]])
        n = xf.size(0)
        target = make_target(y, arch, obj, device=device)
        # replayed labels are legitimately present in this batch, so they are ACTIVE. `counts`
        # is keyed by output unit, the same space as `active`, so this union is well defined.
        av = active_vector(None if active is None else sorted(set(active) | set(counts)),
                           arch, device=device)
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

    # buffer_seen counts examples ENCOUNTERED per label. buffer_x/buffer_labels are the live
    # buffer itself, so len() gives its size and the contents can be inspected or plotted --
    # under Domain-IL the labels alone cannot tell you which task an example came from.
    _publish(handle, p, arch, obj, features, freeze=freeze,
             diag={"buffer_seen": counts, "buffer_labels": mem_y, "buffer_x": mem_x})
    return train_step, predict


# ------------------------------------------------------------------ predictive coding
def make_pc(in_dim=196, hidden=64, out_dim=10, lr=0.05, dt=0.1, steps=50, optimizer="sgd",
            seed=0, device="cpu", arch=None, obj=None, handle=None, **_):
    """steps=0 disables relaxation entirely -> the 'PC without prospective configuration'
       control that should collapse onto backprop (experiment 23)."""
    arch, obj = _spec(arch, obj)
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
                settle_tol=1e-4, gate_frac=None, optimizer="sgd", seed=0, device="cpu",
                arch=None, obj=None, handle=None, settle_patience=None, **_):
    # settle_patience selects the PRE-EXPERIMENT-50 stopping rule and is forwarded, not blocked.
    # Experiments 01-27 and the experiment-12 reproduction pass it, and a superseded rule is
    # still the rule those results were produced under -- removing it would make them
    # unrunnable, and a result that cannot be re-executed cannot be checked. New work should
    # leave it None and take settle_tol; see src/eqprop.py:eqprop_settle for why.
    arch, obj = _spec(arch, obj)
    arch = replace(arch, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    p = init_params(arch, seed=seed, device=device).requires_grad_(True)
    opt = make_optimizer(p.tensors(), optimizer, lr)
    diag, freeze = {}, set()

    def train_step(x, y, active=None):
        d = eqprop_update(x, y, p, opt, arch=arch, obj=obj, beta=beta, dt=dt,
                          max_steps=max_steps, settle_tol=settle_tol,
                          settle_patience=settle_patience,
                          active=active, gate_frac=gate_frac, device=device,
                          return_delta=True, freeze=freeze)
        diag.update(d)

    def predict(x, raw=False):
        states = eqprop_settle(x, p, arch, dt=dt, max_steps=max_steps,
                               settle_tol=settle_tol, settle_patience=settle_patience,
                               device=device)
        return states[-1] if raw else states[-1].argmax(1)

    def features(x):
        from .eqprop import eqprop_features
        return eqprop_features(x, p, arch, dt=dt, max_steps=max_steps,
                               settle_tol=settle_tol, settle_patience=settle_patience,
                               device=device)

    _publish(handle, p, arch, obj, features, diag=diag, freeze=freeze)
    return train_step, predict


def make_eqprop_gated(gate_frac=0.3, **kw):
    return make_eqprop(gate_frac=gate_frac, **kw)


# ------------------------------------------------------------------ dispatch
METHOD_DEFAULTS = {
    "backprop":     dict(lr=0.05),
    # replay_frac=0.5 holds the BATCH SIZE FIXED at the protocol's value: half new, half
    # replayed. The alternative, replay_frac=None, appends the replay batch so the model trains
    # on 64 examples per step while every other rule gets 32 -- more data per update, which is
    # a confound in any comparison the replay control appears in. See make_replay.
    "replay":       dict(lr=0.05, per_class=20, replay_frac=0.5),
    # settle_tol is calibrated for dt=0.3 by experiment 50; see eqprop.eqprop_settle.
    "eqprop":       dict(lr=0.005, beta=0.3, dt=0.3, max_steps=800, settle_tol=1e-4),
    "pc":           dict(lr=0.05, dt=0.1, steps=50),
    "eqprop_gated": dict(lr=0.005, beta=0.3, dt=0.3, max_steps=800, settle_tol=1e-4,
                         gate_frac=0.3),
}

# Nothing needs the dataset at construction time any more -- replay fills its buffer from the
# stream. `train_data`/`class_idx` are still accepted by build_method and ignored, so scripts
# 01-15 keep working unchanged.
_NEEDS_DATA = set()
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
