"""Methods: each builder sets up a model and returns (train_step, predict) closures.
   Add a method = add a make_* function. One model, one function.

   `predict(x, raw=False)` -> class indices, or the raw pre-argmax outputs when raw=True.
   The `raw` kwarg is optional, so older experiment scripts calling predict(x) still work."""
import torch
import torch.nn as nn
from .eqprop import eqprop_init, eqprop_update, eqprop_settle
from .predictive_coding import pc_init, pc_update, pc_predict


def make_mlp(in_dim=196, hidden=64, out_dim=10):
    return nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def make_backprop(in_dim=196, hidden=64, lr=0.1, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = make_mlp(in_dim, hidden).to(device)
    opt, lf = torch.optim.SGD(model.parameters(), lr=lr), nn.CrossEntropyLoss()

    def train_step(x, y):
        opt.zero_grad(); lf(model(x), y).backward(); opt.step()

    def predict(x, raw=False):
        with torch.no_grad():
            out = model(x)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_replay(train_data, class_idx, in_dim=196, hidden=64, lr=0.1, per_class=20, seed=0, device="cpu"):
    """Experience-replay control: stores `per_class` examples the first time each class is seen,
       mixes an equal-sized replay sample into every batch."""
    torch.manual_seed(seed)
    model = make_mlp(in_dim, hidden).to(device)
    opt, lf = torch.optim.SGD(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    mem_x, mem_y, seen = [], [], set()

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
                settle_patience=30, seed=0, device="cpu"):
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)

    def train_step(x, y):
        eqprop_update(x, y, W1, W2, opt, beta=beta, dt=dt, max_steps=max_steps,
                      settle_patience=settle_patience, device=device)

    def predict(x, raw=False):
        _, out = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_pc(in_dim=196, hidden=64, lr=0.05, dt=0.1, steps=50, seed=0, device="cpu"):
    """Predictive coding: settle the hidden activities toward the target, then update weights locally."""
    W1, W2 = pc_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)

    def train_step(x, y):
        pc_update(x, y, W1, W2, lr=lr, dt=dt, steps=steps, device=device)

    def predict(x, raw=False):
        return pc_predict(x, W1, W2, raw=raw)

    return train_step, predict


def make_eqprop_gated(in_dim=196, hidden=64, lr=0.03, beta=0.3, dt=0.3, max_steps=500,
                      settle_patience=30, gate_frac=0.3, seed=0, device="cpu"):
    from .eqprop import eqprop_update_gated
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)

    def train_step(x, y):
        eqprop_update_gated(x, y, W1, W2, opt, beta=beta, dt=dt, max_steps=max_steps,
                            settle_patience=settle_patience, gate_frac=gate_frac, device=device)

    def predict(x, raw=False):
        _, out = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps,
                               settle_patience=settle_patience, device=device)
        return out if raw else out.argmax(1)

    return train_step, predict


def make_eqprop_replay(train_data, class_idx, in_dim=196, hidden=64, lr=0.03, beta=0.3, dt=0.3,
                       max_steps=500, settle_patience=30, per_class=20, seed=0, device="cpu"):
    """EqProp with a stored real-example replay buffer mixed into each batch."""
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)
    mem_x, mem_y, seen = [], [], set()

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
                          settle_patience=30, n_synth=20, gen_steps=200, seed=0, device="cpu"):
    """EqProp with GENERATIVE replay: at each new class, regenerate synthetic examples of the
       already-learned classes from the model itself and mix them in."""
    from .eqprop import eqprop_generate
    W1, W2 = eqprop_init(in_dim=in_dim, hidden=hidden, seed=seed, device=device)
    opt = torch.optim.SGD([W1, W2], lr=lr)
    seen, synth_x, synth_y = [], [], []

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
