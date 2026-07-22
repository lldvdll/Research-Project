"""The EqProp (Equilibrium Propagation) algorithm. One job per function.
   A different energy-based model would be a sibling file (e.g. contrastive.py)."""
import torch


def eqprop_init(in_dim=196, hidden=64, out_dim=10, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    W1 = (torch.randn(in_dim, hidden, generator=g, device=device) / in_dim ** 0.5).requires_grad_(True)
    W2 = (torch.randn(hidden, out_dim, generator=g, device=device) / hidden ** 0.5).requires_grad_(True)
    return W1, W2


def eqprop_energy(x, h, y, W1, W2):
    state = 0.5 * (h ** 2).sum() + 0.5 * (y ** 2).sum()
    align = (h * (x @ W1)).sum() + (y * (torch.tanh(h) @ W2)).sum()
    return state - align


def eqprop_settle(x, W1, W2, target=None, beta=0.0, dt=0.3, max_steps=500,
                  settle_patience=30, min_delta=1e-4, h0=None, y0=None, device="cpu"):
    """Relax (h, y) until per-step movement stops improving for `settle_patience` steps (or max_steps)."""
    with torch.enable_grad():
        x = x.reshape(x.size(0), -1)
        h = (torch.zeros(x.size(0), W1.size(1), device=device) if h0 is None else h0.clone()).requires_grad_(True)
        y = (torch.zeros(x.size(0), W2.size(1), device=device) if y0 is None else y0.clone()).requires_grad_(True)
        best, since = float("inf"), 0
        for _ in range(max_steps):
            gh, gy = torch.autograd.grad(eqprop_energy(x, h, y, W1, W2), [h, y])
            if target is not None:
                gy = gy + beta * torch.where(1 - target * y > 0, -target, torch.zeros_like(target))
            move = (dt * (gh.pow(2).sum() + gy.pow(2).sum()).sqrt()).item()
            h.data -= dt * gh
            y.data -= dt * gy
            if move < best - min_delta:
                best, since = move, 0
            else:
                since += 1
            if since >= settle_patience:
                break
    return h.detach(), y.detach()


def eqprop_update(x, y_labels, W1, W2, opt, beta=0.3, dt=0.3, max_steps=500, settle_patience=30, device="cpu"):
    """One EqProp weight update for a batch. Labels -> ±1 targets. Updates W1, W2 in place."""
    x = x.reshape(x.size(0), -1)
    target = torch.full((x.size(0), W2.size(1)), -1.0, device=device)
    target.scatter_(1, y_labels.unsqueeze(1), 1.0)
    h_f, y_f = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps, settle_patience=settle_patience, device=device)
    h_n, y_n = eqprop_settle(x, W1, W2, target, beta, dt, max_steps, settle_patience, h0=h_f, y0=y_f, device=device)
    opt.zero_grad()
    gW1_f, gW2_f = torch.autograd.grad(eqprop_energy(x, h_f, y_f, W1, W2), [W1, W2])
    gW1_n, gW2_n = torch.autograd.grad(eqprop_energy(x, h_n, y_n, W1, W2), [W1, W2])
    W1.grad = (gW1_n - gW1_f) / (beta * x.size(0))
    W2.grad = (gW2_n - gW2_f) / (beta * x.size(0))
    opt.step()


def eqprop_predict(x, W1, W2, dt=0.3, max_steps=500, settle_patience=30, device="cpu"):
    """Free-phase prediction: settle with no target, take argmax of the output."""
    _, y = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps, settle_patience=settle_patience, device=device)
    return y.argmax(1)


def eqprop_update_gated(x, y_labels, W1, W2, opt, beta=0.3, dt=0.3, max_steps=500,
                        settle_patience=30, gate_frac=0.3, device="cpu"):
    """Like eqprop_update, but only updates the `gate_frac` hidden nodes that move MOST under the nudge
       (the nodes 'responsible' for this stimulus). The rest are frozen this step. (Advisor point 4.)"""
    x = x.reshape(x.size(0), -1)
    target = torch.full((x.size(0), W2.size(1)), -1.0, device=device)
    target.scatter_(1, y_labels.unsqueeze(1), 1.0)
    h_f, y_f = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps, settle_patience=settle_patience, device=device)
    h_n, y_n = eqprop_settle(x, W1, W2, target, beta, dt, max_steps, settle_patience, h0=h_f, y0=y_f, device=device)
    shift = (h_n - h_f).abs().mean(0)                                    # per-node responsibility  [hidden]
    k = max(1, int(gate_frac * shift.numel()))
    mask = torch.zeros_like(shift)
    mask[torch.topk(shift, k).indices] = 1.0                            # 1 for selected nodes, 0 for frozen
    opt.zero_grad()
    gW1_f, gW2_f = torch.autograd.grad(eqprop_energy(x, h_f, y_f, W1, W2), [W1, W2])
    gW1_n, gW2_n = torch.autograd.grad(eqprop_energy(x, h_n, y_n, W1, W2), [W1, W2])
    W1.grad = ((gW1_n - gW1_f) / (beta * x.size(0))) * mask.unsqueeze(0)  # gate hidden columns of W1 [in, hidden]
    W2.grad = ((gW2_n - gW2_f) / (beta * x.size(0))) * mask.unsqueeze(1)  # gate hidden rows of W2   [hidden, out]
    opt.step()


def eqprop_generate(W1, W2, y_class, n, gen_steps=200, dt=0.1, device="cpu"):
    """Generate n synthetic inputs for `y_class`: clamp the output to that class and settle (x, h) from noise.
       Note: this simple energy is a weak generator; inspect the samples before trusting synthetic replay."""
    out_dim, in_dim, hidden = W2.size(1), W1.size(0), W1.size(1)
    y = torch.full((n, out_dim), -1.0, device=device)
    y[:, y_class] = 1.0
    x = torch.rand(n, in_dim, device=device, requires_grad=True)
    h = torch.zeros(n, hidden, device=device, requires_grad=True)
    with torch.enable_grad():
        for _ in range(gen_steps):
            gx, gh = torch.autograd.grad(eqprop_energy(x, h, y, W1, W2), [x, h])
            x.data -= dt * gx
            h.data -= dt * gh
            x.data.clamp_(0, 1)
    return x.detach()
