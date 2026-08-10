"""Shared model specification -- MULTI-LAYER.

Arch       what the network IS      (widths as int OR tuple, nonlinearity, bias, init)
Objective  what the loss SAYS       (loss type, target coding, masking)

output_error(out, target, obj, active) -> e = -dL/dout
Backprop consumes it via out.backward(-e), predictive coding uses it as the final-layer
error, equilibrium propagation uses it as the nudge. All three see an identical output
signal, which is what makes a learning-rule comparison a learning-rule comparison.

BACKWARD COMPATIBILITY
    Arch(hidden=64)          -> one hidden layer, exactly as before
    Arch(hidden=(32, 32))    -> two hidden layers
    Params exposes .W1 .W2 .b1 .b2 ... .Wn .bn as read-only aliases into .Ws / .bs, so
    existing code that does params.W1 or getattr(p, "W2") keeps working unchanged.
    forward() now returns (list_of_pre_activations, out); use hs[-1] for the top hidden layer.
"""
from dataclasses import dataclass, replace
import math
import re
import torch

ACTS = {
    "identity":   (lambda h: h,                              lambda h: torch.ones_like(h)),
    "tanh":       (torch.tanh,                               lambda h: 1.0 - torch.tanh(h) ** 2),
    "sigmoid":    (torch.sigmoid,                            lambda h: torch.sigmoid(h) * (1 - torch.sigmoid(h))),
    "relu":       (lambda h: h.clamp_min(0.0),               lambda h: (h > 0).to(h.dtype)),
    "leaky_relu": (lambda h: torch.where(h > 0, h, 0.01 * h),
                   lambda h: torch.where(h > 0, torch.ones_like(h), torch.full_like(h, 0.01))),
    "hard_sigmoid": (lambda h: h.clamp(0.0, 1.0),
                     lambda h: ((h > 0) & (h < 1)).to(h.dtype)),
}


@dataclass(frozen=True)
class Arch:
    in_dim: int = 196
    hidden: object = 64               # int -> one hidden layer; tuple/list -> several
    out_dim: int = 10
    act: str = "tanh"
    bias: bool = False
    init: str = "scaled_normal"       # "scaled_normal" | "kaiming_uniform" | "xavier_normal"
    init_gain: float = 1.0

    @property
    def widths(self):
        """Full layer sizes, input first: (in, h1, ..., hk, out)."""
        h = (self.hidden,) if isinstance(self.hidden, int) else tuple(self.hidden)
        return (self.in_dim,) + h + (self.out_dim,)

    @property
    def n_weights(self):
        return len(self.widths) - 1

    @property
    def n_hidden(self):
        return len(self.widths) - 2

    def f(self, h):
        return ACTS[self.act][0](h)

    def df(self, h):
        return ACTS[self.act][1](h)


@dataclass(frozen=True)
class Objective:
    loss: str = "mse"                 # "mse" | "ce" | "hinge"
    target: str = "onehot"            # "onehot" -> 1/0 ; "pm1" -> +1/-1
    mask: bool = False
    hinge_margin: float = 1.0
    reduction: str = "mean"           # "mean" over the batch, or "sum"
    # Song & Bogacz's loss_fn is (outputs - target).pow(2).sum() * 0.5 -- summed over BOTH
    # the batch and the output units, with no division. At batch 500 that makes the gradient
    # ~500x larger than a mean-reduced one, which is why their learning-rate grid runs
    # 0.0001-0.005 where ours ran 0.005-0.5. Matching the reduction is what makes the two
    # grids comparable; without it the learning rates are in different units.

    def __post_init__(self):
        if self.loss not in ("mse", "ce", "hinge"):
            raise ValueError(f"unknown loss {self.loss!r}")
        if self.target not in ("onehot", "pm1"):
            raise ValueError(f"unknown target coding {self.target!r}")
        if self.loss == "ce" and self.target != "onehot":
            raise ValueError("cross-entropy requires target='onehot'")


UNIFIED_ARCH = Arch(act="tanh", bias=True, init="scaled_normal")
UNIFIED_OBJ = Objective(loss="mse", target="onehot", mask=False)

# Song & Bogacz Fig 4a-h: sigmoid, Xavier normal, 32 units per hidden layer, batch 32.
#
# STALE, AND KEPT ONLY SO THE ARCHIVED SCRIPTS STILL MEAN WHAT THEY MEANT. "four layers and 32
# hidden neurons in each hidden layer" is ambiguous. This encodes experiment 30's reading,
# in -> 32 -> 32 -> out. Experiment 34 later read it as 784-32-32-32-5 -- three hidden layers
# and FIVE SHARED OUTPUTS, which is what makes Fig 4d Domain-IL rather than Class-IL.
# The value was not corrected because changing it would silently change what
# experiments/archive/bogacz_reproduction/ did. Do not use it for new work; the reproduction is
# archived and did not succeed. See experiments/archive/README.md.
BOGACZ_ARCH = Arch(in_dim=784, hidden=(32, 32), out_dim=10,
                   act="sigmoid", bias=True, init="xavier_normal")
BOGACZ_OBJ = Objective(loss="mse", target="onehot", mask=False)

LEGACY_SPEC = {
    "backprop": (Arch(act="relu", bias=True, init="kaiming_uniform"), Objective("ce", "onehot")),
    "replay":   (Arch(act="relu", bias=True, init="kaiming_uniform"), Objective("ce", "onehot")),
    "pc":       (Arch(act="tanh", bias=False), Objective("mse", "onehot")),
    "eqprop":   (Arch(act="tanh", bias=False), Objective("hinge", "pm1")),
}

_ALIAS = re.compile(r"([Wb])(\d+)$")


class Params:
    """Weight matrices Ws[0..L-1] and optional biases bs[0..L-1].

    .W1 .W2 ... .Wn and .b1 ... .bn are READ-ONLY aliases (1-indexed) into those lists, so
    code written for the two-layer version keeps working. In-place updates must go through
    .Ws[i] / .bs[i]."""

    def __init__(self, Ws, bs=None):
        object.__setattr__(self, "Ws", list(Ws))
        object.__setattr__(self, "bs", list(bs) if bs is not None else [None] * len(Ws))

    def __getattr__(self, name):
        m = _ALIAS.match(name)
        if m:
            seq = self.Ws if m.group(1) == "W" else self.bs
            i = int(m.group(2)) - 1
            if 0 <= i < len(seq):
                return seq[i]
        raise AttributeError(name)

    def named(self):
        """{'W1': tensor, 'b1': tensor or None, ...} -- used by freeze hooks."""
        d = {}
        for i, W in enumerate(self.Ws):
            d[f"W{i + 1}"] = W
            d[f"b{i + 1}"] = self.bs[i]
        return d

    def tensors(self):
        return [t for t in (self.Ws + self.bs) if t is not None]

    def requires_grad_(self, flag=True):
        for t in self.tensors():
            t.requires_grad_(flag)
        return self

    def clone(self):
        return Params([W.detach().clone() for W in self.Ws],
                      [None if b is None else b.detach().clone() for b in self.bs])

    def __len__(self):
        return len(self.Ws)

    def __iter__(self):
        return iter(self.tensors())


def init_params(arch, seed=0, device="cpu"):
    """Deterministic init from a dedicated generator, so weight init is its own RNG stream."""
    g = torch.Generator(device=device).manual_seed(int(seed))
    w, Ws, bs = arch.widths, [], []
    for i in range(arch.n_weights):
        fan_in, fan_out = w[i], w[i + 1]
        if arch.init == "scaled_normal":
            W = torch.randn(fan_in, fan_out, generator=g, device=device) / fan_in ** 0.5
        elif arch.init == "xavier_normal":                     # Song & Bogacz: sd sqrt(2/(in+out))
            W = torch.randn(fan_in, fan_out, generator=g, device=device) * math.sqrt(2.0 / (fan_in + fan_out))
        elif arch.init == "kaiming_uniform":                   # torch.nn.Linear default
            k = 1.0 / math.sqrt(fan_in)
            W = (torch.rand(fan_in, fan_out, generator=g, device=device) * 2 - 1) * k
        else:
            raise ValueError(f"unknown init {arch.init!r}")
        Ws.append(W * arch.init_gain)
        bs.append(torch.zeros(fan_out, device=device) if arch.bias else None)
    return Params(Ws, bs)


def flatten(x):
    return x.reshape(x.size(0), -1)


def _affine(a, W, b):
    o = a @ W
    return o if b is None else o + b


def forward(x, p, arch):
    """Feedforward pass.

    Returns (hs, out) where hs is the list of PRE-ACTIVATION hidden states, one per hidden
    layer. Convention (Song & Bogacz): the nonlinearity is applied on the way OUT of a layer,
    so hs[l] is the raw state and arch.f(hs[l]) is what it transmits.
    """
    a = flatten(x)
    hs = []
    for i in range(arch.n_weights):
        z = _affine(a, p.Ws[i], p.bs[i])
        if i < arch.n_weights - 1:
            hs.append(z)
            a = arch.f(z)
        else:
            return hs, z
    raise RuntimeError("unreachable")


def hidden_code(x, p, arch):
    """Post-activation code of the TOP hidden layer, for probes (NCM, CKA)."""
    hs, _ = forward(x, p, arch)
    return arch.f(hs[-1])


def resolve_freeze(p, freeze):
    """Tensors named in `freeze`. Names are 1-indexed: W1, b1, W2, b2, ... Unknown names are
       ignored rather than raising, so a freeze set written for a 2-layer net is harmless on
       a deeper one."""
    named = p.named()
    return [named[n] for n in freeze if named.get(n) is not None]


def hidden_pre(x, p, arch):
    """Back-compatible: pre-activation of the FIRST hidden layer (predictive coding's mu1)."""
    return _affine(flatten(x), p.Ws[0], p.bs[0])


def make_target(y_labels, arch, obj, device="cpu"):
    fill = 0.0 if obj.target == "onehot" else -1.0
    t = torch.full((y_labels.size(0), arch.out_dim), fill, device=device)
    t.scatter_(1, y_labels.reshape(-1, 1), 1.0)
    return t


def active_vector(active, arch, device="cpu"):
    if active is None:
        return torch.ones(arch.out_dim, device=device)
    m = torch.zeros(arch.out_dim, device=device)
    m[torch.as_tensor(sorted(active), dtype=torch.long, device=device)] = 1.0
    return m


def output_error(out, target, obj, active_vec=None):
    """e = -dL/dout, per example, NOT divided by batch size."""
    if obj.loss == "mse":
        e = target - out
    elif obj.loss == "ce":
        if active_vec is not None:
            out = out.masked_fill(active_vec.unsqueeze(0) == 0, float("-inf"))
        e = target - torch.softmax(out, dim=1)
        e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    elif obj.loss == "hinge":
        e = torch.where(obj.hinge_margin - target * out > 0, target, torch.zeros_like(target))
    else:
        raise ValueError(obj.loss)
    if obj.mask and active_vec is not None:
        e = e * active_vec.unsqueeze(0)
    return e


def batch_scale(obj, n):
    """Divisor applied to a summed-over-batch gradient. 1.0 for sum reduction, n for mean."""
    return 1.0 if obj.reduction == "sum" else float(n)


def loss_value(out, target, obj, active_vec=None):
    """The loss actually being minimised. Honours obj.reduction, so the number reported agrees
       with the gradient applied -- batch_scale divides by n only under reduction='mean'."""
    reduce = (lambda v: v.mean()) if obj.reduction == "mean" else (lambda v: v.sum())
    if obj.loss == "mse":
        d = (target - out) ** 2
        if obj.mask and active_vec is not None:
            d = d * active_vec.unsqueeze(0)
        return 0.5 * reduce(d.sum(1))
    if obj.loss == "ce":
        o = out if active_vec is None else out.masked_fill(active_vec.unsqueeze(0) == 0, float("-inf"))
        return -reduce((target * torch.log_softmax(o, dim=1)).nan_to_num(0.0).sum(1))
    d = (obj.hinge_margin - target * out).clamp_min(0.0)
    if obj.mask and active_vec is not None:
        d = d * active_vec.unsqueeze(0)
    return reduce(d.sum(1))


__all__ = ["Arch", "Objective", "Params", "ACTS", "UNIFIED_ARCH", "UNIFIED_OBJ",
           "BOGACZ_ARCH", "BOGACZ_OBJ", "LEGACY_SPEC", "init_params", "flatten",
           "hidden_pre", "forward", "hidden_code", "resolve_freeze", "make_target", "active_vector",
           "output_error", "loss_value", "batch_scale", "replace"]
