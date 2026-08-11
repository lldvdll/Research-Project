"""The protocol: every setting that must be identical across the four rules, in one object.

WHY THIS EXISTS
    A learning-rule comparison is only a comparison if everything except the rule is held fixed.
    When each script restates the split, the thresholds, the seeds and the eval sets by hand,
    they drift apart, and a difference between two figures stops meaning anything. So the
    settings live here, once, and scripts import them.

HOW TO USE IT -- one deviation at a time

    from dataclasses import replace
    from src.protocol import PROTOCOL, load, run

    proto = replace(PROTOCOL, hidden=64)                  # the protocol
    proto = replace(proto, scenario="domain_il")          # ONE stated deviation
    data  = load(proto)
    out   = run(proto, "pc", seed=0, data=data)

    out["curves"]["argmax"]   # [evals, n_tasks] accuracy
    out["tasks"]              # the class split this seed drew
    out["snapshots"]          # weights at init, task-1 end, task-2 end

`replace` returns a NEW protocol and never mutates the shared one, so a deviation cannot leak
into the next script. Say in the script docstring which line you changed -- that sentence is
also what goes on the slide.

WHAT IS NOT HERE
    Analysis, plotting and saving stay in the experiment script. This module answers "what is
    the setting", not "what did we find". It is deliberately not a harness: `run` is one
    function that fills protocol values into run_classil, and a script that needs something
    different should call run_classil directly rather than growing an option here.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple
import numpy as np
import torch

from .model import Arch, Objective, init_params
from .data import load_mnist, class_indices, make_eval_split
from .methods import build_method
from .runner import run_classil

__all__ = ["Protocol", "PROTOCOL", "Data", "load", "build", "run", "replace",
           "figure_path", "array_path"]

# The one dataset directory, at the repo root. Resolved from this file rather than from the
# working directory: scripts are run from inside experiments/, and a relative "./data" would
# silently re-download MNIST into experiments/data.
DATA_ROOT = str(Path(__file__).resolve().parents[1] / "data")


@dataclass(frozen=True)
class Protocol:
    """Frozen so it cannot be edited in place. Vary it with dataclasses.replace."""

    # ---- data ---------------------------------------------------------------
    img_size: int = 14                  # MNIST downsampled; 196 inputs
    n_tasks: int = 2
    classes_per_task: int = 5           # 2 x 5
    scenario: str = "class_il"          # "class_il" (10 outputs) | "domain_il" (5 shared)
    fashion: bool = False

    # ---- network ------------------------------------------------------------
    # H IS NOT DECIDED. Script 41 sets it on capacity grounds and nothing else, so there is no
    # default here -- inheriting a width from an old experiment is exactly the failure this
    # project is recovering from. `arch` raises with instructions if it is still None.
    hidden: object = None
    n_layers: int = 1                   # depth is a later phase, not an axis of the comparison
    act: str = "tanh"
    bias: bool = True
    init: str = "scaled_normal"

    # ---- output structure ---------------------------------------------------
    # linear readout + squared error + one-hot 1/0. See code_map.md: `act` is the HIDDEN
    # nonlinearity; the output layer is always the raw affine sum.
    loss: str = "mse"
    target: str = "onehot"
    mask: bool = False
    reduction: str = "mean"

    # ---- training -----------------------------------------------------------
    optimizer: str = "sgd"              # plain SGD everywhere
    batch: int = 32
    lr: dict = field(default_factory=dict)   # {method: lr}; empty -> methods.METHOD_DEFAULTS
    stop_threshold: float = 0.9         # BOTH tasks trained to accuracy, not to a step budget
    stop_patience: int = 3
    max_iters_per_task: int = 3000      # cap only; how often it is hit must be reported
    seeds: int = 5

    # ---- evaluation ---------------------------------------------------------
    eval_per_class: int = 100           # per class, in EACH of the two disjoint eval sets
    eval_every: int = 10
    device: str = "cpu"

    # ---- derived ------------------------------------------------------------
    @property
    def in_dim(self):
        return self.img_size ** 2

    @property
    def n_classes(self):
        return self.n_tasks * self.classes_per_task

    @property
    def out_dim(self):
        """Class-IL gives every class its own unit; Domain-IL shares one set across tasks."""
        return self.n_classes if self.scenario == "class_il" else self.classes_per_task

    @property
    def arch(self):
        if self.hidden is None:
            raise ValueError(
                "Protocol.hidden is not set. The hidden width is decided by script 41 on "
                "capacity grounds, not inherited from an earlier experiment. Pass it "
                "explicitly: replace(PROTOCOL, hidden=64)."
            )
        h = (self.hidden,) * self.n_layers if isinstance(self.hidden, int) else tuple(self.hidden)
        return Arch(in_dim=self.in_dim, hidden=h, out_dim=self.out_dim,
                    act=self.act, bias=self.bias, init=self.init)

    @property
    def objective(self):
        return Objective(loss=self.loss, target=self.target, mask=self.mask,
                         reduction=self.reduction)

    def tasks(self, seed):
        """The class split for this seed. Assignment is drawn per seed, so a result is not an
           artefact of which digits happened to share a task."""
        perm = np.random.default_rng(seed).permutation(self.n_classes).tolist()
        k = self.classes_per_task
        return [perm[t * k:(t + 1) * k] for t in range(self.n_tasks)]

    def label_map(self, tasks):
        """Class -> output unit. None for Class-IL (they coincide).

        For Domain-IL the i-th class of every task maps to unit i, so which classes share a
        unit follows the per-seed permutation and is arbitrary -- as it is in [R1] Fig 4d.
        Pairing by sorted rank instead would make each unit's classes more similar than chance,
        which would suppress the very interference being measured (see data.label_remapper)."""
        if self.scenario == "class_il":
            return None
        return {c: i for t in tasks for i, c in enumerate(t)}

    def describe(self):
        """One line for the figure title / slide, stating the setting actually run."""
        data = "fashion-mnist" if self.fashion else "mnist"
        # stop_threshold and max_iters_per_task may each be a scalar or a per-task list
        t = self.stop_threshold
        stop = (f"{self.max_iters_per_task} updates per block" if t is None
                else f"stop at {'/'.join(f'{v:.0%}' for v in t)}"
                if isinstance(t, (list, tuple)) else f"stop at {t:.0%}")
        return (f"{data} {self.img_size}x{self.img_size} | "
                f"{self.scenario} {self.n_tasks}x{self.classes_per_task} | "
                f"{self.in_dim}-{self.hidden}x{self.n_layers}-{self.out_dim} {self.act} | "
                f"{self.loss}/{self.target}{' masked' if self.mask else ''} | "
                f"batch {self.batch}, {stop}")


#: The protocol. Import this, never construct Protocol() in a script.
PROTOCOL = Protocol()


class Data(NamedTuple):
    """Everything a run needs from disk, loaded once and shared across methods and seeds."""
    train: object
    test: object
    class_idx: dict
    stop_eval: tuple        # (x, y) -- decides early stopping
    report_eval: tuple      # (x, y) -- DISJOINT; the number that gets published


def load(proto):
    """Load MNIST at the protocol's resolution and build the two disjoint eval sets.

    Stopping and reporting use different images on purpose: stopping on the set you report
    biases the published number upwards."""
    train, test = load_mnist(size=proto.img_size, root=DATA_ROOT, fashion=proto.fashion)
    stop_eval, report_eval = make_eval_split(
        test, classes=list(range(proto.n_classes)),
        per_class=proto.eval_per_class, device=proto.device)
    return Data(train, test, class_indices(train), stop_eval, report_eval)


def build(proto, method, seed, handle=None, **overrides):
    """(train_step, predict) for `method` under this protocol.

    Every rule gets proto.arch and proto.objective, so the only thing that differs is the rule.
    **overrides reach the builder's own knobs (beta, dt, steps, per_class, ...)."""
    lr = {"lr": proto.lr[method]} if method in proto.lr else {}
    return build_method(method, arch=proto.arch, obj=proto.objective,
                        in_dim=proto.in_dim, hidden=proto.arch.hidden, out_dim=proto.out_dim,
                        optimizer=proto.optimizer, seed=seed, device=proto.device,
                        handle=handle, **lr, **overrides)


def run(proto, method, seed, data=None, readouts=None, handle=None, wrap=None, tasks=None,
        **overrides):
    """One Class-IL / Domain-IL run of one rule at one seed, under this protocol.

    data     : from load(). Pass it in so the dataset is read once, not once per run.
    readouts : {name: fn(x) -> labels} for extra measurements (see src.probes). The reported
               curve is computed under every readout, so one run gives argmax and NCM together.
    wrap     : fn(train_step) -> train_step, for probes that must sit around each update
               (src.probes.alignment_probe, src.probes.weight_path_probe).
    tasks    : override the class split. Defaults to proto.tasks(seed), which is what the
               protocol means. Pass a list to train a schedule the protocol does not describe --
               repeating the two task blocks, [T1, T2, T1, T2, ...], gives an alternating run.
               A block may repeat; the returned curves then have one column per BLOCK, so
               columns for the same class set are duplicates of each other.
               Using this is a stated deviation. Say so in the script docstring.

    Returns run_classil's dict plus:
        tasks      the class split actually trained
        label_map  the class -> unit map actually used
        handle     the method handle (params, features, freeze, diag)
        snapshots  weights at init and at the end of each block, per the protocol's saving rule
    """
    data = load(proto) if data is None else data
    handle = {} if handle is None else handle
    tasks = proto.tasks(seed) if tasks is None else tasks
    lmap = proto.label_map(tasks)

    train_step, predict = build(proto, method, seed, handle=handle, **overrides)
    if wrap is not None:
        train_step = wrap(train_step)

    snapshots = [_snapshot(handle["params"])]
    out = run_classil(
        train_step, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval, readouts=readouts,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, device=proto.device,
        stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
        data_seed=seed, label_map=lmap,
        on_task_end=lambda ti, step: snapshots.append(_snapshot(handle["params"])),
    )
    out.update(tasks=tasks, label_map=lmap, handle=handle, snapshots=snapshots)
    return out


def _snapshot(params):
    """Detached copy of every weight and bias, keyed W1/b1/W2/b2/... for saving to .npz."""
    return {k: v.detach().clone() for k, v in params.named().items() if v is not None}


# ---------------------------------------------------------------- output naming
# Protocol rule: one figure per script, named from __file__, with its .npz beside it.

def figure_path(script_file, suffix=""):
    """`experiments/44_x.py` -> `experiments/44_x.png` (or `44_x_traj.png` with suffix)."""
    from pathlib import Path
    p = Path(script_file)
    return str(p.with_name(p.stem + (f"_{suffix}" if suffix else "") + ".png"))


def array_path(script_file, suffix=""):
    """The .npz that sits beside the figure. Every script writes one, weights included."""
    from pathlib import Path
    p = Path(script_file)
    return str(p.with_name(p.stem + (f"_{suffix}" if suffix else "") + ".npz"))
