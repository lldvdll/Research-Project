"""Training loops. Experiments own their analysis and plots; only the loops live here.

run_joint    : train on a set of classes together (ceilings, learning-rate calibration).
run_classil  : train tasks in sequence, with optional per-task early stopping on accuracy.

Both take the (train_step, predict) pair from src.methods.

New in the 20-series
--------------------
* `train_step(x, y, active=...)` is called with the classes present in the current task, so
  objectives with mask=True can give absent classes zero gradient.
* separate `stop_eval` and `report_eval` sets, so early stopping does not bias the reported
  number.
* `readouts`: a dict of {name: fn(x) -> predicted labels}. The reported curve is computed
  under EVERY readout, so one run yields argmax and nearest-class-mean accuracy together.
* `reached` flags per task, so a method that never met the criterion is visible rather than
  silently reverting to the fixed-step budget.
"""
import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _loader(train_data, class_idx, classes, batch, seed=None):
    idx = torch.cat([torch.as_tensor(class_idx[c]) for c in classes])
    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
    return DataLoader(Subset(train_data, idx.tolist()), batch_size=batch, shuffle=True, generator=g)


def _next(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def _accepts_active(train_step):
    """Does this train_step take an `active` argument? Read off its signature."""
    try:
        return "active" in inspect.signature(train_step).parameters
    except (TypeError, ValueError):
        return True


def _call(train_step, x, y, active):
    """Call train_step, passing `active` only if it accepts it.

    Asking the signature rather than catching TypeError matters. A TypeError raised INSIDE
    train_step -- for any unrelated reason -- would be swallowed and the update silently retried
    WITHOUT `active`, so a masked-loss experiment could quietly run unmasked and still produce a
    plausible-looking curve. Every builder in src.methods accepts `active`; this branch exists
    only for a hand-written 2-argument closure."""
    if _accepts_active(train_step):
        return train_step(x, y, active=active)
    return train_step(x, y)


def _acc(pred, eval_y, classes, label_map=None):
    tgt = (lambda c: c) if label_map is None else (lambda c: label_map[c])
    return [(pred[eval_y == c] == tgt(c)).float().mean().item() for c in classes]


def run_joint(train_step, predict, classes, train_data, class_idx, eval_x, eval_y,
              max_iters=800, batch=32, eval_every=10, device="cpu",
              stop_patience=None, min_delta=1e-3, stop_at=None, data_seed=None):
    """Train on all `classes` at once. Returns (steps, acc) on the held-out eval set."""
    loader = _loader(train_data, class_idx, classes, batch, seed=data_seed)
    it = iter(loader)
    steps, accs = [], []
    best, since = -1.0, 0
    for step in range(1, max_iters + 1):
        (x, y), it = _next(it, loader)
        _call(train_step, x.to(device), y.to(device), classes)
        if step % eval_every == 0:
            a = (predict(eval_x) == eval_y).float().mean().item()
            steps.append(step); accs.append(a)
            if stop_at is not None and a >= stop_at:
                break
            if a > best + min_delta:
                best, since = a, 0
            else:
                since += 1
            if stop_patience is not None and since >= stop_patience:
                break
    return np.array(steps), np.array(accs)


def run_classil(train_step, predict, tasks, train_data, class_idx,
                report_eval, stop_eval=None, readouts=None,
                max_iters_per_task=400, batch=32, eval_every=1, device="cpu",
                stop_threshold=None, stop_patience=3, tail_iters=0, data_seed=None,
                on_task_end=None, label_map=None):
    """Sequential Class-IL training with per-task accuracy logging.

    report_eval : (x, y) used for the reported curves.
    stop_eval   : (x, y) used for the early-stopping decision. Defaults to report_eval,
                  which is mildly optimistic -- pass a disjoint set.
    readouts    : {name: fn(x) -> labels}. Defaults to {"argmax": predict}.
    max_iters_per_task, stop_threshold : int/float, or a LIST with one entry per task. Use a
                  list to train task 1 to a competence threshold and then give task 2 a fixed
                  budget, in a SINGLE call -- which keeps every task as a column in `curves`.
                  Splitting this across two calls silently drops the task-1 column.
    stop_threshold : end a task once ITS accuracy has been >= threshold for `stop_patience`
                  consecutive evals (+ `tail_iters` more updates). This equalises how well
                  each method learns each task, removing learning speed as a confound.
    on_task_end : optional fn(task_index, step) called after each task finishes. Use it to
                  snapshot prototypes or hidden codes at the switch.
    label_map   : optional {global_class: output_unit}. Needed for DOMAIN-IL, where several
                  classes share an output unit -- e.g. Song & Bogacz Fig 4d, where two
                  five-class tasks share five output neurons, so class c of task 1 and class
                  c' of task 2 both map to unit c. Applied to training labels AND to the
                  accuracy check, so a prediction counts as correct when it matches the unit
                  the class was trained on. Leave as None for Class-IL.

    Returns dict with steps, curves {readout: [evals, n_tasks]}, switches, reached.
    """
    n_t = len(tasks)
    iters_per = list(max_iters_per_task) if isinstance(max_iters_per_task, (list, tuple)) \
        else [max_iters_per_task] * n_t
    thr_per = list(stop_threshold) if isinstance(stop_threshold, (list, tuple)) \
        else [stop_threshold] * n_t

    report_x, report_y = report_eval
    stop_x, stop_y = stop_eval if stop_eval is not None else report_eval
    readouts = readouts or {"argmax": predict}

    classes = sorted({c for t in tasks for c in t})
    pos = {c: i for i, c in enumerate(classes)}
    steps_log, switches, reached = [], [], []
    curves = {k: [] for k in readouts}
    step = 0

    for ti, task in enumerate(tasks):
        loader = _loader(train_data, class_idx, task, batch,
                         seed=None if data_seed is None else data_seed + 1000 * ti)
        it = iter(loader)
        hits, countdown, hit_criterion = 0, None, False
        thr = thr_per[ti]
        # `active` indexes OUTPUT UNITS, not classes. Under Class-IL they coincide. Under
        # Domain-IL they do not: task [5,6,7,8,9] trains output units [0,1,2,3,4], and passing
        # the class ids straight through indexes a length-5 vector at position 5.
        active = task if label_map is None else sorted({label_map[int(c)] for c in task})
        for _ in range(iters_per[ti]):
            (x, y), it = _next(it, loader)
            y = y.to(device)
            if label_map is not None:
                y = torch.tensor([label_map[int(v)] for v in y], device=device)
            _call(train_step, x.to(device), y, active)
            step += 1
            if countdown is not None:
                countdown -= 1
            if step % eval_every == 0:
                steps_log.append(step)
                for name, fn in readouts.items():
                    a = _acc(fn(report_x), report_y, classes, label_map)
                    curves[name].append([float(np.mean([a[pos[c]] for c in t])) for t in tasks])
                if thr is not None and countdown is None:
                    a_stop = _acc(predict(stop_x), stop_y, classes, label_map)
                    cur = float(np.mean([a_stop[pos[c]] for c in task]))
                    hits = hits + 1 if cur >= thr else 0
                    if hits >= stop_patience:
                        countdown, hit_criterion = tail_iters, True
            if countdown is not None and countdown <= 0:
                break
        switches.append(step)
        reached.append(hit_criterion if thr is not None else True)
        if on_task_end is not None:
            on_task_end(ti, step)

    return dict(steps=np.array(steps_log),
                curves={k: np.array(v) for k, v in curves.items()},
                switches=switches, reached=reached)


def run_alternating(train_step, predict, tasks, train_data, class_idx, eval_sets,
                    iters_per_task=4, total_iters=160, batch=500, partial_num=None,
                    device="cpu", data_seed=None, n_classes=10, eval_per_update=False,
                    loss_active=None,
                    divergence_check=20, divergence_floor=0.78):
    """Song & Bogacz Fig 4d schedule, matching experiments/nature_forgetting/*.yaml.

    ONE ITERATION = one epoch over that task's training subset. With partial_num=600 and
    batch=500 that is TWO weight updates per iteration (500 + 100, drop_last=False), so the
    per-iteration curve can move further between plotted points than one SGD step allows.
    Tasks alternate every `iters_per_task` iterations.

    Evaluation is ONCE PER ITERATION by default, which is exactly their logging frequency
    and what their figure plots. `eval_per_update=True` evaluates after every weight update
    instead -- six times as many forward passes over the test set, which is where almost all
    of the wall-clock went in the first attempt. Exp 31 showed the per-update panel added
    nothing the block-aligned analysis of per-iteration data did not already show, so the
    default is off.

    Returns dict:
        errors   [n_updates, n_tasks]    test error after each weight update
        iter_of  [n_updates]             iteration each update belongs to
        task_of  [n_updates]             task being trained
        per_iter [total_iters, n_tasks]  error at the end of each iteration
    """
    from torch.utils.data import DataLoader, Subset
    from .data import partial_task_indices, label_remapper

    loaders, remaps = [], []
    for ti, task in enumerate(tasks):
        idx = partial_task_indices(class_idx, task, partial_num,
                                   seed=0 if data_seed is None else data_seed + ti)
        g = None if data_seed is None else torch.Generator().manual_seed(int(data_seed) + 7 * ti)
        loaders.append(DataLoader(Subset(train_data, idx.tolist()), batch_size=batch,
                                  shuffle=True, generator=g, drop_last=False))
        remaps.append(label_remapper(task, device, n_classes))

    def _eval():
        return [1.0 - (predict(ex) == ey).float().mean().item() for ex, ey in eval_sets]

    errors, iter_of, task_of = [], [], []
    per_iter = np.full((total_iters, len(tasks)), np.nan)
    diverged_at = None
    for it in range(total_iters):
        ti = (it // iters_per_task) % len(tasks)
        for x, y in loaders[ti]:
            # loss_active restricts which OUTPUT UNITS contribute to the loss. Their
            # learn_code slices (outputs - target)[:, 0:-1], which silently excludes the
            # LAST output unit -- see exp 33's LOSS_ACTIVE note.
            train_step(x.to(device), remaps[ti][y.to(device)], active=loss_active)
            if eval_per_update:
                errors.append(_eval()); iter_of.append(it); task_of.append(ti)
        per_iter[it] = errors[-1] if eval_per_update else _eval()
        # A diverged learning rate sits pinned at chance and tells us nothing further. Stop
        # it early and hold the last value: the mean over the window is unchanged, and on a
        # six-point learning-rate grid this is a sixth of the total runtime saved.
        if (divergence_check is not None and it >= divergence_check
                and np.nanmin(per_iter[max(0, it - divergence_check):it + 1]) > divergence_floor):
            diverged_at = it
            per_iter[it + 1:] = per_iter[it]
            break
    return dict(errors=np.array(errors) if errors else np.empty((0, len(tasks))),
                iter_of=np.array(iter_of), task_of=np.array(task_of),
                per_iter=per_iter, diverged_at=diverged_at)
