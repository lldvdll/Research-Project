"""Training loops shared by experiments. Each experiment still owns its own analysis and plots;
   only the loops live here, because they are now identical across several scripts.

   run_joint    : train on a set of classes together (no task structure) -> ceiling / speed studies.
   run_classil  : train tasks in sequence, with OPTIONAL per-task early stopping on accuracy.

   Both take the (train_step, predict) pair returned by src.methods builders.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _loader(train_data, class_idx, classes, batch):
    idx = torch.cat([class_idx[c] for c in classes])
    return DataLoader(Subset(train_data, idx.tolist()), batch_size=batch, shuffle=True)


def _next(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def run_joint(train_step, predict, classes, train_data, class_idx, eval_x, eval_y,
              max_iters=800, batch=32, eval_every=10, device="cpu",
              stop_patience=None, min_delta=1e-3, stop_at=None):
    """Train on all `classes` at once. Returns (steps, acc) over the held-out eval set.

    stop_patience : stop when accuracy hasn't improved by `min_delta` for this many evals.
    stop_at       : stop as soon as accuracy reaches this value (used for speed studies).
    """
    loader = _loader(train_data, class_idx, classes, batch)
    it = iter(loader)
    steps, accs = [], []
    best, since = -1.0, 0
    for step in range(1, max_iters + 1):
        (x, y), it = _next(it, loader)
        train_step(x.to(device), y.to(device))
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


def run_classil(train_step, predict, tasks, train_data, class_idx, eval_x, eval_y,
                max_iters_per_task=300, batch=32, eval_every=1, device="cpu",
                stop_threshold=None, stop_patience=3, tail_iters=0):
    """Sequential Class-IL training with per-task accuracy logging.

    stop_threshold : if set, move to the next task once THIS task's accuracy has been
                     >= threshold for `stop_patience` consecutive evals (+ `tail_iters` more
                     updates). This equalises how well each method learns each task, removing
                     learning-speed as a confound when comparing forgetting.
    Returns (steps [evals], task_acc [evals, n_tasks], switches [step each task ended]).
    """
    classes = sorted({c for t in tasks for c in t})
    pos = {c: i for i, c in enumerate(classes)}
    steps_log, task_acc, switches = [], [], []
    step = 0
    for ti, task in enumerate(tasks):
        loader = _loader(train_data, class_idx, task, batch)
        it = iter(loader)
        hits, countdown = 0, None
        for _ in range(max_iters_per_task):
            (x, y), it = _next(it, loader)
            train_step(x.to(device), y.to(device))
            step += 1
            if countdown is not None:
                countdown -= 1
            if step % eval_every == 0:
                pred = predict(eval_x)
                acc = [(pred[eval_y == c] == c).float().mean().item() for c in classes]
                steps_log.append(step)
                task_acc.append([float(np.mean([acc[pos[c]] for c in t])) for t in tasks])
                if stop_threshold is not None and countdown is None:
                    hits = hits + 1 if task_acc[-1][ti] >= stop_threshold else 0
                    if hits >= stop_patience:
                        countdown = tail_iters
            if countdown is not None and countdown <= 0:
                break
        switches.append(step)
    return np.array(steps_log), np.array(task_acc), switches
