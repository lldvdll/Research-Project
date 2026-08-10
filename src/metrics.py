"""Scalar metrics and run-alignment helpers computed from per-task accuracy curves.

All functions are pure numpy/python: no torch, no plotting. Safe to unit-test.
`curves` convention everywhere: array [evals, n_tasks] for one run.
"""
import numpy as np


def first_cross(steps, series, thresh, after=0, rising=True):
    """Steps AFTER `after` until `series` first crosses `thresh`.
       rising=True  -> first value >= thresh   (learning)
       rising=False -> first value <  thresh   (forgetting)
       Returns None if it never crosses."""
    for s, v in zip(steps, series):
        if s <= after:
            continue
        if (v >= thresh) if rising else (v < thresh):
            return s - after
    return None


def crossover(steps, t1, t2, after=0):
    """(step, accuracy) where the t1 and t2 curves cross after `after`, linearly interpolated.
       High accuracy = both tasks held at once; low = a pure trade of one for the other."""
    idx = [i for i, s in enumerate(steps) if s > after]
    for a, b in zip(idx[:-1], idx[1:]):
        d1, d2 = t1[a] - t2[a], t1[b] - t2[b]
        if d1 > 0 >= d2:
            w = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.0
            return steps[a] + w * (steps[b] - steps[a]), t1[a] + w * (t1[b] - t1[a])
    return None, float("nan")


def value_when(steps, trigger, thresh, series, after=0, patience=1):
    """Value of `series` at the first eval after `after` where `trigger` has been >= `thresh`
       for `patience` consecutive evals.  NaN if that never happens.

       Primary matched-accuracy metric: pass trigger=task2 acc, series=task1 acc ->
       'how much of task 1 survives at the moment task 2 is learned to a fixed standard'."""
    run = 0
    for i, s in enumerate(steps):
        if s <= after:
            continue
        run = run + 1 if trigger[i] >= thresh else 0
        if run >= patience:
            return series[i]
    return float("nan")


def align_runs(step_lists, curve_list, switch_list, eval_every=1):
    """Align variable-length runs (per-task early stopping) on their task switch.

    step_lists : list of 1D arrays of eval steps, one per run
    curve_list : list of arrays [evals, n_tasks], one per run
    switch_list: step at which the switch happened, one per run
    Returns (rel_steps, stacked) where rel_steps is 0 at the switch and
    stacked is [runs, len(rel_steps), n_tasks], NaN where a run had no data.
    """
    rels = [np.asarray(s) - sw for s, sw in zip(step_lists, switch_list)]
    lo = int(min(r.min() for r in rels))
    hi = int(max(r.max() for r in rels))
    grid = np.arange(lo, hi + eval_every, eval_every)
    n_tasks = np.asarray(curve_list[0]).shape[-1]
    out = np.full((len(curve_list), len(grid), n_tasks), np.nan)
    for r, (rel, cur) in enumerate(zip(rels, curve_list)):
        cur = np.asarray(cur)
        for j, rs in enumerate(rel):
            k = int(round((rs - lo) / eval_every))
            if 0 <= k < len(grid):
                out[r, k, :] = cur[j]
    return grid, out


def pad_stack(curve_list):
    """Stack runs of differing length into [runs, max_evals, n_tasks], NaN-padded at the end."""
    n = max(np.asarray(c).shape[0] for c in curve_list)
    n_tasks = np.asarray(curve_list[0]).shape[-1]
    out = np.full((len(curve_list), n, n_tasks), np.nan)
    for r, c in enumerate(curve_list):
        c = np.asarray(c)
        out[r, :c.shape[0], :] = c
    return out


def half_life(steps, series, after=0, peak=None, frac=0.5):
    """Updates after `after` for `series` to fall to `frac` of its peak.

    WHY THIS EXISTS: when every condition eventually collapses to zero, the final value has no
    dynamic range and cannot separate methods. How FAST a method falls still can. Returns None
    if it never falls that far (which is itself informative)."""
    idx = [i for i, s in enumerate(steps) if s > after]
    if not idx:
        return None
    pk = max(series[i] for i in idx[:1]) if peak is None else peak
    target = frac * pk
    for i in idx:
        if series[i] <= target:
            return steps[i] - after
    return None


def area_retained(steps, series, after=0, peak=None):
    """Mean task-1 accuracy over the whole of task 2, as a fraction of its peak.

    A single number combining 'how high' and 'how long'. Unlike the final value it does not
    saturate at zero, and unlike half-life it is always defined."""
    idx = [i for i, s in enumerate(steps) if s > after]
    if not idx:
        return float("nan")
    pk = series[idx[0]] if peak is None else peak
    if pk <= 0:
        return float("nan")
    return float(np.mean([series[i] for i in idx]) / pk)


def bootstrap_ci(values, ci=68, n_boot=10000, seed=0):
    """Percentile bootstrap CI of the mean. Song & Bogacz report 68% CIs computed this way."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(v.mean()), float(lo), float(hi)
