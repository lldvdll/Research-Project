"""Metric definitions. Defined ONCE here and imported -- never redefined inside a script.

All functions are pure numpy/python: no torch, no plotting. Safe to unit-test.
`curves` convention everywhere: array [evals, n_tasks] for one run.

    summarise()     the whole scalar grid for one run, in one call -- start here
    inefficiency()  [R31] path length / net displacement, per synapse
    sem()           mean and standard error over seeds, as the protocol reports

The rest are the pieces summarise() is built from, usable on their own.
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


def inefficiency(path, net):
    """[R31] Li & van Rossum: how far a synapse actually travelled / how far it needed to.

        path = sum_t |w(t) - w(t-1)|     accumulated by probes.weight_path_probe
        net  = |w(T) - w(0)|             from the first and last weight snapshots

    1.0 is a perfectly direct path. Higher is more wandering, and by their argument more
    metabolically expensive -- a rule that arrives at the same place having spent less is a
    better hypothesis about a brain under evolutionary pressure.

    Both arguments are per-synapse arrays of the same shape, so this returns the per-synapse
    DISTRIBUTION. Take .mean() for the scalar; the distribution is the more informative object.
    Synapses that barely moved have a near-zero denominator and are returned as NaN rather than
    infinity, since their ratio says nothing about the path taken."""
    path = np.asarray(path, dtype=float)
    net = np.abs(np.asarray(net, dtype=float))
    tiny = net <= (np.nanmax(net) * 1e-6 if np.nanmax(net) > 0 else 0.0)
    out = np.divide(path, net, out=np.full_like(path, np.nan), where=~tiny)
    return out


def summarise(steps, t1, t2, switch, threshold=None):
    """The scalar metric grid for one run, in one call. §2.1 of the presentation plan.

    steps     eval steps        t1, t2  per-task accuracy curves      switch  step of the task change

    No single headline number: each entry answers a different question, and where they disagree
    is analysis material rather than a problem. Returns a flat dict so runs stack into a table.

        peak_t1              how well task 1 was learned -- the ceiling retention is measured against
        final_t1 / final_t2  the endpoint. final_t1 may be 0.0 for every rule in Class-IL; that is
                             a finding, not a broken metric, and it is why the others exist
        forgetting           peak_t1 - final_t1, the standard CL quantity
        crossover_height     accuracy where the curves cross -- were both tasks held at once
        t1_at_threshold      task-1 accuracy when task 2 first reaches `threshold`: retention at a
                             MATCHED standard, so a rule is not rewarded for simply learning less
        area_retained        mean task-1 accuracy over task 2, as a fraction of its peak
        half_life            updates for task 1 to fall to half its peak; None if it never does
    """
    t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
    after = [i for i, s in enumerate(steps) if s > switch]
    peak = float(np.nanmax(t1[:after[0]])) if after else float(np.nanmax(t1))
    x_step, x_height = crossover(steps, t1, t2, after=switch)
    return dict(
        peak_t1=peak,
        final_t1=float(t1[-1]),
        final_t2=float(t2[-1]),
        forgetting=peak - float(t1[-1]),
        crossover_step=x_step,
        crossover_height=x_height,
        t1_at_threshold=(float("nan") if threshold is None
                         else value_when(steps, t2, threshold, t1, after=switch)),
        area_retained=area_retained(steps, t1, after=switch, peak=peak),
        half_life=half_life(steps, t1, after=switch, peak=peak),
    )


def sem(values):
    """Mean and standard error of the mean, over seeds. The protocol reports SEM and says so --
       not the standard deviation, which describes spread rather than uncertainty in the mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


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


def paired_diff(treatment, control):
    """Per-seed difference against a control, as (mean, sem, n_sem). THE DEFAULT COMPARISON.

    Every rule in this project sees the SAME class split and the SAME initialisation at a given
    seed -- `Protocol.tasks(seed)` and `build(..., seed)` both key off it. So most of the
    variance between seeds is shared, and comparing group means throws that away.

    Script 53 measured how much this matters. Backprop retained 38.2% on one seed and 78.0% on
    another, a 40-point range set by which digits the split happened to pair. Compared as group
    means, replay -- the POSITIVE CONTROL, which is supposed to work -- came out at 0.7 sem and
    read as a failure. Paired, the same runs give +11.6 +- 2.3, i.e. 5.1 sem, because every seed
    shows replay ahead. The five-seed budget was never the problem; the statistic was.

    Report the paired difference. Report a group mean only when the two sides genuinely do not
    share a seed, and say so when you do.
    """
    a = np.asarray(treatment, dtype=float)
    b = np.asarray(control, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_diff needs matched runs, got {a.shape} and {b.shape}")
    d = a - b
    d = d[np.isfinite(d)]
    if d.size < 2:
        return (float(d[0]) if d.size else float("nan")), float("nan"), float("nan")
    m, s = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
    return m, s, (abs(m) / s if s > 0 else float("inf"))
