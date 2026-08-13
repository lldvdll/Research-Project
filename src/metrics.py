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


def mean_test_error(steps, curve, after=None, before=None):
    """Mean test ERROR across the whole run -- Song & Bogacz's headline metric [R1].

    They do not report a final accuracy; they report the average test error over training. That
    is a different question from "how much survived at the end", and it rewards two things at
    once: learning the new task quickly AND not destroying the old one. A rule can win it by
    being fast even if it forgets normally, and lose it by being slow even if it forgets less.

    Reported because it is the number [R1]'s claim is made in. Not reported ALONE, for the same
    reason: it cannot distinguish those two ways of winning, which is exactly what this project
    is trying to separate.

    curve : [evals, n_tasks] accuracy in 0-1. Returns mean error over the window, per task.
    """
    s = np.asarray(steps, dtype=float)
    c = np.asarray(curve, dtype=float)
    m = np.ones(len(s), dtype=bool)
    if after is not None:
        m &= s >= after
    if before is not None:
        m &= s <= before
    if not m.any():
        return np.full(c.shape[-1], np.nan)
    return np.nanmean(1.0 - c[m], axis=0)


def acc_bwt_forgetting(steps, curve, switch):
    """The three standard continual-learning endpoint metrics, for a TWO-task sequence.

        ACC         mean final accuracy over both tasks                Lopez-Paz & Ranzato 2017
        BWT         final task-1 accuracy minus its accuracy at the    Lopez-Paz & Ranzato 2017
                    switch; negative means task 2 damaged task 1
        forgetting  peak task-1 accuracy minus its final accuracy      Chaudhry et al. 2018

    All three are read AT THE END, so all three inherit the budget: train task 2 for longer and
    every one of them gets worse, with no change to the learning rule. Michel et al. 2023 call
    this setup-induced forgetting. They are reported for comparability with the literature and
    so that their degeneracy is visible -- in 2x5 Class-IL every rule drives task 1 to ~0, and
    all three collapse onto the same uninformative number.
    """
    s = np.asarray(steps, dtype=float)
    c = np.asarray(curve, dtype=float)
    i_sw = int(np.argmin(np.abs(s - switch)))
    at_switch = c[i_sw, 0]
    peak = np.nanmax(c[:i_sw + 1, 0])
    final = c[-1]
    return dict(acc=float(np.nanmean(final)),
                bwt=float(final[0] - at_switch),
                forgetting=float(peak - final[0]))


def metric_grid(steps, curves, switch):
    """Every metric this project uses, computed PER RUN so paired statistics are available.

    curves : [runs, evals, n_tasks] accuracy in 0-1.  Returns {metric: array[runs]}, in
    percentage points where the quantity is an accuracy.

    The grid is reported in full on every comparison, rather than one headline number, because
    each metric is blind to something and the blind spots do not overlap:

      crossover     the accuracy where the two curves meet. No budget, no threshold, invariant
                    to a uniform rescaling of time. Blind to the asymptote, and UNDEFINED if the
                    curves never cross.
      final_t1      what survived at the end. Blind to nothing except the thing that matters
                    most -- WHEN the end was, which is set by the budget or the threshold.
      final_t2      the check that final_t1 is not "learned less" wearing "forgot less".
      half_life     how fast task 1 falls. A rate, so budget-independent, but it depends on the
                    pre-switch peak and says nothing about where the curve settles.
      area_retained integrated retention. Less noisy than an endpoint because it uses the whole
                    curve, but the integral runs to whenever training stopped.
      mean_err_*    Song & Bogacz's metric. Rewards fast learning and low forgetting together
                    and cannot tell them apart.
      acc/bwt/
      forgetting    the standard trio. All endpoint metrics; all degenerate here.
    """
    A = np.asarray(curves, dtype=float)
    s = np.asarray(steps, dtype=float)
    out = {k: [] for k in ["crossover", "final_t1", "final_t2", "peak_t1", "half_life",
                           "area_retained", "mean_err_t1", "mean_err_t2",
                           "acc", "bwt", "forgetting"]}
    for r in range(A.shape[0]):
        c = A[r]
        ok = np.isfinite(c[:, 0]) & np.isfinite(c[:, 1])
        sr, cr = s[ok], c[ok] * 100.0
        i_sw = int(np.argmin(np.abs(sr - switch)))
        peak = float(np.nanmax(cr[:i_sw + 1, 0]))
        xh = crossover(sr, cr[:, 0], cr[:, 1], after=switch)[1]
        hl = half_life(sr, cr[:, 0], after=switch, peak=peak)
        std = acc_bwt_forgetting(sr, cr / 100.0, switch)
        err = mean_test_error(sr, cr / 100.0, after=switch)
        out["crossover"].append(xh if xh is not None else np.nan)
        out["final_t1"].append(cr[-1, 0])
        out["final_t2"].append(cr[-1, 1])
        out["peak_t1"].append(peak)
        out["half_life"].append(hl if hl is not None else np.nan)
        out["area_retained"].append(area_retained(sr, cr[:, 0], after=switch, peak=peak) * 100)
        out["mean_err_t1"].append(err[0] * 100)
        out["mean_err_t2"].append(err[1] * 100)
        out["acc"].append(std["acc"] * 100)
        out["bwt"].append(std["bwt"] * 100)
        out["forgetting"].append(std["forgetting"] * 100)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


#: Metrics where a HIGHER value is better. The rest (mean_err_*, forgetting) invert, and the
#: paired report uses this so "better" always points the same way on the page.
HIGHER_IS_BETTER = {"crossover": True, "final_t1": True, "final_t2": True, "peak_t1": True,
                    "half_life": True, "area_retained": True, "acc": True, "bwt": True,
                    "mean_err_t1": False, "mean_err_t2": False, "forgetting": False}


def report_grid(grid_by_method, methods, control="backprop", primary="crossover"):
    """Print the full metric grid, paired against `control`, one block per metric.

    grid_by_method : {method: {metric: array[runs]}} from metric_grid.
    Every metric is shown for every method so a claim cannot be made by quoting the one that
    happens to favour it. `primary` is flagged; it is the metric the project reports on.
    """
    keys = [k for k in HIGHER_IS_BETTER if k in grid_by_method[methods[0]]]
    lines = []
    for k in keys:
        tag = "  <- primary" if k == primary else ""
        arrow = "higher better" if HIGHER_IS_BETTER[k] else "LOWER better"
        lines.append(f"\n  {k}   ({arrow}){tag}")
        for m in methods:
            v = grid_by_method[m][k]
            mu = float(np.nanmean(v))
            if m == control:
                lines.append(f"    {m:10s} {mu:8.2f}")
                continue
            d, se, n = paired_diff(v, grid_by_method[control][k])
            better = (d > 0) == HIGHER_IS_BETTER[k]
            mark = ("  BETTER" if better else "  worse") if n > 2 else ""
            lines.append(f"    {m:10s} {mu:8.2f}   vs {control} {d:+7.2f} +-{se:5.2f} "
                         f"{n:4.1f}sem{mark}")
    print("\n".join(lines))
    return keys
