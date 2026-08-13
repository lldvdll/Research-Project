"""Why does the SEED account for a 40-point range -- larger than any effect under study?

THE OBSERVATION
    In script 53, backprop retained 38.2% of task 1 on one seed and 78.0% on another. That
    range is bigger than every rule difference this project has measured. It is shared across
    rules, which is why paired comparison rescues the statistics (`metrics.paired_diff`), but
    nothing explains WHERE it comes from. A nuisance term that large is worth understanding:
    it sets how many seeds any future experiment needs.

THE HYPOTHESIS, AND IT IS TESTABLE
    Under Domain-IL, `Protocol.label_map` sends the i-th class of each task to output unit i.
    So every unit carries one task-1 digit and one task-2 digit, PAIRED BY THE PER-SEED
    PERMUTATION and by nothing else. If a unit's two digits look alike, learning task 2 may
    partly reuse the task-1 feature at that unit; if they look nothing alike, they compete for
    it. Which pairs a seed happens to draw is therefore a candidate source of the spread.

THE CONTROL, WHICH IS THE POINT OF THE SCRIPT
    Pairing similarity is not the only thing a seed changes -- it also decides which five digits
    are in task 1 at all, and some digit sets are simply easier. So two similarities are
    computed per seed:

        sim_paired   mean similarity of the 5 digits that SHARE AN OUTPUT UNIT
        sim_all      mean similarity over all 25 cross-task digit pairs, ignoring the pairing

    `sim_all` depends only on WHICH digits landed in which task; `sim_paired` depends on that
    AND on how they were matched up. If only sim_paired predicts the outcome, the pairing is
    what matters. If sim_all predicts it just as well, the pairing is irrelevant and the story
    is about task difficulty. Reporting sim_paired alone could not tell these apart.

    Similarity is cosine between class-MEAN images, after subtracting the global mean image.
    The centring matters: every MNIST class mean is a centred blob, so raw cosine sits at
    0.6-0.9 for every pair and compresses the differences the test needs. Both are reported.

CENSORING -- READ THIS BEFORE READING THE SCATTER
    Crossover height is undefined when the curves never cross, and for replay that happens
    precisely when it does BEST: task 1 stays above task 2 for the whole run. Dropping those
    runs would bias the control's own metric toward its worst seeds -- in script 52 replay's
    crossover was computed on 3 of 5 seeds for exactly this reason. Here they are counted,
    reported, and drawn as upward arrows at their lower bound (the final task-2 accuracy),
    never silently discarded. Retention and area retained are defined on every run and carry
    the correlation where crossover is censored.

DEVIATIONS FROM THE PROTOCOL, STATED
    * 24 seeds, not 5. A correlation needs points; 5 cannot support one.
    * EqProp dropped. It costs ~350x backprop per update and this question is about the data,
      not the rule -- the same reasoning as script 59.
    * Fixed budgets, as script 52, so the numbers are directly comparable with it.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, run, replace, figure_path as _figure_path, array_path as _array_path
from src.metrics import metric_grid, report_grid

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
METHODS = ["backprop", "replay", "pc"]
SEEDS = 24
EVAL_EVERY = 10
COLORS = {"backprop": "tab:gray", "replay": "tab:brown", "pc": "tab:red"}

# ---- read, not chosen: identical to script 52 so the two are comparable -------
HIDDEN = int(np.load(_array_path(str(ROOT / "experiments" /
                                     "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(_array_path(str(ROOT / "experiments" /
                              "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
PC_STEPS = int(z51["pc_steps"])
T = float(z51["target_steps"])
ITERS = [int(2 * T), int(1.5 * T)]

if SMOKE:
    SEEDS, ITERS = 4, [40, 30]
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | domain-IL | {ITERS[0]}+{ITERS[1]} updates | {SEEDS} seeds | "
      f"{', '.join(METHODS)}")
print("  question: does WHICH DIGITS SHARE AN OUTPUT UNIT predict how much task 1 survives?\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)


def settle_kw(method):
    return dict(steps=PC_STEPS) if method == "pc" else {}


# ---------------------------------------------------------------- similarity
def class_means(train):
    """Global-mean-centred class mean image per digit, flattened. Centring removes the
       'every digit is a centred blob' component that otherwise dominates the cosine."""
    x = train.x.reshape(len(train.x), -1).float()
    y = train.targets
    raw = np.stack([x[y == c].mean(0).numpy() for c in range(10)])
    return raw, raw - raw.mean(0, keepdims=True)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def similarities(tasks, cen, raw):
    """(sim_paired, sim_all, sim_raw_paired, per_pair) for one seed's split.

    sim_paired : the 5 digits that SHARE an output unit -- the hypothesis.
    sim_all    : all 25 cross-task pairs -- the control. Same digits, pairing ignored.
    """
    t1, t2 = tasks[0], tasks[1]
    per_pair = [cos(cen[a], cen[b]) for a, b in zip(t1, t2)]
    allp = [cos(cen[a], cen[b]) for a in t1 for b in t2]
    raw_pair = [cos(raw[a], raw[b]) for a, b in zip(t1, t2)]
    return float(np.mean(per_pair)), float(np.mean(allp)), float(np.mean(raw_pair)), per_pair


def pearson_perm(x, y, n_perm=20000, seed=0):
    """Pearson r plus a PERMUTATION p-value. No scipy dependency, and at n=24 a permutation
       test makes no normality assumption a 24-point scatter could not support."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return float("nan"), float("nan"), int(x.size)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(n_perm)])
    return r, float((null >= abs(r)).mean()), int(x.size)


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    steps, switches = z["steps"], list(z["switches"])
    curves = {m: z[f"argmax_{m}"] for m in METHODS}
    task_list = [list(map(list, t)) for t in z["tasks"]]
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    curves, task_list = {m: [] for m in METHODS}, None
    t0 = time.perf_counter()
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        seed_tasks = []
        for seed in range(SEEDS):
            out = run(proto, m, seed, data=data, **settle_kw(m))
            curves[m].append(out["curves"]["argmax"])
            seed_tasks.append(out["tasks"])
        steps, switches = out["steps"], out["switches"]
        curves[m] = np.stack(curves[m])
        if task_list is None:
            task_list = seed_tasks
        else:
            # every rule must see the same split at a given seed, or the pairing is not a
            # shared nuisance term and pairing the statistics is invalid
            assert seed_tasks == task_list, "rules disagree on the class split -- pairing broken"
        A = curves[m] * 100
        print(f"  {m:10s} t1 kept {A[:, -1, 0].mean():5.1f}% "
              f"(range {A[:, -1, 0].min():5.1f} - {A[:, -1, 0].max():5.1f}, "
              f"spread {A[:, -1, 0].max() - A[:, -1, 0].min():4.1f} points)"
              f"   [{time.perf_counter() - t0:5.0f}s]")

data_for_images = load(base) if REPLOT else data
raw_means, cen_means = class_means(data_for_images.train)
sim_paired, sim_all, sim_raw, pair_detail = [], [], [], []
for tasks in task_list:
    sp, sa, sr, pp = similarities(tasks, cen_means, raw_means)
    sim_paired.append(sp); sim_all.append(sa); sim_raw.append(sr); pair_detail.append(pp)
sim_paired, sim_all = np.array(sim_paired), np.array(sim_all)

grid = {m: metric_grid(steps, curves[m], switches[0]) for m in METHODS}

# ---------------------------------------------------------------- readings
print(f"\n  the spread this script exists to explain, over {SEEDS} seeds:")
print(f"  {'rule':10s} {'t1 kept':>18s} {'crossover':>18s} {'defined on':>11s}")
for m in METHODS:
    k, x = grid[m]["final_t1"], grid[m]["crossover"]
    print(f"  {m:10s} {k.mean():6.1f}% [{k.min():4.1f}-{k.max():4.1f}] "
          f"{np.nanmean(x):6.1f}% [{np.nanmin(x):4.1f}-{np.nanmax(x):4.1f}] "
          f"{int(np.isfinite(x).sum()):6d}/{SEEDS}")
n_cens = {m: int((~np.isfinite(grid[m]["crossover"])).sum()) for m in METHODS}
if any(n_cens.values()):
    print("\n  CENSORED: " + ", ".join(f"{m} {n}/{SEEDS}" for m, n in n_cens.items() if n))
    print("  A missing crossover means task 1 never fell below task 2 -- the BEST outcome, not a")
    print("  failed measurement. Dropping those runs biases the metric against the rule winning.")

print(f"\n  similarity of the digits sharing an output unit, across seeds:")
print(f"    sim_paired (centred)  mean {sim_paired.mean():+.3f}  "
      f"range {sim_paired.min():+.3f} to {sim_paired.max():+.3f}")
print(f"    sim_all    (control)  mean {sim_all.mean():+.3f}  "
      f"range {sim_all.min():+.3f} to {sim_all.max():+.3f}")

print(f"\n  DOES IT PREDICT THE OUTCOME?  Pearson r, permutation p, n")
print(f"  {'':10s} {'metric':14s} {'r(sim_paired)':>22s} {'r(sim_all) = CONTROL':>24s}")
corr = {}
for m in METHODS:
    for k in ["final_t1", "crossover", "area_retained"]:
        rp, pp_, np_ = pearson_perm(sim_paired, grid[m][k])
        ra, pa, na = pearson_perm(sim_all, grid[m][k])
        corr[(m, k)] = (rp, pp_, np_, ra, pa, na)
        star = " *" if pp_ < 0.05 else "  "
        stara = " *" if pa < 0.05 else "  "
        print(f"  {m:10s} {k:14s} {rp:+8.3f} p={pp_:5.3f} n={np_:2d}{star} "
              f"{ra:+8.3f} p={pa:5.3f} n={na:2d}{stara}")

print("\n  How to read it: sim_paired significant AND sim_all not -> the PAIRING matters.")
print("  Both significant -> which digits are in which task matters, not how they were matched.")
print("  Neither -> digit similarity is not the source of the seed spread; look elsewhere.")

# A null control is only evidence if the control COULD have shown something. sim_all averages
# 25 pairs where sim_paired averages 5, so it is inherently less variable across seeds and a
# smaller r could be pure range restriction rather than a real absence. The slope per STANDARD
# DEVIATION OF THE PREDICTOR removes that: it asks how many accuracy points each predictor buys
# for a typical seed-to-seed change in itself, which is comparable between the two.
print(f"\n  Range-restriction check -- sd(sim_paired) = {sim_paired.std(ddof=1):.4f}, "
      f"sd(sim_all) = {sim_all.std(ddof=1):.4f} ({sim_paired.std(ddof=1)/sim_all.std(ddof=1):.1f}x).")
print("  Standardised slope, points of task-1 retained per 1 sd of the predictor:")
for m in METHODS:
    v = grid[m]["final_t1"]
    b_p = np.polyfit(sim_paired, v, 1)[0] * sim_paired.std(ddof=1)
    b_a = np.polyfit(sim_all, v, 1)[0] * sim_all.std(ddof=1)
    print(f"    {m:10s} paired {b_p:+6.1f} pts/sd    control {b_a:+6.1f} pts/sd")
print("  The control is null in EFFECT SIZE, not merely in r, so its narrower range is not")
print("  what makes it null.")

report_grid(grid, METHODS, control="backprop", primary="crossover")

# ---------------------------------------------------------------- figure 1: scatters
METRICS = [("final_t1", "task 1 kept (%)"), ("crossover", "crossover height (%)"),
           ("area_retained", "area retained (%)")]
fig, axes = plt.subplots(len(METRICS), 2, figsize=(10.5, 3.4 * len(METRICS)), squeeze=False)
for row, (k, ylab) in enumerate(METRICS):
    # column 0 reads r from corr[...][0] (vs sim_paired), column 1 from corr[...][3] (vs
    # sim_all). Annotating both columns with the same r would show the control panel wearing
    # the hypothesis's correlation, which is the one number the control exists to contradict.
    for col, (sim, sname, r_at) in enumerate([
            (sim_paired, "paired: digits SHARING a unit", 0),
            (sim_all, "control: all cross-task pairs", 3)]):
        ax = axes[row][col]
        for m in METHODS:
            v = grid[m][k]
            ok = np.isfinite(v)
            ax.plot(sim[ok], v[ok], "o", ms=6, color=COLORS[m], label=m, alpha=0.85)
            if k == "crossover" and (~ok).any():
                # censored: draw at the lower bound (final task-2 accuracy) with an up-arrow,
                # so a run that never crossed is visible instead of absent
                lb = grid[m]["final_t2"][~ok]
                ax.plot(sim[~ok], lb, "^", ms=7, mfc="none", color=COLORS[m])
            if ok.sum() >= 4:
                b = np.polyfit(sim[ok], v[ok], 1)
                xs = np.linspace(sim.min(), sim.max(), 10)
                r = corr[(m, k)][r_at]
                ax.plot(xs, np.polyval(b, xs), color=COLORS[m], lw=1.4, ls="--", alpha=0.8)
                ax.annotate(f"r={r:+.2f}", xy=(0.02, 0.96 - 0.09 * METHODS.index(m)),
                            xycoords="axes fraction", color=COLORS[m], fontsize=8, va="top")
        ax.set_xlabel("mean centred cosine between class means")
        ax.set_ylabel(ylab)
        ax.set_title(sname, fontsize=9)
        ax.grid(alpha=0.25)
axes[0][0].legend(fontsize=8)
fig.suptitle("Does which digits share an output unit explain the seed spread?  "
             "Open triangles = never crossed, drawn at their lower bound.", fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- figure 2: the digits
# One example image per class, arranged so the two digits sharing an output unit are ADJACENT.
# Rows are seeds, ordered by sim_paired, so the eye can check the correlation directly against
# the number: if the hypothesis holds, the bottom rows (similar pairs) should retain more.
train = data_for_images.train
first_of = {c: int((train.targets == c).nonzero()[0][0]) for c in range(10)}
order = np.argsort(sim_paired)
n_units = len(task_list[0][0])
fig2, axes2 = plt.subplots(len(order), 2 * n_units, figsize=(1.05 * 2 * n_units, 0.62 * len(order)))
axes2 = np.atleast_2d(axes2)
for r, si in enumerate(order):
    t1, t2 = task_list[si][0], task_list[si][1]
    for u in range(n_units):
        for j, c in enumerate((t1[u], t2[u])):
            ax = axes2[r][2 * u + j]
            ax.imshow(train.x[first_of[int(c)]].reshape(base.img_size, base.img_size),
                      cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("tab:orange" if j == 0 else "tab:blue")
                s.set_linewidth(1.6)
            if r == 0:
                ax.set_title(f"u{u}\n{'t1' if j == 0 else 't2'}", fontsize=7)
    axes2[r][0].set_ylabel(f"s{si}\n{sim_paired[si]:+.2f}\n{grid['backprop']['final_t1'][si]:.0f}%",
                           fontsize=6, rotation=0, ha="right", va="center", labelpad=16)
fig2.suptitle("Digits sharing each output unit, one pair per column block, rows ordered by "
              "pairing similarity.\nLeft label: seed / similarity / task-1 kept by backprop. "
              "Orange = task 1, blue = task 2.", fontsize=9)
fig2.tight_layout(rect=[0.02, 0, 1, 0.97])
fig2.savefig(figure_path(__file__, "digits"), dpi=140, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'digits')}")

# ---------------------------------------------------------------- save
if not SMOKE:
    np.savez(array_path(__file__),
             steps=steps, switches=switches, methods=np.array(METHODS),
             tasks=np.array(task_list), sim_paired=sim_paired, sim_all=sim_all,
             sim_raw=np.array(sim_raw), pair_detail=np.array(pair_detail),
             **{f"argmax_{m}": curves[m] for m in METHODS})
    print(f"saved {array_path(__file__)}")
