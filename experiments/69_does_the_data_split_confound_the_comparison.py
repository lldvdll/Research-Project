"""Does the data split confound the rule comparison, and does paired comparison remove it?

THE PROBLEM
    Script 60 found that in Domain-IL, WHICH TWO DIGITS SHARE AN OUTPUT UNIT explains ~62% of
    the seed-to-seed variance in task-1 retention (r = +0.787, n = 24) -- a nuisance term larger
    than every rule difference the project has measured. Two things follow that the rule
    comparisons need settled:

      1. Is there an analogous confound in CLASS-IL? There is no output-unit pairing there --
         every class has its own unit -- so if the split still predicts retention it does so
         through a different route, and if it does not, the two scenarios differ in their
         nuisance structure as well as in their forgetting mechanism.
      2. Does PAIRED comparison actually neutralise it? Every rule sees the same split at a
         given seed, so the nuisance is shared and should cancel in a per-seed difference. That
         is the project's standing justification for paired statistics and it has never been
         demonstrated at a sample size where the unpaired comparison could be shown to fail.

WHAT IS MEASURED
    Class-IL, 24 seeds, backprop / replay / pc. Domain-IL is loaded from script 60's saved
    arrays rather than re-run.

    sim_cross   mean centred cosine between the class-mean images of the two task sets, over all
                25 cross-task pairs. In Domain-IL this was the NULL CONTROL (r -0.25 to +0.06);
                in Class-IL, where no pairing exists, it is the natural candidate.
    sim_within  mean similarity within each task's own five classes -- a task of mutually
                similar digits may simply be easier or harder.

    Then, for each rule pair and each scenario, the same comparison computed two ways:
      unpaired  difference of group means, with the between-seed spread in the error bar
      paired    mean of per-seed differences
    The ratio of their standard errors is how much the shared nuisance was costing.

PRE-COMMITTED READINGS
    Class-IL confounded too, by sim_cross  -> the split is a nuisance in both scenarios and
        every comparison must be paired regardless of scenario.
    Class-IL not confounded                -> the Domain-IL confound is specifically the
        output-unit pairing, not "which digits are where", which sharpens 60's claim.
    paired sem << unpaired sem             -> pairing is not a stylistic choice; it is what
        makes the comparison possible at this sample size. Quantifies the standing rule.
    paired sem ~ unpaired sem              -> the nuisance is not shared after all, and the
        project's statistics need revisiting.

DEVIATIONS FROM THE PROTOCOL, STATED
    * 24 seeds, not 5. A correlation and a variance decomposition both need points.
    * EqProp dropped, as in 60: the question is about the data, not the rule.
    * Fixed budgets, as 52/56, so the numbers sit alongside those comparisons.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import (PROTOCOL, load, run, replace,
                          figure_path as _figure_path, array_path as _array_path)
from src.metrics import metric_grid, paired_diff

SMOKE = "--smoke" in sys.argv


def _tag(f, s):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (s + "_SMOKE").lstrip("_") if (SMOKE and own) else s


def figure_path(f, s=""):      # noqa: F811
    return _figure_path(f, _tag(f, s))


def array_path(f, s=""):       # noqa: F811
    return _array_path(f, _tag(f, s))


METHODS = ["backprop", "replay", "pc"]
SEEDS, EVAL_EVERY = 24, 10
COLORS = {"backprop": "tab:gray", "replay": "tab:brown", "pc": "tab:red"}

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

print(f"H = {HIDDEN} | CLASS-IL, {SEEDS} seeds | Domain-IL loaded from script 60")
print("  question: does the data split confound the comparison, and does pairing remove it?\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="class_il", stop_threshold=None,
               max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)


def class_means(train):
    x = train.x.reshape(len(train.x), -1).float()
    y = train.targets
    raw = np.stack([x[y == c].mean(0).numpy() for c in range(10)])
    return raw - raw.mean(0, keepdims=True)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def perm_r(x, y, n_perm=20000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return float("nan"), float("nan"), int(x.size)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(n_perm)])
    return r, float((null >= abs(r)).mean()), int(x.size)


# ---------------------------------------------------------------- run / load
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()
if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    cls = {m: z[f"argmax_{m}"] for m in METHODS}
    steps_c, sw_c = z["steps"], float(z["switch"])
    sim_cross, sim_within = z["sim_cross"], z["sim_within"]
    print("--replot: from saved arrays, no training\n")
else:
    data = load(base)
    cen = class_means(data.train)
    cls, tasks_all, t0 = {}, None, time.perf_counter()
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        rows, tk = [], []
        for seed in range(SEEDS):
            out = run(proto, m, seed, data=data,
                      **(dict(steps=PC_STEPS) if m == "pc" else {}))
            rows.append(out["curves"]["argmax"])
            tk.append(out["tasks"])
        steps_c, sw_c = out["steps"], out["switches"][0]
        cls[m] = np.stack(rows)
        tasks_all = tk if tasks_all is None else tasks_all
        print(f"  {m:10s} done  [{time.perf_counter() - t0:5.0f}s]")
    sim_cross = np.array([np.mean([cos(cen[a], cen[b]) for a in t[0] for b in t[1]])
                          for t in tasks_all])
    sim_within = np.array([np.mean([cos(cen[a], cen[b]) for t_ in t for a in t_ for b in t_
                                    if a != b]) for t in tasks_all])

z60 = np.load(_array_path(str(ROOT / "experiments" /
                              "60_why_does_the_seed_matter_so_much.py")), allow_pickle=True)
dom = {m: z60[f"argmax_{m}"] for m in METHODS}
sw_d = float(np.asarray(z60["switches"]).ravel()[0])
g_dom = {m: metric_grid(z60["steps"], dom[m], sw_d) for m in METHODS}
g_cls = {m: metric_grid(steps_c, cls[m], sw_c) for m in METHODS}
sim_pair_dom = z60["sim_paired"]

# ---------------------------------------------------------------- readings
print(f"\n  1. IS CLASS-IL CONFOUNDED BY THE SPLIT?   r (permutation p), n={SEEDS}")
print(f"  {'rule':10s} {'metric':12s} {'r(sim_cross)':>20s} {'r(sim_within)':>20s}")
for m in METHODS:
    for k in ["final_t1", "crossover"]:
        a = perm_r(sim_cross, g_cls[m][k])
        b = perm_r(sim_within, g_cls[m][k])
        print(f"  {m:10s} {k:12s} {a[0]:+9.3f} p={a[1]:5.3f}{'*' if a[1] < .05 else ' '} "
              f"{b[0]:+12.3f} p={b[1]:5.3f}{'*' if b[1] < .05 else ' '}")
print(f"\n  Domain-IL, for contrast (script 60): r(output-unit pairing, task-1 kept) = "
      f"{perm_r(sim_pair_dom, g_dom['backprop']['final_t1'])[0]:+.3f}")
print(f"  Class-IL retention range: "
      f"{np.nanmin(g_cls['backprop']['final_t1']):.1f} - "
      f"{np.nanmax(g_cls['backprop']['final_t1']):.1f}%   "
      f"Domain-IL: {np.nanmin(g_dom['backprop']['final_t1']):.1f} - "
      f"{np.nanmax(g_dom['backprop']['final_t1']):.1f}%")

print(f"\n  2. DOES PAIRING REMOVE IT?  pc and replay vs backprop, both scenarios")
print(f"  {'scenario':10s} {'rule':8s} {'metric':10s} "
      f"{'unpaired':>22s} {'paired':>22s} {'sem ratio':>10s}")
ratios = {}
for lab, g in (("domain_il", g_dom), ("class_il", g_cls)):
    for m in ["pc", "replay"]:
        for k in ["final_t1", "crossover"]:
            a, b = np.asarray(g[m][k], float), np.asarray(g["backprop"][k], float)
            ok = np.isfinite(a) & np.isfinite(b)
            n = int(ok.sum())
            if n < 3:
                continue
            # unpaired: difference of group means, error from the between-seed spread
            du = float(a[ok].mean() - b[ok].mean())
            su = float(np.sqrt(a[ok].var(ddof=1) / n + b[ok].var(ddof=1) / n))
            dp, sp, _ = paired_diff(a[ok], b[ok])
            ratios[(lab, m, k)] = su / sp if sp > 0 else np.nan
            print(f"  {lab:10s} {m:8s} {k:10s} "
                  f"{du:+8.2f} +-{su:5.2f} ({du / su if su else 0:4.1f}s) "
                  f"{dp:+8.2f} +-{sp:5.2f} ({dp / sp if sp else 0:4.1f}s) "
                  f"{su / sp if sp else np.nan:10.1f}x")
print("\n  'sem ratio' is how many times larger the unpaired error bar is. The paired and")
print("  unpaired point estimates are identical by construction; only the uncertainty differs,")
print("  because the split is shared across rules within a seed and cancels in the difference.")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
ax = axes[0]
for m in METHODS:
    ax.plot(sim_cross, g_cls[m]["final_t1"], "o", ms=6, color=COLORS[m], label=m)
ax.set_xlabel("cross-task class similarity"); ax.set_ylabel("task 1 kept (%)")
ax.set_title(f"Class-IL: does the split predict retention?\n"
             f"range {np.nanmin(g_cls['backprop']['final_t1']):.0f}-"
             f"{np.nanmax(g_cls['backprop']['final_t1']):.0f}%", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axes[1]
for m in METHODS:
    ax.plot(sim_pair_dom, g_dom[m]["final_t1"], "o", ms=6, color=COLORS[m], label=m)
ax.set_xlabel("output-unit pairing similarity"); ax.set_ylabel("task 1 kept (%)")
ax.set_title(f"Domain-IL, script 60: r = "
             f"{perm_r(sim_pair_dom, g_dom['backprop']['final_t1'])[0]:+.2f}\n"
             f"range {np.nanmin(g_dom['backprop']['final_t1']):.0f}-"
             f"{np.nanmax(g_dom['backprop']['final_t1']):.0f}%", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[2]
labels = [f"{s.split('_')[0]}\n{m}\n{k.replace('final_t1', 'retention')}"
          for (s, m, k) in ratios]
ax.bar(range(len(ratios)), list(ratios.values()), color="tab:purple", alpha=0.85)
ax.axhline(1.0, color="k", lw=1, ls=":")
ax.annotate("1.0 = pairing buys nothing", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points", fontsize=8)
ax.set_xticks(range(len(ratios))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("unpaired sem ÷ paired sem")
ax.set_title("How much the shared split was costing", fontsize=10)
ax.grid(alpha=0.25, axis="y")

fig.suptitle("The data split as a confound, and paired comparison as the control", fontsize=11)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

if not SMOKE and not REPLOT:
    np.savez(array_path(__file__), methods=np.array(METHODS), steps=steps_c, switch=sw_c,
             sim_cross=sim_cross, sim_within=sim_within,
             **{f"argmax_{m}": cls[m] for m in METHODS})
    print(f"saved {array_path(__file__)}")
