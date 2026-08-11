"""Do the two energy-based rules actually reach equilibrium, and in how many steps?

Technical intro. This is the first script of the new series, and it runs BEFORE any rule
comparison because it decides whether the rules as configured are the rules we mean.

THE PROBLEM, STATED IN CODE LINES
    Both energy-based rules compute their weight update from a SETTLED state, not from a
    feedforward pass. They stop settling by different criteria:

      predictive_coding.pc_settle    `for _ in range(steps)`         steps = 50, FIXED
      eqprop.eqprop_settle           `move <= settle_tol * |state|`  stop when one step moves
                                                                     the state by almost nothing

    A fixed count is only correct if the relaxation has finished by then. If it has not, the
    weight update is computed from a partially settled state -- and a partially settled state
    is closer to the feedforward state, which is what backprop uses. Millidge et al. 2022 show
    every one of these rules approaches backprop as settling is reduced, and knowledge_base.md
    §3.3 records that FULL relaxation is [R1]'s actual contribution. So if PC's 50 steps are
    short, "PC vs backprop" is partly "backprop vs backprop" and the comparison is weakened
    before it starts.

    An adaptive rule tracks the requirement, but costs whatever it costs. Measured at
    initialisation, EqProp's free phase used 426 steps of a 500 cap, at 0.9 ms each -- 647 ms
    per weight update against backprop's 0.93 ms. That is 700x, and every later experiment
    pays it. So this script also has to say what the cheapest SAFE setting is.

WHAT THIS SCRIPT FOUND, AND WHAT CHANGED BECAUSE OF IT
    PC needs at most 18 settling steps and runs 50, so it is over-settled by 3-6x and is NOT a
    truncated relaxation. The concern that PC's fixed step count makes it a backprop
    approximation (Millidge et al. 2022) does not apply at this size. Its fixed budget stays.

    EqProp's requirement spans 9x -- 529 steps at initialisation, 47-74 once trained -- and its
    ORIGINAL stopping rule was wrong in both directions. That rule stopped once per-step
    movement had failed to improve by min_delta for `settle_patience` steps. At initialisation
    it fired on a temporary plateau: seed 2 needed 529 steps and it stopped at 178, computing
    the weight update from a state a third of the way to equilibrium. Everywhere else it
    overshot by 3-4x, because movement keeps decaying long after the state has arrived.

    It was replaced by a relative-residual test -- stop when one step displaces the state by
    less than `settle_tol` of its own norm -- which is small only once the state has settled.
    The tolerance section below calibrates that constant, which is why this script and not a
    comment is where the number comes from.

WHY THE TOLERANCE IS CALIBRATED HERE AND NOT ASSUMED
    settle_tol is not dimensionless. Near the fixed point the error decays geometrically, so
    stopping at move <= tol*|state| leaves a distance of about tol/(dt*L) from equilibrium,
    where L is the slowest curvature of the energy. It therefore scales with dt and with the
    network. This script runs the real criterion at each candidate tolerance and measures HOW
    FAR FROM THE SETTLED STATE it leaves the network, per seed and per checkpoint. The choice
    is the largest -- cheapest -- tolerance whose WORST case is still settled, with an explicit
    safety factor, because that worst case comes from only a few seeds and under-settling is
    invisible in the output curves. Change dt, the width or the depth, re-run this, and take
    the number it prints.

    Do not calibrate on the STEP COUNT the criterion picks. Step counts are discretised by the
    grid and the distance falls steeply near convergence, so that comparison is over-sensitive
    exactly where the decision is made: it rejected a tolerance that in fact stops 1.5% from
    equilibrium, comfortably inside the 2% this script calls settled.

WHAT IS MEASURED
    For one fixed batch, held identical across every measurement, and at four points during
    task-1 training (0, 50, 200, 500 updates):

      run the relaxation for k steps, for k on a log grid, and record

          distance to equilibrium  =  ||x(k) - x(K_MAX)|| / ||x(K_MAX)||

      where x(K_MAX) is the state after a deliberately over-long relaxation. This is exact at
      every k -- no differencing between neighbouring grid points -- and it says the thing we
      want to know directly: after k steps, how far from the settled state are you still?

      STEPS TO EQUILIBRIUM is then the first k at which that distance falls below TOL.

    The measurement is the same for both rules, over the HIDDEN layers only. In PC the output
    is clamped to the target and is not a free variable; in EqProp it is. Comparing hidden
    states compares the same object: the representation the weight update reads from.

    Forcing exactly k steps needs no new library code. PC takes `steps=k`. EqProp takes
    `max_steps=k, settle_tol=0`, so the stopping test can only fire on exactly zero movement
    and the loop runs exactly k times.

    EqProp settles twice per update. The FREE phase is measured, because it starts from zero
    and does the work; the nudged phase warm-starts from it and is reported as a number only.

THE GATE ON THIS SCRIPT'S OWN VALIDITY
    x(K_MAX) is only the equilibrium if the relaxation has stopped by K_MAX. So the movement
    over the final step, ||x(K_MAX) - x(K_MAX - 1)||, is computed and printed. If that is not
    small, K_MAX is too low, every distance is measured against a moving reference, and the
    curves would fall to zero at the right edge for a purely arithmetic reason. Check that
    line before reading the figure.

READINGS COMMITTED BEFORE RUNNING
    If steps-to-equilibrium is roughly CONSTANT across training for both rules, then a fixed
    step count is defensible and the only question is what the constant is -- set PC's `steps`
    and EqProp's `max_steps` to it and move on.
    If it CHANGES with training -- and it should grow, because settling is driven by the
    weights and the weights grow -- then no single fixed count is right at both ends of a run,
    which is the argument for a patience rule.
    If PC is far from equilibrium at 50 steps, PC as currently configured is a truncated
    relaxation and the settling amount has to become an explicit experimental variable before
    any rule comparison is trusted.
    If EqProp reaches equilibrium in far fewer than 426 steps, its patience rule is
    over-conservative and every later experiment involving it can be made several times
    cheaper.

Deviation from the protocol: no task sequence and no forgetting measurement. One task, trained
under a fixed budget, purely to move the weights so the relaxation can be measured at several
weight magnitudes. Domain-IL and H=32 as the protocol says, so the numbers transfer directly
to the rule comparison that follows.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, run, replace, figure_path, array_path
from src.model import make_target, active_vector
from src.predictive_coding import pc_settle
from src.eqprop import eqprop_settle
from src.methods import METHOD_DEFAULTS

# ---------------------------------------------------------------- settings
CHECKPOINTS = [0, 50, 200, 500]     # updates into task 1 at which settling is measured
SEEDS = 3                           # a convergence diagnostic, not a comparison: 3 shows spread
TOL = 0.02                          # "settled" = within 2% of the equilibrium state
PROBE_BATCH = 32                    # one fixed batch, identical at every measurement

# K_MAX is the reference relaxation, set generously. The last-step movement printed below is
# the check that it is generous ENOUGH; if it is not, raise these.
K_MAX = {"pc": 400, "eqprop": 1200}

# EqProp stops settling when one step moves the state by less than `settle_tol` of its own
# norm. That number is not a universal constant -- it scales with dt and with the curvature of
# the energy (see eqprop.eqprop_settle) -- so it is CALIBRATED HERE rather than assumed, by
# running the criterion at each of these tolerances and comparing what it picks against what
# the convergence curve says is genuinely needed.
TOL_GRID = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5]

# EqProp costs ~650 ms per update here, so the full run is ~20 minutes. `--smoke` runs the
# same code paths in under a minute; it proves the script executes, and nothing else.
if "--smoke" in sys.argv:
    CHECKPOINTS, SEEDS, K_MAX = [0, 5, 10], 1, {"pc": 60, "eqprop": 80}
    print("--smoke: tiny budget, results are NOT meaningful\n")

BUDGET = max(CHECKPOINTS)
GRID = {r: sorted({int(round(v)) for v in np.unique(np.concatenate([
            np.arange(1, 21), np.geomspace(20, K_MAX[r], 26)]))}) for r in K_MAX}

# what each rule currently does, drawn on the figure as the setting under test
CURRENT = {"pc": METHOD_DEFAULTS["pc"]["steps"]}
DEFAULT_TOL = METHOD_DEFAULTS["eqprop"]["settle_tol"]
assert DEFAULT_TOL in TOL_GRID, "the library default must be on the grid it is chosen from"
COLORS = {"pc": "tab:red", "eqprop": "tab:green"}
CP_CMAP = plt.get_cmap("viridis")

cap = np.load(array_path(str(ROOT / "experiments" / "41_capacity_vs_hidden_width.py")))
HIDDEN = int(cap["chosen"])
print(f"H = {HIDDEN} from script 41\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=BUDGET, eval_every=25, seeds=SEEDS)


# ---------------------------------------------------------------- the measurement
def hidden_states(rule, x, y, active, handle, k, arch, obj, device="cpu"):
    """The hidden-layer state after EXACTLY k steps of this rule's relaxation.

    Returned as one flat vector per rule so the two are compared by the same code. PC's output
    is clamped to the target and so is excluded by construction; EqProp's output is free, and
    `arch.n_hidden` slices it off so both sides measure the representation and nothing else.

    `active` is the task's output units, not the units present in this batch -- it is what
    run_classil passes to train_step, so the relaxation measured here is the one the weight
    update actually runs.
    """
    p = handle["params"]
    if rule == "pc":
        target = make_target(y, arch, obj, device=device)
        av = active_vector(active, arch, device=device)
        xs, _ = pc_settle(x, p, arch, obj, target, av,
                          dt=METHOD_DEFAULTS["pc"]["dt"], steps=k)
        return torch.cat([s.reshape(-1) for s in xs])
    # settle_tol=0 means "stop only if the state moves by exactly nothing", so the loop runs
    # all max_steps: exactly k steps.
    states = eqprop_settle(x, p, arch, dt=METHOD_DEFAULTS["eqprop"]["dt"],
                           max_steps=k, settle_tol=0.0, device=device)
    return torch.cat([s.reshape(-1) for s in states[:arch.n_hidden]])


def convergence_curve(rule, x, y, active, handle, arch, obj):
    """distance-to-equilibrium against settling step, plus the two validity numbers."""
    def state(k):
        return hidden_states(rule, x, y, active, handle, k, arch, obj, device=base.device)

    ref, prev = state(K_MAX[rule]), state(K_MAX[rule] - 1)
    scale = ref.norm().item() + 1e-12
    dist = np.array([(state(k) - ref).norm().item() / scale for k in GRID[rule]])
    # first grid point at or below TOL, and everything after it also below -- a single dip
    # under the tolerance on a noisy curve is not arrival.
    ok = dist <= TOL
    settled = next((GRID[rule][i] for i in range(len(ok)) if ok[i:].all()), None)
    return dist, settled, (prev - ref).norm().item() / scale


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    dists = {r: z[f"dist_{r}"] for r in K_MAX}
    settled = {r: z[f"settled_{r}"] for r in K_MAX}
    chose = {r: z[f"chose_{r}"] for r in K_MAX}
    acc = {r: z[f"acc_{r}"] for r in K_MAX}
    ref_move = {r: z[f"refmove_{r}"] for r in K_MAX}
    picked, nudged = z["picked"], z["nudged"]
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    dists = {r: np.full((SEEDS, len(CHECKPOINTS), len(GRID[r])), np.nan) for r in K_MAX}
    settled = {r: np.full((SEEDS, len(CHECKPOINTS)), np.nan) for r in K_MAX}
    chose = {r: np.full((SEEDS, len(CHECKPOINTS)), np.nan) for r in K_MAX}   # as configured
    picked = np.full((SEEDS, len(CHECKPOINTS), len(TOL_GRID)), np.nan)       # eqprop, per tol
    nudged = np.full((SEEDS, len(CHECKPOINTS)), np.nan)                      # eqprop, phase 2
    ref_move = {r: np.full((SEEDS, len(CHECKPOINTS)), np.nan) for r in K_MAX}
    acc = {r: np.full((SEEDS, len(CHECKPOINTS)), np.nan) for r in K_MAX}
    t0 = time.perf_counter()

    for rule in K_MAX:
        for seed in range(SEEDS):
            pair = base.tasks(seed)
            lmap = base.label_map([pair[0]])
            handle = {}

            # one fixed probe batch of task-1 images, drawn once and reused at every
            # checkpoint, so a change in the settling curve is a change in the WEIGHTS
            g = torch.Generator().manual_seed(1000 + seed)
            pool = torch.cat([torch.as_tensor(data.class_idx[c]) for c in pair[0]])
            pick = pool[torch.randperm(len(pool), generator=g)[:PROBE_BATCH]]
            px = torch.stack([data.train[i][0] for i in pick.tolist()]).to(base.device)
            py = torch.tensor([lmap[int(data.train[i][1])] for i in pick.tolist()],
                              device=base.device)

            active = sorted({lmap[int(c)] for c in pair[0]})       # as run_classil computes it

            def measure(ci):
                arch, obj = handle["arch"], handle["obj"]
                d, s, rm = convergence_curve(rule, px, py, active, handle, arch, obj)
                dists[rule][seed, ci] = d
                settled[rule][seed, ci] = np.nan if s is None else s
                ref_move[rule][seed, ci] = rm
                if rule != "eqprop":
                    chose[rule][seed, ci] = CURRENT["pc"]
                    return
                # THE CALIBRATION: run the real stopping criterion at each candidate
                # tolerance and record what it picks. Measured, not estimated from the
                # convergence curve above -- the criterion tests per-step movement, which
                # that curve does not contain.
                kw = dict(dt=METHOD_DEFAULTS["eqprop"]["dt"],
                          max_steps=K_MAX["eqprop"], device=base.device)
                tgt = make_target(py, arch, obj, device=base.device)
                av = active_vector(active, arch, device=base.device)
                for ti, tol in enumerate(TOL_GRID):
                    free, u_f, _ = eqprop_settle(px, handle["params"], arch, settle_tol=tol,
                                                 return_steps=True, **kw)
                    picked[seed, ci, ti] = u_f
                    if tol == DEFAULT_TOL:
                        chose[rule][seed, ci] = u_f
                        _, u_n, _ = eqprop_settle(px, handle["params"], arch, obj=obj,
                                                  target=tgt, active_vec=av, settle_tol=tol,
                                                  beta=METHOD_DEFAULTS["eqprop"]["beta"],
                                                  init=free, return_steps=True, **kw)
                        nudged[seed, ci] = u_n

            # `run` calls wrap(train_step) once, AFTER build has filled `handle` -- so the
            # checkpoint-0 measurement goes here, on the initial weights, before any update.
            n = [0]

            def wrap(train_step):
                measure(0)

                def wrapped(x, y, active=None):
                    train_step(x, y, active=active)
                    n[0] += 1
                    if n[0] in CHECKPOINTS:
                        measure(CHECKPOINTS.index(n[0]))
                return wrapped

            out = run(base, rule, seed, data=data, handle=handle, tasks=[pair[0]], wrap=wrap)
            # context only; the first eval is at step eval_every, so checkpoint 0 clamps to it
            if len(out["steps"]):
                acc[rule][seed] = np.interp(CHECKPOINTS, out["steps"],
                                            out["curves"]["argmax"][:, 0] * 100)
            print(f"  {rule:8s} seed {seed}  settled at "
                  + " ".join(f"{v:>5.0f}" for v in settled[rule][seed])
                  + f"   task-1 acc {acc[rule][seed, -1]:5.1f}%"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- readings
print(f"\n  VALIDITY GATE -- movement over the final reference step, relative. Must be << {TOL}.")
for r in K_MAX:
    m = np.nanmax(ref_move[r])
    print(f"    {r:8s} K_MAX={K_MAX[r]:5d}  worst {m:.2e}   "
          + ("OK" if m < TOL / 10 else "TOO LOW -- raise K_MAX, curves are unreliable"))

print(f"\n  steps to reach within {TOL:.0%} of the settled state, by point in training\n")
print(f"  {'rule':10s}" + "".join(f"{f'@{c}':>10s}" for c in CHECKPOINTS) + "   currently uses")
for r in K_MAX:
    cur = (f"{CURRENT['pc']} fixed" if r == "pc"
           else f"settle_tol {DEFAULT_TOL:.0e} -> {np.nanmean(chose[r]):.0f}")
    print(f"  {r:10s}" + "".join(f"{v:10.0f}" for v in np.nanmean(settled[r], axis=0))
          + f"   {cur}")
for r in K_MAX:
    print(f"  {r:10s} task-1 accuracy " + "".join(f"{v:9.1f}%" for v in np.nanmean(acc[r], 0)))

# ---- the calibration: which tolerance, and why that one -------------------
# THE QUESTION IS NOT "does it stop at the same STEP the convergence curve did". Step counts
# are discretised by GRID, and the distance to equilibrium falls steeply near convergence, so
# comparing step numbers is over-sensitive exactly at the boundary where the decision is made
# -- it rejected a tolerance that stops 1.5% from equilibrium, well inside the 2% that this
# script defines as settled.
#
# The question is HOW FAR FROM THE SETTLED STATE the criterion leaves the network. That is
# measured directly, in the same units as TOL, by reading the convergence curve at the step
# the criterion actually picked.
stop_dist = np.full((SEEDS, len(CHECKPOINTS), len(TOL_GRID)), np.nan)
for i in range(SEEDS):
    for j in range(len(CHECKPOINTS)):
        stop_dist[i, j] = np.interp(picked[i, j], GRID["eqprop"], dists["eqprop"][i, j])
worst_d, mean_d = np.nanmax(stop_dist, axis=(0, 1)), np.nanmean(stop_dist, axis=(0, 1))

# MARGIN is an explicit safety factor on a worst case measured from only SEEDS seeds. The risk
# is asymmetric: under-settling is INVISIBLE in the output curves -- it just quietly makes the
# rule something other than the rule -- while over-settling only costs time. So it is worth
# paying for. A tolerance so tight the criterion never fires is measuring K_MAX, not itself.
MARGIN = 2.0
capped = [i for i in range(len(TOL_GRID)) if np.nanmax(picked[:, :, i]) >= K_MAX["eqprop"]]
safe = [i for i in range(len(TOL_GRID)) if worst_d[i] <= TOL / MARGIN and i not in capped]
best = max(safe, key=lambda i: TOL_GRID[i]) if safe else None

print(f"\n  CALIBRATING settle_tol, over {SEEDS} seeds x {len(CHECKPOINTS)} checkpoints.")
print(f"  How far from the settled state is the network when the criterion stops it?")
print(f"  Settled means <= {TOL:.0%}; a {MARGIN:.0f}x safety factor requires <= {TOL/MARGIN:.0%}.\n")
print(f"  {'settle_tol':>11s} {'worst':>9s} {'mean':>9s} {'steps':>8s} {'ms/update':>10s}")
for i, tol in enumerate(TOL_GRID):
    note = ("  HIT THE CAP, not a measurement" if i in capped else
            "  NOT SETTLED" if worst_d[i] > TOL else
            "  thin margin" if i not in safe else "")
    print(f"  {tol:11.0e} {worst_d[i]:8.3%} {mean_d[i]:8.3%} {np.nanmax(picked[:,:,i]):8.0f}"
          f" {np.nanmean(picked[:,:,i])*1.8*0.9:9.0f}" + note
          + ("  <-- library default" if tol == DEFAULT_TOL else ""))
if best is None:
    print("\n  NO tolerance on the grid is safe. Extend TOL_GRID downward before trusting "
          "any EqProp result.")
else:
    print(f"\n  -> settle_tol = {TOL_GRID[best]:.0e}; the library default is {DEFAULT_TOL:.0e}"
          + ("  (they agree)" if TOL_GRID[best] == DEFAULT_TOL else
             "  MISMATCH: update METHOD_DEFAULTS['eqprop']"))

print(f"\n  cost, at 0.9 ms per settling step and 2 settlings per EqProp update:")
f_, n_ = np.nanmean(chose["eqprop"]), np.nanmean(nudged)
print(f"    eqprop as configured  {f_:.0f} free + {n_:.0f} nudged"
      f" = {(f_+n_)*0.9:.0f} ms/update")
print(f"    lower bound, if it stopped exactly at the settled state: "
      f"{2*np.nanmean(settled['eqprop'])*0.9:.0f} ms/update")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

for a, r in zip(ax[:2], ["pc", "eqprop"]):
    for ci, cp in enumerate(CHECKPOINTS):
        col = CP_CMAP(ci / max(1, len(CHECKPOINTS) - 1))
        a.plot(GRID[r], np.nanmean(dists[r][:, ci], axis=0), lw=2, color=col,
               label=f"after {cp} updates")
        s = np.nanmean(settled[r][:, ci])
        if np.isfinite(s):
            a.plot([s], [TOL], "o", color=col, ms=6)
    a.axhline(TOL, color="gray", ls=":", lw=1)
    used = np.nanmean(chose[r])
    a.axvline(used, color="k", ls="--", lw=1.4)
    a.annotate("what the rule uses", xy=(used, 0.02), xycoords=("data", "axes fraction"),
               xytext=(4, 0), textcoords="offset points", fontsize=8, rotation=90)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("settling steps")
    a.set_title(r)
    a.grid(alpha=0.25, which="both")
ax[0].set_ylabel("distance from the settled state")
ax[0].legend(fontsize=8, loc="lower left")

for r in K_MAX:
    m = np.nanmean(settled[r], axis=0)
    e = np.nanstd(settled[r], axis=0, ddof=1) / np.sqrt(SEEDS)
    ax[2].errorbar(CHECKPOINTS, m, yerr=e, marker="o", capsize=3, lw=2, color=COLORS[r], label=r)
ax[2].plot(CHECKPOINTS, np.nanmean(chose["eqprop"], 0), marker="s", ms=5, ls="--",
           lw=1.4, color=COLORS["eqprop"], label=f"eqprop, settle_tol {DEFAULT_TOL:.0e}")
ax[2].axhline(CURRENT["pc"], ls="--", lw=1.4, color=COLORS["pc"],
              label=f"pc, fixed {CURRENT['pc']}")
ax[2].set_xlabel("training updates completed")
ax[2].set_ylabel("steps to reach the settled state")
ax[2].set_yscale("log")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

np.savez(array_path(__file__), checkpoints=np.asarray(CHECKPOINTS), tol=TOL, hidden=HIDDEN,
         tol_grid=np.asarray(TOL_GRID), picked=picked, nudged=nudged, stop_dist=stop_dist,
         margin=MARGIN, settle_tol=(np.nan if best is None else TOL_GRID[best]),
         **{f"grid_{r}": np.asarray(GRID[r]) for r in K_MAX},
         **{f"dist_{r}": dists[r] for r in K_MAX},
         **{f"settled_{r}": settled[r] for r in K_MAX},
         **{f"chose_{r}": chose[r] for r in K_MAX},
         **{f"refmove_{r}": ref_move[r] for r in K_MAX},
         **{f"acc_{r}": acc[r] for r in K_MAX})
print(f"saved {array_path(__file__)}")
