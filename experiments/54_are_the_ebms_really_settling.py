"""Are PC and EqProp, as parameterised here, computing genuinely different updates from
backprop -- or are they approximating it?

THE THREAT THIS EXISTS TO RULE OUT
    Script 52 found no difference between PC, EqProp and backprop in retention. There are two
    very different reasons that could happen:

      the rules really do differ and the difference does not buy retention at this scale
          -- a real result, and the one we would report;
      or the rules are not really different AS CONFIGURED, and 52 compared backprop to
          backprop three times
          -- in which case 52 measured nothing and every later comparison inherits the fault.

    Millidge et al. 2022 give the mechanism for the second: these rules converge on backprop as
    their relaxation is reduced, and `knowledge_base.md` §3.3 records that FULL relaxation is
    [R1]'s actual contribution. Script 50 already showed both rules reach equilibrium, so the
    relaxation is not truncated. That is necessary but not sufficient -- a rule can settle
    fully and still produce a backprop-shaped update, which is exactly what makes this worth
    measuring rather than assuming.

WHAT IS MEASURED
    From the rule's OWN weights at a point in its OWN training run, on one fixed batch:

      1. the update the rule makes,     dW_rule
      2. the update backprop would make from those identical weights on that identical batch,
         dW_bp -- obtained by copying the weights into a backprop harness, so nothing is
         re-derived by hand,
      3. cos(dW_rule, dW_bp) and |dW_rule| / |dW_bp|, per weight matrix.

    cos = 1 and ratio = 1 means the rule IS backprop. Lower cosine means it is assigning credit
    differently, which is the whole claim being tested.

    Measured along each rule's own trajectory, not from a shared backprop trajectory, because
    the question is whether the rule AS RUN is backprop-like -- not whether it would be if it
    started somewhere else.

THE CONTROL THAT VALIDATES THE INSTRUMENT
    Settling is swept. If the measurement is sound, reducing PC's relaxation must drive its
    update toward backprop's, and `pc_settle`'s docstring makes a sharp prediction for the
    endpoint: at steps=0 every hidden error is zero by construction, so W1 receives NOTHING and
    W2 receives exactly backprop's update. cos(dW2) = 1.000 and |dW1| = 0 at steps=0 is the
    instrument reading a known answer correctly. If that does not appear, the measurement is
    wrong and its other numbers mean nothing.

    Note the sweep is NOT symmetric between the rules. Reducing PC's relaxation approaches
    backprop. Reducing EqProp's does not approach anything in particular -- EqProp's estimate
    approaches the true gradient as beta -> 0 WITH full relaxation, so an under-relaxed EqProp
    is simply a worse estimator, not a backprop approximation. Read its sweep as "how much does
    settling change the update", not as a distance from backprop.

AND WHETHER THE RELAXATION MOVES ANYTHING AT ALL
    Reported alongside, because a rule whose settled state equals its feedforward state is not
    settling whatever its step count says:
      PC       |x_settled - mu| / |mu|, how far relaxation moves the hidden state from the
               feedforward prediction it was initialised at
      EqProp   |s_nudged - s_free| / |s_free|, how far the nudge moves the state -- the
               difference the weight update is computed FROM
    These are different quantities in different dynamical systems and are NOT comparable to
    each other. Each is compared against zero, and that is all.

READINGS COMMITTED BEFORE RUNNING
    If cos is near 1 for both rules at the operating point, 52 compared backprop to itself and
    must be discarded along with its conclusion.
    If cos is clearly below 1 and stays there across training, the rules differ and 52's null
    result stands as a real finding about this scale.
    If cos starts low and RISES toward 1 as the network trains, the rules differ at
    initialisation and converge on backprop by the time task 2 arrives -- which would make the
    comparison valid in principle but empty in practice, and is the outcome that would be
    easiest to miss.

Deviation from the protocol: task 1 only, no forgetting measurement. This is an instrument
check on the rules themselves.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, run, build, replace, figure_path as _figure_path, array_path as _array_path
from src.model import make_target, active_vector
from src.predictive_coding import pc_settle
from src.eqprop import eqprop_settle
from src.methods import METHOD_DEFAULTS

# --smoke writes NOTHING. Its outputs would otherwise overwrite the real .npz that later
# scripts read their settings from -- script 53 takes its task-2 threshold from 52's measured
# ceilings, and a smoke-sized 52 silently poisons it.
SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    """Suffix only THIS script's own outputs, never the .npz files it reads from others."""
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811  - shadows the import, on purpose
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))

# ---------------------------------------------------------------- settings
CHECKPOINTS = [0, 100, 250, 420]        # updates into task 1; 420 is the calibrated switch
SEEDS = 3
PROBE_BATCH = 32
RULES = ["pc", "eqprop"]
COLORS = {"pc": "tab:red", "eqprop": "tab:green"}
WSTYLE = {"W1": "-", "W2": "--"}

# settling swept per rule. PC in relaxation steps; EqProp in FORCED steps (settle_tol=0 makes
# the stopping test unreachable) so the axis is the same quantity for both.
SWEEP = {"pc": [0, 1, 2, 5, 10, 20, 50, 100],
         "eqprop": [1, 2, 5, 10, 20, 50, 100, 200, 400]}

HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(array_path(str(ROOT / "experiments" /
                             "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])

if "--smoke" in sys.argv:
    CHECKPOINTS, SEEDS = [0, 10], 1
    SWEEP = {"pc": [0, 5, 50], "eqprop": [5, 50]}
    print("--smoke: tiny budget, results are NOT meaningful\n")

BUDGET = max(CHECKPOINTS)
OP = {"pc": PC_STEPS, "eqprop": None}   # EqProp's operating point is a tolerance; measured below

print(f"H = {HIDDEN} | learning rates from 51 | operating point: pc {PC_STEPS} steps, "
      f"eqprop settle_tol {SETTLE_TOL:.0e}\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=BUDGET, eval_every=max(25, BUDGET // 4), seeds=SEEDS)


def settle_kw(rule, amount=None):
    """This rule's settling configuration. `amount` forces a fixed number of steps."""
    if rule == "pc":
        return dict(steps=PC_STEPS if amount is None else amount)
    if amount is None:
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return dict(max_steps=amount, settle_tol=0.0)   # tol 0 -> stopping test unreachable


def one_update(rule, weights, x, y, active, seed, **kw):
    """The weight change `rule` makes from exactly `weights`, on exactly this batch.

    A fresh harness each time, with the weights copied in, so measuring never perturbs the
    training trajectory the checkpoint came from."""
    h = {}
    train_step, _ = build(replace(base, lr={rule: LR[rule]}), rule, seed, handle=h, **kw)
    named = h["params"].named()
    for k, v in weights.items():
        named[k].data.copy_(v)
    before = {k: named[k].detach().clone() for k in weights}
    train_step(x, y, active=active)
    return {k: (named[k].detach() - before[k]) for k in weights}


def compare(rule, weights, x, y, active, seed, **kw):
    """cos and magnitude ratio of `rule`'s update against backprop's, from the same weights."""
    d_r = one_update(rule, weights, x, y, active, seed, **kw)
    d_b = one_update("backprop", weights, x, y, active, seed)
    out = {}
    for k in ["W1", "W2"]:
        a, b = d_r[k].flatten(), d_b[k].flatten()
        out[f"cos_{k}"] = float(torch.nn.functional.cosine_similarity(a, b, dim=0)) \
            if a.norm() > 0 and b.norm() > 0 else 0.0
        out[f"mag_{k}"] = float(a.norm() / b.norm()) if b.norm() > 0 else float("nan")
    return out


def displacement(rule, weights, arch, obj, x, y, active, seed):
    """How far the relaxation moves the state. Compared against zero, not against each other."""
    h = {}
    build(replace(base, lr={rule: LR[rule]}), rule, seed, handle=h, **settle_kw(rule))
    named = h["params"].named()
    for k, v in weights.items():
        named[k].data.copy_(v)
    p = h["params"]
    tgt = make_target(y, arch, obj, device=base.device)
    av = active_vector(active, arch, device=base.device)
    if rule == "pc":
        xs, mus = pc_settle(x, p, arch, obj, tgt, av,
                            dt=METHOD_DEFAULTS["pc"]["dt"], steps=PC_STEPS)
        num = sum(((a - b) ** 2).sum() for a, b in zip(xs, mus)).sqrt()
        den = sum((b ** 2).sum() for b in mus).sqrt()
        return float(num / den), np.nan
    kw = dict(dt=METHOD_DEFAULTS["eqprop"]["dt"], max_steps=EQ_MAX_STEPS,
              settle_tol=SETTLE_TOL, device=base.device)
    free, used, _ = eqprop_settle(x, p, arch, return_steps=True, **kw)
    nud = eqprop_settle(x, p, arch, obj=obj, target=tgt, active_vec=av,
                        beta=METHOD_DEFAULTS["eqprop"]["beta"], init=free, **kw)
    nh = arch.n_hidden
    num = sum(((a - b) ** 2).sum() for a, b in zip(nud[:nh], free[:nh])).sqrt()
    den = sum((b ** 2).sum() for b in free[:nh]).sqrt()
    return float(num / den), float(used)


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__))
    op = {r: {k: z[f"op_{r}_{k}"] for k in ["cos_W1", "cos_W2", "mag_W1", "mag_W2"]}
          for r in RULES}
    sweep = {r: {k: z[f"sweep_{r}_{k}"] for k in ["cos_W1", "cos_W2", "mag_W1", "mag_W2"]}
             for r in RULES}
    disp = {r: z[f"disp_{r}"] for r in RULES}
    eq_used = z["eq_used"]
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    KEYS = ["cos_W1", "cos_W2", "mag_W1", "mag_W2"]
    op = {r: {k: np.full((SEEDS, len(CHECKPOINTS)), np.nan) for k in KEYS} for r in RULES}
    sweep = {r: {k: np.full((SEEDS, len(CHECKPOINTS), len(SWEEP[r])), np.nan) for k in KEYS}
             for r in RULES}
    disp = {r: np.full((SEEDS, len(CHECKPOINTS)), np.nan) for r in RULES}
    eq_used = np.full((SEEDS, len(CHECKPOINTS)), np.nan)
    t0 = time.perf_counter()

    for rule in RULES:
        for seed in range(SEEDS):
            proto = replace(base, lr={rule: LR[rule]})
            pair = proto.tasks(seed)
            lmap = proto.label_map([pair[0]])
            active = sorted({lmap[int(c)] for c in pair[0]})
            handle = {}

            g = torch.Generator().manual_seed(2000 + seed)
            pool = torch.cat([torch.as_tensor(data.class_idx[c]) for c in pair[0]])
            pick = pool[torch.randperm(len(pool), generator=g)[:PROBE_BATCH]]
            px = torch.stack([data.train[i][0] for i in pick.tolist()]).to(base.device)
            py = torch.tensor([lmap[int(data.train[i][1])] for i in pick.tolist()],
                              device=base.device)

            def measure(ci):
                w = {k: v.detach().clone() for k, v in handle["params"].named().items()
                     if v is not None}
                arch, obj = handle["arch"], handle["obj"]
                r = compare(rule, w, px, py, active, seed, **settle_kw(rule))
                for k in KEYS:
                    op[rule][k][seed, ci] = r[k]
                for si, amount in enumerate(SWEEP[rule]):
                    rs = compare(rule, w, px, py, active, seed, **settle_kw(rule, amount))
                    for k in KEYS:
                        sweep[rule][k][seed, ci, si] = rs[k]
                d, used = displacement(rule, w, arch, obj, px, py, active, seed)
                disp[rule][seed, ci] = d
                if rule == "eqprop":
                    eq_used[seed, ci] = used

            n = [0]

            def wrap(train_step):
                measure(0)

                def wrapped(x, y, active=None):
                    train_step(x, y, active=active)
                    n[0] += 1
                    if n[0] in CHECKPOINTS:
                        measure(CHECKPOINTS.index(n[0]))
                return wrapped

            run(proto, rule, seed, data=data, handle=handle, tasks=[pair[0]], wrap=wrap,
                **settle_kw(rule))
            print(f"  {rule:8s} seed {seed}  cos(dW1,bp) "
                  + " ".join(f"{v:5.2f}" for v in op[rule]["cos_W1"][seed])
                  + f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- readings
print(f"\n  AT THE OPERATING POINT -- the settings scripts 52 and 53 actually ran\n")
print(f"  {'rule':8s} {'updates':>8s} {'cos dW1':>9s} {'cos dW2':>9s} {'|dW1|/bp':>10s} "
      f"{'|dW2|/bp':>10s} {'relaxation moves':>18s}")
for r in RULES:
    for ci, cp in enumerate(CHECKPOINTS):
        print(f"  {r:8s} {cp:8d} " + " ".join(
            f"{np.nanmean(op[r][k][:, ci]):9.3f}" for k in ["cos_W1", "cos_W2"])
            + " " + " ".join(f"{np.nanmean(op[r][k][:, ci]):10.3f}"
                             for k in ["mag_W1", "mag_W2"])
            + f" {np.nanmean(disp[r][:, ci]):17.1%}")

print(f"\n  INSTRUMENT CHECK -- pc at steps=0 must give cos(dW2)=1.000 and |dW1|=0, because "
      f"\n  every hidden error is zero by construction there (predictive_coding.pc_settle).")
i0 = SWEEP["pc"].index(0) if 0 in SWEEP["pc"] else None
if i0 is not None:
    c2 = np.nanmean(sweep["pc"]["cos_W2"][:, :, i0])
    m1 = np.nanmean(sweep["pc"]["mag_W1"][:, :, i0])
    ok = abs(c2 - 1.0) < 1e-3 and m1 < 1e-6
    print(f"    cos(dW2) = {c2:.4f}   |dW1|/bp = {m1:.2e}   "
          + ("OK, the measurement reads a known answer correctly" if ok else
             "FAILED -- the measurement is wrong and nothing else here can be trusted"))

print(f"\n  VERDICT, read at the task switch ({CHECKPOINTS[-1]} updates)")
for r in RULES:
    c1, c2 = (np.nanmean(op[r]["cos_W1"][:, -1]), np.nanmean(op[r]["cos_W2"][:, -1]))
    worst = min(c1, c2)
    print(f"    {r:8s} cos = {c1:.3f} (W1), {c2:.3f} (W2)  -> "
          + ("INDISTINGUISHABLE from backprop; script 52 compared backprop to itself"
             if worst > 0.99 else
             "differs from backprop, but only slightly" if worst > 0.95 else
             "clearly different credit assignment; 52's null result is about the rules, "
             "not about the setup"))
drift = {r: np.nanmean(op[r]["cos_W1"][:, -1]) - np.nanmean(op[r]["cos_W1"][:, 0])
         for r in RULES}
print("  across training, cos(dW1) moves " + ", ".join(f"{r} {drift[r]:+.3f}" for r in RULES)
      + ("  -- converging on backprop as it trains" if max(drift.values()) > 0.05 else
         "  -- no drift toward backprop"))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

for r in RULES:
    for w in ["W1", "W2"]:
        ax[0].plot(SWEEP[r], np.nanmean(sweep[r][f"cos_{w}"][:, -1], axis=0),
                   WSTYLE[w], marker="o", ms=4, lw=2, color=COLORS[r], label=f"{r} {w}")
    if r == "pc":
        ax[0].axvline(PC_STEPS, color=COLORS[r], ls=":", lw=1.4)
    else:
        ax[0].axvline(np.nanmean(eq_used), color=COLORS[r], ls=":", lw=1.4)
ax[0].axhline(1.0, color="k", lw=1)
ax[0].set_xscale("symlog", linthresh=1)
ax[0].set_xlabel("settling steps (forced)")
ax[0].set_ylabel("cos(update, backprop's update)")
ax[0].set_title("less settling -> more backprop-like")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=0.25)

for r in RULES:
    for w in ["W1", "W2"]:
        m = np.nanmean(op[r][f"cos_{w}"], axis=0)
        e = np.nanstd(op[r][f"cos_{w}"], axis=0, ddof=1) / np.sqrt(SEEDS)
        ax[1].errorbar(CHECKPOINTS, m, yerr=e, fmt=WSTYLE[w], marker="o", ms=4, capsize=3,
                       lw=2, color=COLORS[r], label=f"{r} {w}")
ax[1].axhline(1.0, color="k", lw=1)
ax[1].set_xlabel("training updates completed")
ax[1].set_ylabel("cos(update, backprop's update)")
ax[1].set_title("at the settings 52 and 53 ran")
ax[1].set_ylim(-0.1, 1.1)
ax[1].legend(fontsize=7)
ax[1].grid(alpha=0.25)

for r in RULES:
    m = np.nanmean(disp[r], axis=0) * 100
    e = np.nanstd(disp[r], axis=0, ddof=1) / np.sqrt(SEEDS) * 100
    ax[2].errorbar(CHECKPOINTS, m, yerr=e, marker="o", capsize=3, lw=2, color=COLORS[r],
                   label=r)
ax[2].axhline(0, color="k", lw=1)
ax[2].set_xlabel("training updates completed")
ax[2].set_ylabel("relaxation moves the state (%)")
ax[2].set_title("pc: settled vs feedforward | eqprop: nudged vs free")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")

np.savez(array_path(__file__), checkpoints=np.asarray(CHECKPOINTS), hidden=HIDDEN,
         eq_used=eq_used, pc_steps=PC_STEPS, settle_tol=SETTLE_TOL,
         **{f"sweep_grid_{r}": np.asarray(SWEEP[r]) for r in RULES},
         **{f"op_{r}_{k}": op[r][k] for r in RULES for k in op[r]},
         **{f"sweep_{r}_{k}": sweep[r][k] for r in RULES for k in sweep[r]},
         **{f"disp_{r}": disp[r] for r in RULES})
print(f"\nsaved {figure_path(__file__)}\nsaved {array_path(__file__)}")
