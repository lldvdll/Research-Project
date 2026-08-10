"""31_analyse_reproduction

ANALYSIS ONLY -- reads 30_reproduce_bogacz_fig4de.npz and re-runs nothing.

ONE QUESTION
    Is the switch-driven learning/forgetting trade-off actually present in the exp-30 data,
    and is it merely hidden by the fact that both curves are still descending?

WHY THIS BEFORE CHANGING ANY SETTING
    Exp 30 reproduced the DIRECTION of the claim -- predictive coding better in 10/10 seeds,
    paired difference +0.0222 with a 68% CI of [+0.0202, +0.0241], best learning rate 0.001
    sitting interior to their grid with a clean minimum. What it did not reproduce is the
    SHAPE: their curves show the trained task dropping and plateauing while the untrained task
    rises, and ours look like two noisy descents.

    A sawtooth can only appear once there is something to lose. If at iteration 84 the error
    is still falling steeply, the alternation is a small ripple riding on a large downward
    trend and cannot be seen directly. Exp 30 stored 160 iterations but analysed only 84, and
    it stored every weight update, so all of this is answerable from the file we already have.

WHAT IT COMPUTES
    A. full 160 iterations, both tasks         -- has it converged, or is 84 simply too early?
    B. block-aligned average                   -- every task block laid on top of each other
                                                  and averaged over blocks AND seeds. This is
                                                  the sensitive test: it averages away the
                                                  common trend and the noise, leaving only
                                                  what the switch does. If forgetting exists
                                                  at all, it shows up here.
    C. trade-off index = err(untrained task) - err(trained task)
                                               -- the common descent cancels exactly, so this
                                                  isolates the trade-off from the learning.
                                                  Positive and growing within a block means
                                                  the untrained task is being given up for the
                                                  trained one. This is the cleanest possible
                                                  statement of the claim.
    D. convergence diagnostic                  -- slope of the error over the trailing window,
                                                  reported as points per 10 iterations.

HOW TO READ IT
    D near zero at iteration 84
        -> converged; the flat curves are real and the setup needs changing, not the window.
    D still steeply negative
        -> undertrained. The fix is more weight updates per block, not a different learning
           rate. See the partial_num question in the notes printed at the end.
    B and C show a clear sawtooth
        -> the trade-off IS in our data and the only problem is that panel 4d plots it against
           a descending baseline. Their figure may differ from ours mainly in being converged.
    B and C flat
        -> there genuinely is no switch structure, and something in the setup is wrong rather
           than merely early.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

SRC = ROOT / "experiments" / "30_reproduce_bogacz_fig4de.npz"
FIG = Path(__file__).resolve().with_suffix(".png")
TRAIL = 20                      # trailing window for the convergence slope, in iterations

d = np.load(SRC, allow_pickle=True)
per_iter = d["per_iter"]        # [rule, lr, seed, iteration, task]
errors = d["errors"]            # [rule, lr, seed, update, task]
rules = [str(r) for r in d["rules"]]
lrs = list(d["lrs"])
cfg = json.loads(str(d["config"]))
ITERS_PER_TASK = cfg["iters_per_task"]
ANALYSE = cfg["analyse_iters"]
n_iter = per_iter.shape[3]
n_upd = errors.shape[3]
upd_per_iter = n_upd // n_iter

COL = {"backprop": "tab:red", "pc": "tab:blue"}
LABEL = {"backprop": "backpropagation", "pc": "prospective configuration"}

# best lr by their metric, so this analysis matches exp 30's headline
mean_err = np.nanmean(per_iter[:, :, :, :ANALYSE, :], axis=(3, 4))
best = {r: int(np.nanargmin(np.nanmean(mean_err[i], axis=1))) for i, r in enumerate(rules)}
print(f"source: {SRC.name}")
print(f"  {n_iter} iterations stored, {ANALYSE} analysed; {upd_per_iter} weight updates per "
      f"iteration; {ITERS_PER_TASK} iterations per task block")
print(f"  best lr: " + ", ".join(f"{r}={lrs[best[r]]}" for r in rules))

# ---------------------------------------------------------------- D. convergence
print("\n" + "=" * 78)
print(f"D. CONVERGENCE -- slope of test error over the trailing {TRAIL} iterations")
print(f"{'rule':>28}{'at iter 84':>16}{'at iter ' + str(n_iter):>16}")
converged = {}
for ri, r in enumerate(rules):
    row = f"{LABEL[r]:>28}"
    for end in (ANALYSE, n_iter):
        y = np.nanmean(per_iter[ri, best[r], :, :end, :], axis=(0, 2))
        xs = np.arange(TRAIL)
        slope = np.polyfit(xs, y[-TRAIL:], 1)[0] * 10 * 100     # points per 10 iterations
        converged[(r, end)] = slope
        row += f"{slope:>13.2f} pts"
    print(row)
print("  0 = flat. Large negative = still learning fast, so the alternation is a small")
print("  ripple on a big trend and cannot be seen in a raw plot of test error.")

# ---------------------------------------------------------------- B. block alignment
def block_align(a, iters_per_task):
    """[seed, iteration, task] -> [instance, position_in_block, trained/untrained].

    Every block of `iters_per_task` iterations is cut out and stacked, with the two task
    columns reordered into (task being trained, task not being trained) so that blocks
    belonging to task 0 and task 1 can be averaged together."""
    n_seed, n_it, _ = a.shape
    n_blocks = n_it // iters_per_task
    out = []
    for s in range(n_seed):
        for b in range(n_blocks):
            ti = b % 2
            seg = a[s, b * iters_per_task:(b + 1) * iters_per_task, :]
            out.append(np.stack([seg[:, ti], seg[:, 1 - ti]], axis=1))
    return np.array(out)

print("\n" + "=" * 78)
print("B. BLOCK-ALIGNED AVERAGE -- change across one block, averaged over every block/seed")
print(f"{'rule':>28}{'trained task':>18}{'untrained task':>18}{'trade-off':>12}")
blocks = {}
for ri, r in enumerate(rules):
    B = block_align(per_iter[ri, best[r]], ITERS_PER_TASK)
    blocks[r] = B
    m = np.nanmean(B, axis=0)                     # [position, trained/untrained]
    dtr = (m[-1, 0] - m[0, 0]) * 100
    dun = (m[-1, 1] - m[0, 1]) * 100
    print(f"{LABEL[r]:>28}{dtr:>+15.2f} pts{dun:>+15.2f} pts{dun - dtr:>+9.2f}")
print("  trained task should FALL (negative) and untrained should RISE (positive) within a")
print("  block. 'trade-off' is their difference: bigger = a sharper exchange of one for the")
print("  other. If the untrained column is negative too, both tasks are still improving and")
print("  the run is simply too early to show forgetting.")

# ---------------------------------------------------------------- C. trade-off index
print("\n" + "=" * 78)
print("C. TRADE-OFF INDEX -- err(untrained) - err(trained), mean over the analysed window")
for ri, r in enumerate(rules):
    B = blocks[r]
    idx = np.nanmean(B[:, :, 1] - B[:, :, 0]) * 100
    late = np.nanmean(B[len(B) // 2:, :, 1] - B[len(B) // 2:, :, 0]) * 100
    print(f"{LABEL[r]:>28}  all blocks {idx:>+7.2f} pts   second half {late:>+7.2f} pts")
print("  The common descent cancels in this difference, so a positive and growing value is")
print("  the trade-off itself, independent of how far either curve has descended overall.")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

ax = axes[0, 0]
for ri, r in enumerate(rules):
    for task, ls in ((0, "-"), (1, "--")):
        y = np.nanmean(per_iter[ri, best[r], :, :, task], axis=0)
        ax.plot(np.arange(n_iter), y, ls, color=COL[r], lw=1.8,
                label=f"{LABEL[r]}, task {task + 1}")
ax.axvline(ANALYSE, color="k", lw=1.2, ls=":")
ax.text(ANALYSE, ax.get_ylim()[1] * 0.98, " analysed to here", fontsize=8, va="top")
ax.set_xlabel("iteration"); ax.set_ylabel("test error")
ax.set_title("A. All stored iterations -- is 84 simply too early?")
ax.legend(fontsize=7); ax.grid(alpha=0.2)

ax = axes[0, 1]
for ri, r in enumerate(rules):
    m = np.nanmean(blocks[r], axis=0) * 100
    se = np.nanstd(blocks[r], axis=0) / np.sqrt(len(blocks[r])) * 100
    xs = np.arange(ITERS_PER_TASK)
    ax.plot(xs, m[:, 0], "-o", color=COL[r], lw=2.4, label=f"{LABEL[r]}: trained task")
    ax.plot(xs, m[:, 1], "--s", color=COL[r], lw=2.0, alpha=0.75,
            label=f"{LABEL[r]}: untrained task")
    for j, ls in ((0, "-"), (1, "--")):
        ax.fill_between(xs, m[:, j] - se[:, j], m[:, j] + se[:, j], color=COL[r], alpha=0.12)
ax.set_xlabel("iteration within a task block"); ax.set_ylabel("test error (%)")
ax.set_title("B. Every block averaged -- the switch structure, trend removed")
ax.legend(fontsize=7); ax.grid(alpha=0.2)

ax = axes[1, 0]
for ri, r in enumerate(rules):
    B = blocks[r]
    n_per_seed = B.shape[0] // per_iter.shape[2]
    tr = np.nanmean((B[:, :, 1] - B[:, :, 0]) * 100, axis=1).reshape(per_iter.shape[2], -1)
    m = np.nanmean(tr, axis=0); se = np.nanstd(tr, axis=0) / np.sqrt(tr.shape[0])
    xs = np.arange(len(m))
    ax.plot(xs, m, "-o", color=COL[r], lw=2.2, ms=3, label=LABEL[r])
    ax.fill_between(xs, m - se, m + se, color=COL[r], alpha=0.15)
ax.axhline(0, color="k", lw=1.0)
ax.set_xlabel("block number"); ax.set_ylabel("err(untrained) - err(trained), points")
ax.set_title("C. Trade-off index per block -- above zero = the exchange is happening")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1, 1]
z0 = max(0, n_iter - 4 * ITERS_PER_TASK) * upd_per_iter
for ri, r in enumerate(rules):
    for task, ls in ((0, "-"), (1, "--")):
        y = np.nanmean(errors[ri, best[r], :, z0:, task], axis=0)
        ax.plot(np.arange(z0, z0 + len(y)), y, ls, color=COL[r], lw=1.8,
                label=f"{LABEL[r]}, task {task + 1}")
for b in range(z0, n_upd, ITERS_PER_TASK * upd_per_iter):
    ax.axvline(b, color="k", lw=0.6, alpha=0.35)
ax.set_xlabel("weight update"); ax.set_ylabel("test error")
ax.set_title(f"D. Last four blocks at full update resolution\n"
             f"({upd_per_iter} updates per iteration; lines = task switches)")
ax.legend(fontsize=7); ax.grid(alpha=0.2)

fig.suptitle("Is the learning/forgetting trade-off present in the exp-30 data?  "
             "(analysis only, nothing re-run)")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 78)
print("VERDICT")
still = abs(converged[(rules[0], ANALYSE)]) > 1.0
tr_ok = all(np.nanmean(blocks[r][:, -1, 1] - blocks[r][:, 0, 1]) > 0 for r in rules)
if still:
    print("  The error is still falling steeply at iteration 84. The run is UNDERTRAINED, so")
    print("  the alternation is a ripple on a descent and cannot show as a sawtooth. Fix the")
    print("  number of weight updates per block, not the learning rate.")
else:
    print("  The error has flattened by iteration 84, so the flatness is real.")
if tr_ok:
    print("  The untrained task DOES get worse within a block: the trade-off is present in our")
    print("  data, it is just invisible against the descending baseline in panel 4d.")
else:
    print("  The untrained task does not get worse within a block. There is no forgetting to")
    print("  find yet -- both tasks are still improving together.")
