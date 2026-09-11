"""Retention and crossover vs. task-1 overtraining multiplier -- both scenarios, both rules.

900-series PLOT SCRIPT. Loads 807's saved arrays, trains nothing.

Mode B (shared axes): retention and crossover are different scales and answer different
questions (does the endpoint move vs. does the whole trajectory), but both share the same x-axis
(multiplier) and the comparison is "do they agree", so they sit in one figure, not two, per this
project's own shared-axis convention (a paired-disagreement plot is unreadable split across files).

PROVENANCE: 807_overtrain_task1_{scenario}.npz -- data columns are
    (scenario, method, seed, multiplier, N, final_t1, final_t2, crossover, reached)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff, cohens_d_paired

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
METHODS = ["backprop", "pc"]
COLOR = {"backprop": "black", "pc": "tab:red"}
MULTIPLIERS = [1.0, 1.5, 2.0, 3.0, 4.0]
COLS = ["scenario", "method", "seed", "multiplier", "N", "final_t1", "final_t2", "crossover", "reached"]
CI = {c: i for i, c in enumerate(COLS)}


def load(scenario):
    return np.load(EXP / f"807_overtrain_task1_{scenario}.npz", allow_pickle=True)["data"]


def series(rows, method, field):
    """mean +- sem at each multiplier, for one method."""
    means, sems = [], []
    for m in MULTIPLIERS:
        vals = np.array([r[CI[field]] for r in rows if r[CI["method"]] == method
                          and r[CI["multiplier"]] == m], dtype=float)
        vals = vals[np.isfinite(vals)]
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.array(means), np.array(sems)


def per_seed(rows, method, field):
    """{seed: [value at each multiplier, NaN if missing]} -- for the spaghetti lines."""
    seeds = sorted({r[CI["seed"]] for r in rows if r[CI["method"]] == method})
    out = {}
    for s in seeds:
        vals = []
        for m in MULTIPLIERS:
            hit = [r[CI[field]] for r in rows if r[CI["method"]] == method
                   and r[CI["seed"]] == s and r[CI["multiplier"]] == m]
            vals.append(float(hit[0]) if hit else float("nan"))
        out[s] = np.array(vals)
    return out


JITTER = {"backprop": 0.94, "pc": 1.06}   # multiplicative x-offset so the two rules' spaghetti
                                          # don't sit exactly on top of each other


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), sharex=True)

    for c, field in enumerate(["final_t1", "crossover"]):
        for r, scenario in enumerate(SCENARIOS):
            ax = axes[r][c]
            rows = load(scenario)
            for method in METHODS:
                # thin per-seed lines, jittered, low alpha -- shows whether the same seed tracks
                # consistently across multipliers (as the t1/t2 accuracy plots do) or scatters
                seeds_vals = per_seed(rows, method, field)
                xj = np.asarray(MULTIPLIERS) * JITTER[method]
                for vals in seeds_vals.values():
                    ax.plot(xj, vals, color=COLOR[method], alpha=0.20, lw=0.9, zorder=1)
                # bold mean +- sem on top
                m, s = series(rows, method, field)
                ax.errorbar(xj, m, yerr=s, marker="o", ms=5, capsize=3, lw=2.0,
                            color=COLOR[method], label=method, zorder=5)
            ax.set_xscale("log", base=2)
            ax.set_xticks(MULTIPLIERS)
            ax.set_xticklabels([f"{m:g}x" for m in MULTIPLIERS])
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title("retention (final task-1 %)" if field == "final_t1" else "crossover (%)",
                             fontsize=10)
            if c == 0:
                ax.set_ylabel(NICE[scenario], fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right")

    fig.supxlabel("task-1 training, as a multiple of steps-to-90%", fontsize=9)
    fig.suptitle("Does retention improve if task 1 keeps training past matched competence?",
                 fontsize=10)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    print("\n  paired pc - backprop, at each multiplier (retention / crossover):")
    for scenario in SCENARIOS:
        rows = load(scenario)
        print(f"  {NICE[scenario]}")
        for m in MULTIPLIERS:
            bp = {r[CI["seed"]]: r for r in rows if r[CI["method"]] == "backprop"
                  and r[CI["multiplier"]] == m}
            pc = {r[CI["seed"]]: r for r in rows if r[CI["method"]] == "pc"
                  and r[CI["multiplier"]] == m}
            seeds = sorted(set(bp) & set(pc))
            ret_d = paired_diff([pc[s][CI["final_t1"]] for s in seeds],
                                 [bp[s][CI["final_t1"]] for s in seeds])
            cx_d = paired_diff([pc[s][CI["crossover"]] for s in seeds],
                                [bp[s][CI["crossover"]] for s in seeds])
            print(f"    {m:>4.1f}x  retention {ret_d[0]:+.2f} +- {ret_d[1]:.2f}  "
                  f"crossover {cx_d[0]:+.2f} +- {cx_d[1]:.2f}")

    print("\n  within-method, 4x vs 1x (does overtraining itself help):")
    for scenario in SCENARIOS:
        rows = load(scenario)
        for method in METHODS:
            v1 = {r[CI["seed"]]: r[CI["final_t1"]] for r in rows if r[CI["method"]] == method
                  and r[CI["multiplier"]] == 1.0}
            v4 = {r[CI["seed"]]: r[CI["final_t1"]] for r in rows if r[CI["method"]] == method
                  and r[CI["multiplier"]] == 4.0}
            seeds = sorted(set(v1) & set(v4))
            d = paired_diff([v4[s] for s in seeds], [v1[s] for s in seeds])
            dcohen = cohens_d_paired([v4[s] for s in seeds], [v1[s] for s in seeds])
            print(f"    {NICE[scenario]:10s} {method:9s}  retention 4x-1x {d[0]:+.2f} +- {d[1]:.2f}"
                  f"  (n_sem={d[2]:.2f}, d={dcohen:.2f})")
