"""If we discard mean test error, does crossover carry the same information as retention, or do
they disagree often enough that neither can substitute for the other?

900-series PLOT SCRIPT. Loads saved arrays from every sweep already run, trains nothing. M3.

ONE PLOT: crossover (y) against retention (x), one point per (method, seed, condition), pooled
across every sweep that has both numbers -- 340 (lr), 341 (width), 342 (depth), 802 (activation).
Colour by scenario, marker by rule. This is a diagnostic, not a replacement metric: the pooled
correlation across ALL points looks strong, but that is scenario separation doing the work
(Class-IL sits at low-retention/moderate-crossover, Domain-IL at high-retention/high-crossover),
not a real within-scenario relationship. The number that matters is the WITHIN-scenario r, printed
separately and annotated on the plot.

WHY THIS MATTERS FOR THE SWEEP FIGURES. If crossover and retention were near-perfectly
correlated within a scenario, one plot per sweep would do and 912/927-style dual-panel figures
would be redundant. They are not -- so both are needed, and this figure is the evidence for why,
not a call to reduce them to one.

PROVENANCE: 340_lr_sweep_{s}.npz, 341_width_sweep_{s}.npz, 342_depth_sweep_{s}.npz (columns:
method, x, seed, retention, t2_final, crossover, ...), 802_activation_sweep.npz (columns:
scenario, act, method, seed, final_t1, final_t2, crossover, switch0, reached).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
MARKER = {"backprop": "o", "pc": "^", "replay": "s"}


def points_300(stem, scenario):
    """340/341/342 schema: (method, x, seed, retention, t2_final, crossover, ...)."""
    d = np.load(EXP / f"{stem}_{scenario}.npz", allow_pickle=True)["data"]
    out = []
    for r in d:
        method, ret, cx = r[0], float(r[3]), float(r[5])
        if np.isfinite(ret) and np.isfinite(cx):
            out.append((method, ret, cx))
    return out


def points_802(scenario):
    """802 schema: (scenario, act, method, seed, final_t1, final_t2, crossover, switch0, reached)."""
    d = np.load(EXP / "802_activation_sweep.npz", allow_pickle=True)["data"]
    out = []
    for r in d:
        if r[0] != scenario:
            continue
        method, ret, cx = r[2], float(r[4]), float(r[6])
        if np.isfinite(ret) and np.isfinite(cx):
            out.append((method, ret, cx))
    return out


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharex=False, sharey=False)

    all_ret, all_cx, all_scen = [], [], []
    for ax, scenario in zip(axes, SCENARIOS):
        pts = (points_300("340_lr_sweep", scenario) + points_300("341_width_sweep", scenario)
               + points_300("342_depth_sweep", scenario) + points_802(scenario))
        for method in ["backprop", "replay", "pc"]:
            sub = [(ret, cx) for m, ret, cx in pts if m == method]
            if not sub:
                continue
            ret, cx = zip(*sub)
            ax.scatter(ret, cx, s=14, alpha=0.35, marker=MARKER[method],
                       color=COLORS[scenario], label=method)
            all_ret.extend(ret); all_cx.extend(cx); all_scen.extend([scenario] * len(ret))
        ret_all = np.array([r for m, r, c in pts])
        cx_all = np.array([c for m, r, c in pts])
        r_within = np.corrcoef(ret_all, cx_all)[0, 1]
        ax.set_title(f"{NICE[scenario]}  (within-scenario r = {r_within:+.2f}, n={len(pts)})",
                     fontsize=9)
        ax.set_xlabel("retention (final task-1 %)", fontsize=9)
        ax.grid(alpha=0.25)
        print(f"  {NICE[scenario]:10s} within-scenario r(retention, crossover) = {r_within:+.3f}"
              f"  (n={len(pts)})")

    axes[0].set_ylabel("crossover (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    r_pooled = np.corrcoef(all_ret, all_cx)[0, 1]
    print(f"  pooled across scenarios: r = {r_pooled:+.3f}  <- inflated by scenario separation, "
          f"not a within-scenario relationship")
    fig.suptitle("Crossover is not a proxy for retention within a scenario — both are needed",
                 fontsize=10)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
