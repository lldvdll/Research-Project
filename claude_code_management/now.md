# Now

Dashboard. Detail lives in `current_state.md` — keep this file short.

**Running:** 52, the four-rule comparison at fixed budget. ~50 min, EqProp-bound.
Started 2026-08-11.

## Next three
1. **53** — same comparison, both tasks stopped on accuracy instead of a budget.
2. **B1/B2/B3** — metrics, all three from one set of backprop runs.
3. Decide whether EqProp continues past 53 (see decisions below).

## Settings now fixed, and where they came from
| | value | from |
|---|---|---|
| hidden width | H = 32 | script 41 |
| learning rates | backprop 0.01, replay 0.01, pc 0.02, eqprop 0.01 | script 51 |
| matched at | ~420 updates to 90% on task 1, spread 1.25× | script 51 |
| PC settling | 50 steps fixed (needs ≤18) | script 50 |
| EqProp settling | `settle_tol = 1e-4` (needs ≤529 at init) | script 50 |

## Checklist
- [x] 41 capacity → **H = 32**, ceilings 93.6% Class-IL / 94.3% Domain-IL
- [x] 40 / 42 / 43 backprop forgetting, mechanism split, the measurement trap
- [x] 50 settling → PC fully settled; EqProp's stopping rule was truncating, fixed
- [x] 51 learning-rate calibration → matched to 1.25×
- [ ] 52 four-rule comparison, fixed budget ← running
- [ ] 53 four-rule comparison, accuracy stopping
- [ ] B1 / B2 / B3 metrics
- [ ] C2 factorial (W1 × free/masked/frozen W2), C1 NCM figure
- [ ] D controlled, then E, then F

## Recent decisions
- **2026-08-11 — EqProp stays in through the four-rule comparison, then has to be
  re-earned.** After 53 it is only called in once PC results are established *and* there
  is a specific reason to try it. It costs ~350× backprop per update and that is intrinsic.
- **2026-08-11 — `settle_tol = 1e-4`**, calibrated inside script 50. Not a universal
  constant: re-run 50 if `dt`, width or depth change.
- **2026-08-11 — learning rates matched on updates-to-90% on task 1**, target ~300.
  Never matched on the crossover: that is the dependent variable.
- **2026-08-11 — accuracy-stopped plots keep a step axis**, switch at x=0. Curves are not
  stretched to fill the panel — that destroys the rate, and rate is half of what we measure.
  The trajectory plot is what handles the ragged lengths.
- **2026-08-11 — exp 12's learning rates are void**, not approximate. They were set under
  the legacy per-rule specification (ReLU + cross-entropy vs hinge on ±1 targets).

## Open, not blocking
- PC and EqProp still stop settling by different mechanisms. Verified not to bias the
  comparison; unifying is exposition only.
- β = 0.3 is the one lever with real leverage on EqProp's cost (bias ~42% at 0.3 vs ~1.6%
  at 0.005). Not opened yet.
- Script 40 still needs re-running at H=32; it ran at a provisional H=64.
