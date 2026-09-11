# Narrative plan — Option A (mechanism-first)

Results structure for "anatomy of forgetting." Edit freely — skeleton, not a script list.

**Restructured 2026-09-03**: PC moved from §3 to §2. Old order (info-vs-access, then "does PC
differ") had no hypothesis bridging the two — nothing in §2's finding implied PC specifically
should help, so §3's PC question was unmotivated. New order: establish the BP-vs-PC empirical
fact first (§2), then explain whatever was found (§3, branching). Nothing already built is
wasted — old §2 (info-vs-access) and old §3 (layerwise dynamics) become §3's explanatory
toolkit, just deployed after the PC comparison instead of before it.

## Section summary

| § | Question | Answer / status |
|---|---|---|
| 1 | Does forgetting happen, and does its shape differ by scenario? | Yes — done, 310. |
| 2 | Does PC forget too, and does it differ from BP? | PC's own controls settled (dt=0.4, adaptive stop) and headline figure done (330-334). Paired lr sweep done (340): no universal PC advantage, established defaults near-optimal for both rules. Width/depth sweeps (341/342) required a settle-control correction (dt=0.2, not 0.4) once dt=0.4 was found to genuinely destabilise (not just slow) at depth≥2 and at Class-IL H=4 — corrected reruns done at n=10, both confirmed. Class-IL's PC advantage holds at every width (except the H=4 floor, where it loses) and at every depth, growing then plateauing; Domain-IL's disadvantage holds at large width and at the depth-1/depth-4 bookends, not in the depth middle. |
| 3a | (if §2 differs) Why — trunk or readout? | Gated on §2. Tools: trained linear probe (info vs access), layerwise cosine/path-efficiency/target-alignment. |
| 3b | (if §2 doesn't differ) Is there a useful difference anyway? | Gated on §2. Same tools, checking whether the *route* differs even if the outcome doesn't. |
| 4 | If the readout is where damage concentrates, does protecting it help? | Freezing W2 doesn't; masking does (Class-IL). Done, pre-restructure data. |
| 5 | Synthesis: what does PC actually change? | Gated on §3. |
| 6 (conditional) | Does it generalise with depth? | Not started. Gate on §3 finding something to generalise. |

---

## §1 — Forgetting happens, and its shape differs by scenario

**Done.** 310 (seeds 10-19, canonical block — 300/seeds 0-9 deprecated, seed 9 was an outlier,
see 300_series_log.md). BP only. Two square trajectory+histogram figures, Domain-IL vs Class-IL.
Digit-composition variance (301/311/312/313) characterized separately — methods/robustness note,
not in this figure.

---

## §2 — Does PC forget too, and does it differ from BP? (330 series done; 340 next)

**Done (330 series):** dt swept (adaptive settle-stopping, criterion validated by 333),
settle-steps-vs-dt characterised in full by 334 (dt doesn't move the fixed point, only the speed,
until ~0.7 where it stops converging at all). **dt=0.4 chosen** — minimum settle-steps, flat
accuracy/retention/crossover across the whole stable range (330). Truncating to a small FIXED
step count validated separately (331, dt=0.4, steps 1-50, BP reference): flat from ~steps=5-10
on, adaptive stopping already lands there on its own. **332** is PC's own headline figure (exact
copy of 310) at this settled config, both scenarios: Class-IL retention=6.0% crossover=66.4%;
Domain-IL retention=37.0% crossover=75.1%. Compared informally against 331's BP reference, no
clear uplift either direction — small margins, opposite signs by scenario — but that comparison
used different seed counts/budgets, so it's not the real answer.

**Done (340 series, 2026-09-05)**: paired (same-seed) `paired_diff`/`paired_sign`/`paired_wilcoxon`
comparisons across lr (340), width (341), and depth (342) — three separate sweeps, not a joint
factorial, per this project's one-axis-at-a-time convention. Full detail in 300_series_log.md.

Headline: **no universal PC advantage.** Both rules' absolute retention/crossover peak near the
established lr (0.01 bp / 0.02 pc) and decline past it — the established defaults are close to
each rule's own best point, not a blind spot. Class-IL shows a fairly consistent modest PC edge
across width (all widths except the capacity floor H=4) and toward higher lr; Domain-IL shows the
mirror image (PC edge only at the capacity floor, behind at mid-to-large width, mixed on lr).

**Settle-control correction, mid-series**: dt=0.4 (330-334's established control, validated only
at H=32/depth=1) was found to genuinely **oscillate** (not just slow-converge) at depth≥2 and at
Class-IL's H=4 — confirmed by comparing the oscillation's mean against a genuinely converged
value at the same depth (3-7x higher, not a tight overshoot). **dt=0.2** fixes both problems
simultaneously and is now used for 341/342 specifically; dt=0.4 remains correct everywhere else
(it was never wrong for the H=32/depth=1 work this whole project is otherwise built on). 341/342
corrected and rerun in full (n=5 each; 341's extension to n=10 running).

**Corrected 341 results, final at n=10 (2026-09-05)**: 341's Class-IL H=4 loss survives the dt
correction and is now well-powered (crossover -4.1±0.7, p=0.002), down from the confounded -6.2
under dt=0.4 but not eliminated — not purely a settle-instability artefact. **PC wins significantly
at every Class-IL width except the H=4 capacity floor** (H8 +2.9, H16 +3.0, H32 +1.4, H64 +0.9, all
p≤0.021), and the mirror image holds in Domain-IL (PC wins only at H=4, +2.2 p=0.021; loses at
H32/H64, both p≤0.021; flat at H8/H16).

**342 results, final at n=10 (2026-09-05)**: first-ever valid depth≥2 PC-vs-BP comparison.
Class-IL's PC advantage is significant at every depth (+1.4→+2.9→+3.3→+3.4pp, depths 1-4) —
growing from depth 1 to 3, then **plateauing** rather than continuing to climb (the n=5 pass's
depth-4 value of +5.0 was noise). Domain-IL's disadvantage is significant at the depth-1 and
depth-4 bookends (-0.7, -3.5) but flat/non-significant at depths 2-3 (-0.3, -0.9) — not a clean
"persists at every depth" story. So the effect generalises past H=32/depth=1 in Class-IL cleanly,
and in Domain-IL only partially. ⚠ Depth-4's Domain-IL loss (-3.5, the grid's largest) rests on an
80% PC cap-hit rate (only 2/10 seeds reached matched competence) — crossover is less exposed to
this than retention, but it needs more scrutiny before being cited as a real depth-4-specific
effect. PC's retention (not crossover) is generally unreliable at higher depth — cap-hit climbs
20%→80%(D4) — a second, independent case of retention's known failure mode (112).

**Two things carried in from the 330 series, relevant here or in §3:**
- Class-IL retention (332) splits into a near-zero group and a ~10-30% group, not a smooth
  spread — matches 312/313's established BP finding (tied to which digits land in task-1), now
  also seen under PC. Legitimate seed-dependent variation, read at matched task-2 competence so
  not a convergence-speed artifact; candidate thing for §3 to explain.
- Crossover becomes measurement-unreliable (undefined/censored for a growing fraction of seeds)
  at high lr, hitting backprop before PC in Class-IL (100% vs 60% censored by lr=0.16) — a real
  finding about the metric's own validity boundary, not a null result, worth a methods note.

---

## §3a — Why does PC differ? (gated on §2)

Old §2 (info-vs-access) + old §3 (layerwise dynamics) merge here, now comparative (BP vs PC)
instead of BP-only.

**Info-vs-access tooling correction (2026-09-02/03, see 320 in 300_series_log.md):** bare NCM is
inadequate — it's a centroid-only (first-order) probe, no sensitivity to within-class
spread/covariance, and its ceiling is set by task geometry (raw-pixel NCM on all 10 MNIST digits
= 81.2%, checked directly), giving it little dynamic range. This is a known failure mode in this
project's own history (`knowledge_base.md` §6.6.3: exp 21 invalidated exp 20's NCM probe for
exactly this reason). Needed instead: a **trained/refit linear probe** on frozen features
(random-W1 and post-task-1-W1 ceilings, tracked through task-2 training, compared against
argmax) — flagged in `knowledge_base.md` as "never run." Not yet built.

```
              Domain-IL        Class-IL
          ┌───────────────┬───────────────┐
 cos(Δw)  │      A        │      B        │
 by layer │               │               │
          ├───────────────┼───────────────┤
 path     │      C        │      D        │
 efficiency│              │               │
          ├───────────────┼───────────────┤
 target   │      E        │      F        │
 alignment│               │               │
          ├───────────────┼───────────────┤
 info vs  │      G        │      H        │
 access   │ (trained probe vs argmax,     │
          │  BP vs PC)                    │
          └───────────────┴───────────────┘
```

- A/B: BP–PC update cosine similarity per layer.
- C/D: path efficiency (`|Δw|` net vs `Σ|dw|` accumulated) per layer.
- E/F: target alignment per layer/output (direct re-test of S&B's credited mechanism).
- G/H: trained-probe ceiling vs argmax, BP vs PC — replaces the old NCM panel.

---

## §3b — If §2 finds no difference, is there a useful difference anyway? (gated on §2)

Same toolkit as §3a. Question changes from "explain the difference" to "do BP and PC reach the
same headline number by the same route, or different routes that land in the same place?" Either
answer (same route / different route) is a real result.

---

## §4 — Causal test: does protecting the readout help?

Status: existing data (130 freeze factorial; masking run) — predates the restructure but doesn't
depend on it. Freeze-W1 cells flagged unreliable (9–10/10 seeds non-convergent) — not used here.

```
              Domain-IL        Class-IL
          ┌───────────────┬───────────────┐
 retention│      A        │      B        │
 bars:    │ base/freeze/  │ base/freeze/  │
 base,    │ mask          │ mask          │
 freeze,  │               │               │
 mask     │               │               │
          └───────────────┴───────────────┘
                    C
          learning curves, base vs mask,
          Class-IL only
```

- A: retention under no-intervention / freeze-W2 / mask-W2, Domain-IL — control, little room for
  a readout fix since there's little suppression to fix.
- B: same, Class-IL — freeze fails, mask succeeds.
- C: task-1/task-2 learning curves under masking vs baseline, Class-IL.
- Freeze failing and mask succeeding is the causal confirmation of §3's localisation.

---

## §5 — Synthesis: what does PC actually change?

Status: depends entirely on §3.

```
┌─────────────────────────────────┐
│ A: hypothesis funnel             │
│   target alignment  → ?          │
│   trunk update      → ?          │
│   path efficiency   → ?          │
│   info vs access    → ?          │
├───────────────┬───────────────────┤
│ B: retention  │ C: surviving      │
│ diff, PC−BP,  │ readout metric    │
│ both scenarios│ vs B, both rules  │
└───────────────┴───────────────────┘
```

- A: funnel summarising which candidate explanations from §3 survived.
- B: PC−BP retention difference by scenario.
- C: whichever §3 metric best tracks the readout, plotted directly against B.
- If §3 finds no trunk/readout split, "no explicable difference" is still a legitimate answer.

---

## §6 (conditional) — Does the mechanism generalise with depth?

Status: not started. Gate on §3 finding something worth generalising.

```
              depth=1    depth=2    depth=3
          ┌──────────┬──────────┬──────────┐
 cos(Δw)  │    A     │    B     │    C     │
 by layer │          │          │          │
          ├──────────┼──────────┼──────────┤
 retention│    D     │    E     │    F     │
 (PC−BP)  │          │          │          │
          └──────────┴──────────┴──────────┘
```

- A–C: per-layer BP/PC update cosine at each depth.
- D–F: PC−BP retention difference at each depth.
- Only worth running if §3 found a real effect to generalise.

---

## Everything else (EWC, SI, k-WTA)

Secondary/appendix. EWC flat to λ=100 (non-convergent beyond); SI flat to λ=0.1 (non-convergent
beyond); neither shows a protective effect. k-WTA constrains hidden-layer sparsity, not readout
plasticity — doesn't test §3's question. None earn a place in the main spine.
