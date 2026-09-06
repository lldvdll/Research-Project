"""Report-track audit page. Nine sections; each grid is columns of 1-2 cards.
Story runs left to right across columns; a vertical pair is a direct comparison."""
import base64, html, pathlib

EXP = pathlib.Path(r"c:\Users\dgsan\uni\Research-Project\experiments")
OUT = pathlib.Path(__file__).with_name("report_track.html")

_c = {}
def img(n):
    if n not in _c: _c[n] = base64.b64encode((EXP / n).read_bytes()).decode()
    return f"data:image/png;base64,{_c[n]}"
def pic(n): return f'<img src="{img(n)}" alt="{html.escape(n)}" loading="lazy">'

# ----------------------------------------------------------------- sketch kit
W, H = 420, 250
def sk(body, label):
    return (f'<svg class="sk" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label)}">'
            f'{body}</svg>')
def ax(x0=52, yt=22, x1=404, yb=200):
    return (f'<line x1="{x0}" y1="{yt}" x2="{x0}" y2="{yb}" class="a"/>'
            f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" class="a"/>')
def t(x, y, s, c="l", an="start"):
    return f'<text x="{x}" y="{y}" class="{c}" text-anchor="{an}">{html.escape(s)}</text>'
def rect(x, yb, h, w, c):
    return f'<rect x="{x}" y="{yb-h if h>=0 else yb}" width="{w}" height="{abs(h)}" class="{c}"/>'

# 1 — definitional
SK_DEF = sk(
    ax() + '<line x1="196" y1="22" x2="196" y2="200" class="m"/>' +
    ''.join(f'<polyline points="60,{60+i*7} 196,{46+i*6} 250,{92+i*9} 320,{150+i*7} 396,{178+i*4}" class="f1"/>' for i in range(5)) +
    ''.join(f'<polyline points="60,{194-i*2} 196,{192-i*2} 250,{130-i*8} 320,{74-i*6} 396,{58-i*5}" class="f2"/>' for i in range(5)) +
    '<polyline points="60,74 196,52 250,104 320,158 396,184" class="b1"/>' +
    '<polyline points="60,193 196,191 250,142 320,84 396,66" class="b2"/>' +
    '<circle cx="278" cy="125" r="5" class="cx"/>' + t(286, 120, "crossover", "s") +
    '<path d="M232 66 q26 30 20 62" class="ar"/><path d="M232 168 q26 -30 20 -62" class="ar2"/>' +
    t(150, 40, "task 1", "s") + t(150, 214, "task 2", "s2") +
    t(200, 18, "switch", "s", "middle") +
    t(340, 46, "learning", "s2") + t(300, 208, "forgetting", "s") +
    t(20, 110, "acc %", "l") + t(228, 240, "training step", "l", "middle"),
    "definitional figure sketch")

# 2 — activation insensitivity
SK_ACT = sk(
    ax(62, 22, 404, 186) + '<line x1="62" y1="70" x2="404" y2="70" class="z"/>' +
    t(408, 74, "", "s") +
    rect(92, 186, 92, 42, "p1") + rect(148, 186, 86, 42, "p1") + rect(204, 186, 80, 42, "p1") +
    rect(268, 186, 94, 42, "p2") + rect(324, 186, 88, 42, "p2") +
    t(113, 200, "tanh", "l", "middle") + t(169, 200, "sigmoid", "l", "middle") +
    t(225, 200, "relu", "l", "middle") +
    t(289, 200, "tanh", "l", "middle") + t(345, 200, "sigmoid", "l", "middle") +
    t(169, 218, "Class-IL", "s", "middle") + t(317, 218, "Domain-IL", "s2", "middle") +
    t(233, 60, "flat  =  conclusion does not depend on it", "s", "middle") +
    t(20, 100, "crossover", "l"),
    "activation check sketch")

# 3 — pairing
SK_PAIR = sk(
    ax(70, 22, 404, 196) + '<line x1="70" y1="150" x2="404" y2="150" class="z"/>' +
    '<line x1="160" y1="52" x2="160" y2="176" class="e"/><circle cx="160" cy="114" r="6" class="d1"/>' +
    '<line x1="300" y1="100" x2="300" y2="128" class="e"/><circle cx="300" cy="114" r="6" class="d1"/>' +
    t(160, 212, "unpaired", "l", "middle") + t(300, 212, "paired", "l", "middle") +
    t(160, 42, "wide", "s", "middle") + t(300, 88, "4.5–11.5× tighter", "s", "middle") +
    t(232, 236, "same estimate, different uncertainty", "s", "middle") +
    t(18, 100, "PC − BP", "l"),
    "pairing sketch")

# 4 — 2x2 assembly
SK_CONSOL = sk(
    ax(50, 22, 404, 180) + '<line x1="50" y1="104" x2="404" y2="104" class="z"/>' +
    '<line x1="168" y1="22" x2="168" y2="180" class="m"/><line x1="286" y1="22" x2="286" y2="180" class="m"/>' +
    '<polyline points="72,92 108,98 144,94" class="b1"/>' +
    '<polyline points="190,146 226,80 262,72" class="b1"/>' +
    '<polyline points="308,92 340,74 372,66 396,62" class="b1"/>' +
    '<polyline points="72,116 108,120 144,114" class="b2 dh"/>' +
    '<polyline points="190,80 226,120 262,124" class="b2 dh"/>' +
    '<polyline points="308,116 340,112 372,126 396,166" class="b2 dh"/>' +
    t(108, 200, "lr", "l", "middle") + t(226, 200, "width", "l", "middle") +
    t(345, 200, "depth", "l", "middle") +
    t(74, 42, "Class-IL", "s") + t(74, 176, "Domain-IL", "s2") +
    t(18, 96, "PC − BP", "l"),
    "consolidated sweep sketch")

# 5 — repeated alternation, two geometries
SK_REPEAT = sk(
    # left: class-il closed loop
    '<line x1="34" y1="30" x2="34" y2="186" class="a"/><line x1="34" y1="186" x2="186" y2="186" class="a"/>' +
    '<path d="M44 180 C 60 60, 170 60, 176 108 C 180 158, 90 178, 46 180 Z" class="loop"/>' +
    '<path d="M50 176 C 66 68, 164 66, 170 110 C 174 152, 96 174, 52 176 Z" class="loop2"/>' +
    t(110, 208, "Class-IL", "s", "middle") + t(110, 224, "closed loop", "l", "middle") +
    # right: domain-il inward spiral
    '<line x1="234" y1="30" x2="234" y2="186" class="a"/><line x1="234" y1="186" x2="392" y2="186" class="a"/>' +
    '<path d="M244 180 C 268 74, 372 78, 376 116 C 380 152, 300 168, 268 156 C 246 148, 268 96, 344 100 C 372 102, 366 130, 330 134 C 310 136, 310 118, 330 116" class="spiral"/>' +
    '<circle cx="336" cy="118" r="4.5" class="cx"/>' +
    t(312, 208, "Domain-IL", "s2", "middle") + t(312, 224, "spirals in", "l", "middle") +
    t(210, 20, "task 1 acc  →  ,  task 2 acc  ↑", "l", "middle"),
    "repeated alternation sketch")

# 6 — split tie-out
SK_TIE = sk(
    t(128, 20, "Class-IL", "s", "middle") + t(330, 20, "Domain-IL", "s2", "middle") +
    ''.join(
        f'<line x1="128" y1="{y}" x2="330" y2="{y2}" class="cn"/>'
        f'<circle cx="128" cy="{y}" r="4.5" class="d1"/><circle cx="330" cy="{y2}" r="4.5" class="d2"/>'
        + t(118, y + 4, lb, "l", "end")
        for y, y2, lb in [(46, 82, "PC − BP"), (82, 52, "PC − BP, depth 4"),
                          (118, 122, "replay − BP"), (154, 208, "digit identity"),
                          (190, 194, "freeze-W2"), (216, 212, "argmax − probe")]) +
    t(229, 240, "crossing = splits    parallel = shared", "s", "middle"),
    "split tie-out sketch")

# 7 — retention distribution + interventions
SK_DIST = sk(
    ax(56, 22, 404, 190) +
    ''.join(rect(60 + i * 10, 190, h, 8, "p1") for i, h in enumerate(
        [120, 92, 40, 14, 6, 4, 6, 10, 16, 22, 18, 10, 6, 3])) +
    '<polyline points="64,132 104,150 144,166 204,176 264,182 344,186" class="b2 dh"/>' +
    t(84, 48, "most seeds", "s", "middle") + t(84, 62, "at ~0%", "s", "middle") +
    t(232, 122, "second group", "s2", "middle") + t(232, 136, "15–30%", "s2", "middle") +
    t(300, 154, "after masking", "s2") +
    t(20, 106, "seeds", "l") + t(230, 226, "task-1 retention (%)", "l", "middle"),
    "retention distribution sketch")

# 8 — trained probe
SK_PROBE = sk(
    ax(56, 22, 404, 190) + '<line x1="178" y1="22" x2="178" y2="190" class="m"/>' +
    '<polyline points="64,72 178,44 208,54 268,136 330,182 396,188" class="b1"/>' +
    '<polyline points="64,90 178,56 240,62 320,74 396,80" class="b2 dh"/>' +
    t(340, 172, "argmax", "s") + t(300, 62, "trained probe", "s2") +
    t(182, 18, "switch", "s") +
    t(250, 108, "gap = readout failure", "s", "middle") +
    t(250, 124, "both fall = code destroyed", "s", "middle") +
    t(20, 100, "acc %", "l"),
    "trained probe sketch")

# 9 — per-layer update path under alternation
SK_PATH = sk(
    ax(60, 22, 404, 190) +
    '<polyline points="70,60 110,88 150,120 190,144 230,158 270,164 310,166 350,167 392,167" class="b1"/>' +
    '<polyline points="70,66 110,52 150,96 190,50 230,100 270,48 310,102 350,46 392,100" class="b2 dh"/>' +
    t(320, 182, "W1  settles", "s") + t(180, 36, "W2  keeps moving", "s2") +
    t(18, 96, "weight step", "l") + t(18, 110, "per block", "l") +
    t(232, 226, "alternation block", "l", "middle"),
    "per-layer path sketch")

# 10 — representation drift
SK_DRIFT = sk(
    ax(60, 22, 404, 190) + '<line x1="170" y1="22" x2="170" y2="190" class="m"/>' +
    '<polyline points="70,180 170,174 210,110 280,78 350,68 396,64" class="b1"/>' +
    '<polyline points="70,182 170,178 210,146 280,126 350,120 396,118" class="b2 dh"/>' +
    '<polyline points="70,184 170,183 210,182 280,181 350,181 396,181" class="fz"/>' +
    t(300, 54, "task-1 units", "s") + t(300, 110, "task-2 units", "s2") +
    t(280, 196, "frozen-weight baseline", "l") +
    t(18, 96, "hidden-code", "l") + t(18, 110, "displacement", "l") +
    t(174, 18, "switch", "s"),
    "representation drift sketch")

# 11 — intervention summary
SK_INTERV = sk(
    ax(58, 22, 404, 186) + '<line x1="58" y1="150" x2="404" y2="150" class="z"/>' +
    rect(72, 150, 116, 34, "p3") + rect(122, 150, 74, 34, "p2") + rect(172, 150, 26, 34, "p1") +
    rect(222, 150, 17, 34, "p1") + rect(272, 150, 4, 34, "p1") + rect(322, 150, -58, 34, "p4") +
    t(89, 210, "replay", "l", "middle") + t(139, 210, "mask", "l", "middle") +
    t(189, 210, "SI", "l", "middle") + t(239, 210, "EWC", "l", "middle") +
    t(289, 210, "freeze", "l", "middle") + t(339, 210, "k-WTA", "l", "middle") +
    t(89, 26, "+9.8", "s", "middle") + t(339, 224, "harmful", "s", "middle") +
    t(18, 96, "Δ crossover", "l"),
    "intervention summary sketch")


# ================================================================= cards
S = dict(ok=("Use as-is", "ok"), crop=("Crop", "crop"), amend=("Amend", "amend"),
         rerun=("Re-run", "rerun"), busy=("Too busy", "busy"), build=("Build", "build"),
         appx=("Appendix", "appx"))

def C(status, q, media, cap, note=None):
    lab, cls = S[status]
    n = f'<p class="note"><span class="k">{lab}</span>{note}</p>' if note else ""
    return (f'<figure class="card {cls}"><div class="chip {cls}">{lab}</div><h4>{q}</h4>'
            f'<div class="media">{media}</div><figcaption>{cap}</figcaption>{n}</figure>')

def col(*cards): return '<div class="col">' + "".join(cards) + '</div>'

T = []

# ---------------------------------------------------------------- 1
T.append(dict(n=1, q="What does forgetting look like?", v="build",
  vt="One figure, rebuilt to your spec. What exists today understates its own point.",
  cols=[
    col(C("build", "What happens to the task you just left?", SK_DEF,
      "Class-IL, <strong>one seed, fixed budget</strong> — so the whole track is shown without "
      "averaging across runs that start and stop at different places. Two bold curves, all ten "
      "classes as thin lines, crossover marked, arrows naming forgetting and learning.",
      "the five task-1 classes falling to zero while the five task-2 classes rise <em>is</em> the "
      "output-suppression mechanism. Planting it here means §7 explains something the reader has "
      "already seen rather than introducing it cold.")),
    col(C("amend", "What does the current version show?", pic("102_forgetting_demonstrated.png"),
      "Ten seeds, both scenarios, both rules. Correct in form, and it is where the shape above "
      "comes from — but averaged, and cropped.",
      "the x-window stops ~200 steps after the switch, where the Class-IL mean still reads ≈29%. "
      "The run's actual final task-1 is <strong>3.3%</strong>. As drawn, the opening figure makes "
      "forgetting look milder than it is.")),
  ],
  why="This figure's job is definition, not comparison, and a single fixed-budget seed does that "
      "better than a ten-seed mean: nothing is smoothed, the whole trajectory is visible, and the "
      "per-class lines carry a mechanism the reader will need later. Everything downstream refers "
      "to three things named here — the switch, the plateau, and the crossing."))

# ---------------------------------------------------------------- 2
T.append(dict(n=2, q="The setup, and why each choice was made", v="part",
  vt="Four justifications exist. The stopping rule needs re-deriving without circularity; the activation has none.",
  cols=[
    col(C("ok", "Can the trunk do enough work to ask mechanistic questions?",
      pic("101_problem_complexity.png"),
      "Frozen random projection with a trained head reaches <strong>77.2%</strong> against the "
      "fully-trained <strong>89.7%</strong>. Training the hidden layer is worth ~12 points, so "
      "\"where does forgetting live\" is not being asked of a dead layer.", None),
      C("ok", "Is the width sufficient, and where does it stop being?",
      pic("100_capacity_vs_width.png"),
      "Joint accuracy 84.1 (H=8) → 89.7 (H=32) → 90.3 (H=64). H=32 is off the bottleneck and "
      "flat thereafter. <strong>H=4 and H=8 are capacity-limited</strong> — flagged here, and it "
      "returns in §5 where PC reverses at H=4.", None)),
    col(C("amend", "Does an endpoint metric survive a change of stopping point?",
      pic("111_metric_sensitivity_to_threshold.png"),
      "<strong>Backprop alone — no rule comparison anywhere in this argument.</strong> Final "
      "task-1 falls monotonically to 0.28 as the threshold moves; S&amp;B mean error is U-shaped. "
      "An endpoint number is a statement about where you stopped.",
      "reframe as a single-rule failure and fold in 113's mechanism — the endpoint moves because "
      "<em>task 1's own peak moves with it</em>. That is the reason, and it needs no PC."),
      C("ok", "Does crossover survive a change of learning rate?",
      pic("343_lr_degradation_and_reliability.png"),
      "It does not, at the top of the range: Class-IL crossover becomes <strong>undefined on "
      "10/10 backprop seeds at lr = 0.16</strong>. Both candidate metrics fail, in different "
      "regimes — which is what forces a competence-matched stopping rule rather than either "
      "metric alone.", None)),
    col(C("crop", "Does the conclusion depend on the output maths?",
      pic("120_output_maths_and_masking.png"),
      "Three output specifications. The spec sets the <em>magnitude</em> of suppression — ce 49.94 "
      "&lt; hinge 54.76 &lt; mse 59.31 — but <strong>masking removes it under all three</strong>. "
      "Justification by insensitivity: the choice changes the size, not the finding.",
      "take the spec comparison only. The masking result is an intervention and belongs in §9."),
      C("build", "Does the conclusion depend on the activation?",
      SK_ACT,
      "Proposed: crossover under tanh, sigmoid and ReLU, both scenarios, backprop and PC. "
      "<strong>We currently have no justification for tanh at all</strong> — it was inherited, "
      "and Song &amp; Bogacz use sigmoid.",
      "a cheap three-point sweep, and the same insensitivity argument that works for the output "
      "maths. This is a hole an examiner will find; it costs an afternoon to close.")),
    col(C("build", "Why pair on seeds, and what does pairing buy?", SK_PAIR,
      "Proposed: the same point estimate drawn twice, once with the unpaired error bar and once "
      "with the paired one. Pairing does not move the estimate — it shrinks the SEM "
      "<strong>4.5× in Class-IL and 11.5× in Domain-IL</strong>, because the class split is "
      "shared between rules at a given seed.",
      "numbers exist in 69, never drawn. The headline effects in §5 are one to three points, so "
      "this pre-empts the obvious objection that they sit inside noise.")),
  ],
  why="You were right that fixing the metric on the PC-versus-backprop difference is circular — it "
      "selects the instrument by the answer it gives on the very comparison it will be used for, "
      "which is the error <code>CLAUDE.md</code> already forbids for learning rates. Every "
      "justification here is now <strong>single-rule</strong>: the trunk does work, the width is "
      "adequate, the endpoint fails as the stopping point moves, crossover fails as the learning "
      "rate rises, and the output maths does not change the conclusion. The stopping rule falls "
      "out of the two failures rather than being asserted. The rule comparison is not mentioned "
      "until §4."))

# ---------------------------------------------------------------- 3
T.append(dict(n=3, q="Is PC set up so that what we measure is actually PC?", v="ok",
  vt="Fully answered, and the depth result generalises beyond this project.",
  cols=[
    col(C("ok", "Does the settle step size move where PC settles, or only how fast?",
      pic("334_pc_settle_trace_by_dt_log.png"),
      "Displacement per settle step, every dt on one axis. dt ≤ 0.5 all reach the "
      "<strong>identical plateau</strong> (0.0850 Class-IL, 0.0787 Domain-IL) — the route "
      "differs, the fixed point does not. dt ≥ 0.7 lands somewhere else entirely.", None)),
    col(C("ok", "What does settling cost, and does buying more change the result?",
      pic("330_pc_dt_settling.png"),
      "Settle steps fall <strong>71.2 → 8.5</strong> from dt 0.02 to 0.4 while joint accuracy, "
      "retention and crossover stay flat — 8× cheaper, same answer. Past 0.4 cost rises again "
      "because updates stop converging (0 of 2960 at dt 0.85).", None)),
    col(C("ok", "Does the chosen dt survive going deeper?",
      pic("347_pc_settle_dt_by_depth.png"),
      "dt × depth stability grid. The <strong>only</strong> unstable cell is dt = 0.4 at "
      "depth ≥ 2. dt = 0.2 converges everywhere in ≤ 37 steps, well under the cap.", None),
      C("ok", "…or going narrower?",
      pic("346_pc_settle_trace_by_depth_width_log.png"),
      "<strong>Class-IL H=4 oscillates; Domain-IL H=4 converges.</strong> The instability is "
      "scenario-specific, and the oscillation sits 3–7× above the true fixed point rather than "
      "around it — a wrong state, not overshoot.", None)),
  ],
  why="These are not controls for forgetting; they are controls for PC being a faithful "
      "implementation of itself, and they rule out the alternative reading of everything in §5 — "
      "that a rule difference is really a settling artefact. The pair on the right is the "
      "load-bearing part and the most transferable result the project has: the stable dt band "
      "<em>shrinks with depth</em>, so this is a per-configuration precondition, not a one-time "
      "decision. Ignoring that cost two full sweep re-runs."))

# ---------------------------------------------------------------- 4
T.append(dict(n=4, q="Does each rule forget, and does it look the same in each scenario?", v="ok",
  vt="The four panels exist. They have never been assembled into the one grid that makes the point.",
  cols=[
    col(C("ok", "Backprop, Class-IL — where does a run end up?",
      pic("310_forgetting_by_scenario_class_il.png"),
      "Task-1 against task-2 accuracy, time removed, 10 seeds. Every trajectory turns hard left "
      "and terminates against the axis: final task 1 <strong>5.4%</strong>, crossover 65.0.",
      "this and the three panels beside it should be one 2×2 grid — scenario across, rule down. "
      "Assembly only; no retraining."),
      C("ok", "PC, Class-IL — does the shape change?",
      pic("332_pc_forgetting_by_scenario_class_il.png"),
      "Same protocol, same seeds. Crossover 66.44 against backprop's 65.01. The <em>shape</em> is "
      "indistinguishable; only the crossing height moves, and only slightly.", None)),
    col(C("ok", "Backprop, Domain-IL — where does the same run end up?",
      pic("310_forgetting_by_scenario_domain_il.png"),
      "Identical protocol and seeds, only the output layer differs. Trajectories stop well short "
      "of the axis: final task 1 <strong>38.5%</strong>, crossover 75.8, endpoint spread 13.1–64.2.",
      None),
      C("ok", "PC, Domain-IL — does the shape change?",
      pic("332_pc_forgetting_by_scenario_domain_il.png"),
      "Crossover 75.06 against 75.76 — PC slightly <em>behind</em>. Again the family of shapes is "
      "the same as backprop's; the scenario, not the rule, sets the geometry.", None)),
  ],
  why="Read as a 2×2 these four panels make one point that no single panel makes: <strong>changing "
      "the scenario changes the shape of forgetting; changing the rule does not.</strong> Down a "
      "column the trajectories are near-superimposable; across a row they are different families "
      "entirely — one collapsing onto an axis, one stopping on a shelf. That asymmetry is what §6 "
      "then has to justify formally, and it is why the rule comparison in §5 has to be reported "
      "per scenario rather than pooled."))

# ---------------------------------------------------------------- 5
T.append(dict(n=5, q="Which rule is better, and is that stable?", v="part",
  vt="Answered on three axes. The consolidation that makes it one claim has never been drawn.",
  cols=[
    col(C("ok", "Is the comparison an artefact of the learning rate?",
      pic("340_lr_sweep_accuracy.png"),
      "Shared absolute grid, 10 seeds. <strong>Both rules peak near lr 0.01–0.02 and decline past "
      "it.</strong> The defaults sit close to jointly optimal — the comparison was not read at a "
      "point chosen to favour either rule.", None)),
    col(C("ok", "Does more capacity rescue PC?", pic("341_width_sweep_accuracy.png"),
      "Width 4–64 at the corrected dt = 0.2. PC's Class-IL edge peaks at H=8–16 (+2.9, +3.0) and "
      "shrinks with width. At <strong>H=4 it reverses hard: −4.14 ± 0.70</strong>, and that "
      "survives the settle correction.", None),
      C("ok", "Does depth?", pic("342_depth_sweep_accuracy.png"),
      "Depths 1–4, the first valid depth ≥ 2 data the project has. <strong>Depth amplifies the "
      "scenario split in both directions</strong>: Class-IL +1.42 → +3.44, Domain-IL −0.72 → "
      "−3.49. This contradicts the pre-100 \"depth doesn't matter\" line.", None)),
    col(C("build", "Is the sign reversal stable across every axis we can vary?", SK_CONSOL,
      "Proposed: PC − backprop on one axis against lr, width and depth, two scenario lines, zero "
      "line drawn. <strong>Class-IL above zero and Domain-IL below it everywhere except the "
      "narrowest width.</strong> Currently the reader must hold three figures in mind to see it.",
      "three existing scripts' arrays, one figure, no retraining. This is the section's headline "
      "and it does not yet exist as a single object.")),
    col(C("amend", "How large is a real fix, on the same axes?", pic("340_lr_sweep_diff.png"),
      "Replay − backprop is the upper line everywhere: <strong>+55 to +62 pp Class-IL "
      "retention</strong> against PC's +0 to +2, and +15 to +20 pp in Domain-IL. Flat across the "
      "whole grid, so not a tuning artefact. PC's effect is real and roughly a seventh the size.",
      "replay's magnitude crushes PC's line flat against the zero axis. Give replay its own scale "
      "or a broken axis — the point is the ratio, and right now one of the two terms is unreadable."),
      C("ok", "Does the result depend on which metric we chose?",
      pic("112_which_metric_survives.png"),
      "Five metrics × five stopping thresholds. Class-IL: crossover and crossover-of-peak hold "
      "one sign at 2 SEM throughout; every endpoint metric flips. Domain-IL: nothing survives, "
      "consistent with there being no effect.",
      "<strong>demoted, deliberately.</strong> This was being used to <em>select</em> the metric, "
      "which is circular. As a robustness check reported after the result it is legitimate — and "
      "it is the same figure doing honest work.")),
  ],
  why="The answer is small, real, and sign-flipped by scenario: PC is ahead in Class-IL and behind "
      "in Domain-IL. A one-to-three-point effect at a single architecture invites the dismissal "
      "that it is a quirk of H=32, depth 1 — and the sweeps refute that in an unexpected way. The "
      "effect is not fragile, it is <em>systematic</em>, growing with depth in both directions at "
      "once. The replay panel is what keeps this honest: it establishes that the problem is "
      "solvable and that PC's uplift is roughly a seventh of what a memory mechanism buys, so the "
      "result is reported as a small real effect rather than a headline."))

# ---------------------------------------------------------------- 6
T.append(dict(n=6, q="Why are these two investigations, not one?", v="gap",
  vt="Your repeated-sequence idea is the answer, and it is half-run already.",
  cols=[
    col(C("rerun", "Under repeated alternation, does the network converge on a joint solution?",
      pic("68_what_happens_under_repeated_task_switching_spiral.png"),
      "20 alternations, task-1 against task-2 accuracy, colour = time. <strong>Domain-IL spirals "
      "inward</strong> toward the joint corner — arcs shrinking as time runs — for all three "
      "rules. Exactly the geometry predicted.",
      "<strong>Domain-IL only.</strong> Pre-300 protocol, 5 seeds, includes replay. Re-run as "
      "backprop vs PC under current controls, and add the Class-IL arm — which has never existed."),
     ),
    col(C("build", "Do the two scenarios produce different geometry, not just different numbers?",
      SK_REPEAT,
      "Proposed: the same plot, both scenarios side by side. Prediction — Domain-IL spirals in "
      "(converging on a joint solution), <strong>Class-IL closes a loop</strong> (ping-ponging "
      "between two solutions, learning nothing cumulative). One figure, two qualitatively "
      "different behaviours.",
      "a one-scenario addition to a script that already produces this figure. The separating "
      "quantity is already computed too — the weight-state distance between successive same-task "
      "blocks, which vanishes for a spiral and does not for a loop. Shape becomes a "
      "<em>number</em>, not a label.")),
    col(C("build", "Which measurements separate by scenario, and which do not?", SK_TIE,
      "Proposed: every measurement made in both scenarios, paired and joined. Crossing lines — "
      "PC − backprop, the depth trend, digit identity. <strong>Parallel lines — freeze-W2 "
      "recovery (+0.36 / +0.15) and the argmax-minus-probe gap (+81.6 / +29.5).</strong>",
      "placed at the end of the section, as a tie-out of §5's result into §7 and §8 — not as an "
      "opener. It summarises work the reader has just seen rather than previewing work they have not.")),
  ],
  why="The split has to be earned twice and the honest position is that it is earned decisively on "
      "one axis and not at all on another. <strong>Structurally it is not arguable:</strong> "
      "Class-IL has five output units receiving no positive target during task 2, so suppression "
      "is physically available; Domain-IL's five shared units make it impossible. The scenarios "
      "differ in which mechanisms can exist, not in degree. <strong>Empirically the geometry "
      "argument is the strongest available</strong> — a loop and a spiral are different kinds of "
      "behaviour, not different amounts of one. But two central measurements behave the same in "
      "both, and the tie-out draws them as parallel lines rather than hiding them. A figure that "
      "showed only the crossings would be making a case, not testing one."))

# ---------------------------------------------------------------- 7
T.append(dict(n=7, q="Class-IL — an output and calibration problem", v="gap",
  vt="Four results point one way, one points the other. The tool that separates them was never built.",
  cols=[
    col(C("build", "It collapses to zero most of the time — when doesn't it, and why?",
      SK_DIST,
      "Proposed: the retention <em>distribution</em> over 70 seeds, with each intervention "
      "overlaid as a shifted distribution. Known: two groups, near-zero and 15–30%. "
      "<strong>Averaging reports a number no individual run produces.</strong> Digit identity "
      "explains the split — 8, 9, 5 in task 1 raise retention, 6 lowers it.",
      "replaces the bar chart. The question is what moves the <em>distribution</em>, not what "
      "moves the mean — and it merges the shape question with the why question into one figure."),
      C("appx", "Which digits, specifically?", pic("312_class_il_digit_table.png"),
      "Seeds as rows, digits as columns coloured by task. At Bonferroni only 6, 8 and 9 survive "
      "(digit 8: 11.6 vs 3.0, p = 0.006).",
      "too much detail for the main text — it is the evidence behind the distribution figure, not "
      "a substitute for it.")),
    col(C("amend", "Does protecting a layer during task 2 recover anything?",
      pic("130_freeze_factorial.png"),
      "<strong>Freezing the output layer recovers nothing</strong> — +0.36 ± 0.43 Class-IL, "
      "+0.15 ± 0.24 Domain-IL. Freezing the hidden layer is strongly negative. This is the result "
      "that contradicts the readout story.",
      "fold into the distribution figure as an overlay rather than standing alone as bars. Note "
      "freeze-both is absent because crossover is undefined on 0/10 seeds — a result, not a gap."),
      C("busy", "Does the information survive when the readout fails?",
      pic("320_ncm_both_tasks.png"),
      "Task-1 argmax collapses while a prototype readout holds ~83%. The control that matters: "
      "<strong>on task 2 the probe is worse than argmax</strong> (70.8 vs 91.3), which rules out "
      "\"the probe is simply a better classifier\".",
      "unreadable — four line types × two scenarios × per-seed traces, and the steps near 1700 "
      "are seeds leaving the mean, not behaviour. <strong>Also re-run:</strong> NCM is "
      "centroid-only with almost no dynamic range (69.9% at H=32 against an 81.2% raw-pixel "
      "baseline). Keep the task-2 reversal as a control; replace the probe.")),
    col(C("build", "Is the code intact, or is the probe too blunt to see the damage?",
      SK_PROBE,
      "Proposed: a trained/refit linear probe against argmax, both tasks, across the run. The two "
      "readings are distinguishable — probe high and argmax low means recalibration; both low "
      "means the representation is gone. <strong>The project currently cannot tell them "
      "apart.</strong>",
      "three tools answer three different questions: argmax (what the network reports), a trained "
      "probe (is it linearly decodable at all), masking (does removing the competitors restore "
      "it). NCM answers none of them well. This is the section's blocking dependency.")),
    col(C("build", "Does the trunk settle while the readout keeps thrashing?", SK_PATH,
      "Proposed: per-layer weight step per alternation block, from §6's repeated-sequence runs. "
      "If <strong>W1 converges while W2 keeps oscillating</strong>, that is direct evidence the "
      "trunk has found a joint solution and the readout is being dragged between two — which is "
      "the Class-IL claim, stated in weights rather than accuracy.",
      "same training runs as §6, so it costs one analysis rather than one experiment. 68 stores "
      "only a scalar weight gap per block, so per-layer instrumentation needs adding.")),
  ],
  why="Four things point at the readout: digit identity, the output specification, masking's "
      "success, and the argmax-versus-probe gap. One thing points away from it, and it is the "
      "cleanest confirmation that should have been available — <strong>freezing the output layer "
      "recovers nothing</strong>. This section is built to stage that contradiction rather than "
      "resolve it, because the project cannot currently resolve it: either the damage reaches the "
      "readout by a route freezing does not block, or the surviving-code reading is an artefact of "
      "a probe too blunt to detect representational loss. The trained probe separates those, and "
      "the per-layer path figure tests the same claim from the weight side. Until both exist the "
      "honest verdict is that the localisation is <em>suggested</em>, not established."))

# ---------------------------------------------------------------- 8
T.append(dict(n=8, q="Domain-IL — a representation problem", v="gap",
  vt="Different story, not the same story with smaller numbers. And there is no positive answer to \"where\".",
  cols=[
    col(C("ok", "Does which classes share an output unit predict what survives?",
      pic("311_task_pair_similarity_combined.png"),
      "The leading data-side explanation, and it <strong>does not replicate</strong>: r = +0.686 "
      "(p = 0.029) on seeds 0–9 became <strong>r = +0.183 (p = 0.61)</strong> on seeds 10–19. "
      "Pooled r = +0.509 — real but far weaker than first seen, and absent from an independent "
      "block.", None)),
    col(C("crop", "Does the architecture of the trunk matter more here?",
      pic("342_depth_sweep_accuracy.png"),
      "Depth makes Class-IL <em>better</em> for PC (+1.42 → +3.44) and Domain-IL <em>worse</em> "
      "(−0.72 → −3.49). <strong>Depth is a trunk variable</strong>, so an asymmetry along it is a "
      "representation-side finding — which is why it belongs here rather than as a robustness "
      "check in §5.",
      "take the Domain-IL column. Seed-to-seed spread here is far larger than any hyperparameter "
      "effect, so the variation is a property of the task draw, not the configuration.")),
    col(C("build", "Where does the damage live, when suppression is impossible?", SK_DRIFT,
      "Proposed: hidden-code displacement during task 2, resolved by whether a unit is task-1 or "
      "task-2 selective, against a frozen-weight baseline. <strong>The project has no positive "
      "localisation for Domain-IL at all.</strong> Freezing says the output layer is not it — "
      "which here is expected, not informative.",
      "the largest genuine gap. Path efficiency and layerwise cosine (65, 66) measure the right "
      "kind of thing but ran pre-300 against four rules including one since dropped, and 66 is "
      "flagged as needing re-verification before citation.")),
  ],
  why="Class-IL has five output units receiving no positive target, so it is an <strong>output and "
      "calibration</strong> problem. Domain-IL has none, so whatever happens must be in the "
      "representation or the shared readout — a <strong>different investigation, not the same one "
      "with smaller effects</strong>. The depth asymmetry supports that division directly: a trunk "
      "variable helps one scenario and hurts the other. What this section does not have is a "
      "positive answer to \"where\". The pairing correlation is a clear negative result, "
      "publishable as a correction but not an explanation; freezing rules out the output layer "
      "uninformatively; no drift measurement exists under the current protocol. Reporting that "
      "plainly is worth more than importing §7's readout story by analogy, which the Domain-IL "
      "data does not support."))

# ---------------------------------------------------------------- 9
T.append(dict(n=9, q="What actually helps?", v="part",
  vt="Six interventions measured. The ranking they jointly support has never been drawn.",
  cols=[
    col(C("build", "Of everything we can add, what recovers retention?", SK_INTERV,
      "Proposed: every intervention on one axis against the same unprotected control, both "
      "scenarios, backprop and PC. Replay ≫ masking (+7.6 to +17.5) &gt; SI (+3.15) ≈ EWC (+1.80) "
      "&gt; freezing (nothing) &gt; k-WTA (<strong>−9.8 backprop, −32.4 PC</strong>).",
      "six experiments' arrays on one axis. Masking is Class-IL only — there are no absent classes "
      "to mask in Domain-IL, which is the same structural fact §6 rests on. Censoring annotated "
      "per bar.")),
    col(C("ok", "Does sparsity gating help, as the PC literature suggests?",
      pic("220_kwta_k_sweep.png"),
      "<strong>Harmful, monotonically, and far worse for PC</strong>: Class-IL backprop 59.31 → "
      "49.53 but PC 61.85 → 29.44. Sparsity costs PC 3.3× backprop's loss in Class-IL and 11× in "
      "Domain-IL — PC's relaxation depends on the full hidden code.", None),
      C("ok", "Does consolidation help, and what happens past its optimum?",
      pic("210_si_lambda_sweep.png"),
      "Best at λ=1: +3.15 backprop, +2.39 PC. <strong>Past λ≥10 it deadlocks learning</strong> — "
      "and the evidence is the censoring, not the mean: seeds with a defined crossover fall to "
      "1/10 and 3/10, and those that cross sit exactly at the collapse floor.", None)),
    col(C("crop", "Does removing the competing units restore the readout?",
      pic("120_output_maths_and_masking.png"),
      "Masking the absent classes recovers <strong>+17.49 ± 1.83 (ce), +15.05 ± 3.97 (hinge), "
      "+7.59 ± 1.22 (mse)</strong>. It is the only intervention besides replay that moves "
      "Class-IL substantially — and it works by changing the readout, not the rule.",
      "take the masked-versus-unmasked half; the spec comparison is §2's. Censoring here is "
      "structural: masking keeps task 1 above task 2, so often there is no crossing at all.")),
  ],
  why="This is where the thesis argument lands, and it lands as a ranking rather than a single "
      "result. The two interventions that work substantially — replay and masking — either store "
      "the old data or change the readout. The ones that barely work add a penalty to the "
      "existing rule. The one that actively harms is the one the predictive-coding literature "
      "most often proposes. <strong>Nothing that modifies credit assignment alone recovers "
      "much.</strong> Read against §5, where PC's uplift is roughly a seventh of replay's, that is "
      "the project's conclusion stated in one axis."))


# ================================================================= render
V = {"ok": ("Answered", "v-ok"), "part": ("Partly answered", "v-part"),
     "gap": ("Not answered", "v-gap"), "build": ("Needs rebuilding", "v-part")}

def sect(x):
    lab, cls = V[x["v"]]
    return f'''<section class="tier" id="s{x['n']}">
<header class="th"><div class="tn">{x['n']}</div><div>
<h2>{x['q']}</h2>
<p class="vd"><span class="vc {cls}">{lab}</span>{x['vt']}</p></div></header>
<div class="grid">{"".join(x['cols'])}</div>
<div class="why"><h3>Why these figures answer the question</h3><p>{x['why']}</p></div>
</section>'''

nav = "".join(f'<a href="#s{x["n"]}"><span class="nn">{x["n"]}</span>{html.escape(x["q"])}'
              f'<span class="nv {V[x["v"]][1]}"></span></a>' for x in T)
cnt = {k: sum(1 for x in T if V[x["v"]][0] == k) for k in ("Answered", "Partly answered", "Not answered")}
ncard = sum(c.count('class="card') for x in T for c in x["cols"])
nbuild = sum(c.count('class="chip build"') for x in T for c in x["cols"])

CSS = """
:root{--paper:#f7f4ef;--surface:#fffefb;--ink:#1c1815;--ink2:#4a423a;--mut:#7a7066;
--rule:#e0d8cc;--rule2:#efe8dd;--ac:#b4430c;--ac2:#1a5f96;
--ok:#3f7d3a;--okb:#eaf3e8;--am:#a8710c;--amb:#f8f0dd;--gp:#1a5f96;--gpb:#e6eff7;
--bad:#a33121;--badb:#f8e9e5;--sh:0 1px 2px rgba(40,30,20,.05),0 3px 10px rgba(40,30,20,.045)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--paper:#15120f;--surface:#1e1a16;--ink:#f0eae2;--ink2:#c7bdb1;--mut:#948877;
--rule:#332c25;--rule2:#2a241e;--ac:#e8804a;--ac2:#6fb0e0;
--ok:#8fc78a;--okb:#1e2a1d;--am:#e0b45c;--amb:#2c2517;--gp:#7cb6e4;--gpb:#16232e;
--bad:#e59180;--badb:#2e1c18;--sh:0 1px 2px rgba(0,0,0,.3),0 3px 12px rgba(0,0,0,.28)}}
:root[data-theme="dark"]{--paper:#15120f;--surface:#1e1a16;--ink:#f0eae2;--ink2:#c7bdb1;--mut:#948877;
--rule:#332c25;--rule2:#2a241e;--ac:#e8804a;--ac2:#6fb0e0;
--ok:#8fc78a;--okb:#1e2a1d;--am:#e0b45c;--amb:#2c2517;--gp:#7cb6e4;--gpb:#16232e;
--bad:#e59180;--badb:#2e1c18;--sh:0 1px 2px rgba(0,0,0,.3),0 3px 12px rgba(0,0,0,.28)}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;font-size:15px;line-height:1.55;
font-family:"Charter","Bitstream Charter","Sitka Text",Cambria,Georgia,serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1560px;margin:0 auto;padding:0 24px 72px}
h1,h2,h3,h4{line-height:1.2;text-wrap:balance;margin:0}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;background:var(--rule2);padding:.08em .3em;border-radius:3px}
strong{font-weight:650}
.mast{padding:56px 0 30px;border-bottom:2px solid var(--ink)}
.eb{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;letter-spacing:.16em;
text-transform:uppercase;color:var(--ac);margin:0 0 14px}
.mast h1{font-size:clamp(30px,4.4vw,46px);font-weight:600;letter-spacing:-.018em;max-width:24ch}
.mast .sb{margin:16px 0 0;font-size:17px;color:var(--ink2);max-width:66ch}
.tal{display:flex;flex-wrap:wrap;gap:24px;margin-top:24px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.tal div{display:flex;align-items:baseline;gap:7px;color:var(--mut)}
.tal b{font-size:23px;font-weight:600;color:var(--ink);font-family:inherit}
.toc{margin:32px 0 0;border-top:1px solid var(--rule);padding-top:4px;
columns:2;column-gap:44px}
.toc a{display:flex;align-items:center;gap:13px;padding:8px 3px;break-inside:avoid;
border-bottom:1px solid var(--rule2);color:var(--ink);text-decoration:none;font-size:14px}
.toc a:hover{color:var(--ac)}
.nn{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--mut);min-width:15px;text-align:right}
.nv{margin-left:auto;width:8px;height:8px;border-radius:50%;flex:none}
.nv.v-ok{background:var(--ok)}.nv.v-part{background:var(--am)}.nv.v-gap{background:var(--gp)}
.tier{padding:52px 0 6px;border-bottom:1px solid var(--rule)}
.th{display:flex;gap:17px;align-items:flex-start;margin-bottom:24px}
.tn{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;font-weight:600;
color:var(--surface);background:var(--ink);width:27px;height:27px;flex:none;
display:flex;align-items:center;justify-content:center;border-radius:50%;margin-top:3px}
.th h2{font-size:clamp(21px,2.5vw,27px);font-weight:600;letter-spacing:-.012em;max-width:40ch}
.vd{margin:9px 0 0;font-size:14.5px;color:var(--ink2)}
.vc{display:inline-block;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;
letter-spacing:.09em;text-transform:uppercase;padding:2px 7px;border-radius:3px;margin-right:10px;vertical-align:1px}
.vc.v-ok{background:var(--okb);color:var(--ok)}.vc.v-part{background:var(--amb);color:var(--am)}
.vc.v-gap{background:var(--gpb);color:var(--gp)}
.grid{display:flex;gap:15px;align-items:stretch}
.col{flex:1 1 0;min-width:0;display:flex;flex-direction:column;gap:15px}
@media (max-width:1180px){.grid{flex-wrap:wrap}.col{flex:1 1 340px}}
@media (max-width:720px){.col{flex:1 1 100%}}
.card{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:5px;
padding:13px 14px 12px;box-shadow:var(--sh);display:flex;flex-direction:column;gap:9px;
border-top:3px solid var(--rule);flex:1 1 auto}
.card.ok{border-top-color:var(--ok)}.card.crop,.card.amend{border-top-color:var(--am)}
.card.rerun,.card.busy{border-top-color:var(--bad)}.card.build{border-top-color:var(--gp)}
.card.appx{border-top-color:var(--mut)}
.chip{align-self:flex-start;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9px;
letter-spacing:.1em;text-transform:uppercase;padding:2px 7px;border-radius:3px}
.chip.ok{background:var(--okb);color:var(--ok)}.chip.crop,.chip.amend{background:var(--amb);color:var(--am)}
.chip.rerun,.chip.busy{background:var(--badb);color:var(--bad)}.chip.build{background:var(--gpb);color:var(--gp)}
.chip.appx{background:var(--rule2);color:var(--mut)}
.card h4{font-size:14.5px;font-weight:600;letter-spacing:-.004em}
.media{background:#fff;border:1px solid var(--rule);border-radius:3px;padding:5px;line-height:0;overflow:hidden}
.media img,.media svg{width:100%;height:auto;display:block}
figcaption{font-size:12.5px;color:var(--ink2);line-height:1.5}
.note{margin:0;padding:8px 10px;background:var(--rule2);border-radius:3px;font-size:11.5px;line-height:1.48;color:var(--ink2)}
.k{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;letter-spacing:.07em;
text-transform:uppercase;color:var(--ac);margin-right:6px}
.a{stroke:var(--mut);stroke-width:1.3}.z{stroke:var(--ink2);stroke-width:1.1;stroke-dasharray:5 4}
.m{stroke:var(--rule);stroke-width:1.2;stroke-dasharray:4 4}.cn{stroke:var(--mut);stroke-width:1.2}
.e{stroke:var(--ink2);stroke-width:2}.d1{fill:var(--ac)}.d2{fill:var(--ac2)}.cx{fill:var(--ok)}
.b1{fill:none;stroke:var(--ac);stroke-width:2.3;stroke-linejoin:round}
.b2{fill:none;stroke:var(--ac2);stroke-width:2.3;stroke-linejoin:round}
.f1{fill:none;stroke:var(--ac);stroke-width:.9;opacity:.42}
.f2{fill:none;stroke:var(--ac2);stroke-width:.9;opacity:.42}
.fz{fill:none;stroke:var(--mut);stroke-width:1.6;stroke-dasharray:3 3}
.dh{stroke-dasharray:7 5}
.loop{fill:none;stroke:var(--ac);stroke-width:2.2}.loop2{fill:none;stroke:var(--ac);stroke-width:1.4;opacity:.5}
.spiral{fill:none;stroke:var(--ac2);stroke-width:2.2}
.ar,.ar2{fill:none;stroke:var(--mut);stroke-width:1.4;marker-end:url(#mk)}
.p1{fill:var(--ac);opacity:.8}.p2{fill:var(--ac2);opacity:.8}
.p3{fill:var(--ok);opacity:.8}.p4{fill:var(--bad);opacity:.8}
.l{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10px;fill:var(--mut)}
.s{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;fill:var(--ac)}
.s2{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;fill:var(--ac2)}
.why{margin:22px 0 0;padding:18px 20px;background:var(--rule2);border-radius:5px;border-left:3px solid var(--ac)}
.why h3{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;letter-spacing:.12em;
text-transform:uppercase;color:var(--ac);font-weight:600;margin-bottom:9px}
.why p{margin:0;font-size:15px;line-height:1.62;max-width:100ch;color:var(--ink)}
.key{margin:36px 0 0;padding:20px 22px;border:1px solid var(--rule);border-radius:5px;background:var(--surface)}
.key h3{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;letter-spacing:.12em;
text-transform:uppercase;color:var(--mut);margin-bottom:13px}
.key dl{display:grid;grid-template-columns:auto 1fr;gap:7px 14px;margin:0;font-size:13.5px}
.key dt,.key dd{margin:0}.key dd{color:var(--ink2)}
footer{padding:40px 0 0;color:var(--mut);font-size:13px;max-width:80ch}
"""

DOC = f'''<title>Report track — an honest audit of the evidence</title>
<style>{CSS}</style>
<svg width="0" height="0" style="position:absolute"><defs>
<marker id="mk" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
<path d="M0 0 L8 4 L0 8 z" fill="currentColor" style="fill:var(--mut)"/></marker></defs></svg>
<div class="wrap">
<header class="mast">
<p class="eb">Continual learning · backpropagation vs predictive coding</p>
<h1>Nine questions, and whether the project has actually answered them</h1>
<p class="sb">Built from the track downward, not from the figures back. Each section runs left to
right as an argument; cards stacked in a column are a direct comparison. Every section closes with
why its figures answer the question — or an honest account of why they do not yet.</p>
<div class="tal">
<div><b>{cnt["Answered"]}</b> answered</div><div><b>{cnt["Partly answered"]}</b> partly</div>
<div><b>{cnt["Not answered"]}</b> not answered</div><div><b>{ncard}</b> figure slots</div>
<div><b>{nbuild}</b> to build</div></div>
<nav class="toc">{nav}</nav>
</header>
{"".join(sect(x) for x in T)}
<div class="key"><h3>Status vocabulary</h3><dl>
<dt><span class="chip ok">Use as-is</span></dt><dd>Answers its question in current form.</dd>
<dt><span class="chip crop">Crop</span></dt><dd>Right figure; only part of it belongs here.</dd>
<dt><span class="chip amend">Amend</span></dt><dd>Right figure; data, scale or annotation needs changing.</dd>
<dt><span class="chip rerun">Re-run</span></dt><dd>Right form, wrong controls — protocol, seeds, scenario or a dropped rule.</dd>
<dt><span class="chip busy">Too busy</span></dt><dd>Right content, too dense to make its point.</dd>
<dt><span class="chip build">Build</span></dt><dd>No figure exists; the sketch stands in for what it should be.</dd>
<dt><span class="chip appx">Appendix</span></dt><dd>Real evidence, too much detail for the main text.</dd>
</dl></div>
<footer>Every number is carried from <code>progress.md</code>, re-derived from the saved
<code>.npz</code> arrays rather than from earlier notes. Where a figure and a note disagreed, the
arrays were treated as ground truth. Nothing here requires new training except §2's activation
check and §6's Class-IL alternation arm.</footer>
</div>'''

OUT.write_text(DOC, encoding="utf-8")
print(f"wrote {OUT}  {OUT.stat().st_size/1024/1024:.2f} MB")
print(f"sections={len(T)} cards={ncard} build={nbuild} {cnt}")
