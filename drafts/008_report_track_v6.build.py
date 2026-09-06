"""Report-track audit page. Ten sections; each grid is columns of 1-2 cards.
Story runs left to right across columns; a vertical pair is a direct comparison."""
import base64, html, pathlib

EXP = pathlib.Path(r"c:\Users\dgsan\uni\Research-Project\experiments")
OUT = pathlib.Path(__file__).with_name("008_report_track_v6.html")

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


# 12 — prospective configuration bridge
SK_DX = sk(
    ax(58, 22, 404, 190) +
    ''.join(f'<circle cx="{x}" cy="{y}" r="4" class="d1"/>' for x, y in
            [(88, 168), (112, 150), (136, 158), (162, 132), (188, 140), (214, 118),
             (240, 104), (266, 112), (292, 88), (318, 74), (344, 82), (372, 58)]) +
    '<line x1="76" y1="176" x2="390" y2="56" class="fit"/>' +
    t(232, 42, "does settling displacement predict retention?", "s", "middle") +
    t(18, 96, "task-1", "l") + t(18, 110, "retention", "l") +
    t(232, 226, "D  =  ‖ x*(settled) − x(feedforward) ‖", "l", "middle"),
    "prospective configuration bridge sketch")

# 13 — partial freeze diagnostic
SK_PFREEZE = sk(
    ax(64, 22, 404, 180) + '<line x1="64" y1="150" x2="404" y2="150" class="z"/>' +
    rect(84, 150, 6, 46, "p1") + rect(158, 150, 104, 46, "p2") +
    rect(232, 150, 104, 46, "p3") + rect(306, 150, 104, 46, "p4") +
    '<line x1="290" y1="46" x2="290" y2="150" class="m"/>' +
    t(107, 196, "freeze all W2", "l", "middle") + t(181, 196, "freeze task-1", "l", "middle") +
    t(181, 210, "columns only", "l", "middle") +
    t(255, 196, "mask", "l", "middle") + t(329, 196, "?", "l", "middle") +
    t(107, 138, "≈ 0", "s", "middle") + t(255, 36, "+17.5", "s2", "middle") +
    t(340, 60, "predicted", "s", "middle") + t(340, 74, "if readout", "s", "middle"),
    "partial freeze diagnostic sketch")


# ---------------------------------------------------------------- extra sketches
# target alignment — S&B's own credited mechanism
SK_TA = sk(
    '<line x1="30" y1="26" x2="30" y2="176" class="a"/><line x1="30" y1="176" x2="188" y2="176" class="a"/>' +
    '<polyline points="40,150 70,118 100,102 134,94 178,90" class="b1"/>' +
    '<polyline points="40,146 70,112 100,96 134,88 178,84" class="b2 dh"/>' +
    t(150, 116, "pc", "s2") + t(150, 166, "bp", "s") +
    t(109, 200, "training step", "l", "middle") + t(109, 216, "alignment over time", "l", "middle") +
    t(24, 18, "cos(ΔW, W* − W)", "l") +
    '<line x1="248" y1="26" x2="248" y2="176" class="a"/><line x1="248" y1="176" x2="404" y2="176" class="a"/>' +
    ''.join(f'<circle cx="{x}" cy="{y}" r="3.4" class="d1"/>' for x, y in
            [(268, 60), (284, 132), (300, 74), (318, 148), (334, 66), (350, 120),
             (366, 90), (382, 140), (396, 78)]) +
    '<line x1="258" y1="104" x2="400" y2="100" class="fit"/>' +
    t(326, 200, "alignment", "l", "middle") + t(326, 216, "vs retention: flat", "l", "middle") +
    t(242, 18, "retention", "l"),
    "target alignment sketch")

# PCA of the weight trajectory under alternation
SK_PCA = sk(
    '<line x1="30" y1="28" x2="30" y2="180" class="a"/><line x1="30" y1="180" x2="186" y2="180" class="a"/>' +
    '<path d="M56 168 C 72 62, 168 66, 172 112 C 176 156, 84 172, 58 166" class="loop"/>' +
    '<path d="M62 162 C 78 70, 162 72, 166 114 C 170 150, 90 166, 64 160" class="loop2"/>' +
    '<circle cx="56" cy="168" r="4" class="d1"/>' +
    t(108, 204, "Class-IL", "s", "middle") + t(108, 220, "W1 orbits, no contraction", "l", "middle") +
    '<line x1="246" y1="28" x2="246" y2="180" class="a"/><line x1="246" y1="180" x2="402" y2="180" class="a"/>' +
    '<path d="M262 170 C 288 68, 384 76, 386 118 C 388 152, 306 164, 282 152 C 264 142, 288 100, 348 104 C 372 106, 366 130, 336 132" class="spiral"/>' +
    '<circle cx="340" cy="126" r="4.5" class="cx"/>' +
    t(324, 204, "Domain-IL", "s2", "middle") + t(324, 220, "W1 contracts to a point", "l", "middle") +
    t(216, 18, "PC1  ×  PC2  of the weight trajectory", "l", "middle"),
    "weight-space PCA sketch")

# joint pre-training then sequential
SK_JOINT = sk(
    ax(58, 22, 404, 184) + '<line x1="180" y1="22" x2="180" y2="184" class="m"/>' +
    '<polyline points="66,52 180,44 214,86 268,148 330,176 396,180" class="b1"/>' +
    '<polyline points="66,48 180,42 214,62 268,88 330,100 396,104" class="b2 dh"/>' +
    '<polyline points="66,46 180,40 214,50 268,58 330,62 396,64" class="fz"/>' +
    t(300, 168, "sequential from scratch", "s") +
    t(300, 78, "joint first, then sequential", "s2") +
    t(300, 40, "joint ceiling", "l") +
    t(184, 18, "task-2 phase begins", "s") +
    t(20, 96, "task-1", "l") + t(20, 110, "accuracy", "l") +
    t(232, 220, "does knowing both first protect either?", "l", "middle"),
    "joint pre-training sketch")



# ================================================================= cards
S = dict(ok=("Use as-is", "ok"), crop=("Crop", "crop"), amend=("Amend", "amend"),
         rerun=("Re-run", "rerun"), busy=("Too busy", "busy"), build=("Build", "build"),
         appx=("Appendix", "appx"), tbl=("Table row", "tbl"))

def P(media, n): return (media, n)

def C(status, q, media, cap, note=None):
    pri = None
    if isinstance(media, tuple): media, pri = media
    lab, cls = S[status]
    n = f'<p class="note"><span class="k">{lab}</span>{note}</p>' if note else ""
    pb = f'<span class="pri p{pri}">P{pri}</span>' if pri else ""
    return (f'<figure class="card {cls}"><div class="chiprow"><div class="chip {cls}">{lab}</div>{pb}</div>'
            f'<h4>{q}</h4><div class="media">{media}</div><figcaption>{cap}</figcaption>{n}</figure>')

def col(*cards, feature=False):
    return f'<div class="col{" feature" if feature else ""}">' + "".join(cards) + '</div>'

T = []

# ================================================================== METHODS
T.append(dict(part="Methods", n="M1", w=200, figs=1,
  q="What does forgetting look like?", v="ok",
  vt="Built. 901 replaces the averaged, cropped version and carries its own saturation warning.",
  cols=[
    col(C("ok", "What happens to the task you just left?", pic("901_what_forgetting_looks_like.png"),
      "Class-IL, <strong>one seed, fixed budget</strong>, backprop, every class drawn separately. "
      "Task 1 plateaus at 93.9%, the switch lands at 900 updates, the curves cross at "
      "<strong>71%</strong> 57 updates later, and all five task-1 classes end at "
      "<strong>exactly 0.0%</strong>.",
      "two things the figure states on its face rather than leaving to a caption: this run "
      "<em>saturates</em> because it uses a fixed budget where every measured result uses matched "
      "competence, and <strong>0% is not chance</strong> &mdash; chance on ten classes is 10%, so "
      "zero means argmax has been captured by the task-2 units."), feature=True),
    col(C("appx", "Why not the ten-seed version?", pic("102_forgetting_demonstrated.png"),
      "The averaged form. Correct, and the source of the shape above, but the x-window stops "
      "~200 steps after the switch where the Class-IL mean still reads &asymp;29%.",
      "under matched competence the seeds stop at different steps, so an averaged tail is "
      "computed over a shrinking sample. Kept as appendix context.")),
  ],
  why="This figure defines rather than compares, which is why it sits in Methods and why one "
      "fixed-budget seed beats a ten-seed mean: nothing is smoothed and the per-class collapse "
      "the reader needs for R5 is visible from the start. Everything downstream refers to three "
      "things named here &mdash; the switch, the plateau and the crossing."))

T.append(dict(part="Methods", n="M2", w=500, figs=0,
  q="The setup &mdash; a parameter table, with figures as supporting evidence", v="part",
  vt="The activation hole is now closed by data (802 has run) but 903 is not drawn yet.",
  cols=[
    col(C("tbl", "Can the trunk do enough work to ask mechanistic questions?",
      pic("101_problem_complexity.png"),
      "A frozen random projection with a trained head reaches <strong>77.2%</strong> against a "
      "fully-trained <strong>89.7%</strong>. Training the hidden layer is worth ~12 points.",
      "table row: <em>hidden layer, trained</em>. This number becomes load-bearing in R5, where "
      "the linear probe's random-init floor turns out to be the same phenomenon."),
      C("tbl", "Is the width sufficient?", pic("100_capacity_vs_width.png"),
      "Joint accuracy 84.1 (H=8) &rarr; 89.7 (H=32) &rarr; 90.3 (H=64). H=32 sits off the "
      "bottleneck. <strong>H=4 and H=8 are capacity-limited</strong>.",
      "table row: <em>H = 32</em>. The limitation returns in R2, where PC reverses sign at H=4.")),
    col(C("build", "Does the conclusion depend on the activation?", P(SK_ACT, 2),
      "<strong>802 has now run.</strong> Paired PC &minus; backprop crossover: Class-IL +1.27 "
      "(tanh), +1.51 (sigmoid), <strong>+2.53</strong> (relu). Domain-IL &minus;0.53 (tanh), "
      "<strong>+0.74 (sigmoid)</strong>, &minus;0.88 (relu).",
      "the insensitivity argument <strong>half fails, and the report must say so</strong>. "
      "Class-IL keeps its sign under all three activations. Domain-IL does not &mdash; sigmoid "
      "flips it positive. Since Domain-IL's effect is small (&minus;0.7) that is a sign flip "
      "within a narrow band rather than a reversal of a large effect, but tanh can no longer be "
      "defended as “the choice does not matter” in both scenarios. 903 must draw this."),
      C("tbl", "Does it depend on the output maths?", pic("120_output_maths_and_masking.png"),
      "The spec sets the <em>magnitude</em> &mdash; ce 49.94 &lt; hinge 54.76 &lt; mse 59.31 "
      "&mdash; but masking removes suppression under all three.",
      "table row: <em>MSE / one-hot</em>. The masking half is a result and moves to R5.")),
    col(C("ok", "Does PC's settling reach a fixed point?", pic("334_pc_settle_trace_by_dt_log.png"),
      "dt &le; 0.5 all reach the <strong>identical plateau</strong> (0.0850 Class-IL, 0.0787 "
      "Domain-IL). dt changes the route, not the destination.",
      "and the control is <em>not</em> a one-time gate: dt = 0.4 is correct at H=32/depth 1 and "
      "silently wrong at depth &ge; 2 and Class-IL H=4, sitting 3&ndash;7&times; above the true "
      "fixed point. That cost two full sweep re-runs. 341/342 use dt = 0.2 for this reason.")),
  ],
  why="Every row of the parameter table is defended by a measurement rather than by convention, "
      "and the one row that had no defence at all &mdash; the activation &mdash; now has data "
      "behind it. The honest version of that row is weaker than hoped: insensitivity holds in "
      "Class-IL and fails in Domain-IL under sigmoid."))

T.append(dict(part="Methods", n="M3", w=350, figs=1,
  q="How forgetting is measured, and why that metric", v="ok",
  vt="Justified from single-rule failures only. The circular version is demoted to a robustness check.",
  cols=[
    col(C("ok", "Why not read the endpoint?", pic("111_metric_sensitivity_to_threshold.png"),
      "Backprop alone, stopping threshold swept 75&rarr;95%. Every endpoint metric drifts with "
      "where you stop; crossover does not.",
      "<strong>this is the whole justification, and it uses one rule measured against itself.</strong> "
      "It does not reference the PC comparison, so it cannot be selecting a metric by the answer "
      "that metric gives.")),
    col(C("appx", "The robustness check, deliberately demoted", pic("112_which_metric_survives.png"),
      "Which metric preserves the sign of PC &minus; backprop across thresholds.",
      "<strong>this figure cannot justify the metric choice</strong> &mdash; it picks the "
      "instrument by the answer it gives on the very comparison it will then be used for. It runs "
      "<em>after</em> the result as a robustness check and is cited that way. An earlier draft had "
      "this the wrong way round.")),
  ],
  why="A metric chosen because it flatters the comparison is not evidence. The argument therefore "
      "runs entirely on single-rule behaviour: endpoint metrics inherit the stopping rule, "
      "crossover does not, and a censored crossover means task 1 never fell below task 2 &mdash; "
      "the best outcome, so it is ranked rather than dropped."))

# ================================================================== RESULTS
T.append(dict(part="Results", n="R1", w=250, figs=1,
  q="Two forgetting phenotypes", v="ok",
  vt="Built. The hinge of the report: columns differ, rows barely do.",
  cols=[
    col(C("ok", "Does the scenario change the shape, and does the rule?",
      pic("911_two_forgetting_phenotypes.png"),
      "Phase plot, time removed. Final task 1 <strong>5.4%</strong> (Class-IL) against "
      "<strong>38.5%</strong> (Domain-IL); crossover 65.0 against 75.8. Ceilings are 93.6 and "
      "94.3, so retention is read against those.",
      "read the columns first, then the rows. <strong>The scenario changes the phenotype; the "
      "learning rule barely changes anything.</strong> Same protocol, same seeds &mdash; only the "
      "output layer differs. This is why the mechanism investigation splits by scenario and not "
      "by rule."), feature=True),
  ],
  why="The structural argument matters more than the numbers and comes first: Class-IL has ten "
      "output units, five of which receive no positive target during task 2, so output "
      "suppression is <em>available</em>. Domain-IL has five shared units and suppression "
      "<em>cannot occur</em>. The scenarios differ in which mechanisms are physically possible, "
      "not merely in degree."))

T.append(dict(part="Results", n="R2", w=500, figs=2,
  q="The rule comparison &mdash; small, systematic, scenario-dependent", v="ok",
  vt="Built. 912 asks whether the effect is real; 913 asks whether it is large.",
  cols=[
    col(C("ok", "Is the sign reversal stable across every axis we can vary?",
      pic("912_pc_minus_backprop_sweeps.png"),
      "Paired PC &minus; backprop crossover across lr, width and depth, both scenarios, zero line. "
      "At the working point <strong>+1.42 &plusmn; 0.39</strong> (Class-IL) and "
      "<strong>&minus;0.72 &plusmn; 0.19</strong> (Domain-IL). Depth widens the split both ways "
      "(+3.44 / &minus;3.49 at four layers); Class-IL H=4 reverses to &minus;4.14.",
      "the lower row <strong>draws the censoring instead of averaging over it</strong>. At lr 0.16 "
      "the Class-IL crossover is undefined on 10/10 backprop seeds &mdash; an absent measurement, "
      "not a null result, and marked as such."), feature=True),
    col(C("ok", "How large is a real fix, on the same axes?",
      pic("913_replay_sets_the_scale_a.png"),
      "Replay, the positive control: <strong>+9.84 &plusmn; 0.77</strong> (Class-IL) and "
      "<strong>+3.43 &plusmn; 0.50</strong> (Domain-IL) against PC's +1.42 and &minus;0.72.",
      "about <strong>seven times</strong> PC's effect where PC wins, and positive where PC is "
      "negative. Separate from 912 on purpose: on 912's &plusmn;4pp axis replay would compress "
      "the entire PC result into a few pixels."),
      C("ok", "Is that a tuning artefact?", pic("913_replay_sets_the_scale_b.png"),
      "The same three sweeps with replay included. Replay stays positive and roughly flat across "
      "every lr, width and depth.",
      "so the gap is not an artefact of where the sweep was sampled. What replay costs is stated "
      "in the text: it stores and re-presents task-1 data, the exact resource a continual-learning "
      "rule is supposed not to need.")),
  ],
  why="Order matters. 912 establishes the effect is real and systematic; 913 puts it on a scale. "
      "Answering them the other way round would make the first look like special pleading. The "
      "conclusion the section can defend: the problem is solvable, and changing the "
      "credit-assignment rule is not what solves it."))

T.append(dict(part="Results", n="R3", w=500, figs=2,
  q="Does prospective configuration explain the difference?", v="ok",
  vt="Built, and the answer is sharper than expected: the mechanism replicates and costs the other task.",
  cols=[
    col(C("ok", "Does PC aim its updates better? (Song &amp; Bogacz Fig. 3b)",
      pic("915_target_alignment_a.png"),
      "Target alignment on the batch being trained. PC is <strong>higher</strong> than backprop: "
      "paired <strong>+0.0245 &plusmn; 0.0046</strong> (5.4 sem) in Class-IL.",
      "<strong>their claim replicates on our networks.</strong> This belongs in Results whichever "
      "way it fell &mdash; an earlier draft demoted it to an appendix negative, which was wrong."),
      C("ok", "&hellip;and what does that cost the task it is not training?",
      pic("915_target_alignment_b.png"),
      "The same cosine measured on a fixed task-1 batch that is never trained on. Both rules go "
      "<strong>negative</strong> after the switch and PC is <strong>more</strong> negative: "
      "&minus;0.0175 &plusmn; 0.0053 (3.3 sem).",
      "the interference is a <strong>transient</strong>, not a drift &mdash; it plunges to "
      "&minus;0.48 within ~10 updates and recovers by ~40, and the plunge is 2.4&times; deeper in "
      "Class-IL. This is forgetting caught per update rather than inferred from an endpoint, and "
      "because it is a rate it does not inherit the training budget."), feature=True),
    col(C("ok", "Does the settling displacement go anywhere?",
      pic("916_displacement_to_retention_b.png"),
      "Total distance each layer travels over task 2. PC moves the <strong>output</strong> weights "
      "markedly less &mdash; paired &minus;0.767 &plusmn; 0.191 (4.0 sem) Class-IL, &minus;0.846 "
      "&plusmn; 0.200 (4.2 sem) Domain-IL &mdash; while the <strong>trunk path is "
      "indistinguishable</strong> (0.4 and 0.2 sem).",
      "PC takes a shorter route through weight space and does not convert it into proportionally "
      "better retention. Note the difference is the <em>same size in both scenarios</em> even "
      "though the retention outcome flips sign between them."),
      C("ok", "⚠ The confound that reverses this answer",
      pic("916_displacement_to_retention_c.png"),
      "Matched competence makes task-2 length a dependent variable (119&ndash;4999 updates). Long "
      "runs have small <em>mean</em> per-update &#8214;&Delta;W&#8214; <em>and</em> forget more, so "
      "mean step size correlates with retention at <strong>r = +0.93</strong>.",
      "that reads as “bigger updates preserve task 1” and is an artefact. The "
      "<strong>total path</strong> has no such problem and gives the interpretable sign, "
      "<strong>r = &minus;0.56</strong>. Every claim here is made on totals.")),
  ],
  why="The mechanism the source paper credits is real and measurable on our networks &mdash; and "
      "the same measurement shows what it costs. PC configures itself more prospectively for the "
      "task in front of it, moves the readout less, and pushes the absent task's outputs further "
      "away while doing so. That is a coherent mechanistic story, and it explains why the "
      "retention benefit is small rather than large."))

T.append(dict(part="Results", n="R4", w=750, figs=3,
  q="Why these are two investigations, not one", v="part",
  vt="Two of three built. 917 supplies the strongest scenario separation in the project; the tie-out figure is not drawn.",
  cols=[
    col(C("ok", "Are the weights converging, orbiting, or drifting?",
      pic("917_alternation_geometry.png"),
      "Repeated alternation, 10 blocks, matched competence per block. <strong>Neither scenario "
      "traces a closed loop.</strong> Task-1 accuracy at the end of each task-2 block: Class-IL "
      "4.8&rarr;35.9, Domain-IL 37.6&rarr;63.1.",
      "<strong>the pre-registered prediction is refuted and the section says so.</strong> The plan "
      "predicted Class-IL would loop and Domain-IL would spiral in. Both converge; what differs is "
      "where they stop."), feature=True),
    col(C("ok", "The measurement that separates the scenarios most sharply",
      pic("917_alternation_geometry.png"),
      "Panel (d). At a <em>single</em> switch PC &minus; backprop is &minus;0.72 &plusmn; 0.19. At "
      "the <strong>fifth task-2 block</strong> it is <strong>&minus;16.71 &plusmn; 2.45</strong> "
      "(6.8 sem) in Domain-IL and &minus;1.78 &plusmn; 3.09 (nothing) in Class-IL.",
      "PC <em>plateaus</em> in Domain-IL while backprop keeps improving. A two-task protocol "
      "understates this by more than twenty times &mdash; an argument about the benchmark as much "
      "as about the rule, and the single largest scenario separation the project has."),
      C("ok", "Is forgetting about where you start in weight space?",
      pic("919_joint_then_sequential_a.png"),
      "Train on the joint distribution to convergence, then run the sequential protocol from "
      "there. Class-IL 4.8&rarr;<strong>16.2</strong>, Domain-IL 37.6&rarr;<strong>44.3</strong>.",
      "protection is real and <strong>partial</strong>. The solution can be handed to the network "
      "and it still leaves, so forgetting here is not mainly a failure to <em>find</em> a joint "
      "solution &mdash; it is a failure to stay at one.")),
    col(C("ok", "⚠ The confound in that arm, drawn rather than hidden",
      pic("919_joint_then_sequential_b.png"),
      "Both arms stop task 1 at the same competence, but the joint arm is already above threshold "
      "when its task-1 phase begins, so that phase is far shorter (402&rarr;182 updates).",
      "a shorter phase means less to lose. Correlation between the phase-length difference and the "
      "retention difference: Class-IL <strong>r = &minus;0.64</strong> (confound present), "
      "Domain-IL backprop <strong>r = &minus;0.09</strong> (clean). <strong>Lean on the Domain-IL "
      "backprop arm.</strong> The two cannot be separated within this design."),
      C("build", "Which measurements separate by scenario, and which do not?", P(SK_TIE, 2),
      "Proposed: every measurement made in both scenarios on one axis, rows = measurement, "
      "columns = scenario, so sign flips read as crossings. Pure re-analysis, no new training.",
      "this is the figure that makes the section's title an argument rather than an assertion. It "
      "must include the measurements that do <em>not</em> separate &mdash; freezing recovers "
      "nothing in either (+0.36 and +0.15) and the argmax&ndash;probe gap is large in both &mdash; "
      "or it is one-sided.")),
  ],
  why="The split is justified on two independent grounds. <em>Structural</em>: output suppression "
      "is available in Class-IL and physically impossible in Domain-IL, so the two scenarios admit "
      "different explanations. <em>Empirical</em>: the measurements separate by sign and by "
      "magnitude &mdash; PC &minus; backprop is +1.42 against &minus;0.72 at one switch and "
      "&minus;1.78 against &minus;16.71 after five, depth trends run opposite ways, and masking is "
      "only definable in one. The honest qualifier stays: the split is <em>not</em> justified by "
      "the readout gap, which behaves similarly in both."))

T.append(dict(part="Results", n="R5", w=1000, figs=4,
  q="Where does the damage live?", v="part",
  vt="The two decisive figures are built, and they overturn the readout account this section used to assume.",
  cols=[
    col(C("ok", "Is it the task-1 weights, or the competition at argmax?",
      pic("922_partial_column_freeze_a.png"),
      "Four conditions, ten seeds. Control 4.8, freeze all of W&#8322; 3.2, <strong>freeze the "
      "task-1 columns 6.7</strong>, <strong>mask 50.9</strong>.",
      "<strong>this landed on the second of three pre-committed readings.</strong> Masking "
      "recovers +46.11 &plusmn; 4.52; freezing exactly the weights masking spares recovers "
      "+1.87 &plusmn; 1.03. Twenty-five times smaller &mdash; so <strong>masking does not work by "
      "sparing those weights</strong>. The remaining candidate is that masking changes what "
      "W&#8321; learns."), feature=True),
    col(C("ok", "&hellip;and they do not even act at the same time",
      pic("922_partial_column_freeze_b.png"),
      "Paired against control: freezing the task-1 columns moves <em>crossover</em> by "
      "<strong>+1.16 &plusmn; 0.23</strong> (5.0 sem, 10W&ndash;0L) but the endpoint barely at all. "
      "Masking moves the <em>endpoint</em> by +46 and crossover by +0.20 (0.1 sem).",
      "freezing changes <strong>where the curves cross</strong>; masking changes <strong>where "
      "they end</strong>. One mechanism at two strengths would differ in size but agree in shape. "
      "A second, independent reason to reject the readout account."),
      C("ok", "Is the code still readable when the readout fails?",
      pic("923_probe_vs_argmax.png"),
      "Refit linear probe against argmax, both tasks, both scenarios. Class-IL task 1: probe "
      "<strong>82.6</strong> against argmax <strong>21.1</strong> &mdash; a +61 point gap.",
      "<strong>but the probe reads 80.2% on the untrained network of the same seed</strong>, so "
      "only <strong>~2.4 points</strong> come from anything the trunk learned. Drawn without that "
      "floor line this figure says “the representation survives” when it mostly shows the "
      "probe barely needed one. The Domain-IL task-2 panel is the control that keeps it honest: "
      "there the gap <em>reverses</em> (&minus;2.3, argmax wins), so the probe is not simply a "
      "better classifier.")),
    col(C("amend", "The freezing result underneath it", pic("130_freeze_factorial.png"),
      "Freezing the output layer recovers nothing in either scenario &mdash; +0.36 &plusmn; 0.43 "
      "and +0.15 &plusmn; 0.24.",
      "922 explains why this null is not the paradox it looked like: freezing all of W&#8322; also "
      "blocks task 2 from learning through the readout and forces it into the shared trunk."),
      C("build", "What moves the outcome distribution?", P(SK_DIST, 2),
      "Proposed: retention as a distribution over 70 seeds, not a mean, with interventions "
      "overlaid. Class-IL retention is bimodal &mdash; near-zero and 15&ndash;30% &mdash; so "
      "<strong>the mean reports a number no individual run produces</strong>.",
      "still worth building: it is the figure that stops every other number in R5 being read as if "
      "runs were interchangeable.")),
    col(C("build", "What do the weights themselves do, per layer?", P(SK_PATH, 1),
      "Proposed: per-layer weight path and hidden-code drift from 803 and 804.",
      "916 already answers half of this &mdash; PC's trunk path is indistinguishable from "
      "backprop's while its readout path is much shorter. What is missing is the <em>code</em> "
      "drift that would localise Domain-IL's damage."),
      C("gapx" if False else "build", "&hellip;and where does Domain-IL's damage sit?", P(SK_DRIFT, 1),
      "Proposed: hidden-code drift measured on task-1 data during task 2.",
      "<strong>this is the largest genuine gap in the project.</strong> Suppression is "
      "structurally impossible in Domain-IL, so the explanation must be representation drift "
      "&mdash; and freezing only says where it is <em>not</em>. No positive localisation exists.")),
  ],
  why="This section changed its mind, and the report is stronger for saying so. It was written "
      "around a readout account: the code survives, argmax misreads it, protect the readout and "
      "retention returns. Two independent measurements now argue against that. Freezing exactly "
      "the weights masking spares recovers almost nothing, and the interventions act at different "
      "points of the curve; and the probe that appeared to show a surviving representation reads "
      "within ~2.4 points of its own random-init floor. What survives is the observation, not the "
      "explanation &mdash; and the honest position is that Class-IL's mechanism is not yet "
      "localised and Domain-IL's is not localised at all."))

# ================================================================== DISCUSSION
T.append(dict(part="Discussion", n="D1", w=300, figs=1,
  q="Added mechanisms &mdash; a preliminary evaluation", v="part",
  vt="Data exists for every arm; the consolidated figure is not drawn.",
  cols=[
    col(C("build", "Of everything we can add, what recovers retention?", P(SK_INTERV, 2),
      "Proposed: every intervention on one axis with its SEM and its censoring. Replay +9.84 / "
      "+3.43 &middot; masking +7.6 to +17.5 (Class-IL only) &middot; SI +3.15 &middot; EWC +1.80 "
      "&middot; PC +1.42 / &minus;0.72 &middot; freezing nothing &middot; k-WTA "
      "<strong>&minus;9.8 / &minus;32.4</strong>.",
      "<strong>⚠ EWC's &lambda; grid never bracketed its optimum</strong>, so its number is a "
      "lower bound and the figure must say so rather than ranking it as though it were tuned."),
      C("appx", "Sparsity gating", pic("220_kwta_k_sweep.png"),
      "k-WTA is the one intervention that makes things markedly <em>worse</em>, and worse for PC "
      "than for backprop.",
      "kept because a negative result on a plausible mechanism is informative, and because the "
      "asymmetry between the rules is itself a datum.")),
  ],
  why="Placing the learning-rule result on one axis with every alternative is what stops it being "
      "read as larger than it is. The defensible claim: prospective configuration produces a "
      "small, real, scenario-dependent difference, roughly a seventh of what replay buys, and "
      "therefore the credit-assignment rule is not the lever. That does not refute Song &amp; "
      "Bogacz &mdash; it locates their effect at a size, under a protocol that reports its "
      "censoring and pairs its seeds."))


V = {"ok": ("Answered", "v-ok"), "part": ("Partly answered", "v-part"),
     "gap": ("Not answered", "v-gap"), "build": ("Needs rebuilding", "v-part")}

def sect(x, prev):
    lab, cls = V[x["v"]]
    bar = ""
    if x["part"] != prev:
        n = sum(1 for y in T if y["part"] == x["part"])
        w = sum(y["w"] for y in T if y["part"] == x["part"])
        f = sum(y["figs"] for y in T if y["part"] == x["part"])
        bar = (f'<div class="partbar"><h2>{x["part"]}</h2>'
               f'<span>{n} section{"s" if n>1 else ""} &middot; {f} main-text figure'
               f'{"s" if f!=1 else ""} &middot; approx. {w:,} words</span></div>')
    return bar + f'''<section class="tier" id="s{x['n']}">
<header class="th"><div class="tn">{x['n']}</div><div>
<h2>{x['q']}</h2>
<p class="vd"><span class="vc {cls}">{lab}</span>{x['vt']}</p>
<p class="cost">approx. <b>{x['w']}</b> words &middot; <b>{x['figs']}</b> main-text figure{"s" if x['figs']!=1 else ""}</p></div></header>
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
.partbar{margin:56px 0 -18px;padding:16px 0 12px;border-top:2px solid var(--ink);
display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
.partbar h2{font-size:15px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;margin:0}
.partbar span{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--mut)}
.cost{margin:7px 0 0;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--mut)}
.cost b{color:var(--ink);font-weight:600}
.chip.tbl{background:var(--rule2);color:var(--ac)}
.card.tbl{border-top-color:var(--ac)}
.chiprow{display:flex;align-items:center;gap:7px}
.pri{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9px;letter-spacing:.08em;
padding:2px 6px;border-radius:3px;border:1px solid currentColor}
.pri.p1{color:var(--bad)}.pri.p2{color:var(--am)}.pri.p3{color:var(--mut)}
.map{margin:34px 0 0;padding:22px 24px;border:1px solid var(--rule);border-radius:5px;background:var(--surface)}
.map h3{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;letter-spacing:.12em;
text-transform:uppercase;color:var(--ac);margin-bottom:6px}
.map .lede{margin:0 0 16px;font-size:14px;color:var(--ink2);max-width:82ch}
.map ol{margin:0;padding:0;list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.map li{border-left:2px solid var(--ac);padding:2px 0 2px 12px;font-size:13.5px;line-height:1.5}
.map b{display:block;font-size:11px;font-family:ui-monospace,Menlo,Consolas,monospace;
letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:500;margin-bottom:3px}
.map .sx{color:var(--ac);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px}
.punch{margin:34px 0 0;padding:26px 28px;border-radius:5px;background:var(--ink);color:var(--paper)}
.punch h3{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9.5px;letter-spacing:.12em;
text-transform:uppercase;opacity:.65;margin-bottom:12px}
.punch p{margin:0;font-size:18px;line-height:1.58;max-width:74ch}
.punch em{opacity:.8;font-size:14.5px;display:block;margin-top:14px;font-style:normal}
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
<h1>Nine questions, and what the evidence now says</h1>
<p class="sb">Built from the track downward, not from the figures back. Each section runs left to
right as an argument; cards stacked in a column are a direct comparison. Every section closes with
why its figures answer the question — or an honest account of why they do not yet.</p>
<div class="tal">
<div><b>{cnt["Answered"]}</b> answered</div><div><b>{cnt["Partly answered"]}</b> partly</div>
<div><b>{cnt["Not answered"]}</b> not answered</div><div><b>{sum(x["figs"] for x in T)}</b> main-text figures</div>
<div><b>{sum(x["w"] for x in T):,}</b> words</div>
<div><b>{nbuild}</b> to build</div></div>
<nav class="toc">{nav}</nav>
</header>

<div class="map">
<h3>What changed between v5 and this version</h3>
<p class="lede"><b>R3 answered, and the answer cuts both ways.</b> Song &amp; Bogacz's target
alignment replicates &mdash; PC aims its updates better than backprop on the batch it is training
(+0.0245, 5.4&nbsp;sem) &mdash; and the same measurement shows it pushes the untrained task's
outputs further away (&minus;0.0175, 3.3&nbsp;sem). The mechanism is real and it has a cost.</p>
<p class="lede"><b>R4's pre-registered prediction is refuted.</b> Neither scenario traces a closed
loop under repeated alternation; both converge, and they differ in where they stop. The same run
produced the project's sharpest scenario separation: PC &minus; backprop is &minus;0.72 after one
switch and <b>&minus;16.71 &plusmn; 2.45</b> after five, in Domain-IL only.</p>
<p class="lede"><b>R5 changed its mind.</b> It was written around a readout account. Freezing
exactly the weights masking spares recovers +1.87 against masking's +46.11, and the linear probe
that appeared to show a surviving representation sits ~2.4 points above its own random-init floor.
The observation survives; the explanation does not.</p>
<h3>Fitting an 8,000-word budget</h3>
<p class="lede">Written out at 150&ndash;225 words per working figure, the earlier layout came to
roughly 26 main-text figures and would have overrun by about 40%. This version puts the setup in a
<strong>grouped parameter table</strong> with figures as appendix evidence, merges the two
scenario sections into one organised by tool, and demotes the added-mechanism sweeps to
Discussion. Fifteen main-text figures; the appendix carries the rest and is meant to be
load-bearing, not a dumping ground.</p>
<ol>
<li><b>Methods &middot; approx. 1,050</b><span class="sx">M1&ndash;M3</span> &mdash; what forgetting
is, the parameter table, and the derivation of the metric. No rule comparison appears here.</li>
<li><b>Results &middot; approx. 3,000</b><span class="sx">R1&ndash;R5</span> &mdash; the two
phenotypes, the comparison, its mechanism, why the scenarios split, and where the damage lives.</li>
<li><b>Discussion &middot; approx. 300 of it</b><span class="sx">D1</span> &mdash; added
mechanisms as a preliminary evaluation, plus the unresolved contradiction and the Domain-IL gap.</li>
</ol>
<p class="lede" style="margin:14px 0 0"><strong>One thing to check before the 800 list is
frozen:</strong> whether captions count against the 8,000. If they do not, a caption can carry the
&ldquo;what was done&rdquo; and &ldquo;what it shows&rdquo; load &mdash; roughly 100 words per
figure &mdash; and several appendix demotions above become unnecessary.</p>
</div>

{"".join(sect(x, T[i-1]["part"] if i else None) for i, x in enumerate(T))}
<div class="key"><h3>Status vocabulary</h3><dl>
<dt><span class="chip ok">Use as-is</span></dt><dd>Answers its question in current form.</dd>
<dt><span class="chip crop">Crop</span></dt><dd>Right figure; only part of it belongs here.</dd>
<dt><span class="chip amend">Amend</span></dt><dd>Right figure; data, scale or annotation needs changing.</dd>
<dt><span class="chip rerun">Re-run</span></dt><dd>Right form, wrong controls — protocol, seeds, scenario or a dropped rule.</dd>
<dt><span class="chip busy">Too busy</span></dt><dd>Right content, too dense to make its point.</dd>
<dt><span class="chip build">Build</span></dt><dd>No figure exists; the sketch stands in for what it should be.</dd>
<dt><span class="chip appx">Appendix</span></dt><dd>Real evidence, too much detail for the main text.</dd>
<dt><span class="chip tbl">Table row</span></dt><dd>Becomes a row in the Methods parameter table; figure cited as supporting evidence.</dd>
<dt><span class="pri p1">P1</span></dt><dd>Build first — the argument does not close without it.</dd>
<dt><span class="pri p2">P2</span></dt><dd>Worth building; cheap, or it is a chapter's headline.</dd>
<dt><span class="pri p3">P3</span></dt><dd>Only if time allows.</dd>
</dl></div>

<div class="punch">
<h3>What the whole thing adds up to</h3>
<p>Predictive coding does not provide a general solution to catastrophic forgetting. Its
prospective-configuration dynamics produce a small, scenario-dependent change in continual-learning
performance — ahead in Class-IL, behind in Domain-IL, amplifying with depth in both directions —
but the dominant source of forgetting lies in the interaction between task structure, output
competition and representation stability. Explicit memory mechanisms are substantially more
effective than modifying the credit-assignment rule alone.</p>
<em>Six of the thirteen builds are P1, and five of those six are instrumentation of runs that
already exist rather than new experiments.</em>
</div>
<footer>Every number is carried from <code>progress.md</code>, re-derived from the saved
<code>.npz</code> arrays rather than from earlier notes. Where a figure and a note disagreed, the
arrays were treated as ground truth. Nothing here requires new training except §2's activation
check and §6's Class-IL alternation arm.</footer>
</div>'''

OUT.write_text(DOC, encoding="utf-8")
print(f"wrote {OUT}  {OUT.stat().st_size/1024/1024:.2f} MB")
print(f"sections={len(T)} cards={ncard} build={nbuild} {cnt}")
