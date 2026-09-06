"""Report-track audit page. Ten sections; each grid is columns of 1-2 cards.
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
  q="What does forgetting look like?", v="build",
  vt="Definitional. One figure, rebuilt — what exists today understates its own point.",
  cols=[
    col(C("build", "What happens to the task you just left?", P(SK_DEF, 2),
      "Class-IL, <strong>one seed, fixed budget</strong> — the whole track shown without averaging "
      "over runs that start and stop in different places. Two bold curves, all ten classes as "
      "thin lines, crossover marked, arrows naming forgetting and learning.",
      "the five task-1 classes falling to zero while the five task-2 classes rise <em>is</em> the "
      "output-competition mechanism. Planting it in Methods means R5 explains something the "
      "reader has already seen.")),
    col(C("amend", "What does the current version show?", pic("102_forgetting_demonstrated.png"),
      "Ten seeds, both scenarios, both rules — correct in form and the source of the shape above, "
      "but averaged and cropped.",
      "the x-window stops ~200 steps after the switch, where the Class-IL mean still reads ≈29%. "
      "The run's actual final task-1 is <strong>3.3%</strong>. As drawn, it makes forgetting look "
      "milder than it is.")),
  ],
  why="This figure defines rather than compares, which is why it sits in Methods and why a single "
      "fixed-budget seed beats a ten-seed mean: nothing is smoothed, the whole trajectory is "
      "visible, and the per-class lines carry a mechanism the reader needs later. Everything "
      "downstream refers to three things named here — the switch, the plateau, and the crossing."))

T.append(dict(part="Methods", n="M2", w=500, figs=0,
  q="The setup — a parameter table, with figures as supporting evidence", v="part",
  vt="Six justifications become one grouped table plus appendix panels. The largest word saving available.",
  cols=[
    col(C("tbl", "Can the trunk do enough work to ask mechanistic questions?",
      pic("101_problem_complexity.png"),
      "A frozen random projection with a trained head reaches <strong>77.2%</strong> against the "
      "fully-trained <strong>89.7%</strong>. Training the hidden layer is worth ~12 points, so "
      "\"where does forgetting live\" is not asked of a dead layer.",
      "table row: <em>hidden layer, trained</em> — supporting evidence, appendix."),
      C("tbl", "Is the width sufficient, and where does it stop being?",
      pic("100_capacity_vs_width.png"),
      "Joint accuracy 84.1 (H=8) → 89.7 (H=32) → 90.3 (H=64). H=32 is off the bottleneck and flat "
      "thereafter. <strong>H=4 and H=8 are capacity-limited</strong> — which returns in R2 where "
      "PC reverses at H=4.",
      "table row: <em>H = 32</em> — supporting evidence, appendix.")),
    col(C("build", "Does the conclusion depend on the activation?", P(SK_ACT, 2),
      "Proposed: crossover under tanh, sigmoid and ReLU, both scenarios. <strong>There is "
      "currently no justification for tanh at all</strong> — it was inherited, and Song &amp; "
      "Bogacz use sigmoid.",
      "table row: <em>tanh</em>. Still worth running — a cheap three-point sweep turns an "
      "unjustified inheritance into a justification by insensitivity, and it is a hole an "
      "examiner will find."),
      C("tbl", "Does it depend on the output maths?",
      pic("120_output_maths_and_masking.png"),
      "The spec sets the <em>magnitude</em> of suppression — ce 49.94 &lt; hinge 54.76 &lt; mse "
      "59.31 — but <strong>masking removes it under all three</strong>. The choice changes the "
      "size, not the finding.",
      "table row: <em>MSE / one-hot</em>. The masking half of this figure is a result and appears "
      "in R5.")),
    col(C("ok", "Does PC's settling reach a fixed point, and does dt move it?",
      pic("334_pc_settle_trace_by_dt_log.png"),
      "dt ≤ 0.5 all reach the <strong>identical plateau</strong> (0.0850 Class-IL, 0.0787 "
      "Domain-IL) — dt changes the route, not the destination. That is the whole control: a "
      "stable band exists and the choice inside it is free.",
      "the one settling figure that stays in the main text. Cut from four to one."),
      C("appx", "Does that band survive a change of architecture?",
      pic("347_pc_settle_dt_by_depth.png"),
      "The <strong>only</strong> unstable cell is dt = 0.4 at depth ≥ 2; dt = 0.2 converges "
      "everywhere. The band <em>shrinks with depth</em>, so this is a per-configuration "
      "precondition — ignoring it cost two sweep re-runs.",
      "appendix, cited from the table. Methodologically the most transferable result in the "
      "project, but it does not carry an argument the report is making.")),
    col(C("appx", "Why pair on seeds?", P(SK_PAIR, 3),
      "Pairing does not move the point estimate — it shrinks the SEM <strong>4.5× in Class-IL and "
      "11.5× in Domain-IL</strong>, because the class split is shared between rules at a given "
      "seed.",
      "one sentence in Methods plus an appendix panel. Demoted from a main figure: it is a "
      "statistical footnote, not a result.")),
  ],
  why="Grouped parameter decisions belong in a table with \"supporting evidence: Fig. N\" "
      "cross-references, not in six figures each costing 150 words to explain. That converts this "
      "section from roughly 900 words to about 500, and it reads better — a reader checking "
      "whether the width was justified wants a row and a pointer, not a page. What stays in the "
      "main text is the one settling panel, because \"dt changes the route, not the destination\" "
      "is the sentence that licenses every PC number in the report."))

T.append(dict(part="Methods", n="M3", w=350, figs=1,
  q="How forgetting is measured, and why that metric", v="ok",
  vt="Two single-rule failures, and the stopping rule falls out of them. No rule comparison appears here.",
  cols=[
    col(C("amend", "Does an endpoint metric survive a change of stopping point?",
      pic("111_metric_sensitivity_to_threshold.png"),
      "<strong>Backprop alone.</strong> Final task-1 falls monotonically to 0.28 as the threshold "
      "moves; S&amp;B mean error is U-shaped. An endpoint number is a statement about where you "
      "stopped.",
      "fold in 113's mechanism — the endpoint moves because <em>task 1's own peak moves with "
      "it</em>. Merge both panels into one figure with 343."),
      C("ok", "Does crossover survive a change of learning rate?",
      pic("343_lr_degradation_and_reliability.png"),
      "It does not, at the top of the range: Class-IL crossover is <strong>undefined on 10/10 "
      "backprop seeds at lr = 0.16</strong>. Two candidate metrics, two different failure "
      "regimes.", None), feature=True),
    col(C("appx", "Does the choice change the reported result?",
      pic("112_which_metric_survives.png"),
      "Five metrics × five stopping thresholds. Class-IL: crossover and crossover-of-peak hold one "
      "sign at 2 SEM throughout; every endpoint metric flips. Domain-IL: nothing survives.",
      "<strong>appendix, not Methods.</strong> This is inherently a paired PC−backprop figure, so "
      "putting it in Methods would show a rule-comparison result before the rule comparison "
      "exists — the circularity we removed. Cite it from R2 as robustness.")),
  ],
  why="The stopping rule is <em>derived</em> here rather than asserted: the endpoint fails as the "
      "stopping point moves, crossover fails as the learning rate rises, and a competence-matched "
      "rule is what survives both. Critically, every step uses backprop alone. Choosing the metric "
      "because it preserves the PC-versus-backprop sign would select the instrument by the answer "
      "it gives on the comparison it will then be used for — the error <code>CLAUDE.md</code> "
      "already forbids for learning rates. That is why 112 is cited from Results and not shown "
      "here."))

# ================================================================== RESULTS
T.append(dict(part="Results", n="R1", w=250, figs=1,
  q="Two forgetting phenotypes", v="ok",
  vt="The core figure. Four panels that exist, never assembled into the one grid that makes the point.",
  cols=[
    col(C("ok", "Backprop, Class-IL — where does a run end up?",
      pic("310_forgetting_by_scenario_class_il.png"),
      "Task-1 against task-2 accuracy, time removed, 10 seeds. Every trajectory turns hard left "
      "and terminates against the axis: final task 1 <strong>5.4%</strong>, crossover 65.0.",
      "this and the three beside it become one 2×2 — scenario across, rule down. Assembly only."),
      C("ok", "PC, Class-IL — does the shape change?",
      pic("332_pc_forgetting_by_scenario_class_il.png"),
      "Same protocol, same seeds. Crossover 66.44 against 65.01. The <em>shape</em> is "
      "indistinguishable; only the crossing height moves, and only slightly.", None)),
    col(C("ok", "Backprop, Domain-IL — where does the same run end up?",
      pic("310_forgetting_by_scenario_domain_il.png"),
      "Identical protocol and seeds, only the output layer differs. Trajectories stop well short "
      "of the axis: final task 1 <strong>38.5%</strong>, crossover 75.8, spread 13.1–64.2.", None),
      C("ok", "PC, Domain-IL — does the shape change?",
      pic("332_pc_forgetting_by_scenario_domain_il.png"),
      "Crossover 75.06 against 75.76 — PC slightly <em>behind</em>. Again the family of shapes "
      "matches backprop's.", None)),
  ],
  why="Read as a 2×2 these four panels make a point no single panel makes: <strong>changing the "
      "scenario changes the shape of forgetting; changing the rule does not.</strong> Down a "
      "column the trajectories are near-superimposable; across a row they are different families "
      "— one collapsing onto an axis, one stopping on a shelf. Every later section is a "
      "consequence of that asymmetry, which is why this is the first and load-bearing result."))

T.append(dict(part="Results", n="R2", w=500, figs=2,
  q="The rule comparison — small, systematic, scenario-dependent", v="part",
  vt="Answered on three axes. The consolidation that makes it one claim has never been drawn.",
  cols=[
    col(C("build", "Is the sign reversal stable across every axis we can vary?", P(SK_CONSOL, 1),
      "Proposed: PC − backprop against lr, width and depth, two scenario lines, zero line drawn. "
      "<strong>Class-IL above zero and Domain-IL below it everywhere except the narrowest "
      "width.</strong> At the working point: <strong>+1.42 ± 0.39</strong> and "
      "<strong>−0.72 ± 0.19</strong>, defined 10/10.",
      "the section's headline, and it replaces the three sweep figures rather than joining them — "
      "they go to appendix. Three existing scripts' arrays, one figure, no retraining.")),
    col(C("appx", "The lr sweep behind it", pic("340_lr_sweep_accuracy.png"),
      "<strong>Both rules peak near lr 0.01–0.02 and decline past it</strong>, so the defaults sit "
      "close to jointly optimal and the comparison was not read at a point favouring either rule.",
      "appendix. The consolidated figure carries the claim; this supports it."),
      C("appx", "…and the width and depth sweeps",
      pic("342_depth_sweep_accuracy.png"),
      "<strong>Depth amplifies the split in both directions</strong>: Class-IL +1.42 → +3.44, "
      "Domain-IL −0.72 → −3.49. Width reverses at H=4 (−4.14 ± 0.70). Contradicts the pre-100 "
      "\"depth doesn't matter\" line.",
      "appendix, with 341. The depth asymmetry is quoted in R5 as trunk-side evidence, so it is "
      "cited twice and drawn once.")),
    col(C("amend", "How large is a real fix, on the same axes?", pic("340_lr_sweep_diff.png"),
      "Replay − backprop is the upper line everywhere: <strong>+55 to +62 pp Class-IL "
      "retention</strong> against PC's +0 to +2; paired Δcrossover <strong>+9.84 ± 0.77</strong> "
      "against PC's +1.42. Flat across the whole grid, so not a tuning artefact.",
      "<strong>stays in Results, not Discussion.</strong> Without it the reader finishes the "
      "headline not knowing whether +1.42 is large. Trim to backprop/PC/replay at the working "
      "point and give replay its own scale — its magnitude currently crushes PC's line flat.")),
  ],
  why="The answer is small, real, and sign-flipped by scenario. A one-to-three-point effect at a "
      "single architecture invites the dismissal that it is a quirk of H=32, depth 1 — and the "
      "sweeps refute that in an unexpected way: the effect is not fragile, it is "
      "<em>systematic</em>, growing with depth in both directions at once. The replay panel is "
      "what keeps the claim proportionate. Reporting +1.42 without it risks the Results overselling; "
      "reporting it here rather than in Discussion means the scale arrives with the claim rather "
      "than two sections later. The full intervention ranking is a separate, weaker argument and "
      "goes to Discussion."))

T.append(dict(part="Results", n="R3", w=500, figs=2,
  q="Does prospective configuration explain the difference?", v="gap",
  vt="The mechanism the source paper credits — and our own — measured against the outcome for the first time.",
  cols=[
    col(C("rerun", "Does PC align its updates toward the solution better than backprop?",
      P(SK_TA, 1),
      "<strong>Song &amp; Bogacz's own credited mechanism.</strong> Measured once at 5 seeds "
      "across four rules: alignment <em>does not track forgetting</em>, and ranks the rules "
      "opposite to retention. Two panels — alignment through training, and alignment against "
      "retention, which is flat.",
      "re-run as backprop vs PC at 10 seeds under the 300-series protocol. A direct test of the "
      "source paper's mechanism belongs in Results whichever way it falls.")),
    col(C("build", "Does how far the state moves predict how much is forgotten?", P(SK_DX, 1),
      "Proposed: per-seed settling displacement <strong>D = ‖x*(settled) − x(feedforward)‖</strong> "
      "against task-1 retention and against ‖ΔW‖, both scenarios. The missing link — <strong>PC "
      "dynamics → internal configuration → weight change → forgetting</strong>.",
      "<code>handle[\"diag\"][\"displacement\"]</code> is already published on every train step, so "
      "this instruments existing runs. 344's finding becomes a panel of it — see right."),
      C("ok", "Where does settling reach the weights?",
      pic("344_weight_step_vs_lr.png"),
      "W1 shows <strong>no differential damping</strong> (0.93 vs 0.90); W2 does "
      "(<strong>0.99 vs 0.78</strong>). PC's output update multiplies the same error against the "
      "<em>settled</em> hidden activity, so settling reaches W2 by a route the hypothesis did not "
      "predict.",
      "becomes a panel of the bridge figure rather than standing alone. It refuted its own "
      "pre-registered prediction, which is why it belongs in the argument.")),
  ],
  why="Everything before this is behavioural. None of it touches what makes PC <em>PC</em> — that "
      "the hidden state relaxes to a configuration the feedforward pass would not produce, and the "
      "update is computed against that. Without this section an examiner can fairly ask where the "
      "evidence is that prospective configuration has anything to do with the result. "
      "<strong>Target alignment is the mechanism the source paper credits</strong>, so testing it "
      "is not optional and its failure is a result about their claim. The displacement is our own "
      "measure of the same thing and is currently discarded on every update as a convergence "
      "check. Kept as its own short section rather than folded into R2 because target alignment is "
      "the most citable thing in the report and would be invisible as a sub-panel."))

T.append(dict(part="Results", n="R4", w=750, figs=3,
  q="Why these are two investigations, not one", v="gap",
  vt="The strongest section available, and almost none of it is built.",
  cols=[
    col(C("rerun", "Under repeated alternation, does the network converge on a joint solution?",
      P(SK_REPEAT, 1),
      "20 alternations, task-1 against task-2 accuracy, colour = time, <strong>both scenarios side "
      "by side</strong>. Domain-IL already shows the inward spiral. Prediction: <strong>Class-IL "
      "closes a loop</strong> — ping-ponging between incompatible solutions, learning nothing "
      "cumulative.",
      "one figure, not two. The existing run is Domain-IL only, 5 seeds, <strong>fixed "
      "budget</strong> — re-run at the 90% threshold as backprop vs PC and add the Class-IL arm. "
      "Note matched competence makes block length a dependent variable, which is itself the "
      "\"gradual relearn\" result."),
      C("appx", "What the existing Domain-IL run looks like",
      pic("68_what_happens_under_repeated_task_switching_spiral.png"),
      "All three rules spiral inward toward the joint corner — the geometry is already there, at "
      "the wrong protocol.",
      "appendix or superseded entirely by the re-run above.")),
    col(C("build", "Are the weights converging, orbiting, or drifting?", P(SK_PCA, 1),
      "Proposed: PCA of the weight trajectory under alternation, W1 and W2 separately, both "
      "scenarios. Accuracy can look settled while parameters wander, so <strong>this is the "
      "stronger version of the panel to its left</strong> — and it asks whether the trunk "
      "contracts while the readout orbits.",
      "68 already stores the full W1 trajectory (600 × 6272 = 196×32, seed 0), so the Domain-IL W1 "
      "panel is computable today. W2 and Class-IL ride along with the re-run — one experiment, "
      "both figures.")),
    col(C("build", "Is forgetting about where you start in weight space?", P(SK_JOINT, 1),
      "Proposed: train on the <strong>joint</strong> distribution first, then run the sequential "
      "protocol from there, against sequential-from-scratch. If a network that already solves both "
      "tasks still collapses on task 1, forgetting is not a failure to <em>find</em> a joint "
      "solution — it is a failure to <em>stay</em> in one.",
      "sits awkwardly here and we keep it anyway; where it belongs will be obvious once it runs. "
      "Pairs with the PCA panel: does joint initialisation turn the Class-IL loop into a spiral?")),
    col(C("build", "Which measurements separate by scenario, and which do not?", P(SK_TIE, 2),
      "Proposed: every measurement made in both scenarios, paired and joined. Crossings — PC − "
      "backprop, the depth trend, digit identity. <strong>Parallels — freeze-W2 recovery "
      "(+0.36 / +0.15) and the argmax-minus-probe gap (+81.6 / +29.5).</strong>",
      "the structural half of the justification is <strong>Methods</strong>: Class-IL has five "
      "output units receiving no positive target, Domain-IL has none, so suppression is available "
      "in one and impossible in the other. That follows from the architecture description, so this "
      "figure only has to carry the empirical half.")),
  ],
  why="The split is earned twice. <strong>Structurally it is not arguable</strong>, and that "
      "argument lives in Methods with the architecture — the two scenarios differ in which "
      "mechanisms can physically exist, not in degree. <strong>Empirically the geometry argument "
      "is the strongest available</strong>: a loop and a spiral are different kinds of behaviour, "
      "not different amounts of one. The weight-space panel is what stops it being a pretty "
      "picture — a scalar contraction measure turns \"it looks like a loop\" into \"the network "
      "returns to a substantially different state after each switch, while Domain-IL's "
      "progressively contracts\". The tie-out then draws the two measurements that behave the "
      "<em>same</em> in both scenarios as parallel lines rather than hiding them."))

T.append(dict(part="Results", n="R5", w=1000, figs=4,
  q="Where does the damage live?", v="gap",
  vt="Merged from two scenario sections into one, organised by tool. Both scenarios in every figure.",
  cols=[
    col(C("build", "What moves the outcome distribution?", P(SK_DIST, 2),
      "Proposed: the retention <em>distribution</em> over 70 seeds, both scenarios, with each "
      "readout intervention overlaid. Class-IL is two groups, near-zero and 15–30%; Domain-IL is a "
      "broad unimodal spread. <strong>Averaging Class-IL reports a number no run produces.</strong> "
      "Masking recovers +17.5; freezing recovers nothing.",
      "absorbs the freeze bar chart and the masking half of 120. The question is what moves the "
      "<em>distribution</em>, not the mean."),
      C("amend", "The freezing result underneath it", pic("130_freeze_factorial.png"),
      "<strong>Freezing the output layer recovers nothing</strong> — +0.36 ± 0.43 Class-IL, "
      "+0.15 ± 0.24 Domain-IL. Freezing the hidden layer is strongly negative.",
      "folds in as an overlay. Freeze-both is absent because crossover is undefined on 0/10 seeds "
      "— a result, not a gap.")),
    col(C("build", "Is it the task-1 weights, or the competition at argmax?", P(SK_PFREEZE, 1),
      "Proposed: freeze <strong>only the task-1 columns of W2</strong>. Masking is train-time only "
      "— <code>active_vector</code> zeroes the error in <code>output_error</code> while "
      "<code>predict</code> still argmaxes over all ten units. Masking spares those weights "
      "<em>while letting task-2 units learn</em>; freezing all of W2 spares them but blocks task "
      "2's readout path, forcing it into W1.",
      "the decisive experiment for the contradiction to its left, and cheap. If it reproduces "
      "+17.5 the mechanism is established; if not, masking works another way. <strong>Needs "
      "approval — <code>_apply_freeze</code> is whole-matrix.</strong>")),
    col(C("build", "Is the code still readable when the readout fails?", P(SK_PROBE, 1),
      "Proposed: a trained/refit linear probe against argmax, both tasks, both scenarios. Probe "
      "high and argmax low means recalibration; both low means the representation is gone. "
      "<strong>The project cannot currently tell them apart.</strong>",
      "three tools, three questions: argmax (what the network reports), a trained probe (is it "
      "linearly decodable), masking (does removing competitors restore it)."),
      C("busy", "What the current probe shows", pic("320_ncm_both_tasks.png"),
      "Task-1 argmax collapses while a prototype readout holds ~83%. The control that matters: "
      "<strong>on task 2 the probe is worse than argmax</strong> (70.8 vs 91.3), ruling out \"the "
      "probe is simply a better classifier\".",
      "unreadable, and the steps near 1700 are seeds leaving the mean. NCM is centroid-only with "
      "almost no dynamic range. Keep the task-2 reversal as a control; replace the probe.")),
    col(C("build", "What do the weights themselves do, per layer?", P(SK_PATH, 1),
      "Proposed: per-layer weight step per alternation block, both scenarios, from R4's runs. If "
      "<strong>W1 converges while W2 keeps oscillating</strong> in Class-IL, that is the "
      "output-competition claim stated in weights rather than accuracy.",
      "shares R4's training runs and its PCA panel — one analysis, not one experiment."),
      C("build", "…and where does Domain-IL's damage sit?", P(SK_DRIFT, 1),
      "Proposed: hidden-code displacement during task 2, resolved by whether a unit is task-1 or "
      "task-2 selective, against a frozen-weight baseline. <strong>There is no positive "
      "localisation for Domain-IL at all</strong> — and that is because it has not been measured, "
      "not because it cannot be.",
      "the largest genuine gap, and it is cheap: instrumentation of runs that already exist.")),
    col(C("ok", "Does task structure predict what survives?",
      pic("311_task_pair_similarity_combined.png"),
      "Domain-IL's leading data-side explanation <strong>does not replicate</strong>: r = +0.686 "
      "(p = 0.029) on seeds 0–9 became <strong>r = +0.183 (p = 0.61)</strong> on seeds 10–19. "
      "Class-IL's does: 8, 9 and 5 in task 1 raise retention, 6 lowers it, surviving Bonferroni at "
      "70 seeds.",
      "one figure, both scenarios, opposite outcomes. The Class-IL digit table (312) is the "
      "appendix evidence behind the second half.")),
  ],
  why="Two sections that both ended in \"we don't know\" made a weak back half. Organised by tool "
      "instead of by scenario, the contrast becomes the point of each figure rather than the "
      "section boundary, and the section gains a spine: intervene on the readout, read the "
      "representation, watch the weights, test the data dependence. <strong>The contradiction at "
      "its centre narrows rather than muddies.</strong> Masking recovers +17.5 acting only on the "
      "training error; freezing all of W2 spares the same weights and recovers nothing — but also "
      "blocks task 2 from learning through the readout. That one difference is testable by "
      "freezing only the task-1 columns. On the Domain-IL side the honest statement is that the "
      "measurement has not been made, not that it cannot be — the tools exist and the displacement "
      "is already logged."))

# ================================================================== DISCUSSION
T.append(dict(part="Discussion", n="D1", w=300, figs=1,
  q="Added mechanisms — a preliminary evaluation", v="part",
  vt="Demoted from Results. Real runs, but the grids were not prepared carefully enough to claim.",
  cols=[
    col(C("build", "Of everything we can add, what recovers retention?", P(SK_INTERV, 2),
      "Proposed: every intervention on one axis. Replay ≫ masking (+7.6 to +17.5) &gt; SI (+3.15) "
      "≈ EWC (+1.80) &gt; freezing (nothing) &gt; k-WTA (<strong>−9.8 backprop, −32.4 PC</strong>).",
      "framed as future work, not result. Replay, masking and freezing are results-grade and "
      "appear in R2 and R5; <strong>EWC, SI and k-WTA are what belongs here</strong>.")),
    col(C("appx", "Consolidation, and what happens past its optimum",
      pic("210_si_lambda_sweep.png"),
      "Best at λ=1: +3.15 backprop, +2.39 PC. Past λ≥10 it <strong>deadlocks learning</strong> — "
      "and the evidence is the censoring, not the mean: defined crossovers fall to 1/10 and 3/10.",
      "⚠ EWC's λ grid was <strong>still rising at its top end</strong>, so its optimum was never "
      "bracketed. That is the specific reason this is preliminary."),
      C("appx", "Sparsity gating", pic("220_kwta_k_sweep.png"),
      "<strong>Harmful, monotonically, and far worse for PC</strong>: Class-IL backprop 59.31 → "
      "49.53 but PC 61.85 → 29.44. An intervention motivated by the predictive-coding literature "
      "fails here, and fails PC hardest.",
      "the reading that PC's relaxation depends on the full hidden code is a <strong>hypothesis "
      "this is consistent with</strong>, not something it establishes.")),
  ],
  why="These are real runs at 10 seeds, but two things stop them being results. EWC's grid never "
      "bracketed its optimum, and the head-to-head comparison at each mechanism's best setting "
      "(230) was only smoke-tested. Presented as a preliminary evaluation they still do useful "
      "work: they establish that <strong>nothing which modifies credit assignment alone recovers "
      "much</strong>, and that the one intervention the predictive-coding literature most often "
      "proposes actively harms. That is a legitimate closing observation and a clear statement of "
      "what a follow-up should run."))


# ================================================================= render
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
<h1>Nine questions, and whether the project has actually answered them</h1>
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
<h3>Fitting an 8,000-word budget</h3>
<p class="lede">Written out at 150&ndash;225 words per working figure, the earlier layout came to
roughly 26 main-text figures and would have overrun by about 40%. This version puts the setup in a
<strong>grouped parameter table</strong> with figures as appendix evidence, merges the two
scenario sections into one organised by tool, and demotes the added-mechanism sweeps to
Discussion. Sixteen main-text figures; the appendix carries the rest and is meant to be
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
