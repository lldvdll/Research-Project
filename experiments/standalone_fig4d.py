"""standalone_fig4d.py -- Song & Bogacz (2024) Fig 4d, in one file, no src/ imports.

WHY THIS EXISTS
    The framework version (exps 30-34) produces noisy curves with no clear anticorrelation
    between the two tasks, where the paper's figure shows a sharp exchange. There are two
    possible causes and they need separating:
        (a) the framework is doing something wrong
        (b) our reading of their configuration is wrong
    This file is ~200 lines of explicit torch that you can read top to bottom. If it
    reproduces the figure, the framework is at fault. If it does not, our understanding of
    their setup is at fault and no amount of framework work would have helped.

WHAT IT VARIES -- the two things I am least sure of, crossed
    SPLIT   how the ten classes are divided into two tasks
      "fixed"   task 0 = classes 0-4, task 1 = classes 5-9, on every seed.
                Their data loader filters with partial_targets = range(5*task_i, 5*(task_i+1))
                BEFORE the label shuffle is applied, so the task composition never changes.
      "random"  a fresh random 5/5 split per seed. What exps 30-34 did. If task composition
                varies by seed, so does task difficulty -- Fashion-MNIST's garment classes
                (0,2,3,4,6) are far harder to separate than its shoes and bags -- and that
                variance would swamp everything when averaged over ten seeds.

    UNITS   how a class is assigned to one of the five shared output units
      "mod5"      shuffle all ten labels, then unit = shuffled_label % 5. This is literally
                  what their two mappers do in sequence. It COLLIDES: five distinct labels
                  taken mod 5 give about 3.9 distinct units on average, so typically two
                  classes share a unit and one unit is never a target at all.
      "distinct"  task[i] -> unit i. No collisions. What we assumed.

    2 splits x 2 unit schemes x {backprop, PC} x SEEDS.

READING THE OUTPUT
    The number that matters is ANTICORRELATION: over the analysed window, the correlation
    between task-1 error and task-2 error. Their figure shows a strong exchange, so this
    should be clearly NEGATIVE. Ours has been near zero, which is the noise you are seeing.
    Whichever variant makes it most negative is the one that matches their setup.
"""
import sys, time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ============================== config ==============================
DATA_DIR   = str(Path(__file__).resolve().parent.parent / "data")
FIG        = Path(__file__).resolve().with_suffix(".png")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

WIDTHS         = [784, 32, 32, 32, 5]   # num_layers 4 -> four weight matrices
PARTIAL_NUM    = 600                    # per class -> 3000 per task
BATCH          = 500                    # -> six updates per iteration
ITERS_PER_TASK = 4
TOTAL_ITERS    = 84
EVAL_PER_CLASS = 200
LR_W           = 0.001                  # SGD on the SUMMED loss
T_INFER        = 64                     # PC inference steps
LR_X           = 0.1                    # PC inference step size
X_DISCOUNT     = 0.9                    # multiply LR_X by this when the energy fails to fall
SEEDS          = [0, 1]

VARIANTS = [("fixed", "mod5"), ("fixed", "distinct"),
            ("random", "mod5"), ("random", "distinct")]
# ====================================================================


def load_data():
    tf = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.reshape(-1))])
    tr = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    te = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    return tr, te


def stack(ds, idx):
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx])
    return xs.to(DEVICE), ys.to(DEVICE)


def make_tasks(split, units, seed):
    """-> (tasks, unit_of) where tasks[t] is a list of original class labels and
       unit_of[c] is the output unit that class c is trained on."""
    rng = np.random.default_rng(seed)
    if split == "fixed":
        tasks = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]      # filtered before any shuffling
    else:
        p = rng.permutation(10).tolist()
        tasks = [p[:5], p[5:]]
    unit_of = {}
    if units == "mod5":
        shuffled = rng.permutation(10)                   # their shuffle_mapper
        for c in range(10):
            unit_of[c] = int(shuffled[c]) % 5            # their mapper, collisions and all
    else:
        for t in tasks:
            for i, c in enumerate(t):
                unit_of[c] = i
    return tasks, unit_of


# ------------------------------ the model ------------------------------
def init_weights(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Ws = []
    for a, b in zip(WIDTHS[:-1], WIDTHS[1:]):
        W = torch.randn(a, b, generator=g) * np.sqrt(2.0 / (a + b))   # xavier normal
        Ws.append(W.to(DEVICE))
    return Ws


def forward(x, Ws):
    a = x
    for W in Ws[:-1]:
        a = torch.sigmoid(a @ W)
    return a @ Ws[-1]


def energy(x0, xs, target, Ws):
    """Total squared prediction error. xs are the free hidden states, one per hidden layer.
       The PC node sits between the Linear and the Sigmoid, matching their
       structure ['Linear', 'PCLayer', 'Acf']."""
    F, a = 0.0, x0
    for l, W in enumerate(Ws[:-1]):
        F = F + 0.5 * ((xs[l] - a @ W) ** 2).sum()
        a = torch.sigmoid(xs[l])
    return F + 0.5 * ((a @ Ws[-1] - target) ** 2).sum()


def bp_update(x, target, Ws):
    """Plain backprop on 0.5 * sum((out - target)^2), summed over batch AND outputs."""
    for W in Ws:
        W.requires_grad_(True)
        W.grad = None
    loss = 0.5 * ((forward(x, Ws) - target) ** 2).sum()
    loss.backward()
    with torch.no_grad():
        for W in Ws:
            W -= LR_W * W.grad
    for W in Ws:
        W.requires_grad_(False)


def pc_update(x, target, Ws):
    """Settle the hidden states with the output clamped, then one weight step at that state."""
    with torch.no_grad():                                   # feedforward initialisation
        xs, a = [], x
        for W in Ws[:-1]:
            z = a @ W
            xs.append(z.clone())
            a = torch.sigmoid(z)
    xs = [z.requires_grad_(True) for z in xs]

    lr_x, last = LR_X, None
    for _ in range(T_INFER):
        F = energy(x, xs, target, Ws)
        g = torch.autograd.grad(F, xs)
        if last is not None:
            lr_x = lr_x * (1.0 if F < last else X_DISCOUNT)
        last = F.detach()
        with torch.no_grad():
            for z, gz in zip(xs, g):
                z -= lr_x * gz

    xs = [z.detach() for z in xs]
    for W in Ws:
        W.requires_grad_(True)
        W.grad = None
    energy(x, xs, target, Ws).backward()
    with torch.no_grad():
        for W in Ws:
            W -= LR_W * W.grad
    for W in Ws:
        W.requires_grad_(False)


# ------------------------------ one run ------------------------------
def run(rule, split, units, seed, train, test, tr_idx, te_idx):
    tasks, unit_of = make_tasks(split, units, seed)
    rng = np.random.default_rng(seed)

    train_sets, eval_sets = [], []
    for t in tasks:
        idx = np.concatenate([rng.permutation(tr_idx[c])[:PARTIAL_NUM] for c in t])
        x, y = stack(train, idx)
        train_sets.append((x, torch.tensor([unit_of[int(c)] for c in y.cpu()]).to(DEVICE)))
        eidx = np.concatenate([te_idx[c][:EVAL_PER_CLASS] for c in t])
        ex, ey = stack(test, eidx)
        eval_sets.append((ex, torch.tensor([unit_of[int(c)] for c in ey.cpu()]).to(DEVICE)))

    Ws = init_weights(seed)
    step = bp_update if rule == "bp" else pc_update
    errs = np.zeros((TOTAL_ITERS, 2))

    for it in range(TOTAL_ITERS):
        ti = (it // ITERS_PER_TASK) % 2
        X, Y = train_sets[ti]
        order = torch.randperm(len(Y), device=DEVICE)
        for s in range(0, len(Y), BATCH):
            b = order[s:s + BATCH]
            tgt = torch.zeros(len(b), WIDTHS[-1], device=DEVICE)
            tgt[torch.arange(len(b)), Y[b]] = 1.0
            step(X[b], tgt, Ws)
        with torch.no_grad():
            for j, (ex, ey) in enumerate(eval_sets):
                errs[it, j] = 1.0 - (forward(ex, Ws).argmax(1) == ey).float().mean().item()
    return errs, tasks, unit_of


# ------------------------------ main ------------------------------
train, test = load_data()
tr_idx = {c: (train.targets == c).nonzero(as_tuple=True)[0].numpy() for c in range(10)}
te_idx = {c: (test.targets == c).nonzero(as_tuple=True)[0].numpy() for c in range(10)}
NAMES = ["tshirt", "trouser", "pullover", "dress", "coat",
         "sandal", "shirt", "sneaker", "bag", "boot"]

print(f"device {DEVICE}   {len(VARIANTS)} variants x 2 rules x {len(SEEDS)} seeds")
res, t0 = {}, time.time()
for split, units in VARIANTS:
    for rule in ("bp", "pc"):
        acc = []
        for seed in SEEDS:
            e, tasks, unit_of = run(rule, split, units, seed, train, test, tr_idx, te_idx)
            acc.append(e)
            if rule == "bp" and seed == SEEDS[0]:
                u = [sorted(unit_of[c] for c in t) for t in tasks]
                print(f"\n[{split}/{units}] seed {seed}")
                print(f"   task1 {[NAMES[c] for c in tasks[0]]} -> units {u[0]}")
                print(f"   task2 {[NAMES[c] for c in tasks[1]]} -> units {u[1]}")
                print(f"   distinct units: task1 {len(set(u[0]))}, task2 {len(set(u[1]))} of 5")
        res[(split, units, rule)] = np.array(acc)
        print(f"   {rule}: final t1 {res[(split,units,rule)][:,-1,0].mean():.3f}  "
              f"t2 {res[(split,units,rule)][:,-1,1].mean():.3f}  ({time.time()-t0:4.0f}s)")

print("\n" + "=" * 78)
print("ANTICORRELATION between the two tasks (their figure shows a strong exchange,")
print("so this should be clearly NEGATIVE; near zero means the curves are just noise)")
print(f"{'variant':>22}{'backprop':>12}{'pc':>12}{'bp final t1':>14}{'pc final t1':>14}")
best = None
for split, units in VARIANTS:
    row = f"{split + '/' + units:>22}"
    cs = {}
    for rule in ("bp", "pc"):
        A = res[(split, units, rule)]
        cs[rule] = np.mean([np.corrcoef(a[:, 0], a[:, 1])[0, 1] for a in A])
        row += f"{cs[rule]:>12.3f}"
    row += f"{res[(split,units,'bp')][:,-1,0].mean():>14.3f}"
    row += f"{res[(split,units,'pc')][:,-1,0].mean():>14.3f}"
    print(row)
    if best is None or cs["pc"] < best[1]:
        best = ((split, units), cs["pc"])
print(f"\nmost anticorrelated for PC: {best[0][0]}/{best[0][1]}  (r = {best[1]:.3f})")
print("If one variant is strongly negative and the others are not, that variant is their")
print("setup and the framework should be changed to match it. If NONE is negative, the")
print("problem is not the task/unit construction and the next suspects are the learning")
print("rate and the number of iterations.")

fig, axes = plt.subplots(2, len(VARIANTS), figsize=(4.6 * len(VARIANTS), 7),
                         sharex=True, sharey=True)
for j, (split, units) in enumerate(VARIANTS):
    for i, rule in enumerate(("bp", "pc")):
        ax = axes[i, j]
        A = res[(split, units, rule)]
        for task, ls in ((0, "-"), (1, "--")):
            ax.plot(A[:, :, task].mean(0), ls, lw=1.8,
                    color="tab:red" if rule == "bp" else "tab:blue",
                    label=f"task {task + 1}")
        for b in range(ITERS_PER_TASK, TOTAL_ITERS, ITERS_PER_TASK):
            ax.axvline(b, color="k", lw=0.3, alpha=0.25)
        ax.set_title(f"{split}/{units}  {'backprop' if rule == 'bp' else 'PC'}", fontsize=9)
        ax.grid(alpha=0.2)
        if j == 0:
            ax.set_ylabel("test error"); ax.legend(fontsize=7)
        if i == 1:
            ax.set_xlabel("iteration")
fig.suptitle("Standalone Fig 4d: which task split and output-unit mapping gives the "
             "anticorrelated exchange their figure shows?")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}   total {time.time() - t0:.0f}s")
np.savez_compressed(FIG.with_suffix(".npz"),
                    **{f"{s}|{u}|{r}": res[(s, u, r)] for s, u in VARIANTS for r in ("bp", "pc")})
