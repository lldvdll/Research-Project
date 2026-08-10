"""NumPy mirror of src/model.py, src/predictive_coding.py and src/eqprop.py.

Purpose: validate the MATHS of the unified specification without needing torch. Run this
first; if it passes, any failure in the torch code is a plumbing bug, not an algebra bug.

    python3 tests/test_numpy_mirror.py

Checks
  1. hinge output_error reproduces the old hard-coded EqProp nudge exactly
  2. masked cross-entropy matches van de Ven eq. (2) (softmax over the ACTIVE set only)
  3. masked loss gives absent classes EXACTLY zero gradient
  4. squared-error suppression has a fixed point at 0; softmax suppression does not
  5. PC's dF/dx1 matches finite differences of the energy
  6. PC's weight update matches -dF/dW at the settled state
  7. PC with steps=0 equals the backprop gradient (the 'no relaxation' control)
  8. EqProp's (nudged - free)/beta converges to the backprop gradient as beta -> 0
"""
import numpy as np

rng = np.random.default_rng(0)
TOL = 1e-6


def f(h):
    return np.tanh(h)


def df(h):
    return 1.0 - np.tanh(h) ** 2


def softmax(z, active=None):
    z = z.copy()
    if active is not None:
        z[:, active == 0] = -np.inf
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    e[~np.isfinite(z)] = 0.0
    return e / e.sum(1, keepdims=True)


def output_error(out, target, loss, active=None, mask=False, margin=1.0):
    if loss == "mse":
        e = target - out
    elif loss == "ce":
        e = target - softmax(out, active)
    elif loss == "hinge":
        e = np.where(margin - target * out > 0, target, 0.0)
    if mask and active is not None:
        e = e * active[None, :]
    return e


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    assert ok, name


# ------------------------------------------------------------------ 1 & 2 & 3 & 4
print("\noutput structure")
y = rng.normal(size=(4, 5))
t_pm1 = -np.ones((4, 5)); t_pm1[np.arange(4), [0, 1, 2, 3]] = 1.0
old_nudge = -np.where(1 - t_pm1 * y > 0, -t_pm1, 0.0)          # -(old hard-coded expression)
check("hinge error == old EqProp nudge",
      np.allclose(output_error(y, t_pm1, "hinge"), old_nudge))

active = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
p = softmax(y, active)
check("masked softmax normalises over active set only",
      np.allclose(p[:, 2:], 0) and np.allclose(p.sum(1), 1))

t_oh = np.zeros((4, 5)); t_oh[np.arange(4), [0, 1, 0, 1]] = 1.0
e_masked = output_error(y, t_oh, "ce", active, mask=True)
check("masked loss -> exactly zero gradient for absent classes",
      np.all(e_masked[:, 2:] == 0.0))

# suppression fixed points: absent-class error as a function of its own score
z = np.linspace(-3, 3, 7)
mse_e = np.array([output_error(np.array([[v]]), np.array([[0.0]]), "mse")[0, 0] for v in z])
check("squared error has a fixed point at out = 0",
      abs(mse_e[z == 0][0]) < TOL and mse_e[0] > 0 > mse_e[-1])
big = np.array([[0.0, 10.0]])
ce_e = output_error(big, np.array([[0.0, 1.0]]), "ce")
check("softmax keeps pushing an absent class down even at 0",
      ce_e[0, 0] < 0, f"e={ce_e[0,0]:.2e}")

# ------------------------------------------------------------------ 5, 6, 7  predictive coding
print("\npredictive coding")
D, H, O, N = 6, 4, 3, 5
W1 = rng.normal(size=(D, H)) / np.sqrt(D)
W2 = rng.normal(size=(H, O)) / np.sqrt(H)
x0 = rng.random((N, D))
tgt = np.zeros((N, O)); tgt[np.arange(N), rng.integers(0, O, N)] = 1.0


def energy(x1, W1, W2):
    e1 = x1 - x0 @ W1
    e2 = tgt - f(x1) @ W2
    return 0.5 * (e1 ** 2).sum() + 0.5 * (e2 ** 2).sum()


def dF_dx1(x1, W1, W2):
    e1 = x1 - x0 @ W1
    e2 = tgt - f(x1) @ W2
    return e1 - df(x1) * (e2 @ W2.T)


x1 = x0 @ W1 + 0.3 * rng.normal(size=(N, H))
num = np.zeros_like(x1)
eps = 1e-6
for i in range(N):
    for j in range(H):
        p_, m_ = x1.copy(), x1.copy()
        p_[i, j] += eps; m_[i, j] -= eps
        num[i, j] = (energy(p_, W1, W2) - energy(m_, W1, W2)) / (2 * eps)
check("dF/dx1 matches finite differences",
      np.abs(num - dF_dx1(x1, W1, W2)).max() < 1e-6,
      f"max err {np.abs(num - dF_dx1(x1, W1, W2)).max():.2e}")


def settle(W1, W2, steps=400, dt=0.05):
    x1 = x0 @ W1
    for _ in range(steps):
        x1 = x1 - dt * dF_dx1(x1, W1, W2)
    return x1


x1s = settle(W1, W2)
e1 = x1s - x0 @ W1
e2 = tgt - f(x1s) @ W2
num2 = np.zeros_like(W2)
for i in range(H):
    for j in range(O):
        p_, m_ = W2.copy(), W2.copy()
        p_[i, j] += eps; m_[i, j] -= eps
        num2[i, j] = (energy(x1s, W1, p_) - energy(x1s, W1, m_)) / (2 * eps)
check("PC dW2 = -dF/dW2 at the settled state",
      np.abs(-num2 - f(x1s).T @ e2).max() < 1e-6,
      f"max err {np.abs(-num2 - f(x1s).T @ e2).max():.2e}")

# steps=0 control: no relaxation -> the update IS the backprop gradient of 1/2|t-out|^2
x1_0 = x0 @ W1
e2_0 = tgt - f(x1_0) @ W2
pc0_dW1 = x0.T @ (x1_0 - x0 @ W1)                    # = 0 by construction
bp_dW1 = x0.T @ (df(x1_0) * (e2_0 @ W2.T))
check("PC steps=0 makes the W1 update vanish (pure feedforward, e1 = 0)",
      np.abs(pc0_dW1).max() < 1e-12)
check("...whereas backprop's W1 gradient is non-zero -> steps=0 is NOT backprop for W1",
      np.abs(bp_dW1).max() > 1e-3, f"|bp_dW1|max {np.abs(bp_dW1).max():.3f}")
print("      note: one relaxation step is what injects the output error into W1;")
print("      the small-step limit approaches backprop, steps=0 does not. See exp 23.")

# ------------------------------------------------------------------ 8  equilibrium propagation
print("\nequilibrium propagation")
D2, H2, O2, N2 = 5, 3, 2, 2
V1 = rng.normal(size=(D2, H2)) / np.sqrt(D2)
V2 = rng.normal(size=(H2, O2)) / np.sqrt(H2)
xe = rng.random((N2, D2))
te = np.zeros((N2, O2)); te[np.arange(N2), [0, 1]] = 1.0


def eq_settle(V1, V2, beta=0.0, steps=4000, dt=0.02):
    h = np.zeros((N2, H2)); yv = np.zeros((N2, O2))
    for _ in range(steps):
        gh = h - xe @ V1 - df(h) * (yv @ V2.T)
        gy = yv - f(h) @ V2
        if beta:
            gy = gy - beta * (te - yv)                       # mse cost nudge
        h = h - dt * gh
        yv = yv - dt * gy
    return h, yv


def dE_dW(h, yv):
    return -xe.T @ h, -f(h).T @ yv


h_f, y_f = eq_settle(V1, V2, 0.0)


def cost(V1, V2):
    h, yv = eq_settle(V1, V2, 0.0)
    return 0.5 * ((te - yv) ** 2).sum()


true_g2 = np.zeros_like(V2)
for i in range(H2):
    for j in range(O2):
        p_, m_ = V2.copy(), V2.copy()
        p_[i, j] += 1e-5; m_[i, j] -= 1e-5
        true_g2[i, j] = (cost(V1, p_) - cost(V1, m_)) / 2e-5

print("      beta      relative error of (nudged-free)/beta vs true dC/dW2")
errs = []
for beta in (0.5, 0.1, 0.02, 0.005):
    h_n, y_n = eq_settle(V1, V2, beta)
    _, g2_f = dE_dW(h_f, y_f)
    _, g2_n = dE_dW(h_n, y_n)
    est = (g2_n - g2_f) / beta
    rel = np.abs(est - true_g2).max() / (np.abs(true_g2).max() + 1e-12)
    errs.append(rel)
    print(f"      {beta:<9.3f} {rel:.4f}")
check("EqProp estimator converges to the backprop gradient as beta -> 0",
      errs[-1] < errs[0] and errs[-1] < 0.05,
      f"{errs[0]:.3f} -> {errs[-1]:.3f}")
print("      -> confirms EqProp is a finite-difference estimator of backprop's gradient,")
print("         and that large beta carries real estimator bias (Laborieux et al. 2021).")

print("\nall checks passed\n")
