"""does_eqprop_forget_classil10x1
Q: Does EqProp forget under Class-IL (10x1)?
   Controls: backprop (should forget), replay (should fix it). Under test: EqProp.
   Runs this one comparison and saves one figure named after this script.
"""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data import load_mnist, class_indices, make_eval_set
from src.eqprop import eqprop_init, eqprop_update, eqprop_generate


# ============================ constants ============================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR        = ROOT / "data"                    # <project>/data
FIG             = Path(__file__).resolve().with_suffix(".png")   # figure named after this script
IMG_SIZE        = 14
SEED            = 0
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]   # 5x2 - learning 2 classes per task
ITERS           = 100                              # weight updates per task
BATCH           = 64
EVAL_EVERY      = 1                               # evaluate every N updates
EVAL_PER_CLASS  = 100                              # held-out TEST images per class for evaluation

BP_LR                          = 0.1               # backprop control
RP_LR, RP_PER_CLASS            = 0.1, 20           # replay control
EQP_LR, EQP_BETA, EQP_DT       = 0.03, 0.3, 0.3    # eqprop (under test)
EQP_MAX_STEPS, EQP_SETTLE_PAT  = 500, 30
GATE_FRAC                      = 0.3               # eqprop gating control: 
# ==================================================================

IN_DIM = IMG_SIZE * IMG_SIZE
train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)
eval_x, eval_y = make_eval_set(test, per_class=EVAL_PER_CLASS, device=DEVICE)

W1, W2 = eqprop_init(device=DEVICE)
opt = torch.optim.SGD([W1, W2], lr=EQP_LR)
for i, (x, y) in enumerate(DataLoader(train, batch_size=64, shuffle=True)):
    if i >= 300: break
    eqprop_update(x.to(DEVICE), y.to(DEVICE), W1, W2, opt, beta=EQP_BETA, dt=EQP_DT, max_steps=EQP_MAX_STEPS, settle_patience=EQP_SETTLE_PAT, device=DEVICE)
fig, ax = plt.subplots(10, 8, figsize=(9, 11))
for c in range(10):
    s = eqprop_generate(W1, W2, c, 8, device=DEVICE)
    for j in range(8): ax[c, j].imshow(s[j].reshape(14, 14).cpu(), cmap="gray"); ax[c, j].axis("off")
plt.title("Is Eqprop a reasonable generator?", fontsize=16)
plt.tight_layout()
plt.savefig(FIG, dpi=120)
plt.show()