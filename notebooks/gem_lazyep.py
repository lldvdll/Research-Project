import torch
import torchvision
import torchvision.transforms as transforms

# =============================================================================
# 1. Hyperparameters and Setup
# =============================================================================
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Network dimensions
INPUT_DIM = 28 * 28  # MNIST images are 28x28 pixels
HIDDEN_DIM = 500     # Hidden neurons
OUTPUT_DIM = 10      # 10 digits (0-9)

# Physics & Training parameters
BATCH_SIZE = 128
BETA = 0.5           # Nudging force multiplier
DT = 0.1             # Time step for physics integration
STEPS = 150           # Number of simulation steps to find equilibrium
LR = 0.01           # Learning rate

# Load MNIST Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Weights (Using slightly smaller init for stability in large networks)


W1 = (torch.randn(INPUT_DIM, HIDDEN_DIM, device=device) * 0.01).requires_grad_(True)
W2 = (torch.randn(HIDDEN_DIM, OUTPUT_DIM, device=device) * 0.01).requires_grad_(True)


optimizer = torch.optim.SGD([W1, W2], lr=LR)

# =============================================================================
# 2. The Energy Function (Vectorized for Batches)
# =============================================================================
def compute_energy(x, h, y_pred, w1, w2):
    """
    x: [Batch, 784], h: [Batch, 500], y_pred: [Batch, 10]
    Returns a single scalar energy value for the entire batch.
    """
    # 1. Penalty for firing (keeps activations from exploding to infinity)
    state_penalty = 0.5 * (h**2).sum() + 0.5 * (y_pred**2).sum()

    # 2. Reward for aligning with synapses (forward pass emulation)
    alignment_h = (h * (x @ w1)).sum()
    alignment_y = (y_pred * (torch.tanh(h) @ w2)).sum()

    return state_penalty - alignment_h - alignment_y

# =============================================================================
# 3. The Physical Dynamics Simulator
# =============================================================================
def settle_to_equilibrium(x, w1, w2, target=None, beta=0.0, init_h=None, init_y=None):
    batch_size = x.size(0)

    # Initialize neurons
    if init_h is None:
        h = torch.zeros(batch_size, HIDDEN_DIM, device=device, requires_grad=True)
    else:
        h = init_h.clone().detach().requires_grad_(True)

    if init_y is None:
        y_pred = torch.zeros(batch_size, OUTPUT_DIM, device=device, requires_grad=True)
    else:
        y_pred = init_y.clone().detach().requires_grad_(True)

    # Calculate mask for Lazy Learning
    # If a target is provided, we check which specific images in the batch need a nudge
    if target is not None:
        with torch.no_grad():
            # Calculate Hinge loss per image: max(0, 1 - y_true * y_pred)
            loss_per_item = torch.clamp(1.0 - target * y_pred, min=0.0).sum(dim=1)
            # Create a [Batch, 1] mask: 1.0 if loss > 0 (needs nudge), 0.0 if margin is clear
            active_mask = (loss_per_item > 0).float().unsqueeze(1)

            # Track how many images we are physically skipping this batch!
            skipped_this_batch = (active_mask == 0).sum().item()
    else:
        active_mask = 1.0
        skipped_this_batch = 0

    # Simulate the physics
    for _ in range(STEPS):
        E = compute_energy(x, h, y_pred, w1, w2)

        # Apply external Hinge Loss force only to the masked (incorrect/unsure) items
        if target is not None:
            hinge_matrix = torch.clamp(1.0 - target * y_pred, min=0.0)
            # The mask ensures beta is 0 for confident predictions
            E = E + (beta * active_mask * hinge_matrix).sum()

        grad_h, grad_y = torch.autograd.grad(E, [h, y_pred])

        # Step down the gradient
        h.data -= DT * grad_h
        y_pred.data -= DT * grad_y

    return h, y_pred, skipped_this_batch

# =============================================================================
# 4. Training Loop
# =============================================================================
print("Starting EqProp + Multi-Class Hinge Loss Training on MNIST...\n")

epochs = 50

for epoch in range(epochs):
    epoch_skipped = 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Convert labels (0-9) to Multi-Class Hinge Targets (+1 for correct, -1 for others)
        targets = torch.full((batch_size, OUTPUT_DIM), -1.0, device=device)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)

        # --- PHASE 1: Free Phase ---
        h_free, y_free, _ = settle_to_equilibrium(images, W1, W2)

        # Calculate accuracy for logging
        predictions = y_free.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

        # --- PHASE 2: Nudged Phase ---
        # The active_mask inside this function handles the lazy learning!
        h_nudged, y_nudged, skipped = settle_to_equilibrium(
            images, W1, W2, target=targets, beta=BETA, init_h=h_free, init_y=y_free
        )
        epoch_skipped += skipped
    # --- PHASE 3: Contrastive Weight Update ---
        if skipped == batch_size:
            continue

        optimizer.zero_grad()

        E_free = compute_energy(images, h_free, y_free, W1, W2)
        grad_W1_free, grad_W2_free = torch.autograd.grad(E_free, [W1, W2])

        E_nudged = compute_energy(images, h_nudged, y_nudged, W1, W2)
        grad_W1_nudged, grad_W2_nudged = torch.autograd.grad(E_nudged, [W1, W2])

        # Update weights (averaged over the batch size for stability)
        W1.grad = (grad_W1_nudged - grad_W1_free) / (BETA * batch_size)
        W2.grad = (grad_W2_nudged - grad_W2_free) / (BETA * batch_size)

        # NaN Catcher (Loudly crash if things explode)
        if torch.isnan(W1.grad).any() or torch.isnan(y_free).any():
            print(f"\n[CRASH] NaNs detected at Epoch {epoch+1}, Batch {batch_idx}!")
            import sys; sys.exit(1)

        #Gradient Clipping (Forces the update step to remain physically stable)
        torch.nn.utils.clip_grad_norm_([W1, W2], max_norm=0.1)

# --- DIAGNOSTICS BLOCK ---
        if batch_idx == 0:
            print("\n--- NETWORK HEALTH DIAGNOSTICS ---")

            # 1. Check for Tanh Saturation
            tanh_h = torch.tanh(h_free)
            saturated_percent = (tanh_h.abs() > 0.95).float().mean().item() * 100
            print(f"Hidden Node Saturation: {saturated_percent:.1f}% (Ideal: < 30%)")

            # 2. Check Gradient Flow
            w1_grad_norm = W1.grad.norm().item()
            w2_grad_norm = W2.grad.norm().item()
            print(f"W1 Grad Norm: {w1_grad_norm:.6f} | W2 Grad Norm: {w2_grad_norm:.6f}")

            # 3. Check Physics Equilibrium (Recalculate Energy for a fresh graph!)
            E_diag = compute_energy(images, h_free, y_free, W1, W2)
            force_h, force_y = torch.autograd.grad(E_diag, [h_free, y_free])
            print(f"Residual Force on H: {force_h.norm().item():.4f} (Ideal: close to 0)")
            print(f"Residual Force on Y: {force_y.norm().item():.4f} (Ideal: close to 0)")
            print("----------------------------------\n")

        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Batch Acc: {(predictions == labels).float().mean().item()*100:.1f}% | "
                  f"Images Skipped: {skipped}/{batch_size}")

    epoch_acc = 100.0 * correct_predictions / total_samples
    print(f"\n=> End of Epoch {epoch+1} | Total Acc: {epoch_acc:.2f}% | Total Skips: {epoch_skipped}/{total_samples}\n")
