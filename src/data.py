"""Data: MNIST at a chosen resolution, class-IL splits, fixed eval sets.

`make_eval_split` returns two DISJOINT held-out sets. Stop on one, report on the other, so early
stopping does not bias the number you publish.

USE `load_mnist` -- it caches. The resize happens once, ever, and is then stored at the working
resolution. See below for why that matters.
"""
from pathlib import Path
import torch, torchvision, torchvision.transforms as T


class TensorMNIST(torch.utils.data.Dataset):
    """MNIST already resized and converted, held as one tensor.

    Interchangeable with the torchvision dataset for everything this project does: indexing
    gives (image, label), and `.targets` is the label vector that class_indices() reads.
    """

    def __init__(self, x, targets):
        self.x, self.targets = x, targets

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def _load_mnist_raw(size=14, root="./data", fashion=False):
    """The slow path: torchvision, with the transform re-run on every single access."""
    # Skip the resize when the source is already the requested size: Resize((28,28)) on a
    # 28x28 PIL image still runs the interpolation filter and can perturb pixel values.
    steps = ([] if size == 28 else [T.Resize((size, size), antialias=True)]) + [T.ToTensor()]
    tf = T.Compose(steps)
    ds = torchvision.datasets.FashionMNIST if fashion else torchvision.datasets.MNIST
    return (ds(root, train=True, download=True, transform=tf),
            ds(root, train=False, download=True, transform=tf))


def load_mnist(size=14, root="./data", fashion=False):
    """MNIST (or Fashion-MNIST) at size x size, scaled to [0,1]. Returns (train, test).

    THE RESIZE HAPPENS ONCE, EVER. torchvision applies its transform inside __getitem__, so
    every batch re-runs a PIL resize on 32 images: measured at 8.1 ms per batch, which across a
    width sweep is tens of minutes of doing the same arithmetic again and again. Here the whole
    set is converted once and written to <root>/preprocessed/<name>_<size>.pt as a single
    tensor; later runs load that file and a batch becomes an index.

    Pixel values are identical either way -- the transform is deterministic, so this changes
    only the cost. Delete the .pt to force a rebuild.
    """
    name = f"{'fashion' if fashion else 'mnist'}_{size}.pt"
    path = Path(root) / "preprocessed" / name
    if path.exists():
        blob = torch.load(path, weights_only=True)
    else:
        tr, te = _load_mnist_raw(size=size, root=root, fashion=fashion)
        blob = {
            "train_x": torch.stack([tr[i][0] for i in range(len(tr))]),
            "train_y": tr.targets.clone(),
            "test_x": torch.stack([te[i][0] for i in range(len(te))]),
            "test_y": te.targets.clone(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)
    return (TensorMNIST(blob["train_x"], blob["train_y"]),
            TensorMNIST(blob["test_x"], blob["test_y"]))


def class_indices(dataset):
    """Dict: class -> tensor of that class's sample indices."""
    return {c: (dataset.targets == c).nonzero(as_tuple=True)[0] for c in range(10)}


def balanced_indices(class_idx, classes, per_class, seed=0):
    """Equal number of training examples per class, sampled with a dedicated RNG.

    MNIST class counts differ (~5400-6700), which is a prior-probability shift layered on
    top of the class-IL shift. Equalising removes it."""
    g = torch.Generator().manual_seed(int(seed))
    out = {}
    for c in classes:
        pool = class_idx[c]
        out[c] = pool[torch.randperm(len(pool), generator=g)[:per_class]]
    return out


def make_eval_set(test, classes=None, per_class=100, device="cpu", offset=0, seed=None):
    """Fixed held-out set: `per_class` TEST images per class, starting at `offset`.
       `seed` shuffles the pool first, so different seeds give different (still disjoint) sets."""
    classes = list(range(10)) if classes is None else list(classes)
    idx = class_indices(test)
    xs, ys = [], []
    for c in classes:
        pool = idx[c]
        if seed is not None:
            pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(int(seed)))]
        sel = pool[offset:offset + per_class]
        xs.append(torch.stack([test[i][0] for i in sel.tolist()]))
        ys.append(torch.full((len(sel),), c))
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def make_eval_split(test, classes=None, per_class=100, device="cpu", seed=0):
    """Two disjoint eval sets: (stop_x, stop_y), (report_x, report_y)."""
    stop = make_eval_set(test, classes, per_class, device, offset=0, seed=seed)
    report = make_eval_set(test, classes, per_class, device, offset=per_class, seed=seed)
    return stop, report


def label_remapper(task, device="cpu", n_classes=10):
    """Lookup tensor mapping an original class label to its within-task output unit.

    `task[i]` -> unit `i`. THE ORDER OF `task` IS HONOURED AND MUST NOT BE SORTED HERE.

    Song & Bogacz shuffle all ten labels first (shuffle_mapper) and then map with `i % 5`, so
    which two classes share an output unit is random. Sorting instead pairs the i-th SMALLEST
    class of task 1 with the i-th smallest of task 2. Fashion-MNIST's class order is
    semantically structured (0-4 garments, 5-9 footwear and bags), so rank-pairing makes each
    unit's two classes more visually similar than chance -- and when they are similar,
    training task 2 partly REINFORCES task 1 rather than overwriting it, which suppresses the
    very interference the experiment exists to measure.
    """
    remap = torch.full((n_classes,), -1, dtype=torch.long, device=device)
    for i, c in enumerate(task):
        remap[int(c)] = i
    return remap


def make_domain_il_eval(test, tasks, per_class=1000, device="cpu"):
    """One (x, y) eval set per task, labels remapped to within-task indices.

    per_class=1000 is the whole Fashion-MNIST / MNIST test set for that class.
    """
    out = []
    for task in tasks:
        # images are selected by class (order irrelevant); the LABEL mapping must use the
        # unsorted task order, so the remapper is built from `task` and not from sorted(task)
        x, y = make_eval_set(test, classes=sorted(task), per_class=per_class, device=device)
        out.append((x, label_remapper(task, device)[y]))
    return out


def partial_task_indices(class_idx, classes, partial_num=None, seed=0):
    """Indices for `classes`, cut to `partial_num` examples PER CLASS.

    Their dataset_utils.partial_dateset docstring is explicit:
        "partial_num (int): The number of datapoints to extract from each class."
    So partial_num=600 with five classes per task is 3000 examples per task, not 600.
    v1 of this function read it as a total, which gave a THIRD of the training per
    iteration and left exp 30 badly undertrained.

    It also matters for a second reason. 3000/500 is six whole batches, so every update
    uses the full 500 examples. 600/500 leaves a 100-example remainder, and under the
    summed loss that final update carries a fifth of the gradient of the others -- a
    systematic large-then-small alternation within every iteration.
    """
    idx = torch.cat([
        torch.as_tensor(class_idx[c])[:partial_num] if partial_num is not None
        else torch.as_tensor(class_idx[c])
        for c in sorted(classes)
    ])
    g = torch.Generator().manual_seed(int(seed))
    return idx[torch.randperm(len(idx), generator=g)]
