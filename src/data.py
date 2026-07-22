"""Data: load MNIST at a chosen resolution, build class-IL splits, make a fixed eval set."""
import torch, torchvision, torchvision.transforms as T
from torch.utils.data import Subset


def load_mnist(size=14, root="./data"):
    """MNIST resized to size×size, scaled to [0,1]. Returns (train, test)."""
    tf = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor()])
    train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)
    return train, test


def class_indices(dataset):
    """Dict: class -> tensor of that class's sample indices."""
    return {c: (dataset.targets == c).nonzero(as_tuple=True)[0] for c in range(10)}


def make_eval_set(test, per_class=100, device="cpu"):
    """Fixed held-out eval set: `per_class` TEST images per class, stacked once."""
    idx = class_indices(test)
    xs, ys = [], []
    for c in range(10):
        sel = idx[c][:per_class]
        xs.append(torch.stack([test[i][0] for i in sel]))
        ys.append(torch.full((len(sel),), c))
    return torch.cat(xs).to(device), torch.cat(ys).to(device)
