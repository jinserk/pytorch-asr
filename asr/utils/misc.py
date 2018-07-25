#!python
from pathlib import Path

import torch
import torch.nn as nn

from . import params as p


def get_model_file_path(log_dir, prefix, desc):
    path = Path(log_dir).resolve()
    return path / f"{prefix}_{desc}.{p.MODEL_SUFFIX}"


def onehot2int(onehot, dim=1):
    _, idx = torch.topk(onehot, dim)
    #idx = idx.squeeze()
    if idx.dim() == 0:
        return int(idx)
    else:
        return idx


def int2onehot(idx, num_classes):
    if not torch.is_tensor(idx):
        onehot = torch.zeros(1, num_classes)
        idx = torch.LongTensor([idx])
        onehot = onehot.scatter_(1, idx.unsqueeze(0), 1.0)
    else:
        onehot = torch.zeros(idx.size(0), num_classes)
        onehot = onehot.scatter_(1, idx.long().unsqueeze(1), 1.0)
    return onehot


class View(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, *args):
        return x.view(*self.dim)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = x.size()
        return x.view(x.size(0), -1)


class MultiOut(nn.ModuleList):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        return (m.forward(*args, **kwargs) for m in self)


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


if __name__ == "__main__":
    i = 5
    o = int2onehot(i, 10)
    print(o)

    i = torch.IntTensor([2, 8])
    o = int2onehot(i, 10)
    print(o)

    o = torch.IntTensor([0, 0, 1, 0, 0])
    i = onehot2int(o)
    print(i)

    o = torch.IntTensor([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
    i = onehot2int(o)
    print(i)

