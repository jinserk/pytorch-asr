#!python
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import params as p


def get_model_file_path(log_dir, prefix, desc):
    path = Path(log_dir).resolve()
    return path / f"{prefix}_{desc}.{p.MODEL_SUFFIX}"


def get_num_lines(filename):
    #import mmap
    #with open(filename, "r+") as f:
    #    buf = mmap.mmap(f.fileno(), 0)
    #    lines = 0
    #    while buf.readline():
    #        lines += 1
    with open(filename, "r") as f:
        lines = sum(1 for line in f if line.strip())
    return lines


def onehot2int(onehot, dim=1):
    _, idx = torch.topk(onehot, dim)
    #idx = idx.squeeze()
    if idx.dim() == 0:
        return int(idx)
    else:
        return idx


def int2onehot(idx, num_classes, floor=0.):
    value = 1. - floor * (num_classes - 1)
    assert value > floor
    if not torch.is_tensor(idx):
        onehot = torch.full((1, num_classes), floor)
        idx = torch.LongTensor([idx])
        onehot = onehot.scatter_(1, idx.unsqueeze(0), value)
    else:
        onehot = torch.full((idx.size(0), num_classes), floor)
        onehot = onehot.scatter_(1, idx.long().unsqueeze(1), value)
    return onehot


def remove_duplicates(labels, blank=-1):
    p = -1
    for x in labels:
        if x != blank and x != p:
            p = x
            yield x


def edit_distance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


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


class InferenceBatchSoftmax(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if not self.training:
            return self.softmax(x)
        else:
            return x


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

