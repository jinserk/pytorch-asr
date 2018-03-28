#!python
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

from . import params as p


def get_model_file_path(log_dir, prefix, desc):
    path = Path(log_dir).resolve()
    return path / f"{prefix}_{desc}.{p.MODEL_SUFFIX}"


class View(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, *args):
        return x.view(*self.dim)


class MultiOut(nn.ModuleList):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        return (m.forward(*args, **kwargs) for m in self)


class Swish(nn.Module):

    def forward(self, x):
        return x * F.sigmoid(x)


