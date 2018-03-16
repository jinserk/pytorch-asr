import sys
from pathlib import Path

import torch
from torch.autograd import Function

from .._path import KALDI_ROOT
from .._ext import latgen_lib


GRAPH_PATH = Path(__file__).parents[1].joinpath("graph")
if not GRAPH_PATH.exists():
    print("ERROR: no graph path found. please run build.py first")
    sys.exit(1)

DEFAULT_TOKEN = GRAPH_PATH.joinpath("tokens.txt")
DEFAULT_GRAPH = GRAPH_PATH.joinpath("TLG.fst")
DEFAULT_WORDS = GRAPH_PATH.joinpath("words.txt")


class LatGenDecoder(Function):

    def __init__(self, beam=16.0, max_active=8000, min_active=200,
                 acoustic_scale=1.0, allow_partial=True,
                 token_file=str(DEFAULT_TOKEN),
                 fst_file=str(DEFAULT_GRAPH), wd_file=str(DEFAULT_WORDS)):
        # store number of tokens
        lines = list()
        with open(token_file, "r") as f:
            for line in f:
                lines.append(line.strip().split())
        self.num_token = len(lines)
        # initialize
        fst_in_filename = fst_file.encode('ascii')
        wd_in_filename = wd_file.encode('ascii')
        latgen_lib.initialize(beam, max_active, min_active, acoustic_scale,
                              allow_partial, fst_in_filename, wd_in_filename)

    def forward(self, loglikes):
        assert loglikes.dim() == 3 and loglikes.shape[-1] == self.num_token
        with torch.no_grad():
            # N: batch size, RxC: R frames for C classes
            words = torch.IntTensor().zero_()
            alignments = torch.IntTensor().zero_()
            # actual decoding
            latgen_lib.decode(loglikes, words, alignments)
        return words, alignments

    def backward(self, grad_output):
        pass
