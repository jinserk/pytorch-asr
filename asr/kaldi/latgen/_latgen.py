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

DEFAULT_GRAPH = GRAPH_PATH.joinpath("CLG.fst")
DEFAULT_LABEL = GRAPH_PATH.joinpath("phones.txt")
DEFAULT_WORDS = GRAPH_PATH.joinpath("words.txt")


class LatGenDecoder(Function):
    """ decoder using position-dependent phones as the acoustic model labels """

    def __init__(self, beam=16.0, max_active=8000, min_active=200,
                 acoustic_scale=1.0, allow_partial=True,
                 label_file=str(DEFAULT_LABEL),
                 fst_file=str(DEFAULT_GRAPH), wd_file=str(DEFAULT_WORDS)):
        # store number of labels
        lines = list()
        with open(label_file, "r") as f:
            for line in f:
                lines.append(line.strip().split())
        self.num_labels = len(lines)
        # initialize
        fst_in_filename = fst_file.encode('ascii')
        wd_in_filename = wd_file.encode('ascii')
        latgen_lib.initialize(beam, max_active, min_active, acoustic_scale,
                              allow_partial, fst_in_filename, wd_in_filename)
        # load words table
        self.words = list()
        self.wordi = dict()
        with open(wd_file, "r") as f:
            for line in f:
                record = line.strip().split()
                self.words.append(record[0])
                self.wordi[record[0]] = int(record[1])

    def forward(self, loglikes):
        assert loglikes.dim() == 3 and loglikes.size(2) == self.num_labels
        with torch.no_grad():
            # N: batch size, RxC: R frames for C classes
            words = torch.IntTensor().zero_()
            alignments = torch.IntTensor().zero_()
            w_sizes = torch.IntTensor().zero_()
            a_sizes = torch.IntTensor().zero_()
            # actual decoding
            latgen_lib.decode(loglikes, words, alignments, w_sizes, a_sizes)
        return words, alignments, w_sizes, a_sizes

    def backward(self, grad_output):
        pass


DEFAULT_CTC_GRAPH = GRAPH_PATH.joinpath("TLG.fst")
DEFAULT_CTC_LABEL = GRAPH_PATH.joinpath("labels.txt")

class LatGenCTCDecoder(LatGenDecoder):
    """ decoder using CTC labels with blank label in the acoustic model """

    def __init__(self, label_file=str(DEFAULT_CTC_LABEL), fst_file=str(DEFAULT_CTC_GRAPH),
                 *args, **kwargs):
        super().__init__(label_file=label_file, fst_file=fst_file, *args, **kwargs)

