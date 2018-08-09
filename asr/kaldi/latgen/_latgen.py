import sys
from pathlib import Path

import torch
from torch.autograd import Function

from asr.utils.misc import get_num_lines

from .._path import KALDI_ROOT
from .._ext import latgen_lib


GRAPH_PATH = Path(__file__).parents[1].joinpath("graph")
if not GRAPH_PATH.exists():
    print("ERROR: no graph path found. please run build.py first")
    sys.exit(1)

DEFAULT_GRAPH = GRAPH_PATH.joinpath("CLG.fst")
DEFAULT_LABEL = GRAPH_PATH.joinpath("phones.txt")
DEFAULT_WORDS = GRAPH_PATH.joinpath("words.txt")
DEFAULT_LEXICON = GRAPH_PATH.joinpath("align_lexicon.int")


class Labeler:
    """ provides phone to pid, pid to phone, word to wid, wid to word """

    def __init__(self, label_file=str(DEFAULT_LABEL), word_file=str(DEFAULT_WORDS),
                 lex_file=str(DEFAULT_LEXICON)):
        self.label_file = label_file
        self.word_file = word_file
        self.lex_file = lex_file

        self.__load_label_file()
        self.__load_word_file()
        self.__load_lex_file()

    def __load_label_file(self):
        self.p2i, self.i2p = dict(), dict()
        with open(self.label_file, "r") as f:
            for line in f:
                p, i = line.strip().split()
                p, i = p.strip(), int(i.strip())
                self.p2i[p] = i
                self.i2p[i] = p

    def __load_word_file(self):
        self.w2i, self.i2w = dict(), dict()
        with open(self.word_file, "r") as f:
            for line in f:
                w, i = line.strip().split()
                w, i = w.strip(), int(i.strip())
                self.w2i[w] = i
                self.i2w[i] = w

    def __load_lex_file(self):
        self.wi2l = dict()
        with open(self.lex_file, "r") as f:
            for line in f:
                token = line.strip().split()
                wi, l = int(token[0].strip()), [int(i.strip()) for i in token[2:]]
                if wi in self.wi2l:
                    self.wi2l[wi].append(l)
                else:
                    self.wi2l[wi] = [l]

    def get_num_labels(self):
        return len(self.p2i)

    def get_num_words(self):
        return len(self.w2i)

    def phone2idx(self, phone):
        return self.p2i[phone]

    def idx2phone(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.i2p[idx]

    def idx2word(self, idx, unk='<unk>'):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.i2w[idx] if idx in self.i2w else unk

    def word2idx(self, word, unk='<unk>'):
        return self.w2i[word] if word in self.w2i else self.w2i[unk]

    def word2lex(self, word):
        """ return list of lexicons for a single word to support multiple definitions """
        return self.wi2l[self.word2idx(word)]


class LatGenDecoder(Function):
    """ decoder using position-dependent phones as the acoustic model labels """

    def __init__(self, beam=16.0, max_active=8000, min_active=200, acoustic_scale=1.0, allow_partial=True,
                 fst_file=str(DEFAULT_GRAPH), label_file=str(DEFAULT_LABEL),
                 lexicon_file=str(DEFAULT_LEXICON), wd_file=str(DEFAULT_WORDS)):
        # labels info
        self.labeler = Labeler(label_file, wd_file, lexicon_file)

        # initialize decoder
        fst_in_filename = fst_file.encode('ascii')
        wd_in_filename = wd_file.encode('ascii')
        latgen_lib.initialize(beam, max_active, min_active, acoustic_scale,
                              allow_partial, fst_in_filename, wd_in_filename)

    def forward(self, loglikes, frame_lens):
        # loglikes should NxTxH (N: batch size, T: frames, H: classes)
        assert loglikes.dim() == 3 and loglikes.size(2) == self.labeler.get_num_labels()
        assert frame_lens.dim() == 1 and loglikes.size(0) == frame_lens.size(0)

        with torch.no_grad():
            words = torch.IntTensor().zero_()
            alignments = torch.IntTensor().zero_()
            w_sizes = torch.IntTensor().zero_()
            a_sizes = torch.IntTensor().zero_()
            # actual decoding
            latgen_lib.decode(loglikes, frame_lens, words, alignments, w_sizes, a_sizes)

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

