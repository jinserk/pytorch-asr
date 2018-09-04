#!python
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from asr.utils.logger import logger
from asr.utils.misc import onehot2int, remove_duplicates
from asr.utils import params as p

from asr.kaldi.latgen import LatGenCTCDecoder


class NonSplitPredictor:

    def __init__(self, model, use_cuda=False, continue_from=None, verbose=False,
                 *args, **kwargs):
        assert continue_from is not None
        self.use_cuda = use_cuda
        self.verbose = verbose

        # load from args
        self.model = model
        if self.use_cuda:
            logger.info("using cuda")
            self.model.cuda()

        self.load(continue_from)

        # prepare kaldi latgen decoder
        self.decoder = LatGenCTCDecoder()

    def decode(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            for xs, frame_lens, filenames in data_loader:
                # predict phones using AM
                if self.use_cuda:
                    xs = xs.cuda(non_blocking=True)
                ys_hat = self.model(xs)
                #frame_lens = torch.ceil(frame_lens.float() / FRAME_REDUCE_FACTOR).int()
                # decode using Kaldi's latgen decoder
                # no need to normalize posteriors with state priors when we use CTC
                # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43908.pdf
                loglikes = torch.log(ys_hat)
                if self.use_cuda:
                    loglikes = loglikes.cpu()
                words, alignment, w_sizes, a_sizes = self.decoder(loglikes, frame_lens)
                # print results
                loglikes = [l[:s] for l, s in zip(loglikes, frame_lens)]
                words = [w[:s] for w, s in zip(words, w_sizes)]
                for results in zip(filenames, loglikes, words):
                    self.print_result(*results)

    def print_result(self, filename, loglikes, words):
        logger.info(f"decoding wav file: {str(Path(filename).resolve())}")
        if self.verbose:
            labels = onehot2int(loglikes).squeeze()
            logger.info(f"labels: {' '.join([str(x) for x in labels.tolist()])}")
            symbols = [self.decoder.labeler.idx2phone(x) for x in remove_duplicates(labels, blank=0)]
            logger.info(f"symbols: {' '.join(symbols)}")
        words = words.squeeze()
        text = ' '.join([self.decoder.labeler.idx2word(i) for i in words]) \
               if words.dim() else '<null output from decoder>'
        logger.info(f"decoded text: {text}")

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.info(f"loading the model from {file_path}")
        to_device = f"cuda:{torch.cuda.current_device()}" if self.use_cuda else "cpu"
        states = torch.load(file_path, map_location=to_device)
        self.model.load_state_dict(states["model"])


class SplitPredictor(NonSplitPredictor):

    def decode(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            for i, (data) in enumerate(data_loader):
                # predict phones using AM
                xs, frame_lens, filenames = data
                if self.use_cuda:
                    xs = xs.cuda(non_blocking=True)
                ys_hat = self.model(xs)
                ys_hat = ys_hat.unsqueeze(dim=0).transpose(1, 2)
                pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(frame_lens, dim=0)))
                ys_hats = [ys_hat.narrow(2, p, l).clone() for p, l in zip(pos[:-1], frame_lens)]
                max_len = torch.max(frame_lens)
                ys_hats = [nn.ConstantPad1d((0, max_len-yh.size(2)), 0)(yh) for yh in ys_hats]
                ys_hat = torch.cat(ys_hats).transpose(1, 2)
                # latgen decoding
                loglikes = torch.log(ys_hat)
                if self.use_cuda:
                    loglikes = loglikes.cpu()
                words, alignment, w_sizes, a_sizes = self.decoder(loglikes, frame_lens)
                # print results
                loglikes = [l[:s] for l, s in zip(loglikes, frame_lens)]
                words = [w[:s] for w, s in zip(words, w_sizes)]
                for results in zip(filenames, loglikes, words):
                    self.print_result(*results)


if __name__ == "__main__":
    pass
