#!python
import sys
from pathlib import Path

import numpy as np
import torch

from ..utils.dataset import NonSplitDataset
from ..utils.dataloader import PredictDataLoader
from ..utils.logger import logger, set_logfile
from ..utils.misc import onehot2int
from ..utils import params as p
from ..kaldi.latgen import LatGenCTCDecoder

from .network import *


class Predictor:

    def __init__(self, use_cuda=False, continue_from=None, *args, **kwargs):
        self.use_cuda = use_cuda

        self.dataset = NonSplitDataset()
        self.data_loader = PredictDataLoader(dataset=self.dataset)

        # load from args
        assert continue_from is not None
        self.load(continue_from)

        # prepare kaldi latgen decoder
        self._load_labels()
        self.decoder = LatGenCTCDecoder()

    def _load_labels(self):
        file_path = Path(__file__).parents[1].joinpath("kaldi", "graph", "labels.txt")
        self.labels = list()
        with open(file_path, "r") as f:
            for line in f:
                self.labels.append(line.strip().split()[0])

    def __setup_networks(self):
        # setup networks
        self.encoder = resnet101(num_classes=p.NUM_CTC_LABELS)
        if self.use_cuda:
            self.encoder.cuda()

    def predict(self, xs):
        self.encoder.eval()
        if self.use_cuda:
            xs = xs.cuda()
        ys_hat = self.encoder(xs, softmax=True)
        return ys_hat

    def decode(self, wav_file, verbose=False):
        # predict phones using AM
        xs, ys, txt = self.data_loader.load(wav_file)
        ys_hat = self.predict(xs)
        # no need to normalize posteriors with state priors when we use CTC
        # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43908.pdf
        loglikes = torch.log(ys_hat)

        if verbose:
            labels = onehot2int(loglikes).squeeze()
            logger.info(f"labels: {' '.join([str(x) for x in labels.tolist()])}")
            def remove_duplicates(labels):
                p = -1
                for x in labels:
                    if x != p:
                        p = x
                        yield x
            symbols = [self.labels[x] for x in remove_duplicates(labels) if self.labels[x] != "<blk>"]
            logger.info(f"symbols: {' '.join(symbols)}")

        # decode using Kaldi's latgen decoder
        if self.use_cuda:
            loglikes = loglikes.cpu()
        words, alignment = self.decoder(loglikes)
        # convert into text
        words = words.squeeze()
        if words.dim():
            text = ' '.join([self.decoder.words[i] for i in words])
            logger.info(f"decoded text: {text}")
        else:
            logger.info("decoded text: <null output from decoder>")

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.info(f"loading the model from {file_path}")
        if not self.use_cuda:
            states = torch.load(file_path, map_location='cpu')
        else:
            states = torch.load(file_path)

        self.__setup_networks()
        self.encoder.load_state_dict(states["model"])
        if self.use_cuda:
            self.encoder.cuda()


def predict(argv):
    import argparse
    parser = argparse.ArgumentParser(description="ResNet prediction")
    parser.add_argument('--verbose', default=False, action='store_true', help="set true if you need to check AM output")
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    parser.add_argument('--continue-from', type=str, help="model file path to make continued from")
    parser.add_argument('wav_files', type=str, nargs='+', help="list of wav_files for prediction")
    args = parser.parse_args(argv)

    print(f"begins logging to file: {str(Path(args.log_dir).resolve() / 'predict.log')}")
    set_logfile(Path(args.log_dir, "predict.log"))

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Prediction started with command: {' '.join(sys.argv)}")
    args_str = [f"{k}={v}" for (k, v) in vars(args).items()]
    logger.info(f"args: {' '.join(args_str)}")

    if args.use_cuda:
        logger.info("using cuda")

    if args.continue_from is None:
        logger.error("model name is missing: add '--continue-from <model-name>' in options")
        #parser.print_help()
        sys.exit(1)

    # run prediction
    predict = Predictor(**vars(args))
    for i, wav_file in enumerate(args.wav_files):
        wav_file = str(Path(wav_file).resolve())
        logger.info(f"[{i}] decoding wav file: {wav_file}")
        predict.decode(wav_file, verbose=args.verbose)


if __name__ == "__main__":
    pass
