#!python
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

from ..utils.logger import logger, set_logfile
from ..utils.dataset import NonSplitDataset
from ..utils.dataloader import PredictDataLoader
from ..utils.misc import onehot2int
from ..utils import params as p
from ..kaldi.latgen import LatGenCTCDecoder

from .model import ResNetEdModel


class Predict(object):

    def __init__(self, args):
        self.dataset = NonSplitDataset()
        self.data_loader = PredictDataLoader(dataset=self.dataset)
        self.model = ResNetEdModel(x_dim=p.NUM_PIXELS, y_dim=p.NUM_CTC_LABELS, **vars(args))
        self.use_cuda = args.use_cuda

        self._load_labels()
        self._load_label_counts()

        # prepare kaldi latgen decoder
        self.decoder = LatGenCTCDecoder()

    def _load_labels(self):
        file_path = Path(__file__).parents[1].joinpath("kaldi", "graph", "labels.txt")
        self.labels = list()
        with open(file_path, "r") as f:
            for line in f:
                self.labels.append(line.strip().split()[0])

    def _load_label_counts(self):
        file_path = Path(__file__).parents[2].joinpath("data", "aspire", "ctc_count.txt")
        priors = np.loadtxt(file_path, dtype="double", ndmin=1)
        total = np.sum(priors)
        priors = np.divide(priors, total)
        priors = torch.log(torch.FloatTensor(priors))
        self.priors[np.where(priors < 1e-15)] = 1e30 # to prevent divided zero error
        if self.use_cuda:
            self.priors = self.priors.cuda()
        print(self.priors)

    def decode(self, wav_file, verbose=False):
        # predict phones using AM
        xs, ys, txt = self.data_loader.load(wav_file)
        ys_hat = self.model.predict(xs)
        #eps = torch.zeros(ys_hat.size(0), ys_hat.size(1), 1)
        #ys_hat = torch.cat((eps, ys_hat), 2)
        # devide by priors
        loglikes = torch.log(ys_hat) - self.priors

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


def predict(argv):
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
    predict = Predict(args)
    for i, wav_file in enumerate(args.wav_files):
        logger.info(f"[{i}] decoding wav file: {wav_file}")
        predict.decode(wav_file, verbose=args.verbose)


if __name__ == "__main__":
    pass
