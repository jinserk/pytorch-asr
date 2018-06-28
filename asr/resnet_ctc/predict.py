#!python
import sys
import argparse
from pathlib import Path

import torch
from pyro.shim import parse_torch_version

from ..utils.logger import logger, set_logfile
from ..utils.audio import AudioCTCDataset, PredictDataLoader
from ..utils.misc import onehot2int
from ..utils import params as p
from ..kaldi.latgen import LatGenDecoder

from .model import ResNetCTCModel


class Predict(object):

    def __init__(self, args):
        self.dataset = AudioCTCDataset()
        self.data_loader = PredictDataLoader(dataset=self.dataset, use_cuda=args.use_cuda)
        self.model = ResNetCTCModel(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, **vars(args))
        self.use_cuda = args.use_cuda

        self._load_ctc_token_table()

    def _load_ctc_token_table(self):
        filepath = Path(__file__).parents[1] / "kaldi/graph/tokens.txt"
        self.tokens = list()
        with open(filepath, "r") as f:
            for line in f:
                self.tokens.append(line.strip().split()[0])

    def predict(self, wav_files, logging=False):
        for i, wav_file in enumerate(wav_files):
            xs, ys, txt = self.data_loader.load(wav_file)
            ys_hat = self.model.predict(xs)
            loglikes = -torch.log(ys_hat)
            labels = onehot2int(ys_hat).squeeze()
            if self.use_cuda:
                labels = labels.cpu()
            tokens = list(labels.numpy())

            if logging:
                logger.info(f"prediction of {wav_files[i]}: {tokens}")
                token_sbls = [self.tokens[x+1] for x, y in zip(tokens[:-1], tokens[1:]) if x != y and x != 0]
                logger.info(f"token symbols: {token_sbls}")

    def decode(self, wav_files):
        # prepare kaldi latgen decoder
        decoder = LatGenDecoder()
        # decode per each wav file
        for wav_file in wav_files:
            xs = self.dataloader.load(wav_file)
            with torch.no_grad():
                alpha = self.model.encoder(xs, softmax=True)
            # phones -> tokens
            #tmp = torch.ones(alpha.shape[0], 1) * p.EPS
            #alpha = torch.cat((alpha[:, 0].unsqueeze(1), tmp, alpha[:, 1:]), dim=1)
            loglikes = torch.log(alpha).unsqueeze(0)
            print(loglikes)
            # decoding
            words, alignment = decoder(loglikes)
            # convert into text
            text = ' '.join([decoder.words[i] for i in words.squeeze()])
            print(text)


def predict(argv):
    parser = argparse.ArgumentParser(description="ResNet prediction")
    parser.add_argument('--decode', default=False, action='store_true', help="retrieve Kaldi's latgen decoder")
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
    if args.decode:
        predict.decode(args.wav_files)
    else:
        predict.predict(args.wav_files, logging=True)


if __name__ == "__main__":
    pass
