#!python
import sys
import argparse
from pathlib import Path

import torch
from torch.autograd import Variable
from pyro.shim import parse_torch_version

from ..utils.logger import logger, set_logfile
from ..utils.audio import AudioDataset, PredictDataLoader
from ..utils import params as p
from ..kaldi.latgen import LatGenDecoder

from .model import DenseNetCTCModel


class Predict(object):

    def __init__(self, args):
        self.dataset = AudioDataset(resample=True, sample_rate=p.SAMPLE_RATE,
                                    frame_margin=p.FRAME_MARGIN, unit_frames=p.HEIGHT)
        self.dataloader = PredictDataLoader(dataset=self.dataset, use_cuda=args.use_cuda)
        self.model = DenseNetModel(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, **vars(args))
        self.use_cuda = args.use_cuda

        self._load_phn_table()

    def _load_phn_table(self):
        filepath = Path(__file__).parents[1] / "kaldi/graph/phones.txt"
        self.phns = list()
        with open(filepath, "r") as f:
            for line in f:
                self.phns.append(line.strip().split()[0])

    def predict(self, wav_files, logging=False):
        for wav_file in wav_files:
            xs = self.dataloader.load(wav_file)
            xs = Variable(xs)
            # classify phones
            with torch.no_grad():
                _, phn_idx = self.model.classifier(xs)

            if self.use_cuda:
                phns = list(torch.squeeze(phn_idx).cpu().numpy())
            else:
                phns = list(torch.squeeze(phn_idx).numpy())

            if logging:
                logger.info(f"prediction of {wav_file}: {phns}")
                phn_sbls = [self.phns[i] for i, j in zip(phns[:-1], phns[1:]) if i != j]
                logger.info(f"phone symbols: {phn_sbls}")

    def decode(self, wav_files):
        # prepare kaldi latgen decoder
        decoder = LatGenDecoder()
        # decode per each wav file
        for wav_file in wav_files:
            xs = self.dataloader.load(wav_file)
            xs = Variable(xs)
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
    parser = argparse.ArgumentParser(description="DenseNet prediction")
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
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

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
