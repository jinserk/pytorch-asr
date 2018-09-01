#!python
import torch
from torch.autograd import Variable
from pyro.shim import parse_torch_version

from utils.logger import logger, set_logfile
from utils.audio import AudioDataset
import utils.params as p

from model import SsVae
from conv import ConvAM

MODEL_SUFFIX = "pth.tar"


class PredictLoader(AudioDataset):

    def __init__(self, use_cuda=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cuda = use_cuda

    def load(self, wav_file):
        # read and transform wav file
        if self.transform is not None:
            tensor = self.transform(wav_file)
        tensor = torch.stack(tensor)
        if self.use_cuda:
            tensor = tensor.cuda()
        return tensor


class Predict(object):

    def __init__(self, args):
        if args.model == "conv":
            self.model = ConvAM(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, **vars(args))
        else:
            self.model = SsVae(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, **vars(args))

        self.data_loader = PredictLoader(use_cuda=args.use_cuda, resample=True, sample_rate=p.SAMPLE_RATE,
                                         frame_margin=p.FRAME_MARGIN, unit_frames=p.HEIGHT)

    def __call__(self, wav_files, verbose=False):
        for wav_file in wav_files:
            self._predict(wav_file, verbose)

    def _predict(self, wav_file, verbose=False):
        # prepare data
        xs = self.data_loader.load(wav_file)
        xs = Variable(xs)
        # classify phones
        with torch.no_grad():
            alpha = self.model.classifier(xs)

        res, phn_idx = torch.topk(alpha, 1)
        if args.use_cuda:
            phns = list(torch.squeeze(phn_idx).cpu().numpy())
        else:
            phns = list(torch.squeeze(phn_idx).numpy())
        if verbose:
            logger.info(f"prediction of {wav_file}: {phns}")
        return phns

    def decode(self, wav_files):
        for wav_file in wav_files:
            phns = _predict(wav_file)


if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    import torch

    parser = argparse.ArgumentParser(description="SS-VAE model prediction")
    subparsers = parser.add_subparsers(dest="model", description="choose AM models")

    ## CNN AM command line options
    conv_parser = subparsers.add_parser('conv', help="prediction with CNN AM")
    conv_parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    conv_parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    conv_parser.add_argument('--continue-from', type=str, help="model file path to make continued from")
    conv_parser.add_argument('wav_files', type=str, nargs='+', help="list of wav_files for prediction")

    ## SS-VAE AM command line options
    ssvae_parser = subparsers.add_parser('ssvae', help="prediction with SS-VAE AM")
    ssvae_parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    ssvae_parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    ssvae_parser.add_argument('--continue-from', type=str, help="model file path to make continued from")
    ssvae_parser.add_argument('wav_files', type=str, nargs='+', help="list of wav_files for prediction")

    args = parser.parse_args()

    if args.model is None:
        parser.print_help()
        sys.exit(1)

    # some assertions to make sure that batching math assumptions are met
    assert parse_torch_version() >= (0, 2, 1), "you need pytorch 0.2.1 or later"

    set_logfile(Path(args.log_dir, "predict.log"))

    logger.info(f"Prediction started with command: {' '.join(sys.argv)}")
    args_str = [f"{k}={v}" for (k, v) in vars(args).items()]
    logger.info(f"args: {' '.join(args_str)}")

    if args.use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # run prediction
    predict = Predict(args)
    predict(args.wav_files, verbose=True)

