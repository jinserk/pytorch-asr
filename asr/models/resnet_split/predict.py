#!python
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from asr.utils.dataset import SplitPredictDataset
from asr.utils.dataloader import SplitPredictDataLoader
from asr.utils.logger import logger, set_logfile, version_log
from asr.utils import params as p
from asr.kaldi.latgen import LatGenCTCDecoder

from ..predictor import Predictor
from .network import resnet50, resnet101


class SplitPredictor(Predictor):

    def decode(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            for i, (data) in enumerate(data_loader):
                # predict phones using AM
                xs, frame_lens, filenames = data
                if self.use_cuda:
                    xs = xs.cuda()
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


def predict(argv):
    parser = argparse.ArgumentParser(description="ResNet prediction")
    parser.add_argument('--verbose', default=False, action='store_true', help="set true if you need to check AM output")
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--batch-size', default=8, type=int, help="number of simultaneous decoding")
    parser.add_argument('--log-dir', default='./logs_resnet_ctc', type=str, help="filename for logging the outputs")
    parser.add_argument('--continue-from', type=str, help="model file path to make continued from")
    parser.add_argument('wav_files', type=str, nargs='+', help="list of wav_files for prediction")

    args = parser.parse_args(argv)

    set_logfile(Path(args.log_dir, "predict.log"))
    version_log(args)

    if args.continue_from is None:
        logger.error("model name is missing: add '--continue-from <model-name>' in options")
        #parser.print_help()
        sys.exit(1)

    model = resnet50(num_classes=p.NUM_CTC_LABELS)
    predictor = SplitPredictor(model, **vars(args))

    dataset = SplitPredictDataset(args.wav_files)
    dataloader = SplitPredictDataLoader(dataset=dataset, batch_size=args.batch_size,
                                        pin_memory=args.use_cuda)

    # run prediction
    predictor.decode(dataloader)


if __name__ == "__main__":
    pass
