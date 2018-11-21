#!python
import sys
import argparse
from pathlib import Path

from asr.utils.dataset import NonSplitPredictDataset
from asr.utils.dataloader import NonSplitPredictDataLoader
from asr.utils.logger import logger, init_logger
from asr.utils.misc import onehot2int
from asr.utils import params as p

from ..predictor import NonSplitPredictor
from .network import ListenAttendSpell


class LASPredictor(NonSplitPredictor):

    def print_result(self, filename, ys_hat, words):
        logger.info(f"decoding wav file: {str(Path(filename).resolve())}")
        if self.verbose:
            labels = onehot2int(ys_hat)
            logger.info(f"labels: {' '.join([str(x) for x in labels.tolist()])}")
            symbols = [self.decoder.labeler.idx2phone(x.item()) for x in labels]
            logger.info(f"symbols: {' '.join(symbols)}")
        words = words.squeeze()
        text = ' '.join([self.decoder.labeler.idx2word(i) for i in words]) \
               if words.dim() else '<null output from decoder>'
        logger.info(f"decoded text: {text}")


def predict(argv):
    parser = argparse.ArgumentParser(description="DeepSpeech prediction")
    parser.add_argument('--verbose', default=False, action='store_true', help="set true if you need to check AM output")
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--batch-size', default=8, type=int, help="number of simultaneous decoding")
    parser.add_argument('--log-dir', default='./logs_deepspeech_ctc', type=str, help="filename for logging the outputs")
    parser.add_argument('--continue-from', type=str, help="model file path to make continued from")
    parser.add_argument('wav_files', type=str, nargs='+', help="list of wav_files for prediction")
    args = parser.parse_args(argv)

    init_logger(log_file="predict.log", **vars(args))

    if args.continue_from is None:
        logger.error("model name is missing: add '--continue-from <model-name>' in options")
        sys.exit(1)

    input_folding = 3
    model = ListenAttendSpell(label_vec_size=p.NUM_CTC_LABELS, input_folding=input_folding)
    predictor = LASPredictor(model, **vars(args))

    dataset = NonSplitPredictDataset(wav_files=args.wav_files, stride=input_folding)
    dataloader = NonSplitPredictDataLoader(dataset=dataset, sort=True, batch_size=args.batch_size,
                                           pin_memory=args.use_cuda)

    # run prediction
    predictor.decode(dataloader)


if __name__ == "__main__":
    pass
