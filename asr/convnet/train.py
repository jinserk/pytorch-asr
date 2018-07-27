#!python
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

from ..utils.dataset import AudioSplitDataset
from ..utils.dataloader import AudioSplitDataLoader
from ..utils.logger import logger, set_logfile
from ..utils import misc
from ..utils import params as p

from .model import ConvNetModel


def parse_options(argv):
    parser = argparse.ArgumentParser(description="CNN AM with fully supervised training")
    # for training
    parser.add_argument('--data-path', default='data/aspire', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=15., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--num-workers', default=16, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=1000, type=int, help="number of epochs to run")
    parser.add_argument('--batch-size', default=1024, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--init-lr', default=0.0001, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='conv_aspire', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=False, action='store_true', help="save checkpoint")
    parser.add_argument('--num-ckpt', default=10000, type=int, help="number of batch-run to save checkpoints")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")

    args = parser.parse_args(argv)

    print(f"begins logging to file: {str(Path(args.log_dir).resolve() / 'train.log')}")
    set_logfile(Path(args.log_dir, "train.log"))

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Training started with command: {' '.join(sys.argv)}")
    args_str = [f"{k}={v}" for (k, v) in vars(args).items()]
    logger.info(f"args: {' '.join(args_str)}")

    if args.use_cuda:
        logger.info("using cuda")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)

    return args


def train(argv):
    args = parse_options(argv)

    # batch_size: number of images (and labels) to be considered in a batch
    model = ConvNetModel(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, **vars(args))

    # initializing local variables to maintain the best validation accuracy
    # seen across epochs over the supervised training set
    # and the corresponding testing set and the state of the networks
    best_valid_acc = 0.0

    # if you want to limit the datasets' entry size
    sizes = { "train": 10, "dev": 1 }

    # prepare data loaders
    datasets, data_loaders = dict(), dict()
    for mode in ["train", "dev"]:
        datasets[mode] = AudioSplitDataset(root=args.data_path, mode=mode, data_size=sizes[mode],
                                           min_len=args.min_len, max_len=args.max_len)
        data_loaders[mode] = AudioSplitDataLoader(datasets[mode], batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=True,
                                                  pin_memory=args.use_cuda)

    # run inference for a certain number of epochs
    for i in range(model.epoch, args.num_epochs):
        # get the losses for an epoch
        model.train_epoch(data_loaders["train"])
        # validate
        model.test(data_loaders["dev"], "validating")

        # update the best validation accuracy and the corresponding
        # testing accuracy and the state of the parent module (including the networks)
        #if best_valid_acc < validation_accuracy:
        #    best_valid_acc = validation_accuracy

    # test
    #model.test(data_loaders["test"], "testing   ")

    #logger.info(f"best validation accuracy {best_valid_acc:5.3f} "
    #            f"test accuracy {test_accuracy:5.3f}")


if __name__ == "__main__":
    pass
