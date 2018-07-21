#!python
import sys
import argparse
from pathlib import Path, PurePath

import numpy as np
import torch

from ..utils.logger import logger, set_logfile, VisdomLog, TensorboardLog
from ..utils.audio import AudioCTCDataLoader
from ..utils import params as p
from ..utils import misc
from ..dataset.aspire import AspireEdDataset

from .model import ResNetEdModel


def parse_options(argv):
    parser = argparse.ArgumentParser(description="ResNet AM with fully supervised training with edit distance loss")
    # for training
    parser.add_argument('--data-path', default='data/aspire', type=str, help="dataset path to use in training")
    parser.add_argument('--num-workers', default=4, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--batch-size', default=8, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='resnet_aspire', type=str, help="model file prefix to store")
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

    def get_model_file_path(desc):
        return str(misc.get_model_file_path(args.log_dir, args.model_prefix, desc))

    viz = None
    if args.visdom:
        try:
            logger.info("using visdom")
            from visdom import Visdom
            viz = VisdomLog(Visdom())
        except:
            viz = None

    tbd = None
    if args.tensorboard:
        #try:
            logger.info("using tensorboard")
            tbd = TensorboardLog(PurePath(args.log_dir, 'tensorboard'))
        #except:
        #    tbd = None


    # batch_size: number of images (and labels) to be considered in a batch
    model = ResNetEdModel(x_dim=p.NUM_PIXELS, y_dim=p.NUM_CTC_LABELS, viz=viz, tbd=tbd, **vars(args))

    # initializing local variables to maintain the best validation accuracy
    # seen across epochs over the supervised training set
    best_valid_acc = 0.0

    # if you want to limit the datasets' entry size
    sizes = { "train": 1600000, "dev": 1600 }
    #sizes = { "train": 10000, "dev": 100 }

    # prepare data loaders
    datasets, data_loaders = dict(), dict()
    for mode in ["train", "dev"]:
        datasets[mode] = AspireEdDataset(root=args.data_path, mode=mode, data_size=sizes[mode],
                                         tempo=True, gain=True, noise=True)
        data_loaders[mode] = AudioCTCDataLoader(datasets[mode], batch_size=args.batch_size,
                                                num_workers=args.num_workers, shuffle=True,
                                                use_cuda=args.use_cuda, pin_memory=True)

    # run inference for a certain number of epochs
    for i in range(model.epoch, args.num_epochs):
        # get the losses for an epoch
        model.train_epoch(data_loaders["train"])
        # validate
        model.test(data_loaders["dev"], "validating")

        # update the best validation accuracy and the corresponding
        # testing accuracy and the state of the parent module (including the networks)
        #if best_valid_acc < model.meter_accuracy.value()[0]:
        #    best_valid_acc = model.meter_accuracy.value()[0]

    # test
    #model.test(data_loaders["test"], "testing   ")

    #logger.info(f"best validation accuracy {best_valid_acc:6.3f} "
    #            f"test accuracy {model.meter_accuracy.value()[0]:6.3f}")


if __name__ == "__main__":
    pass
