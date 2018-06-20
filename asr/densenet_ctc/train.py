#!python
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

from ..utils.logger import logger, set_logfile
from ..utils.audio import AudioCTCDataLoader
from ..utils import params as p
from ..utils import misc
from ..dataset.aspire import AspireCTCDataset

from .model import DenseNetCTCModel


def parse_options(argv):
    parser = argparse.ArgumentParser(description="DenseNet AM with fully supervised training")
    # for training
    parser.add_argument('--data-path', default='data/aspire', type=str, help="dataset path to use in training")
    parser.add_argument('--num-workers', default=4, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=1000, type=int, help="number of epochs to run")
    parser.add_argument('--batch-size', default=2, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--init-lr', default=5e-7, type=float, help="initial learning rate for Adam optimizer")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='dense_aspire', type=str, help="model file prefix to store")
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
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)

    return args, device


def train(argv):
    args, device = parse_options(argv)

    def get_model_file_path(desc):
        return misc.get_model_file_path(args.log_dir, args.model_prefix, desc)

    # batch_size: number of images (and labels) to be considered in a batch
    model = DenseNetCTCModel(x_dim=p.NUM_PIXELS, y_dim=p.NUM_LABELS, device=device, **vars(args))

    # initializing local variables to maintain the best validation accuracy
    # seen across epochs over the supervised training set
    best_valid_acc = 0.0

    # if you want to limit the datasets' entry size
    sizes = { "train": 600000, "dev": 1000 }
    #sizes = { "train": 10000, "dev": 100 }

    # prepare data loaders
    datasets, data_loaders = dict(), dict()
    for mode in ["train", "dev"]:
        datasets[mode] = AspireCTCDataset(root=args.data_path, mode=mode, data_size=sizes[mode])
        data_loaders[mode] = AudioCTCDataLoader(datasets[mode], batch_size=args.batch_size,
                                                num_workers=args.num_workers, shuffle=True,
                                                use_cuda=args.use_cuda, pin_memory=True)

    # run inference for a certain number of epochs
    for i in range(model.epoch, args.num_epochs):
        # get the losses for an epoch
        model.train_epoch(data_loaders["train"])
        logger.info(f"epoch {model.epoch:03d}: "
                    f"training loss {model.meter_loss.value()[0]:5.3f} ")
                    #f"training accuracy {model.meter_accuracy.value()[0]:6.3f}")

        # validate
        model.test(data_loaders["dev"], "validating")
        logger.info(f"epoch {model.epoch:03d}: "
                    f"validating loss {model.meter_loss.value()[0]:5.3f} ")
                    #f"validating accuracy {model.meter_accuracy.value()[0]:6.3f}")

        # update the best validation accuracy and the corresponding
        # testing accuracy and the state of the parent module (including the networks)
        #if best_valid_acc < model.meter_accuracy.value()[0]:
        #    best_valid_acc = model.meter_accuracy.value()[0]
        # save
        model.save(get_model_file_path(f"epoch_{model.epoch:04d}"))
        # increase epoch num
        model.epoch += 1

    # test
    #model.test(data_loaders["test"])

    #logger.info(f"best validation accuracy {best_valid_acc:6.3f} "
    #            f"test accuracy {model.meter_accuracy.value()[0]:6.3f}")

    #save final model
    model.save(get_model_file_path("final"), epoch=model.epoch)


if __name__ == "__main__":
    pass
