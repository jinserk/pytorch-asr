#!python
import sys
import argparse
from pathlib import Path, PurePath

import torch
import torch.distributed as dist
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from warpctc_pytorch import CTCLoss

from asr.utils.dataset import NonSplitTrainDataset, AudioSubset
from asr.utils.dataloader import NonSplitTrainDataLoader
from asr.utils.logger import *
from asr.utils import params as p
from asr.kaldi.latgen import LatGenCTCDecoder

from ..trainer import FRAME_REDUCE_FACTOR, OPTIMIZER_TYPES, set_seed, init_distributed, NonSplitTrainer
from .network import DeepSpeech


def batch_train(argv):
    parser = argparse.ArgumentParser(description="DeepSpeech AM with batch training")
    # for training
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_deepspeech_ctc', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='deepspeech_ctc', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=True, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="sgdr", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")

    args = parser.parse_args(argv)

    # init distributed env
    init_distributed()
    if dist.get_world_size() > 1:
        LOG_STREAM = False
        logfile = f"train_rank{dist.get_rank()}.log"
    else:
        logfile = "train.log"

    init_logger(args, logfile)
    set_seed(args.seed)

    # prepare trainer object
    model = DeepSpeech(num_classes=p.NUM_CTC_LABELS)
    trainer = NonSplitTrainer(model, **vars(args))
    labeler = trainer.decoder.labeler

    train_datasets = [
        NonSplitTrainDataset(labeler=labeler, manifest_file="/d1/jbaik/ics-asr/data/aspire/train.csv"),
        NonSplitTrainDataset(labeler=labeler, manifest_file="/d1/jbaik/ics-asr/data/aspire/dev.csv"),
        NonSplitTrainDataset(labeler=labeler, manifest_file="/d1/jbaik/ics-asr/data/aspire/test.csv"),
        NonSplitTrainDataset(labeler=labeler, manifest_file="/d1/jbaik/ics-asr/data/swbd/train.csv"),
    ]

    datasets = {
        #"train3" : ConcatDataset([AudioSubset(d, max_len=3) for d in train_datasets]),
        "train5" : ConcatDataset([AudioSubset(d, max_len=5) for d in train_datasets]),
        "train10": ConcatDataset([AudioSubset(d, max_len=10) for d in train_datasets]),
        "train15": ConcatDataset([AudioSubset(d, max_len=15) for d in train_datasets]),
        "dev"    : NonSplitTrainDataset(labeler=labeler, manifest_file="/d1/jbaik/ics-asr/data/swbd/eval2000.csv"),
        "test"   : NonSplitTrainDataset(labeler=labeler, manifest_file="/d1/jbaik/ics-asr/data/swbd/rt03.csv"),
    }
    dataloaders = {
        #"train3" : NonSplitTrainDataLoader(datasets["train3"],
        #                                   sampler=DistributedSampler(datasets["train3"]),
        #                                   batch_size=32, num_workers=32,
        #                                   shuffle=False, pin_memory=args.use_cuda),
        "train5" : NonSplitTrainDataLoader(datasets["train5"],
                                           sampler=DistributedSampler(datasets["train5"]),
                                           batch_size=32, num_workers=32,
                                           shuffle=False, pin_memory=args.use_cuda),
        "train10": NonSplitTrainDataLoader(datasets["train10"],
                                           sampler=DistributedSampler(datasets["train10"]),
                                           batch_size=32, num_workers=32,
                                           shuffle=False, pin_memory=args.use_cuda),
        "train15": NonSplitTrainDataLoader(datasets["train15"],
                                           sampler=DistributedSampler(datasets["train15"]),
                                           batch_size=24, num_workers=24,
                                           shuffle=False, pin_memory=args.use_cuda),
        "dev"    : NonSplitTrainDataLoader(datasets["dev"],
                                           batch_size=16, num_workers=16,
                                           shuffle=False, pin_memory=args.use_cuda),
        "test"   : NonSplitTrainDataLoader(datasets["test"],
                                           batch_size=16, num_workers=16,
                                           shuffle=False, pin_memory=args.use_cuda),
    }

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        #if i < 5:
        #    dataloaders["train3"].sampler.set_epoch(i)
        #    trainer.train_epoch(dataloaders["train3"])
        #    trainer.validate(dataloaders["dev"])
        if i < 11:
            dataloaders["train5"].sampler.set_epoch(i)
            trainer.train_epoch(dataloaders["train5"])
            trainer.validate(dataloaders["dev"])
        elif i < 40:
            dataloaders["train10"].sampler.set_epoch(i)
            trainer.train_epoch(dataloaders["train10"])
            trainer.validate(dataloaders["dev"])
        else:
            dataloaders["train15"].sampler.set_epoch(i)
            trainer.train_epoch(dataloaders["train15"])
            trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def train(argv):
    parser = argparse.ArgumentParser(description="DeepSpeech AM with fully supervised training")
    # for training
    parser.add_argument('--data-path', default='data/aspire', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=15., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--batch-size', default=8, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--num-workers', default=8, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=400, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_deepspeech_ctc', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='deepspeech_ctc', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=True, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="sgdr", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")

    args = parser.parse_args(argv)

    set_logfile(Path(args.log_dir, "train.log"))
    version_log(args)
    set_seed(args.seed)

    # prepare trainer object
    model = DeepSpeech(num_classes=p.NUM_CTC_LABELS)
    trainer = NonSplitTrainer(model=model, **vars(args))
    labeler = trainer.decoder.labeler

    data_opts = {
        "train" : (f"{args.data_path}/train.csv", 0),
        "dev"   : (f"{args.data_path}/dev.csv", 0),
        "test"  : (f"{args.data_path}/test.csv", 0),
    }
    datasets, dataloaders = dict(), dict()
    for k, (v) in data_opts.items():
        manifest_file, data_size = v
        datasets[k] = AudioSubset(NonSplitTrainDataset(labeler=labeler, manifest_file=manifest_file),
                                  data_size=data_size, min_len=args.min_len, max_len=args.max_len)
        dataloaders[k] = NonSplitTrainDataLoader(datasets[k], batch_size=args.batch_size,
                                                 num_workers=args.num_workers, shuffle=True,
                                                 pin_memory=args.use_cuda)

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        trainer.train_epoch(dataloaders["train"])
        trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def test(argv):
    parser = argparse.ArgumentParser(description="DeepSpeech AM testing")
    # for testing
    parser.add_argument('--data-path', default='data/swbd', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=20., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--num-workers', default=0, type=int, help="number of dataloader workers")
    parser.add_argument('--batch-size', default=4, type=int, help="number of images (and labels) to be considered in a batch")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--log-dir', default='./logs_deepspeech_ctc', type=str, help="filename for logging the outputs")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")

    args = parser.parse_args(argv)

    set_logfile(Path(args.log_dir, "test.log"))
    version_log(args)

    assert args.continue_from is not None

    model = DeepSpeech(num_classes=p.NUM_CTC_LABELS)
    trainer = NonSplitTrainer(model, **vars(args))
    labeler = trainer.decoder.labeler

    manifest = f"{args.data_path}/eval2000.csv"
    dataset = AudioSubset(NonSplitTrainDataset(labeler=labeler, manifest_file=manifest),
                          max_len=args.max_len, min_len=args.min_len)
    dataloader = NonSplitTrainDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=True, pin_memory=args.use_cuda)

    trainer.test(dataloader)


if __name__ == "__main__":
    pass
