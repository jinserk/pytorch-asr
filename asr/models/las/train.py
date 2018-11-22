#!python
import sys
import argparse
from pathlib import Path, PurePath

import numpy as np

import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as vutils

from asr.utils.dataset import NonSplitTrainDataset, AudioSubset
from asr.utils.dataloader import NonSplitTrainDataLoader
from asr.utils.logger import logger, init_logger
from asr.utils.misc import register_nan_checks
from asr.utils import params as p
from asr.kaldi.latgen import LatGenCTCDecoder

from ..trainer import *
from .network import TFRScheduler, ListenAttendSpell


class LASTrainer(NonSplitTrainer):
    """Trainer for ListenAttendSpell model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss = nn.NLLLoss()

        self.tfr_scheduler = TFRScheduler(self.model, ranges=(0.9, 0.1), warm_up=5, epochs=32)
        if self.states is not None and "tfr_scheduler" in self.states:
            self.tfr_scheduler.load_state_dict(self.states["tfr_scheduler"])

    def train_loop_before_hook(self):
        self.tfr_scheduler.step()
        logger.debug(f"current tfr = {self.model.tfr:.3e}")

    def train_loop_checkpoint_hook(self):
        self.plot_attention_heatmap()

    def train_loop_after_hook(self):
        self.plot_attention_heatmap()

    def plot_attention_heatmap(self):
        if is_distributed() and dist.get_rank > 0:
            return
        if logger.visdom is not None and self.model.attentions is not None:
            # pick up random attention in batch size, plot each of num of heads
            for head in range(self.model.num_heads):
                a = self.model.attentions[0, head, :, :]
                logger.visdom.plot_heatmap(title=f'attention_head{head}', tensor=a)
        if logger.tensorboard is not None and self.model.attentions is not None:
            if self.model.attentions.size(0) == 1:
                logger.tensorboard.add_heatmap('attention', self.global_step, self.model.attentions[0])
            else:
                batch_size = self.model.attentions.size(0)
                logger.tensorboard.add_heatmap('attention batch0', self.global_step, self.model.attentions[0])
                logger.tensorboard.add_heatmap(f'attention batch{batch_size-1}', self.global_step, self.model.attentions[-1])

    def unit_train(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        try:
            if self.use_cuda:
                xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
            ys_hat, ys_hat_lens, ys = self.model(xs, frame_lens, ys, label_lens)
            if self.fp16:
                ys_hat = ys_hat.float()
            loss = self.loss(ys_hat.transpose(1, 2), ys.long())
            #if ys_hat_lens is None:
            #    logger.debug("the batch includes a data with label_lens > max_seq_lens: ignore the entire batch")
            #    loss.mul_(0)
            loss_value = loss.item()
            self.optimizer.zero_grad()
            if self.fp16:
                #self.optimizer.backward(loss)
                #self.optimizer.clip_master_grads(self.max_norm)
                with self.optimizer.scale_loss(loss) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            if self.use_cuda:
                torch.cuda.synchronize()
            del loss
            return loss_value
        except Exception as e:
            print(e)
            print(filenames, frame_lens, label_lens)
            raise
            #return 0

    def unit_validate(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        if self.use_cuda:
            xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
        ys_hat, ys_hat_lens = self.model(xs, frame_lens, ys, label_lens)
        if self.fp16:
            ys_hat = ys_hat.float()
        # convert likes to ctc labels
        hyps = [onehot2int(yh[:s, :]) for yh, s in zip(ys_hat, ys_hat_lens)]
        # slice the targets
        pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(label_lens, dim=0)))
        refs = [ys[s:l] for s, l in zip(pos[:-1], pos[1:])]
        return hyps, refs

    def save_hook(self):
        self.states["tfr_scheduler"] = self.tfr_scheduler.state_dict()


def batch_train(argv):
    parser = argparse.ArgumentParser(description="ListenAttendSpell AM with batch training")
    # for training
    parser.add_argument('--data-path', default='/d1/jbaik/ics-asr/data', type=str, help="dataset path to use in training")
    parser.add_argument('--num-epochs', default=200, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for optimizer")
    parser.add_argument('--max-norm', default=1e-2, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--fp16', default=False, action='store_true', help="use FP16 model")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--slack', default=False, action='store_true', help="use slackclient logging (need to set SLACK_API_TOKEN and SLACK_API_USER env_var")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_las', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='las', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=False, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="adamw", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")
    args = parser.parse_args(argv)

    init_distributed(args.use_cuda)
    init_logger(log_file="train.log", rank=get_rank(), **vars(args))
    set_seed(args.seed)

    # prepare trainer object
    input_folding = 3
    model = ListenAttendSpell(label_vec_size=p.NUM_CTC_LABELS, input_folding=input_folding)

    amp_handle = get_amp_handle(args)
    trainer = LASTrainer(model, amp_handle, **vars(args))
    labeler = trainer.decoder.labeler

    train_datasets = [
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/train.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/train.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/dev.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/test.csv", stride=input_folding),
    ]

    datasets = {
        "warmup5" : AudioSubset(train_datasets[0], max_len=5),
        "warmup10": AudioSubset(train_datasets[0], max_len=10),
        "train5" : ConcatDataset([AudioSubset(d, max_len=5) for d in train_datasets]),
        "train10": ConcatDataset([AudioSubset(d, max_len=10) for d in train_datasets]),
        "train15": ConcatDataset([AudioSubset(d, max_len=15) for d in train_datasets]),
        "dev"    : NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/eval2000.csv", stride=input_folding),
        "test"   : NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/rt03.csv", stride=input_folding),
    }

    dataloaders = {
        "warmup5" : NonSplitTrainDataLoader(datasets["warmup5"],
                                            sort=True,
                                            sampler=(DistributedSampler(datasets["warmup5"]) if is_distributed() else None),
                                            batch_size=32, num_workers=16,
                                            shuffle=(not is_distributed()),
                                            pin_memory=args.use_cuda),
        "warmup10": NonSplitTrainDataLoader(datasets["warmup10"],
                                            sort=True,
                                            sampler=(DistributedSampler(datasets["warmup10"]) if is_distributed() else None),
                                            batch_size=16, num_workers=16,
                                            shuffle=(not is_distributed()),
                                            pin_memory=args.use_cuda),
        "train5" : NonSplitTrainDataLoader(datasets["train5"],
                                           sort=True,
                                           sampler=(DistributedSampler(datasets["train5"]) if is_distributed() else None),
                                           batch_size=32, num_workers=16,
                                           shuffle=(not is_distributed()),
                                           pin_memory=args.use_cuda),
        "train10": NonSplitTrainDataLoader(datasets["train10"],
                                           sort=True,
                                           sampler=(DistributedSampler(datasets["train10"]) if is_distributed() else None),
                                           batch_size=16, num_workers=16,
                                           shuffle=(not is_distributed()),
                                           pin_memory=args.use_cuda),
        "train15": NonSplitTrainDataLoader(datasets["train15"],
                                           sort=True,
                                           sampler=(DistributedSampler(datasets["train15"]) if is_distributed() else None),
                                           batch_size=8, num_workers=16,
                                           shuffle=(not is_distributed()),
                                           pin_memory=args.use_cuda),
        "dev"    : NonSplitTrainDataLoader(datasets["dev"],
                                           sort=True,
                                           batch_size=32, num_workers=16,
                                           shuffle=False, pin_memory=args.use_cuda),
        "test"   : NonSplitTrainDataLoader(datasets["test"],
                                           sort=True,
                                           batch_size=32, num_workers=16,
                                           shuffle=False, pin_memory=args.use_cuda),
    }

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        #if i < 2:
        #    trainer.train_epoch(dataloaders["train3"])
        #    trainer.validate(dataloaders["dev"])
        if i < 2:
            trainer.train_epoch(dataloaders["warmup5"])
            trainer.validate(dataloaders["dev"])
        elif i < 5:
            trainer.train_epoch(dataloaders["warmup10"])
            trainer.validate(dataloaders["dev"])
        elif i < 30:
            trainer.train_epoch(dataloaders["train10"])
            trainer.validate(dataloaders["dev"])
        else:
            trainer.train_epoch(dataloaders["train15"])
            trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def train(argv):
    parser = argparse.ArgumentParser(description="ListenAttendSpell AM with fully supervised training")
    # for training
    parser.add_argument('--data-path', default='/d1/jbaik/ics-asr/data', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=5., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--batch-size', default=32, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--num-workers', default=16, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=100, type=int, help="number of epochs to run")
    parser.add_argument('--init-lr', default=1e-4, type=float, help="initial learning rate for Adam optimizer")
    parser.add_argument('--max-norm', default=1e-2, type=int, help="norm cutoff to prevent explosion of gradients")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--fp16', default=False, action='store_true', help="use FP16 model")
    parser.add_argument('--visdom', default=False, action='store_true', help="use visdom logging")
    parser.add_argument('--visdom-host', default="127.0.0.1", type=str, help="visdom server ip address")
    parser.add_argument('--visdom-port', default=8097, type=int, help="visdom server port")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="use tensorboard logging")
    parser.add_argument('--slack', default=False, action='store_true', help="use slackclient logging (need to set SLACK_API_TOKEN and SLACK_API_USER env_var")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--log-dir', default='./logs_las', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='las', type=str, help="model file prefix to store")
    parser.add_argument('--checkpoint', default=False, action='store_true', help="save checkpoint")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--opt-type', default="adamw", type=str, help=f"optimizer type in {OPTIMIZER_TYPES}")
    args = parser.parse_args(argv)

    init_distributed(args.use_cuda)
    init_logger(log_file="train.log", rank=get_rank(), **vars(args))
    set_seed(args.seed)

    # prepare trainer object
    input_folding = 3
    model = ListenAttendSpell(label_vec_size=p.NUM_CTC_LABELS, input_folding=input_folding)

    amp_handle = get_amp_handle(args)
    trainer = LASTrainer(model, amp_handle, **vars(args))
    labeler = trainer.decoder.labeler

    train_datasets = [
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/train.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/train.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/dev.csv", stride=input_folding),
        NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/aspire/test.csv", stride=input_folding),
    ]

    datasets = {
        "warmup": AudioSubset(train_datasets[0], data_size=0, min_len=args.min_len, max_len=args.max_len),
        "train" : ConcatDataset([AudioSubset(d, data_size=0, min_len=args.min_len, max_len=args.max_len) for d in train_datasets]),
        "dev"   : AudioSubset(NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/eval2000.csv", stride=input_folding),
                              data_size=0),
        "test"  : NonSplitTrainDataset(labeler=labeler, manifest_file=f"{args.data_path}/swbd/rt03.csv", stride=input_folding),
    }

    dataloaders = {
        "warmup": NonSplitTrainDataLoader(datasets["warmup"],
                                          sort=True,
                                          sampler=(DistributedSampler(datasets["warmup"]) if is_distributed() else None),
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=(not is_distributed()),
                                          pin_memory=args.use_cuda),
        "train": NonSplitTrainDataLoader(datasets["train"],
                                         sort=True,
                                         sampler=(DistributedSampler(datasets["train"]) if is_distributed() else None),
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=(not is_distributed()),
                                         pin_memory=args.use_cuda),
        "dev"  : NonSplitTrainDataLoader(datasets["dev"],
                                         sort=True,
                                         batch_size=16, num_workers=8,
                                         shuffle=False, pin_memory=args.use_cuda),
        "test" : NonSplitTrainDataLoader(datasets["test"],
                                         sort=True,
                                         batch_size=16, num_workers=8,
                                         shuffle=False, pin_memory=args.use_cuda),
    }

    # run inference for a certain number of epochs
    for i in range(trainer.epoch, args.num_epochs):
        trainer.train_epoch(dataloaders["warmup"])
        trainer.validate(dataloaders["dev"])

    # final test to know WER
    trainer.test(dataloaders["test"])


def test(argv):
    parser = argparse.ArgumentParser(description="ListenAttendSpell AM testing")
    # for testing
    parser.add_argument('--data-path', default='data/swbd', type=str, help="dataset path to use in training")
    parser.add_argument('--min-len', default=1., type=float, help="min length of utterance to use in secs")
    parser.add_argument('--max-len', default=20., type=float, help="max length of utterance to use in secs")
    parser.add_argument('--num-workers', default=0, type=int, help="number of dataloader workers")
    parser.add_argument('--batch-size', default=4, type=int, help="number of images (and labels) to be considered in a batch")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--fp16', default=False, action='store_true', help="use FP16 model")
    parser.add_argument('--log-dir', default='./logs_las', type=str, help="filename for logging the outputs")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")
    parser.add_argument('--validate', default=False, action='store_true', help="test LER instead of WER")
    args = parser.parse_args(argv)

    init_logger(log_file="test.log", **vars(args))

    assert args.continue_from is not None

    input_folding = 3
    model = ListenAttendSpell(label_vec_size=p.NUM_CTC_LABELS, input_folding=input_folding)

    amp_handle = get_amp_handle(args)
    trainer = LASTrainer(model, amp_handle, **vars(args))
    labeler = trainer.decoder.labeler

    if args.validate:
        manifest = f"{args.data_path}/eval2000.csv"
    else:
        manifest = f"{args.data_path}/rt03.csv"

    dataset = AudioSubset(NonSplitTrainDataset(labeler=labeler, manifest_file=manifest, stride=input_folding),
                          max_len=args.max_len, min_len=args.min_len)
    dataloader = NonSplitTrainDataLoader(dataset, sort=True, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=True, pin_memory=args.use_cuda)

    if args.validate:
        trainer.validate(dataloader)
    else:
        trainer.test(dataloader)


if __name__ == "__main__":
    pass
