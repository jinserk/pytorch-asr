#!python
import os
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import apex.parallel
from apex import amp

import torchvision.utils as tvu
import torchnet as tnt
import Levenshtein as Lev

from asr.utils.logger import logger
from asr.utils.misc import onehot2int, int2onehot, remove_duplicates, get_model_file_path
from asr.utils.lr_scheduler import CosineAnnealingWithRestartsLR
from asr.utils import params as p

from asr.kaldi.latgen import LatGenCTCDecoder


OPTIMIZER_TYPES = set([
    "sgd",
    "sgdr",
    "adam",
    "rmsprop",
])


def init_distributed(use_cuda, backend="nccl", init="slurm"):
    try:
        mp.set_start_method('spawn') # spawn, forkserver, and fork
    except RuntimeError:
        pass

    try:
        if init == "slurm":
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            local_rank = int(os.environ['SLURM_LOCALID'])
            #maser_node = os.environ['SLURM_TOPOLOGY_ADDR']
            #maser_port = '23456'
        elif init == "ompi":
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

        if use_cuda:
            torch.cuda.set_device(local_rank)
            print(f"set cuda device to cuda:{local_rank}")

        master_node = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        init_method = f"tcp://{master_node}:{master_port}"
        #init_method = "env://"
        dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
        print(f"initialized as {rank}/{world_size} via {init_method}")
    except:
        print(f"initialized as single process")


def is_distributed():
    try:
        return (dist.get_world_size() > 1)
    except:
        return False


def get_rank():
    try:
        return dist.get_rank()
    except:
        return None


def set_seed(seed=None):
    if seed is not None:
        logger.info(f"set random seed to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.use_cuda:
            torch.cuda.manual_seed(seed)


class Trainer:

    def __init__(self, model, amp_handle=None, init_lr=1e-2, max_norm=100, use_cuda=False,
                 fp16=False, log_dir='logs', model_prefix='model',
                 checkpoint=False, continue_from=None, opt_type="sgdr",
                 *args, **kwargs):
        if fp16:
            if not use_cuda:
                raise RuntimeError
        self.amp_handle = amp_handle

        # training parameters
        self.init_lr = init_lr
        self.max_norm = max_norm
        self.use_cuda = use_cuda
        self.fp16 = fp16
        self.log_dir = log_dir
        self.model_prefix = model_prefix
        self.checkpoint = checkpoint
        self.opt_type = opt_type
        self.epoch = 0

        # prepare visdom
        if logger.visdom is not None:
            logger.visdom.add_plot(title=f'train', xlabel='epoch', ylabel='loss')
            logger.visdom.add_plot(title=f'validate', xlabel='epoch', ylabel='LER')

        # setup model
        self.model = model
        if self.use_cuda:
            logger.debug("using cuda")
            self.model.cuda()

        # setup loss
        self.loss = nn.CTCLoss(blank=0, reduction='elementwise_mean')

        # setup optimizer
        assert opt_type in OPTIMIZER_TYPES
        parameters = self.model.parameters()
        if opt_type == "sgdr":
            logger.debug("using SGDR")
            self.optimizer = torch.optim.SGD(parameters, lr=self.init_lr, momentum=0.9, weight_decay=5e-4)
            #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
            self.lr_scheduler = CosineAnnealingWithRestartsLR(self.optimizer, T_max=5, T_mult=2)
        elif opt_type == "adam":
            logger.debug("using Adam")
            self.optimizer = torch.optim.Adam(parameters, lr=self.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
            self.lr_scheduler = None
        elif opt_type == "rmsprop":
            logger.debug("using RMSprop")
            self.optimizer = torch.optim.RMSprop(parameters, lr=self.init_lr, alpha=0.95, eps=1e-8, weight_decay=5e-4, centered=True)
            self.lr_scheduler = None

        # setup decoder for test
        self.decoder = LatGenCTCDecoder()
        self.labeler = self.decoder.labeler

        # load from pre-trained model if needed
        if continue_from is not None:
            self.load(continue_from)

        # FP16 and distributed after load
        if self.fp16:
            #self.model = network_to_half(self.model)
            #self.optimizer = FP16_Optimizer(self.optimizer, static_loss_scale=128.)
            self.optimizer = self.amp_handle.wrap_optimizer(self.optimizer)

        if is_distributed():
            if self.use_cuda:
                local_rank = torch.cuda.current_device()
                if fp16:
                    self.model = apex.parallel.DistributedDataParallel(self.model)
                else:
                    self.model = nn.parallel.DistributedDataParallel(self.model,
                                                                     device_ids=[local_rank],
                                                                     output_device=local_rank)
            else:
                self.model = nn.parallel.DistributedDataParallel(self.model)


    def __get_model_name(self, desc):
        return str(get_model_file_path(self.log_dir, self.model_prefix, desc))

    def __remove_ckpt_files(self, epoch):
        for ckpt in Path(self.log_dir).rglob(f"*_epoch_{epoch:03d}_ckpt_*"):
            ckpt.unlink()

    def train_loop_before_hook(self):
        raise NotImplementedError

    def unit_train(self, data):
        raise NotImplementedError

    def train_epoch(self, data_loader):
        self.model.train()
        meter_loss = tnt.meter.MovingAverageValueMeter(len(data_loader) // 100 + 1)
        #meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        #meter_confusion = tnt.meter.ConfusionMeter(p.NUM_CTC_LABELS, normalized=True)
        logger.debug(f"current lr = {self.optimizer.param_groups[0]['lr']}")
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if is_distributed() and data_loader.sampler is not None:
            data_loader.sampler.set_epoch(self.epoch)
        ckpts = iter(len(data_loader) * np.arange(0.1, 1.1, 0.1))
        ckpt = next(ckpts)
        self.train_loop_before_hook()
        # count the number of supervised batches seen in this epoch
        t = tqdm(enumerate(data_loader), total=len(data_loader), desc="training")
        for i, (data) in t:
            loss_value = self.unit_train(data)
            meter_loss.add(loss_value)
            t.set_description(f"training (loss: {meter_loss.value()[0]:.3f})")
            t.refresh()
            #self.meter_accuracy.add(ys_int, ys)
            #self.meter_confusion.add(ys_int, ys)

            if i > ckpt:
                title = "train"
                x = self.epoch + i / len(data_loader)
                if logger.visdom is not None:
                    logger.visdom.add_point(title=title, x=x, y=meter_loss.value()[0])
                if logger.tensorboard is not None:
                    logger.tensorboard.add_graph(self.model, xs)
                    xs_img = tvu.make_grid(xs[0, 0], normalize=True, scale_each=True)
                    logger.tensorboard.add_image('xs', x, xs_img)
                    ys_hat_img = tvu.make_grid(ys_hat[0].transpose(0, 1), normalize=True, scale_each=True)
                    logger.tensorboard.add_image('ys_hat', x, ys_hat_img)
                    logger.tensorboard.add_scalars(title, x, { 'loss': meter_loss.value()[0], })
                if self.checkpoint:
                    logger.info(f"training loss at epoch_{self.epoch:03d}_ckpt_{i:07d}: "
                                f"{meter_loss.value()[0]:5.3f}")
                    if not is_distributed() or (is_distributed() and dist.get_rank() == 0):
                        self.save(self.__get_model_name(f"epoch_{self.epoch:03d}_ckpt_{i:07d}"))
                ckpt = next(ckpts)
            #input("press key to continue")

        self.epoch += 1
        logger.info(f"epoch {self.epoch:03d}: "
                    f"training loss {meter_loss.value()[0]:5.3f} ")
                    #f"training accuracy {meter_accuracy.value()[0]:6.3f}")
        if not is_distributed() or (is_distributed() and dist.get_rank() == 0):
            self.save(self.__get_model_name(f"epoch_{self.epoch:03d}"))
            self.__remove_ckpt_files(self.epoch-1)

    def unit_validate(self, data):
        raise NotImplementedError

    def validate(self, data_loader):
        "validate with label error rate by the edit distance between hyps and refs"
        self.model.eval()
        with torch.no_grad():
            N, D = 0, 0
            t = tqdm(enumerate(data_loader), total=len(data_loader), desc="validating")
            for i, (data) in t:
                hyps, refs = self.unit_validate(data)
                # calculate ler
                N += self.edit_distance(refs, hyps)
                D += sum(len(r) for r in refs)
                ler = N * 100. / D
                t.set_description(f"validating (LER: {ler:.2f} %)")
                t.refresh()
            logger.info(f"validating at epoch {self.epoch:03d}: LER {ler:.2f} %")

            title = f"validate"
            x = self.epoch - 1 + i / len(data_loader)
            if logger.visdom is not None:
                logger.visdom.add_point(title=title, x=x, y=ler)
            if logger.tensorboard is not None:
                logger.tensorboard.add_scalars(title, x, { 'LER': ler, })

    def unit_test(self, data):
        raise NotImplementedError

    def test(self, data_loader):
        "test with word error rate by the edit distance between hyps and refs"
        self.model.eval()
        with torch.no_grad():
            N, D = 0, 0
            t = tqdm(enumerate(data_loader), total=len(data_loader), desc="testing")
            for i, (data) in t:
                hyps, refs = self.unit_test(data)
                # calculate wer
                N += self.edit_distance(refs, hyps)
                D += sum(len(r) for r in refs)
                wer = N * 100. / D
                t.set_description(f"testing (WER: {wer:.2f} %)")
                t.refresh()
            logger.info(f"testing at epoch {self.epoch:03d}: WER {wer:.2f} %")

    def edit_distance(self, refs, hyps):
        assert len(refs) == len(hyps)
        n = 0
        for ref, hyp in zip(refs, hyps):
            r = [chr(c) for c in ref]
            h = [chr(c) for c in hyp]
            n += Lev.distance(''.join(r), ''.join(h))
        return n

    def target_to_loglikes(self, ys, label_lens):
        max_len = max(label_lens.tolist())
        num_classes = self.labeler.get_num_labels()
        ys_hat = [torch.cat((torch.zeros(1).int(), ys[s:s+l], torch.zeros(max_len-l).int()))
                  for s, l in zip([0]+label_lens[:-1].cumsum(0).tolist(), label_lens.tolist())]
        ys_hat = [int2onehot(torch.IntTensor(z), num_classes, floor=1e-3) for z in ys_hat]
        ys_hat = torch.stack(ys_hat)
        ys_hat = torch.log(ys_hat)
        return ys_hat

    def save(self, file_path, **kwargs):
        Path(file_path).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        logger.debug(f"saving the model to {file_path}")
        states = kwargs
        states["epoch"] = self.epoch
        states["opt_type"] = self.opt_type
        if is_distributed():
            model_state_dict = self.model.state_dict()
            strip_prefix = 9 if self.fp16 else 7
            # remove "module.1." prefix from keys
            states["model"] = {k[strip_prefix:]: v for k, v in model_state_dict.items()}
        else:
            states["model"] = self.model.state_dict()
        states["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            states["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(states, file_path)

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.debug(f"loading the model from {file_path}")
        to_device = f"cuda:{torch.cuda.current_device()}" if self.use_cuda else "cpu"
        states = torch.load(file_path, map_location=to_device)
        self.epoch = states["epoch"]
        self.model.load_state_dict(states["model"])
        if "opt_type" in states and self.opt_type == states["opt_type"]:
            self.optimizer.load_state_dict(states["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in states:
            self.lr_scheduler.load_state_dict(states["lr_scheduler"])
        #for _ in range(self.epoch-1):
        #    self.lr_scheduler.step()


class NonSplitTrainer(Trainer):
    """training model for overall utterance spectrogram as a single image"""

    def train_loop_before_hook(self):
        pass

    def unit_train(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        try:
            batch_size = xs.size(0)
            if self.use_cuda:
                xs = xs.cuda(non_blocking=True)
            ys_hat, frame_lens = self.model(xs, frame_lens)
            if self.fp16:
                ys_hat = ys_hat.float()
            ys_hat = ys_hat.transpose(0, 1).contiguous()  # TxNxH
            #frame_lens = torch.ceil(frame_lens.float() / FRAME_REDUCE_FACTOR).int()
            #torch.set_printoptions(threshold=5000000)
            #print(ys_hat.shape, frame_lens, ys.shape, label_lens)
            #print(onehot2int(ys_hat).squeeze(), ys)
            frame_lens = frame_lens.new_full((batch_size, ), fill_value=ys_hat.size(0))  # for CUDNN ctc_loss backend
            loss = self.loss(ys_hat, ys, frame_lens, label_lens)
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
            raise
            print(e)
            print(filenames, frame_lens, label_lens)
            return 0

    def unit_validate(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        if self.use_cuda:
            xs = xs.cuda(non_blocking=True)
        ys_hat, frame_lens = self.model(xs, frame_lens)
        if self.fp16:
            ys_hat = ys_hat.float()
        # convert likes to ctc labels
        #frame_lens = torch.ceil(frame_lens.float() / FRAME_REDUCE_FACTOR).int()
        hyps = [onehot2int(yh[:s]).squeeze() for yh, s in zip(ys_hat, frame_lens)]
        hyps = [remove_duplicates(h, blank=0) for h in hyps]
        # slice the targets
        pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(label_lens, dim=0)))
        refs = [ys[s:l] for s, l in zip(pos[:-1], pos[1:])]
        return hyps, refs

    def unit_test(self, data, target_test=False):
        xs, ys, frame_lens, label_lens, filenames, texts = data
        if not target_test:
            if self.use_cuda:
                xs = xs.cuda(non_blocking=True)
            ys_hat, frame_lens = self.model(xs, frame_lens)
            if self.fp16:
                ys_hat = ys_hat.float()
            #frame_lens = torch.ceil(frame_lens.float() / FRAME_REDUCE_FACTOR).int()
        else:
            ys_hat = self.target_to_loglikes(ys, label_lens)
        # latgen decoding
        if self.use_cuda:
            ys_hat = ys_hat.cpu()
        words, alignment, w_sizes, a_sizes = self.decoder(ys_hat, frame_lens)
        w2i = self.labeler.word2idx
        num_words = self.labeler.get_num_words()
        words.masked_fill_(words.ge(num_words), w2i('<unk>'))
        words.masked_fill_(words.lt(0), w2i('<unk>'))
        hyps = [w[:s] for w, s in zip(words, w_sizes)]
        # convert target texts to word indices
        refs = [[w2i(w.strip()) for w in t.strip().split()] for t in texts]
        return hyps, refs


class SplitTrainer(Trainer):
    """ training model for splitting utterance into multiple images
        single image stands for localized timing segment corresponding to frame output
    """

    def train_loop_after_hook(self):
        pass

    def unit_train(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        try:
            if self.use_cuda:
                xs = xs.cuda(non_blocking=True)
            ys_hat = self.model(xs)
            if self.fp16:
                ys_hat = ys_hat.float()
            ys_hat = ys_hat.unsqueeze(dim=0).transpose(1, 2)
            pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(frame_lens, dim=0)))
            ys_hats = [ys_hat.narrow(2, p, l).clone() for p, l in zip(pos[:-1], frame_lens)]
            max_len = torch.max(frame_lens)
            ys_hats = [nn.ConstantPad1d((0, max_len-yh.size(2)), 0)(yh) for yh in ys_hats]
            ys_hat = torch.cat(ys_hats).transpose(1, 2).transpose(0, 1)
            loss = self.loss(ys_hat, ys, frame_lens, label_lens)
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
            del loss
        except Exception as e:
            print(filenames, frame_lens, label_lens)
            raise
        return loss_value

    def unit_validate(self, data):
        xs, ys, frame_lens, label_lens, filenames, _ = data
        if self.use_cuda:
            xs = xs.cuda(non_blocking=True)
        ys_hat = self.model(xs)
        if self.fp16:
            ys_hat = ys_hat.float()
        pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(frame_lens, dim=0)))
        ys_hat = [ys_hat.narrow(0, p, l).clone() for p, l in zip(pos[:-1], frame_lens)]
        # convert likes to ctc labels
        hyps = [onehot2int(yh[:s]).squeeze() for yh, s in zip(ys_hat, frame_lens)]
        hyps = [remove_duplicates(h, blank=0) for h in hyps]
        # slice the targets
        pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(label_lens, dim=0)))
        refs = [ys[s:l] for s, l in zip(pos[:-1], pos[1:])]
        return hyps, refs

    def unit_test(self, data, target_test=False):
        xs, ys, frame_lens, label_lens, filenames, texts = data
        if not target_test:
            if self.use_cuda:
                xs = xs.cuda(non_blocking=True)
            ys_hat = self.model(xs)
            if self.fp16:
                ys_hat = ys_hat.float()
            ys_hat = ys_hat.unsqueeze(dim=0).transpose(1, 2)
            pos = torch.cat((torch.zeros((1, ), dtype=torch.long), torch.cumsum(frame_lens, dim=0)))
            ys_hats = [ys_hat.narrow(2, p, l).clone() for p, l in zip(pos[:-1], frame_lens)]
            max_len = torch.max(frame_lens)
            ys_hats = [nn.ConstantPad1d((0, max_len-yh.size(2)), 0)(yh) for yh in ys_hats]
            ys_hat = torch.cat(ys_hats).transpose(1, 2)
        else:
            ys_hat = self.target_to_loglikes(ys, label_lens)
        # latgen decoding
        if self.use_cuda:
            ys_hat = ys_hat.cpu()
        words, alignment, w_sizes, a_sizes = self.decoder(ys_hat, frame_lens)
        w2i = self.labeler.word2idx
        num_words = self.labeler.get_num_words()
        words.masked_fill_(words.ge(num_words), w2i('<unk>'))
        words.masked_fill_(words.lt(0), w2i('<unk>'))
        hyps = [w[:s] for w, s in zip(words, w_sizes)]
        # convert target texts to word indices
        refs = [[w2i(w.strip()) for w in t.strip().split()] for t in texts]
        return hyps, refs

if __name__ == "__main__":
    pass
