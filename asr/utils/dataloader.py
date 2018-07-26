import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler, SequentialSampler
import torchaudio

from .logger import logger
from . import params as p


class SplitCollateFn(object):
    idx = -1
    tensor = None
    target = None

    def __call__(self, dataset, indices):
        tensors, targets = list(), list()
        for idx, fidx in indices:
            if self.idx != idx:
                self.tensor, self.target = dataset[idx]
                self.idx = idx
            tensors.append(self.tensor[fidx])
            if self.target is not None:
                targets.append(self.target[fidx])
        if targets:
            batch = (torch.stack(tensors), torch.stack(targets))
        else:
            batch = torch.stack(tensors)
        return batch


class NonSplitCollateFn(object):

    def __init__(self, frame_shift=0):
        self.frame_shift = frame_shift

    def __call__(self, batch):
        batch_size = len(batch)
        longest_tensor = max(batch, key=lambda x: x[0].size(2))[0]
        #shape = (*tuple(longest_tensor.shape)[:-1], 2000)
        #longest_target = max(batch, key=lambda x: x[1].size(0))[1]
        #tensors = torch.zeros(batch_size, *shape)
        tensors = torch.zeros(batch_size, *longest_tensor.shape)
        targets = []
        tensor_lens = []
        target_lens = []
        filenames = []
        for i in range(batch_size):
            tensor, target, filename = batch[i]
            if self.frame_shift > 0:
                offset = random.randint(0, self.frame_shift)
                if offset == 0:
                    tensors[i].narrow(2, 0, tensor.size(2)).copy_(tensor)
                else:
                    tensors[i].narrow(2, 0, offset).copy_(tensor[:, :, -offset:])
                    tensors[i].narrow(2, 0, tensor.size(2)-offset).copy_(tensor[:, :, :-offset])
            else:
                tensors[i].narrow(2, 0, tensor.size(2)).copy_(tensor)
            targets.append(target)
            tensor_lens.append(tensor.size(2))
            target_lens.append(target.size(0))
            filenames.append(filename)
        targets = torch.cat(targets)
        tensor_lens = torch.IntTensor(tensor_lens)
        target_lens = torch.IntTensor(target_lens)
        return tensors, targets, tensor_lens, target_lens, filenames


class SplitBatchSampler(BatchSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = self.sampler.data_source.entry_frames

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            frame = np.arange(self.frames[idx])
            np.random.shuffle(frame)
            for fidx in frame:
                batch.append((idx, fidx))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        total = sum(self.frames)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size


class AudioSplitDataLoader(DataLoader):

    def __init__(self, dataset, batch_size,
                 shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 drop_last=True, pin_memory=False, *args, **kwargs):
        collate_fn = SplitCollateFn()
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = SplitBatchSampler(sampler, batch_size, drop_last)

        super().__init__(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                         collate_fn=collate_fn, pin_memory=pin_memory, timeout=0, *args, **kwargs)

    #def __iter__(self):
    #    return AudioDataLoaderIter(self)


class AudioNonSplitDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        if "frame_shift" in kwargs:
            frame_shift = kwargs["frame_shift"]
            del(kwargs["frame_shift"])
        else:
            frame_shift = 0

        collate_fn = NonSplitCollateFn(frame_shift)
        super().__init__(collate_fn=collate_fn, *args, **kwargs)


class PredictDataLoader:

    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset

    def load(self, wav_file):
        # read and transform wav file
        if self.dataset.transform is not None:
            tensor = self.dataset.transform(wav_file)
        if isinstance(tensor, tuple):
            tensor = torch.stack(tensor)
        else:
            tensor = tensor.unsqueeze(0)

        ctc_file = wav_file.replace('wav', 'ctc')
        ctc_target = None
        if Path(ctc_file).exists():
            ctc_target = np.loadtxt(ctc_file, dtype="int", ndmin=1)
            ctc_target = torch.IntTensor(ctc_target)

        txt_file = wav_file.replace('wav', 'txt')
        txt_target = None
        if Path(txt_file).exists():
            with open(txt_file, 'r') as f:
                txt_target = ' '.join(f.readlines()).strip().replace('\n', '')

        return tensor, ctc_target, txt_target


def test_plot():
    from ..util.audio import AudioDataLoader, NonSplitDataLoader
    train_dataset = AsrDataset(mode="test")
    loader = AudioDataLoader(train_dataset, batch_size=10, num_workers=4, shuffle=True)
    logger.info(f"num_workers={loader.num_workers}")

    for i, data in enumerate(loader):
        tensors, targets = data
        #for tensors, targets in data:
        logger.info("f{tensors}, {targets}")
        if False:
            import matplotlib
            matplotlib.use('TkAgg')
            matplotlib.interactive(True)
            import matplotlib.pyplot as plt

            for tensor, target in zip(tensors, targets):
                tensor = tensor.view(-1, p.CHANNEL, p.WIDTH, p.HEIGHT)
                t = np.arange(0, tensor.size(3)) / 8000
                f = np.linspace(0, 4000, tensor.size(2))

                fig = plt.figure(1)
                p = plt.pcolormesh(t, f, np.log10(10 ** tensor[0][0] - 1), cmap='plasma')
                plt.colorbar(p)
                plt.show(block=True)
        if i == 2:
            break
    #plt.close('all')


if __name__ == "__main__":
    # test Augment
    if False:
        transformer = Augment(resample=True, sample_rate=p.SAMPLE_RATE)
        wav_file = Path("/home/jbaik/src/enf/stt/test/conan1-8k.wav")
        audio = transformer(wav_file)

    # test Spectrogram
    if True:
        import matplotlib
        matplotlib.use('TkAgg')
        matplotlib.interactive(True)
        import matplotlib.pyplot as plt

        nperseg = int(p.SAMPLE_RATE * p.WINDOW_SIZE)
        noverlap = int(p.SAMPLE_RATE * (p.WINDOW_SIZE - p.WINDOW_SHIFT))

        wav_file = Path("../data/aspire/000/fe_03_00047-A-025005-025135.wav")
        audio, _ = torchaudio.load(wav_file)

        # pyplot specgram
        audio = torch.squeeze(audio)
        fig = plt.figure(0)


if __name__ == "__main__":
    # test Augment
    if False:
        transformer = Augment(resample=True, sample_rate=p.SAMPLE_RATE)
        wav_file = Path("/home/jbaik/src/enf/stt/test/conan1-8k.wav")
        audio = transformer(wav_file)

    # test Spectrogram
    if True:
        import matplotlib
        matplotlib.use('TkAgg')
        matplotlib.interactive(True)
        import matplotlib.pyplot as plt

        nperseg = int(p.SAMPLE_RATE * p.WINDOW_SIZE)
        noverlap = int(p.SAMPLE_RATE * (p.WINDOW_SIZE - p.WINDOW_SHIFT))

        wav_file = Path("../data/aspire/000/fe_03_00047-A-025005-025135.wav")
        audio, _ = torchaudio.load(wav_file)

        # pyplot specgram
        audio = torch.squeeze(audio)
        fig = plt.figure(0)
        plt.specgram(audio, Fs=p.SAMPLE_RATE, NFFT=p.NFFT, noverlap=noverlap, cmap='plasma')

        # implemented transformer - scipy stft
        transformer = Spectrogram(sample_rate=p.SAMPLE_RATE, window_stride=p.WINDOW_SHIFT,
                                  window_size=p.WINDOW_SIZE, nfft=p.NFFT)
        data, f, t = transformer(audio)
        print(data.shape)
        mag = data[0]
        fig = plt.figure(1)
        plt.pcolormesh(t, f, np.log10(np.expm1(data[0])), cmap='plasma')
        fig = plt.figure(2)
        plt.pcolormesh(t, f, data[1], cmap='plasma')
        #print(max(data[0].view(257*601)), min(data[0].view(257*601)))
        #print(max(data[1].view(257*601)), min(data[1].view(257*601)))

        # scipy spectrogram
        f, t, z = sp.signal.spectrogram(audio, fs=p.SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap,
                                        nfft=p.NFFT, mode='complex')
        spect, phase = np.abs(z), np.angle(z)
        fig = plt.figure(3)
        plt.pcolormesh(t, f, 20*np.log10(spect), cmap='plasma')
        fig = plt.figure(4)
        plt.pcolormesh(t, f, phase, cmap='plasma')

        plt.show(block=True)
        plt.close('all')
