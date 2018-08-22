import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

from .logger import logger
from . import params as p


class SplitTrainCollateFn(object):

    def __call__(self, batch):
        tensors = list()
        targets = list()
        tensor_lens = list()
        target_lens = list()
        filenames = list()
        texts = list()
        for tensor, target, filename, text in batch:
            tensors.append(tensor)
            targets.append(target)
            tensor_lens.append(tensor.size(0))
            target_lens.append(target.size(0))
            filenames.append(filename)
            texts.append(text)
        tensors = torch.cat(tensors)
        targets = torch.cat(targets)
        tensor_lens = torch.IntTensor(tensor_lens)
        target_lens = torch.IntTensor(target_lens)
        return tensors, targets, tensor_lens, target_lens, filenames, texts


class SplitTrainDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=SplitTrainCollateFn(), *args, **kwargs)


class NonSplitTrainCollateFn(object):

    def __call__(self, batch):
        longest_tensor = max(batch, key=lambda x: x[0].size(2))[0]
        tensors = list()
        targets = list()
        tensor_lens = list()
        target_lens = list()
        filenames = list()
        texts = list()
        for tensor, target, filename, text in batch:
            tensors.append(F.pad(tensor, (0, longest_tensor.size(2)-tensor.size(2))).unsqueeze(dim=0))
            targets.append(target)
            tensor_lens.append(tensor.size(2))
            target_lens.append(target.size(0))
            filenames.append(filename)
            texts.append(text)
        tensors = torch.cat(tensors)
        targets = torch.cat(targets)
        tensor_lens = torch.IntTensor(tensor_lens)
        target_lens = torch.IntTensor(target_lens)
        return tensors, targets, tensor_lens, target_lens, filenames, texts


class NonSplitTrainDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=NonSplitTrainCollateFn(), *args, **kwargs)


class SplitPredictCollateFn(object):

    def __call__(self, batch):
        tensors = list()
        tensor_lens = list()
        filenames = list()
        for tensor, filename in batch:
            tensors.append(tensor)
            tensor_lens.append(tensor.size(0))
            filenames.append(filename)
        tensors = torch.cat(tensors)
        tensor_lens = torch.IntTensor(tensor_lens)
        return tensors, tensor_lens, filenames


class SplitPredictDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['shuffle'] = False
        super().__init__(collate_fn=SplitPredictCollateFn(), *args, **kwargs)


class NonSplitPredictCollateFn(object):

    def __call__(self, batch):
        longest_tensor = max(batch, key=lambda x: x[0].size(2))[0]
        tensors = list()
        tensor_lens = list()
        filenames = list()
        for tensor, filename in batch:
            tensors.append(F.pad(tensor, (0, longest_tensor.size(2)-tensor.size(2))).unsqueeze(dim=0))
            tensor_lens.append(tensor.size(2))
            filenames.append(filename)
        tensors = torch.cat(tensors)
        tensor_lens = torch.IntTensor(tensor_lens)
        return tensors, tensor_lens, filenames


class NonSplitPredictDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['shuffle'] = False
        super().__init__(collate_fn=NonSplitPredictCollateFn(), *args, **kwargs)


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
