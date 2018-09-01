import os
import sys
import random
from pathlib import Path
import tempfile

import numpy as np
import scipy.io.wavfile
from scipy.signal import tukey
from pysndfx import AudioEffectsChain

import torch
import torch.nn as nn
from torch._C import _set_worker_signal_handlers
from torch.utils.data import Dataset, Subset
import torchaudio

from .logger import logger
from . import params as p


WIN_SAMP_SIZE = p.SAMPLE_RATE * p.WINDOW_SIZE
WIN_SAMP_SHIFT = p.SAMPLE_RATE * p.WINDOW_SHIFT
#SAMPLE_MARGIN = WIN_SAMP_SHIFT * p.FRAME_MARGIN  # samples
SAMPLE_MARGIN = 0

#np.seterr(all='raise')

# transformer: resampling and augmentation
class Augment(object):

    def __init__(self, resample, sample_rate, tempo, tempo_range, pitch, pitch_range,
                 noise, noise_range, offset, offset_range, padding, num_padding):
        self.resample = resample
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.tempo_range = tempo_range
        self.pitch = pitch
        self.pitch_range = pitch_range
        self.noise = noise
        self.noise_range = noise_range
        self.offset = offset
        self.offset_range=offset_range
        self.padding = padding
        self.num_padding=num_padding

    def __call__(self, wav_file):
        if not Path(wav_file).exists():
            print(wav_file)
            raise IOError

        sr, wav = scipy.io.wavfile.read(wav_file)
        if wav.ndim > 1 and wav.shape[1] > 1:
            logger.error("wav file has two or more channels")
            sys.exit(1)
        if type(wav[0]) is np.int32:
            wav = wav.astype('float32', copy=False) / 2147483648.0
        elif type(wav[0]) is np.int16:
            wav = wav.astype('float32', copy=False) / 32768.0
        elif type(wav[0]) is np.uint8:
            wav = wav.astype('float32', copy=False) / 256.0 - 128.0

        fx = AudioEffectsChain()

        if self.resample:
            if self.sample_rate > sr:
                ratio = int(self.sample_rate / sr)
                fx.upsample(ratio)
            elif self.sample_rate < sr:
                ratio = int(sr / self.sample_rate)
                fx.custom(f"downsample {ratio}")

        if self.tempo:
            tempo_change = np.random.uniform(*self.tempo_range)
            fx.tempo(tempo_change, opt_flag="s")

        if self.pitch:
            pitch_change = np.random.uniform(*self.pitch_range)
            fx.pitch(pitch_change)

        # dithering
        fx.custom(f"dither -s")

        wav = fx(wav, sample_in=sr, sample_out=self.sample_rate)
        #wav = wav / max(abs(wav))

        # normalize audio power
        gain = 0.1
        wav_energy = np.sqrt(np.sum(np.power(wav, 2)) / wav.size)
        wav = gain * wav / wav_energy

        # sample-domain padding
        if self.padding:
            wav = np.pad(wav, self.num_padding, mode='constant')

        # sample-domain offset
        if self.offset:
            offset = np.random.randint(*self.offset_range)
            wav = np.roll(wav, offset, axis=0)

        if self.noise:
            snr = 10.0 ** (np.random.uniform(*self.noise_range) / 10.0)
            noise = np.random.normal(0, 1, wav.shape)
            noise_energy = np.sqrt(np.sum(np.power(noise, 2)) / noise.size)
            wav = wav + snr * gain * noise / noise_energy

        #filename = wav_file.replace(".wav", "_augmented.wav")
        #scipy.io.wavfile.write(filename, self.sample_rate, wav)
        return torch.FloatTensor(wav)


# transformer: spectrogram
class Spectrogram(object):

    def __init__(self, sample_rate, window_shift, window_size, nfft, window=tukey):
        self.nfft = nfft
        self.window_size = int(sample_rate * window_size)
        self.window_shift = int(sample_rate * window_shift)
        self.window = torch.FloatTensor(window(self.window_size))

    def __call__(self, wav):
        with torch.no_grad():
            # STFT
            data = torch.stft(wav, n_fft=self.nfft, hop_length=self.window_shift,
                              win_length=self.window_size, window=self.window)
            data /= self.window.pow(2).sum().sqrt_()
            #mag = data.pow(2).sum(-1).log1p_()
            #ang = torch.atan2(data[:, :, 1], data[:, :, 0])
            ## {mag, phase} x n_freq_bin x n_frame
            #data = torch.cat([mag.unsqueeze_(0), ang.unsqueeze_(0)], dim=0)
            ## FxTx2 -> 2xFxT
            data = data.transpose(1, 2).transpose(0, 1)
            return data


# transformer: frame splitter
class FrameSplitter(object):
    """ split C x H x W frames to M x C2 x H x U where U is unit frames in time
        C2 = stride x C, M = floor((W - U) / stride)
    """
    def __init__(self, unit_frames, padding=0, stride=1, split=True):
        self.padding = padding
        self.pad = nn.ZeroPad2d((padding, padding, 0, 0))
        self.stride = stride
        self.split = split
        if split:
            assert unit_frames % 2 == 1, "unit_frames should be odd integer"
            self.unit_frames = unit_frames

    def __call__(self, tensor):
        with torch.no_grad():
            tensor = tensor.unsqueeze(dim=0)
            if self.padding > 0:
                tensor = self.pad(tensor)
            M, C, H, W = tensor.size()
            Wp = W // self.stride
            sWp = Wp * self.stride
            sC = C * self.stride
            folded = tensor[:, :, :, :sWp].view(M, C, H, Wp, self.stride)
            folded = folded.transpose(3, 4).transpose(2, 3).contiguous().view(M, sC, H, Wp)
            if not split:
                return folded
            pos = [p for p in range(0, Wp - self.unit_frames)]
            splits = [folded.narrow(3, p, self.unit_frames).clone() for p in pos]
            frames = torch.cat(splits)
            return frames


# transformer: convert int to one-hot vector
class Int2OneHot(object):

    def __init__(self, num_labels):
        self.num_labels = num_labels

    def __call__(self, targets):
        one_hots = list()
        for t in targets:
            one_hot = torch.LongTensor(self.num_labels).zero_()
            one_hot[t] = 1
            one_hots.append(one_hot)
        return one_hots


class NonSplitTransformer(torchaudio.transforms.Compose):

    def __init__(self,
                 resample=True, sample_rate=p.SAMPLE_RATE,
                 tempo=True, tempo_range=p.TEMPO_RANGE,
                 pitch=True, pitch_range=p.PITCH_RANGE,
                 noise=True, noise_range=p.NOISE_RANGE,
                 offset=True, offset_range=None,
                 padding=True, num_padding=None,
                 window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE, nfft=p.NFFT,
                 stride=1):
        if offset and offset_range is None:
            offset_range = (0, stride * WIN_SAMP_SHIFT)
        if padding and num_padding is None:
            pad = int(((p.WIDTH * stride) // 2 - 1) * WIN_SAMP_SHIFT)
            num_padding = (pad, pad)
        super().__init__([
            Augment(resample=resample, sample_rate=sample_rate,
                    tempo=tempo, tempo_range=tempo_range,
                    pitch=pitch, pitch_range=pitch_range,
                    noise=noise, noise_range=noise_range,
                    offset=offset, offset_range=offset_range,
                    padding=padding, num_padding=num_padding),
            Spectrogram(sample_rate=sample_rate, window_shift=window_shift,
                        window_size=window_size, nfft=nfft),
        ])


class SplitTransformer(NonSplitTransformer):

    def __init__(self, unit_frames=p.WIDTH, stride=1, *args, **kwargs):
        super().__init__(stride=stride, *args, **kwargs)
        self.transforms.append(FrameSplitter(unit_frames=unit_frames, padding=0, stride=stride))


def _smp2frm(samples):
    num_samples = samples - 2 * SAMPLE_MARGIN
    return int((num_samples - WIN_SAMP_SIZE) // WIN_SAMP_SHIFT + 1)


def _load_manifest(manifest_file):
    if not manifest_file.exists():
        logger.error(f"no such manifest file {manifest_file} found. "
                     f"need to prepare data first.")
        sys.exit(1)
    logger.info(f"loading dataset manifest {str(manifest_file)} ...")
    with open(manifest_file, "r") as f:
        manifest = f.readlines()
    entries = [tuple(x.strip().split(',')) for x in manifest]
    entry_frames = [_smp2frm(int(e[2])) for e in entries]
    logger.info(f"{len(entries)} entries, {sum(entry_frames)} frames are loaded.")
    return entries, entry_frames


def _text_to_labels(labeler, text, sil_prop=(0.2, 0.8)):
    """ choosing a uniformly random lexicon definition, after inserting sil phones
        with sil_prop[0] between words and with sil_prop[1] at the beginning and the end
        of the sentences
    """
    sil = labeler.phone2idx('sil')
    words = [w.strip() for w in text.strip().split()]
    labels = list()
    if random.random() < sil_prop[1]:
        labels.append(sil)
    for word in words[:-1]:
        lex = labeler.word2lex(word)
        labels.extend(lex[int(len(lex)*random.random())] if len(lex) > 1 else lex[0])
        if random.random() < sil_prop[0]:
            labels.append(sil)
    lex = labeler.word2lex(words[-1])
    labels.extend(lex[int(len(lex)*random.random())] if len(lex) > 1 else lex[0])
    if random.random() < sil_prop[1]:
        labels.append(sil)
    return labels


class TrainDataset(Dataset):

    def __init__(self, labeler, manifest_file, *args, **kwargs):
        self.labeler = labeler
        self.manifest_file = Path(manifest_file).resolve()
        super().__init__(*args, **kwargs)
        self.entries, self.entry_frames = _load_manifest(self.manifest_file)

    def __getitem__(self, index):
        uttid, wav_file, samples, txt_file = self.entries[index]
        # read and transform wav file
        if self.transformer is not None:
            tensors = self.transformer(wav_file)
        # read txt file
        with open(txt_file, 'r') as f:
            text = f.read()
        targets = _text_to_labels(self.labeler, text)
        targets = torch.IntTensor(targets)
        if self.target_transformer is not None:
            targets = self.target_transformer(targets)
        return tensors, targets, wav_file, text

    def __len__(self):
        return len(self.entries)


class PredictDataset(Dataset):

    def __init__(self, wav_files, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = wav_files

    def __getitem__(self, index):
        wav_file = self.entries[index]
        # read and transform wav file
        if self.transformer is not None:
            tensors = self.transformer(wav_file)
        return tensors, wav_file

    def __len__(self):
        return len(self.entries)


class NonSplitTrainDataset(TrainDataset):

    def __init__(self,
                 transformer=None, target_transformer=None,
                 resample=True, sample_rate=p.SAMPLE_RATE,
                 tempo=True, tempo_range=p.TEMPO_RANGE,
                 pitch=True, pitch_range=p.PITCH_RANGE,
                 noise=True, noise_range=p.NOISE_RANGE,
                 offset=True, padding=True,
                 window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE, nfft=p.NFFT,
                 stride=3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transformer is None:
            self.transformer = NonSplitTransformer(resample=resample, sample_rate=sample_rate,
                                                   tempo=tempo, tempo_range=tempo_range,
                                                   pitch=pitch, pitch_range=pitch_range,
                                                   noise=noise, noise_range=noise_range,
                                                   offset=offset, padding=padding,
                                                   window_shift=window_shift, window_size=window_size, nfft=nfft,
                                                   stride=stride)
        else:
            self.transformer = transformer
        self.target_transformer = target_transformer


class NonSplitPredictDataset(PredictDataset):

    def __init__(self,
                 transformer=None, target_transformer=None,
                 resample=True, sample_rate=p.SAMPLE_RATE,
                 noise=True, noise_range=(-20, -20),
                 padding=False,
                 window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE, nfft=p.NFFT,
                 stride=3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transformer is None:
            self.transformer = NonSplitTransformer(resample=resample, sample_rate=sample_rate,
                                                   tempo=False, pitch=False,
                                                   noise=noise, noise_range=noise_range,
                                                   offset=False, padding=padding,
                                                   window_shift=p.WINDOW_SHIFT, window_size=window_size, nfft=nfft,
                                                   stride=stride)
        else:
            self.transformer = transformer
        self.target_transformer = target_transformer


class SplitTrainDataset(TrainDataset):

    def __init__(self,
                 transformer=None, target_transformer=None,
                 resample=True, sample_rate=p.SAMPLE_RATE,
                 tempo=True, tempo_range=p.TEMPO_RANGE,
                 pitch=True, pitch_range=p.PITCH_RANGE,
                 noise=True, noise_range=p.NOISE_RANGE,
                 offset=True, padding=True,
                 window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE, nfft=p.NFFT,
                 unit_frames=p.WIDTH, stride=3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transformer is None:
            self.transformer = SplitTransformer(resample=resample, sample_rate=sample_rate,
                                                tempo=tempo, tempo_range=tempo_range,
                                                pitch=pitch, pitch_range=pitch_range,
                                                noise=noise, noise_range=noise_range,
                                                offset=offset, padding=padding,
                                                window_shift=window_shift, window_size=window_size, nfft=nfft,
                                                unit_frames=unit_frames, stride=stride)
        else:
            self.transformer = transformer
        self.target_transformer = target_transformer


class SplitPredictDataset(PredictDataset):

    def __init__(self,
                 transformer=None, target_transformer=None,
                 resample=True, sample_rate=p.SAMPLE_RATE,
                 noise=True, noise_range=(-20, -20),
                 padding=False,
                 window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE, nfft=p.NFFT,
                 unit_frames=p.WIDTH, stride=3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transformer is None:
            self.transformer = SplitTransformer(resample=resample, sample_rate=sample_rate,
                                                tempo=False, pitch=False,
                                                noise=noise, noise_range=noise_range,
                                                offset=False, padding=padding,
                                                window_shift=p.WINDOW_SHIFT, window_size=window_size, nfft=nfft,
                                                unit_frames=unit_frames, stride=stride)
        else:
            self.transformer = transformer
        self.target_transformer = target_transformer


class AudioSubset(Subset):

    def __init__(self, dataset, data_size=0, min_len=1., max_len=10.):
        indices = self._pick_indices(dataset.entries, data_size, min_len, max_len)
        super().__init__(dataset, indices)

    def _pick_indices(self, entries, data_size, min_len, max_len):
        full_indices = range(len(entries))
        # pick up entries of time length from min_len to max_len secs
        MIN_FRAME = min_len / p.WINDOW_SHIFT
        MAX_FRAME = max_len / p.WINDOW_SHIFT
        indices = [i for i in full_indices if MIN_FRAME < _smp2frm(int(entries[i][2])) < MAX_FRAME ]
        # randomly choose a number of data_size
        size = min(data_size, len(indices)) if data_size > 0 else len(indices)
        selected = random.sample(indices, size)
        return selected


if __name__ == "__main__":
    test = 1
    # test Augment
    if test == 1:
        transformer = Augment(resample=True, sample_rate=8000,
                              tempo=True, tempo_range=(0.9, 1.1),
                              pitch=True, pitch_range=(-150., -150.),
                              noise=True, noise_range=(-20., -5.),
                              offset=False, offset_range=(0, 40),
                              padding=False, num_padding=0)
        wav_file = "/d1/jbaik/ics-asr/temp/conan1-8k.wav"
        audio = transformer(wav_file)
    # test Spectrogram
    elif test == 2:
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
    elif test == 3:
        s = FrameSplitter(unit_frames=5, stride=2)
        x = torch.rand((2, 3, 20))
        y = s(x)
        breakpoint()

