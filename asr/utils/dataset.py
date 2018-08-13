import os
import sys
import random
from pathlib import Path
import tempfile

import numpy as np
import scipy.io.wavfile
from scipy.signal import tukey
import sox

import torch
import torch.nn as nn
from torch._C import _set_worker_signal_handlers
from torch.utils.data import Dataset, Subset
import torchaudio

from .logger import logger
from . import params as p


# transformer: resampling and augmentation
class Augment(object):

    def __init__(self, resample, sample_rate, tempo, tempo_range,
                 gain, gain_range, noise, noise_range):
        self.resample = resample
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.tempo_range = tempo_range
        self.last_tempo = 1.
        self.gain = gain
        self.gain_range = gain_range
        self.noise = noise
        self.noise_range = noise_range

    def __call__(self, wav_file, tar_file=None):
        if not Path(wav_file).exists():
            print(wav_file)
            raise IOError

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        if self.resample:
            tfm.rate(self.sample_rate)

        if self.tempo:
            tempo = np.random.uniform(*self.tempo_range)
            if tempo < 0.9 or tempo > 1.1:
                tfm.tempo(tempo, audio_type='s')
            else:
                tfm.stretch(tempo)
            self.last_tempo = tempo

        if self.gain:
            gain = np.random.uniform(*self.gain_range)
            tfm.gain(gain, normalize=True)

        if tar_file is None:
            tmp_dir = tempfile._get_default_tempdir()
            tmp_name = next(tempfile._get_candidate_names())
            tmp_file = Path(tmp_dir, tmp_name + ".wav")
            tfm.build(str(wav_file), str(tmp_file))
            sr, wav = scipy.io.wavfile.read(tmp_file)
            os.unlink(tmp_file)
        else:
            Path(tar_file).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            tfm.build(str(wav_file), str(tar_file))
            sr, wav = scipy.io.wavfile.read(tar_file)

        if wav.ndim > 1 and wav.shape[1] > 1:
            logger.error("wav file has two or more channels")
            sys.exit(1)

        # normalize audio power
        # (TODO: how about tfm.gain above?)
        gain = 0.1
        wav = wav.astype(np.float32)
        wav_energy = np.sqrt(np.sum(np.power(wav, 2)) / wav.size)
        wav = gain * wav / wav_energy

        if self.noise:
            snr = 10.0 ** (np.random.uniform(*self.noise_range) / 10.0)
            noise = np.random.normal(0, 1, wav.shape)
            noise_energy = np.sqrt(np.sum(np.power(noise, 2)) / noise.size)
            wav = wav + snr * gain * noise / noise_energy

        #scipy.io.wavfile.write("test.wav", sr, wav)
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
    """ split C x H x W frames to M x C x H x U where U is unit frames in time
        padding and stride are only applied to W, the time axis
    """

    def __init__(self, unit_frames, padding = 0, stride = 1):
        assert unit_frames % 2 == 1, "unit_frames should be odd integer"
        self.unit_frames = unit_frames
        self.pad = nn.ZeroPad2d((padding, padding, 0, 0))
        self.stride = stride

    def __call__(self, tensor):
        with torch.no_grad():
            tensor = self.pad(tensor.unsqueeze(dim=0))
            bins = [(x, self.unit_frames) for x in range(0, tensor.size(3)-self.unit_frames, self.stride)]
            frames = torch.cat([tensor.narrow(3, s, l).clone() for s, l in bins])
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


class SplitDataset(Dataset):

    def __init__(self,
                 transform=None, target_transform=None,
                 resample=False, sample_rate=p.SAMPLE_RATE,
                 tempo=False, tempo_range=p.TEMPO_RANGE,
                 gain=False, gain_range=p.GAIN_RANGE,
                 noise=False, noise_range=p.NOISE_RANGE,
                 window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE, nfft=p.NFFT,
                 frame_margin=0, unit_frames=p.HEIGHT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transform is None:
            self.transform = torchaudio.transforms.Compose([
                Augment(resample=resample, sample_rate=sample_rate,
                        tempo=tempo, tempo_range=tempo_range,
                        gain=gain, gain_range=gain_range,
                        noise=noise, noise_range=noise_range),
                Spectrogram(sample_rate=sample_rate, window_shift=window_shift,
                            window_size=window_size, nfft=nfft),
                FrameSplitter(frame_margin=frame_margin, unit_frames=unit_frames),
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform
        #if target_transform is None:
        #    self.target_transform = Int2OneHot(p.NUM_LABELS)
        #else:
        #    self.target_transform = target_transform


class NonSplitDataset(Dataset):

    def __init__(self,
                 transform=None, target_transform=None,
                 resample=False, sample_rate=p.SAMPLE_RATE,
                 tempo=False, tempo_range=p.TEMPO_RANGE,
                 gain=False, gain_range=p.GAIN_RANGE,
                 noise=False, noise_range=p.NOISE_RANGE,
                 window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE, nfft=p.NFFT,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transform is None:
            self.transform = torchaudio.transforms.Compose([
                Augment(resample=resample, sample_rate=sample_rate,
                        tempo=tempo, tempo_range=tempo_range,
                        gain=gain, gain_range=gain_range,
                        noise=noise, noise_range=noise_range),
                Spectrogram(sample_rate=sample_rate, window_shift=window_shift,
                            window_size=window_size, nfft=nfft),
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform


WIN_SAMP_SIZE = p.SAMPLE_RATE * p.WINDOW_SIZE
WIN_SAMP_SHIFT = p.SAMPLE_RATE * p.WINDOW_SHIFT
#SAMPLE_MARGIN = WIN_SAMP_SHIFT * p.FRAME_MARGIN  # samples
SAMPLE_MARGIN = 0


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


class AudioCTCDataset(NonSplitDataset):

    def __init__(self, labeler, manifest_file, *args, **kwargs):
        self.labeler = labeler
        self.manifest_file = Path(manifest_file).resolve()
        super().__init__(tempo=True, gain=True, noise=True, *args, **kwargs)
        self.entries, self.entry_frames = _load_manifest(self.manifest_file)

    def __getitem__(self, index):
        uttid, wav_file, samples, txt_file = self.entries[index]
        # read and transform wav file
        if self.transform is not None:
            tensors = self.transform(wav_file)
        # read ctc file
        #ctc_file = phn_file.replace('phn', 'ctc')
        #targets = np.loadtxt(ctc_file, dtype="int", ndmin=1)
        #targets = torch.IntTensor(targets)
        # read txt file
        with open(txt_file, 'r') as f:
            text = f.read()
        targets = _text_to_labels(self.labeler, text)
        targets = torch.IntTensor(targets)
        return tensors, targets, wav_file, text

    def __len__(self):
        return len(self.entries)


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


class PredictDataset(NonSplitDataset):

    def __init__(self, wav_files, *args, **kwargs):
        super().__init__(noise=True, noise_range=(-20, -20), *args, **kwargs)
        self.entries = wav_files

    def __getitem__(self, index):
        wav_file = self.entries[index]
        # read and transform wav file
        if self.transform is not None:
            tensors = self.transform(wav_file)
        return tensors, wav_file

    def __len__(self):
        return len(self.entries)


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
