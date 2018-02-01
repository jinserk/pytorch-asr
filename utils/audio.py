import os
from pathlib import Path
import tempfile as tmp

import numpy as np
import scipy as sp

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

import sox


AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_manifest(path):
    audios = []
    path = Path(path).resolve()
    walk = [x for x in path.glob("**/*") if x.is_file()]
    for f in walk:
        if is_audio_file(f):
            audios.append(f)
    return audios


# transformer: resampling and augmentation
class Augment(object):

    def __init__(self,
                 resample=False, sample_rate=8000,
                 tempo=False, tempo_range=(0.85, 1.15),
                 gain=False, gain_range=(-6., 8.)):
        self.resample = resample
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.tempo_range = tempo_range
        self.last_tempo = 1.
        self.gain = gain
        self.gain_range = gain_range

    def __call__(self, wav_file, tar_file=None):
        if not Path(wav_file).exists():
            raise IOError

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        if self.resample:
            tfm.rate(self.sample_rate)

        if self.tempo:
            tempo = np.random.uniform(*self.tempo_range)
            tfm.tempo(tempo, audio_type='s')
            self.last_tempo = tempo

        if self.gain:
            gain = np.random.uniform(*self.gain_range)
            tfm.gain(gain, normalize=True)

        if tar_file is None:
            tmp_dir = tmp._get_default_tempdir()
            tmp_name = next(tmp._get_candidate_names())
            tmp_file = Path(tmp_dir, tmp_name + ".wav")
            tfm.build(str(wav_file), str(tmp_file))
            audio, _ = torchaudio.load(tmp_file)
            os.unlink(tmp_file)
        else:
            Path(tar_file).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            tfm.build(str(wav_file), str(tar_file))
            audio, _ = torchaudio.load(tar_file)

        return audio


# transformer: spectrogram
class Spectrogram(object):

    def __init__(self, sample_rate=8000, window_shift=0.01, window_size=0.025,
                 window=sp.signal.tukey, nfft=512, normalize=False, max_samples=0):
        self.sample_rate = sample_rate
        self.window_shift = window_shift
        self.window_size = window_size
        self.window = window
        self.nfft = nfft
        self.normalize = normalize
        self.max_samples = max_samples

    def __call__(self, tensor):
        if len(tensor.shape) > 1:
            tensor = torch.squeeze(tensor)
        # add white noise as the length of max samples (will be croped when make mini-batch)
        if self.max_samples > 0:
            noise = torch.randn(self.max_samples)
            noise.narrow(0, 0, tensor.size(0)).add_(tensor)
            tensor = noise
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()

        nperseg = int(self.sample_rate * self.window_size)
        noverlap = int(self.sample_rate * (self.window_size - self.window_shift))
        window = self.window(nperseg)
        # STFT
        bins, frames, z = sp.signal.stft(tensor, nfft=self.nfft, nperseg=nperseg, noverlap=noverlap,
                                         window=window, boundary=None, padded=False)
        mag, ang = np.abs(z), np.angle(z)
        if self.normalize:
            mag -= mag.mean()
            mag /= mag.std()
            ang = ang / (2 * np.pi) + 0.5
        spect = torch.FloatTensor(np.log10(mag + 1))
        phase = torch.FloatTensor(ang)
        # {mag, phase} x n_freq_bin x n_frame
        data = torch.cat((torch.unsqueeze(spect, 0), torch.unsqueeze(phase, 0)), 0)
        return data


class AudioDataset(Dataset):
    """
    Dataset for audio signals with augmentation and spectrogram output
    """
    def __init__(self,
                 transform=None, target_transform=None,
                 resample=False, sample_rate=8000,
                 tempo=False, tempo_range=(0.85, 1.15),
                 gain=False, gain_range=(-6., 8.),
                 window_shift=0.01, window_size=0.025,
                 window=sp.signal.tukey, nfft=512,
                 normalize=False, max_samples=0, unit_frames=9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transform is None:
            self.transform = torchaudio.transforms.Compose([
                Augment(resample=resample, sample_rate=sample_rate,
                        tempo=tempo, tempo_range=tempo_range,
                        gain=gain, gain_range=gain_range),
                Spectrogram(sample_rate=sample_rate, window_shift=window_shift,
                            window_size=window_size, window=window,
                            nfft=nfft, normalize=normalize, max_samples=max_samples),
            ])
        self.target_transform = target_transform


def _collate_fn(batch):
    # batch = [(uttid, spectogram, phn_seq), ...]
    batch_size = len(batch)
    _, max_spect, max_tar= max(batch, key=lambda x: len(x[2]))
    chs, bins, max_frms = max_spect.size(0), max_spect.size(1), len(max_tar)

    inputs = torch.FloatTensor(batch_size, chs, bins, max_frms).zero_()
    targets = torch.IntTensor(batch_size, max_frms).zero_() + 1  # padded frame has sil phone (id 1)
    frame_sizes = torch.IntTensor(batch_size).zero_()
    for i in range(batch_size):
        uttid, tensor, target = batch[i]
        inputs[i].copy_(tensor.narrow(2, 0, max_frms))
        frame_sizes[i] = len(target)
        targets[i].narrow(0, 0, frame_sizes[i]).copy_(target)
    return inputs, targets, frame_sizes


class AudioDataLoader(DataLoader):
    """
    DataLoader for audio signals, to make mini-batches with different frame-length samples
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


if __name__ == "__main__":
    # test Augment
    if False:
        transformer = Augment(resample=True, sample_rate=16000)
        wav_file = Path("/home/jbaik/src/enf/stt/test/conan1-8k.wav")
        audio = transformer(wav_file)

    # test Spectrogram
    if True:
        import matplotlib
        matplotlib.use('TkAgg')
        matplotlib.interactive(True)
        import matplotlib.pyplot as plt

        sr = 8000
        nfft = 512
        window_size = 0.025
        window_stride = 0.01
        nperseg = int(sr * window_size)
        noverlap = int(sr * (window_size-window_stride))

        wav_file = Path("../data/aspire/000/fe_03_00047-A-025005-025135.wav")
        audio, _ = torchaudio.load(wav_file)

        # pyplot specgram
        audio = torch.squeeze(audio)
        fig = plt.figure(0)
        plt.specgram(audio, Fs=sr, NFFT=nfft, noverlap=noverlap, cmap='plasma')

        # implemented transformer - scipy stft
        transformer = Spectrogram(sample_rate=sr, window_stride=window_stride, window_size=window_size, nfft=nfft)
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
        f, t, z = sp.signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, nfft=nfft, mode='complex')
        spect, phase = np.abs(z), np.angle(z)
        fig = plt.figure(3)
        plt.pcolormesh(t, f, 20*np.log10(spect), cmap='plasma')
        fig = plt.figure(4)
        plt.pcolormesh(t, f, phase, cmap='plasma')

        plt.show(block=True)
        plt.close('all')
