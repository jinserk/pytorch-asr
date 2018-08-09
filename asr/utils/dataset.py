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
from torch._C import _set_worker_signal_handlers
from torch.utils.data import Dataset
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

    def __init__(self, frame_margin, unit_frames):
        self.frame_margin = frame_margin
        self.unit_frames = unit_frames
        self.half = (self.unit_frames - 1) // 2
        assert unit_frames % 2 == 1, "unit_frames should be odd integer"
        assert frame_margin >= 0, "frame_margin should be >= 0"
        assert self.half <= frame_margin, "frame_margin is too small for the unit_frames"

    def __call__(self, tensor):
        with torch.no_grad():
            bins = [(x - self.half, self.unit_frames) for x in
                    range(self.frame_margin, tensor.size(2) - self.frame_margin)]
            frames = [tensor.narrow(2, s, l).clone() for s, l in bins]
            #c, w, h = frames[0].shape
            #frames = [x.view(c * w * h) for x in frames]
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
                 frame_margin=p.FRAME_MARGIN, unit_frames=p.HEIGHT, *args, **kwargs):
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



def _load_manifest(data_root, mode, data_size, min_len=0., max_len=100.):
    manifest_file = data_root.joinpath(f"{mode}.csv")
    if not data_root.exists() or not manifest_file.exists():
        logger.error(f"no such path {data_root} or manifest file {manifest_file} found. "
                     f"need to prepare data first.")
        sys.exit(1)

    logger.info(f"loading dataset manifest {str(manifest_file)} ...")
    with open(manifest_file, "r") as f:
        manifest = f.readlines()
    entries = [tuple(x.strip().split(',')) for x in manifest]

    def _smp2frm(samples):
        WIN_SAMP_SIZE = p.SAMPLE_RATE * p.WINDOW_SIZE
        WIN_SAMP_SHIFT = p.SAMPLE_RATE * p.WINDOW_SHIFT
        SAMPLE_MARGIN = WIN_SAMP_SHIFT * p.FRAME_MARGIN  # samples
        num_samples = samples - 2 * SAMPLE_MARGIN
        return int((num_samples - WIN_SAMP_SIZE) // WIN_SAMP_SHIFT + 1)

    # pick up entries of 1 to 15 secs
    MIN_FRAME = min_len / p.WINDOW_SHIFT
    MAX_FRAME = max_len / p.WINDOW_SHIFT
    entries = [e for e in entries if MIN_FRAME < _smp2frm(int(e[2])) < MAX_FRAME ]
    # randomly choose a number of data_size
    size = min(data_size, len(entries)) if data_size > 0 else len(entries)
    selected_entries = random.sample(entries, size)
    # count each entry's number of frames
    entry_frames = [_smp2frm(int(e[2])) for e in selected_entries]
    logger.info(f"{len(selected_entries)} entries, {sum(entry_frames)} frames are loaded.")
    return selected_entries, entry_frames


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


class AudioSplitDataset(SplitDataset):

    def __init__(self, root, mode, data_size=1e30, min_len=1., max_len=15., *args, **kwargs):
        self.root = Path(root).resolve()
        self.mode = mode
        self.data_size = data_size
        super().__init__(*args, **kwargs)
        self.entries, self.entry_frames = _load_manifest(self.root, mode, data_size, min_len, max_len)
        self.frame_map = list()
        for i, (frames) in enumerate(self.entry_frames):
            self.frame_map.extend([(i, f) for f in range(frames)])

    def __getitem__(self, index):
        ei, fi = self.frame_map[index]
        uttid, wav_file, samples, txt_file = self.entries[ei]
        # read and transform wav file
        if self.transform is not None:
            tensors = self.transform(wav_file)
        tensor = tensors[fi]
        if self.mode == "train_unsup":
            return tensor, None
        else:
            # read phn file
            targets = np.loadtxt(phn_file, dtype="int").tolist()
            if self.target_transform is not None:
                targets = self.target_transform(targets)
            # manipulating when the length of data and targets are mismatched
            #l0, l1 = len(tensors), len(targets)
            #if l0 > l1:
            #    tensors = tensors[:l1]
            #elif l0 < l1:
            #    tensors.extend([torch.zeros_like(tensors[0]) for i in range(l1 - l0)])
            target = targets[fi]
            return tensor, target

    def __len__(self):
        return len(self.frame_map)


class AudioCTCDataset(NonSplitDataset):

    def __init__(self, labeler, root, mode, data_size=1e30, min_len=1., max_len=15., *args, **kwargs):
        self.labeler = labeler
        self.root = Path(root).resolve()
        self.mode = mode
        self.data_size = data_size
        super().__init__(tempo=True, gain=True, noise=True, *args, **kwargs)
        self.entries, self.entry_frames = _load_manifest(self.root, mode, data_size, min_len, max_len)

    def __getitem__(self, index):
        uttid, wav_file, samples, txt_file = self.entries[index]
        # read and transform wav file
        if self.transform is not None:
            tensors = self.transform(wav_file)
        if self.mode == "train_unsup":
            return tensors, None
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


class AudioCEDataset(NonSplitDataset):

    def __init__(self, root, mode, data_size=1e30, min_len=1., max_len=15., *args, **kwargs):
        self.root = Path(root).resolve()
        self.mode = mode
        self.data_size = data_size
        super().__init__(tempo=False, gain=True, noise=True, *args, **kwargs)
        self.entries, self.entry_frames = _load_manifest(self.root, mode, data_size, min_len, max_len)

    def __getitem__(self, index):
        uttid, wav_file, samples, phn_file, num_phns, txt_file = self.entries[index]
        # read and transform wav file
        if self.transform is not None:
            tensors = self.transform(wav_file)
        if self.mode == "train_unsup":
            return tensors, None
        # read phn file
        targets = np.loadtxt(phn_file, dtype="int", ndmin=1)
        targets = torch.IntTensor(targets)
        targets_len = len(targets)
        start = (tensors.size(2) - targets_len) // 2
        # read txt file
        with open(txt_file, 'r') as f:
            text = f.read()
        return tensors[:, :, start:start+targets_len], targets, wav_file, text

    def __len__(self):
        return len(self.entries)


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
