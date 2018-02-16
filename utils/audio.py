import os
import sys
import threading
import collections
from pathlib import Path
import tempfile as tmp

import numpy as np
import scipy as sp

import torch
import torch.multiprocessing as multiprocessing
from torch._C import _set_worker_signal_handlers, _update_worker_pids, _remove_worker_pids
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _worker_manager_loop, _set_SIGCHLD_handler, \
                                        pin_memory_batch, ExceptionWrapper
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import torchaudio

import sox

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


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
                 gain=False, gain_range=(-6., 8.),
                 noise=False, noise_range=(-30, -10)):
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
            sr, wav = sp.io.wavfile.read(tmp_file)
            os.unlink(tmp_file)
        else:
            Path(tar_file).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            tfm.build(str(wav_file), str(tar_file))
            sr, wav = sp.io.wavfile.read(tar_file)

        if self.noise:
            noise = np.random.normal(0, 1, wav.shape)  # TODO: noise range?
            wav += noise
        return wav


# transformer: spectrogram
class Spectrogram(object):

    def __init__(self, sample_rate=8000, window_shift=0.01, window_size=0.025,
                 window=sp.signal.tukey, nfft=256, normalize=True):
        self.sample_rate = sample_rate
        self.window_shift = window_shift
        self.window_size = window_size
        self.window = window
        self.nfft = nfft
        self.normalize = normalize

    def __call__(self, data):
        nperseg = int(self.sample_rate * self.window_size)
        noverlap = int(self.sample_rate * (self.window_size - self.window_shift))
        window = self.window(nperseg)
        # STFT
        bins, frames, z = sp.signal.stft(data, nfft=self.nfft, nperseg=nperseg, noverlap=noverlap,
                                         window=window, boundary=None, padded=False)
        mag, ang = np.abs(z), np.angle(z)
        spect = torch.FloatTensor(np.log1p(mag))
        phase = torch.FloatTensor(ang)
        # TODO: is this normalization correct?
        if self.normalize:
            m, s = spect.mean(), spect.std()
            spect.sub_(m)
            spect.div_(s)
            phase.div_(np.pi)
        # {mag, phase} x n_freq_bin x n_frame
        data = torch.cat([spect.unsqueeze_(0), phase.unsqueeze_(0)], 0)
        return data


# transformer: frame splitter
class FrameSplitter(object):

    def __init__(self, frame_margin=10, unit_frames=21):
        self.frame_margin = frame_margin
        self.unit_frames = unit_frames
        self.half = (self.unit_frames - 1) // 2
        assert unit_frames % 2 == 1, "unit_frames should be odd integer"
        assert frame_margin >= 0, "frame_margin should be >= 0"
        assert self.half <= frame_margin, "frame_margin is too small for the unit_frames"

    def __call__(self, tensor):
        bins = [(x - self.half, self.unit_frames) for x in
                range(self.frame_margin, tensor.size(2) - self.frame_margin)]
        frames = [tensor.narrow(2, s, l).clone() for s, l in bins]
        c, w, h = frames[0].shape
        frames = [x.view(c * w * h) for x in frames]
        return frames


# transformer: convert int to one-hot vector
class Int2OneHot(object):

    def __init__(self, num_labels):
        self.num_labels = num_labels

    def __call__(self, targets):
        one_hots = list()
        for t in targets:
            one_hot = torch.FloatTensor(self.num_labels).zero_()
            one_hot[t] = 1.
            one_hots.append(one_hot)
        return one_hots


class AudioDataset(Dataset):

    def __init__(self,
                 transform=None, target_transform=None,
                 resample=False, sample_rate=8000,
                 tempo=False, tempo_range=(0.85, 1.15),
                 gain=False, gain_range=(-6., 8.),
                 window_shift=0.01, window_size=0.025,
                 window=sp.signal.tukey, nfft=256, normalize=True,
                 frame_margin=0, unit_frames=21, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if transform is None:
            self.transform = torchaudio.transforms.Compose([
                Augment(resample=resample, sample_rate=sample_rate,
                        tempo=tempo, tempo_range=tempo_range,
                        gain=gain, gain_range=gain_range),
                Spectrogram(sample_rate=sample_rate, window_shift=window_shift,
                            window_size=window_size, window=window, nfft=nfft,
                            normalize=normalize),
                FrameSplitter(frame_margin=frame_margin, unit_frames=unit_frames),
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform


class AudioBatchSampler(BatchSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = self.sampler.data_source.entry_frames

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            for fidx in iter(torch.randperm(self.frames[idx]).long()):
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
            return (total // self.batch_size) + 1


class AudioCollateFn(object):

    def __call__(self, dataset, indices):
        tensors, targets = list(), list()
        pwidx = -1
        for widx, fidx in indices:
            if pwidx != widx:
                tensor, target = dataset[widx]
            tensors.append(tensor[fidx])
            if target is not None:
                targets.append(target[fidx])
            pwidx = widx
        if targets:
            batch = (torch.stack(tensors), torch.stack(targets))
        else:
            batch = torch.stack(tensors)
        return batch


def _worker_loop(dataset, index_queue, data_queue, collate_fn, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True

    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    torch.manual_seed(seed)

    if init_fn is not None:
        init_fn(worker_id)

    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            samples = collate_fn(dataset, batch_indices)
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


class AudioDataLoaderIter(object):

    def __init__(self, loader, use_cuda=False):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()
        self.use_cuda = loader.use_cuda

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queue = multiprocessing.SimpleQueue()
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            base_seed = torch.LongTensor(1).random_()[0]
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.worker_result_queue, self.collate_fn,
                          base_seed + i, self.worker_init_fn, i))
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_worker_manager_loop,
                    args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                          maybe_device_id))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def _get_batch(self):
        if self.timeout > 0:
            try:
                return self.data_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        else:
            return self.data_queue.get()

    def __next__(self):
        if self.num_workers == 0:
            batch = self._next_without_worker()
        else:
            batch = self._next_with_worker()

        if self.use_cuda:
            if isinstance(batch, collections.Sequence):
                return (x.cuda() for x in batch)
            else:
                return batch.cuda()
        else:
            return batch

    def _next_without_worker(self):
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = self.collate_fn(self.dataset, indices)
        if self.pin_memory:
            batch = pin_memory_batch(batch)
        return batch

    def _next_with_worker(self):
        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        try:
            if not self.shutdown:
                self.shutdown = True
                self.done_event.set()
                # if worker_manager_thread is waiting to put
                while not self.data_queue.empty():
                    self.data_queue.get()
                for _ in self.workers:
                    self.index_queue.put(None)
                # done_event should be sufficient to exit worker_manager_thread,
                # but be safe here and put another None
                self.worker_result_queue.put(None)
        finally:
            # removes pids no matter what
            if self.worker_pids_set:
                _remove_worker_pids(id(self))
                self.worker_pids_set = False

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class AudioDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, drop_last=False, pin_memory=True, use_cuda=False,
                 *args, **kwargs):
        collate_fn = AudioCollateFn()
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = AudioBatchSampler(sampler, batch_size, drop_last)
        self.use_cuda = use_cuda

        super().__init__(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                         collate_fn=collate_fn, pin_memory=pin_memory, *args, **kwargs)

    def __iter__(self):
        return AudioDataLoaderIter(self)


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
        nfft = 256
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
