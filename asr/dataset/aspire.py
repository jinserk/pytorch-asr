import sys
import argparse
from pathlib import Path
import subprocess as sp
import random

import numpy as np

from tqdm import tqdm
import torch

from ..utils.audio import AudioDataset, AudioCTCDataset
from ..utils.kaldi_io import smart_open, read_string, read_vec_int
from ..utils.logger import logger
from ..utils import params as p
from ..kaldi._path import KALDI_ROOT


"""
This recipe requires Kaldi's egs/aspire/s5 recipe directory containing the result
of its own scripts, especially the data/ and the exp/
"""

KALDI_PATH = Path(KALDI_ROOT).resolve()
ASPIRE_PATH = Path(KALDI_PATH, "egs/aspire/ics").resolve()
DATA_PATH = Path(__file__).parents[2].joinpath("data", "aspire").resolve()

assert KALDI_PATH.exists(), f"no such path \"{str(KALDI_PATH)}\" not found"
assert ASPIRE_PATH.exists(), f"no such path \"{str(ASPIRE_PATH)}\" not found"

WIN_SAMP_SIZE = p.SAMPLE_RATE * p.WINDOW_SIZE
WIN_SAMP_SHIFT = p.SAMPLE_RATE * p.WINDOW_SHIFT
SAMPLE_MARGIN = WIN_SAMP_SHIFT * p.FRAME_MARGIN  # samples


def get_num_lines(filename):
    import mmap

    with open(filename, "r+") as f:
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
    return lines


def strip_text(text):
    mask = "abcdefghijklmnopqrstuvwxyz'- "
    stripped = [x for x in text.lower() if x in mask]
    return ''.join(stripped)


def split_wav(mode, target_dir):
    import io
    import wave

    data_dir = Path(ASPIRE_PATH, "data", mode).resolve()
    segments_file = Path(data_dir, "segments")
    logger.info(f"processing {str(segments_file)} file ...")
    segments = dict()
    with smart_open(segments_file, "r") as f:
        for line in tqdm(f, total=get_num_lines(segments_file)):
            split = line.strip().split()
            uttid, wavid, start, end = split[0], split[1], float(split[2]), float(split[3])
            if wavid in segments:
                segments[wavid].append((uttid, start, end))
            else:
                segments[wavid] = [(uttid, start, end)]

    wav_scp = Path(data_dir, "wav.scp")
    logger.info(f"processing {str(wav_scp)} file ...")
    manifest = dict()
    with smart_open(wav_scp, "r") as rf:
        for line in tqdm(rf, total=get_num_lines(wav_scp)):
            wavid, cmd = line.strip().split(" ", 1)
            if not wavid in segments:
                continue
            cmd = cmd.strip().rstrip(' |').split()
            p = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            fp = io.BytesIO(p.stdout)
            with wave.openfp(fp, "rb") as wav:
                fr = wav.getframerate()
                nf = wav.getnframes()
                for uttid, start, end in segments[wavid]:
                    fs, fe = int(fr * start - SAMPLE_MARGIN), int(fr * end + SAMPLE_MARGIN)
                    if fs < 0 or fe > nf:
                        continue
                    wav.rewind()
                    wav.setpos(fs)
                    signal = wav.readframes(fe - fs)
                    tar_dir = Path(target_dir).joinpath(uttid[6:9])
                    tar_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
                    wav_file = tar_dir.joinpath(uttid + ".wav")
                    with wave.open(str(wav_file), "wb") as wf:
                        wf.setparams(wav.getparams())
                        wf.writeframes(signal)
                    manifest[uttid] = (str(wav_file), fe - fs)
    return manifest


def get_transcripts(mode, target_dir):
    data_dir = Path(ASPIRE_PATH, "data", mode).resolve()
    texts_file = Path(data_dir, "text")
    logger.info(f"processing {str(texts_file)} file ...")
    manifest = dict()
    with smart_open(Path(data_dir, "text"), "r") as f:
        for line in tqdm(f, total=get_num_lines(texts_file)):
            uttid, text = line.strip().split(" ", 1)
            text = strip_text(text)
            tar_dir = Path(target_dir).joinpath(uttid[6:9])
            Path(tar_dir).mkdir(mode=0o755, parents=True, exist_ok=True)
            txt_file = tar_dir.joinpath(uttid + ".txt")
            with open(str(txt_file), "w") as txt:
                txt.write(text + "\n")
            manifest[uttid] = (str(txt_file), text)
    return manifest


def get_alignments(target_dir):
    import io
    import pipes
    import gzip

    exp_dir = Path(ASPIRE_PATH, "exp", "tri5a").resolve()
    models = exp_dir.glob("*.mdl")
    model = sorted(models, key=lambda x: x.stat().st_mtime)[-1]

    logger.info("processing alignment files ...")
    logger.info(f"using the trained kaldi model: {model}")
    manifest = dict()
    alis = [x for x in exp_dir.glob("ali.*.gz")]
    for ali in tqdm(alis):
        cmd = [ str(Path(KALDI_PATH, "src", "bin", "ali-to-phones")),
                "--per-frame", f"{model}", f"ark:-", f"ark,f:-" ]
        with gzip.GzipFile(ali, "rb") as a:
            p = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, input=a.read())
            with io.BytesIO(p.stdout) as f:
                while True:
                    # mkdir
                    try:
                        uttid = read_string(f)
                    except ValueError:
                        break
                    tar_dir = Path(target_dir).joinpath(uttid[6:9])
                    tar_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
                    # store phn file
                    phn_file = tar_dir.joinpath(uttid + ".phn")
                    phones = read_vec_int(f)
                    np.savetxt(str(phn_file), phones, "%d")
                    # prepare manifest elements
                    num_frms = len(phones)
                    manifest[uttid] = (str(phn_file), num_frms)
    return manifest


def make_ctc_labels(target_dir):
    def remove_duplicates(labels):
        p = -1
        for x in labels:
            if p != x:
                p = x
                yield x
    phn_files = [str(x) for x in Path(target_dir).rglob("*.phn")]
    for phn_file in tqdm(phn_files):
        phn_sbls = np.loadtxt(phn_file, dtype="int", ndmin=1)
        # make ctc labelings by removing duplications
        ctc_file = phn_file.replace("phn", "ctc")
        ctc_sbls = np.array([x for x in remove_duplicates(phn_sbls)])
        np.savetxt(str(ctc_file), ctc_sbls, "%d")


def process(target_dir=None):
    """
    since the target time-alignment exists only on the train set,
    we split the train set into train and dev set
    """
    if target_dir is None:
        target_path = DATA_PATH
    else:
        target_path = Path(target_dir).resolve()
    logger.info(f"target data path : {target_path}")

    train_wav_manifest = split_wav("train", target_path)
    train_txt_manifest = get_transcripts("train", target_path)
    phn_manifest = get_alignments(target_path)
    make_ctc_labels(target_path)

    logger.info("generating manifest files ...")
    with open(Path(target_path, "train.csv"), "w") as f1:
        with open(Path(target_path, "dev.csv"), "w") as f2:
            for k, v in train_wav_manifest.items():
                if not k in train_txt_manifest:
                    continue
                if not k in phn_manifest:
                    continue
                wav_file, samples = v
                txt_file, _ = train_txt_manifest[k]
                phn_file, num_frms = phn_manifest[k]
                if 0 < int(k[6:11]) < 60:
                    f2.write(f"{k},{wav_file},{samples},{phn_file},{num_frms},{txt_file}\n")
                else:
                    f1.write(f"{k},{wav_file},{samples},{phn_file},{num_frms},{txt_file}\n")
    logger.info("data preparation finished.")


def reconstruct_manifest(target_dir):
    import wave

    logger.info("reconstructing manifest files ...")
    target_path = Path(target_dir).resolve()
    with open(target_path.joinpath("train.csv"), "w") as f1:
        with open(target_path.joinpath("dev.csv"), "w") as f2:
            for wav_file in target_path.rglob("*.wav"):
                uttid = wav_file.stem
                txt_file = str(wav_file).replace("wav", "txt")
                phn_file = str(wav_file).replace("wav", "phn")
                if not Path(txt_file).exists() or not Path(phn_file).exists():
                    continue
                with wave.openfp(str(wav_file), "rb") as wav:
                    samples = wav.getnframes()
                num_frms = get_num_lines(phn_file)
                if 0 < int(uttid[6:11]) < 60:
                    f2.write(f"{uttid},{wav_file},{samples},{phn_file},{num_frms},{txt_file}\n")
                else:
                    f1.write(f"{uttid},{wav_file},{samples},{phn_file},{num_frms},{txt_file}\n")
    logger.info("data preparation finished.")


def prepare(argv):
    parser = argparse.ArgumentParser(description="Prepare ASpIRE dataset")
    parser.add_argument('--manifest-only', default=False, action='store_true', help="if you want to reconstruct manifest only instead of the overall processing")
    parser.add_argument('--path', default=None, type=str, help="path to store the processed data")
    args = parser.parse_args(argv)

    if args.manifest_only:
        reconstruct_manifest(args.path)
    else:
        process(args.path)



MODES = ["train_sup", "train_unsup", "train", "dev", "test"]


def load_manifest(data_root, mode, data_size, min_len=1, max_len=15):
    assert mode in MODES, f"invalid mode options: either one of {MODES}"
    manifest_file = data_root / f"{mode}.csv"
    if not data_root.exists() or not manifest_file.exists():
        logger.error(f"no such path {data_root} or manifest file {manifest_file} found. "
                     f"need to run 'python prepare.py aspire' first")
        sys.exit(1)

    logger.info(f"loading dataset manifest {manifest_file} ...")
    with open(manifest_file, "r") as f:
        manifest = f.readlines()
    entries = [tuple(x.strip().split(',')) for x in manifest]

    def _smp2frm(samples):
        num_samples = samples - 2 * SAMPLE_MARGIN
        return int((num_samples - WIN_SAMP_SIZE) // WIN_SAMP_SHIFT + 1)

    # pick up entries of 1 to 15 secs
    MIN_FRAME = min_len / p.WINDOW_SHIFT
    MAX_FRAME = max_len / p.WINDOW_SHIFT
    entries = [e for e in entries if _smp2frm(int(e[2])) > MIN_FRAME and _smp2frm(int(e[2])) < MAX_FRAME ]
    # randomly choose a number of data_size
    size = min(data_size, len(entries))
    selected_entries = random.sample(entries, size)
    # count each entry's number of frames
    if mode == "train_unsup":
        entry_frames = [_smp2frm(int(e[2])) for e in selected_entries]
    else:
        entry_frames = [int(e[4]) for e in selected_entries]

    logger.info(f"{len(selected_entries)} entries, {sum(entry_frames)} frames are loaded.")
    return selected_entries, entry_frames


class AspireDataset(AudioDataset):
    """
    Kaldi's ASpIRE recipe (LDC Fisher dataset) for frame based training
    loading Kaldi's frame-aligned phones target and the corresponding audio files
    """
    root = DATA_PATH
    entries = list()
    entry_frames = list()

    def __init__(self, root=None, mode=None, data_size=1e30, *args, **kwargs):
        self.mode = mode
        self.data_size = data_size
        if root is not None:
            self.root = Path(root).resolve()
        super().__init__(frame_margin=p.FRAME_MARGIN, unit_frames=p.HEIGHT,
                         window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE,
                         *args, **kwargs)
        self.entries, self.entry_frames = load_manifest(self.root, mode, data_size)

    def __getitem__(self, index):
        uttid, wav_file, samples, phn_file, num_phns, txt_file = self.entries[index]
        # read and transform wav file
        if self.transform is not None:
            tensors = self.transform(wav_file)
        if self.mode == "train_unsup":
            return tensors, None
        # read phn file
        targets = np.loadtxt(phn_file, dtype="int").tolist()
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        # manipulating when the length of data and targets are mismatched
        l0, l1 = len(tensors), len(targets)
        if l0 > l1:
            tensors = tensors[:l1]
        elif l0 < l1:
            tensors.extend([torch.zeros_like(tensors[0]) for i in range(l1 - l0)])
        return tensors, targets

    def __len__(self):
        return len(self.entries)


class AspireCTCDataset(AudioCTCDataset):
    """
    Kaldi's ASpIRE recipe (LDC Fisher dataset) for frame based training
    loading CTC blank-inserted token target and the corresponding audio files
    """
    root = DATA_PATH
    entries = list()
    entry_frames = list()

    def __init__(self, root=None, mode=None, data_size=1e30, *args, **kwargs):
        self.mode = mode
        self.data_size = data_size
        if root is not None:
            self.root = Path(root).resolve()
        super().__init__(frame_margin=p.FRAME_MARGIN, unit_frames=p.HEIGHT,
                         window_shift=p.WINDOW_SHIFT, window_size=p.WINDOW_SIZE,
                         *args, **kwargs)
        self.entries, self.entry_frames = load_manifest(self.root, mode, data_size)

    def __getitem__(self, index):
        uttid, wav_file, samples, phn_file, num_phns, txt_file = self.entries[index]
        ctc_file = phn_file.replace('phn', 'ctc')
        # read and transform wav file
        if self.transform is not None:
            tensors = self.transform(wav_file)
        if self.mode == "train_unsup":
            return tensors, None
        # read phn file
        targets = np.loadtxt(ctc_file, dtype="int", ndmin=1)
        targets = torch.IntTensor(targets)
        return tensors, targets, ctc_file

    def __len__(self):
        return len(self.entries)


class AspireEdDataset(AspireCTCDataset):

    def __init__(self, *args, **kwargs):
        if 'tempo' in kwargs:
            kwargs['tempo'] = False
        super().__init__(*args, **kwargs)

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
        return tensors[:, :, start:start+targets_len], targets, phn_file


def test_plot():
    from ..util.audio import AudioDataLoader, AudioCTCDataLoader
    train_dataset = Aspire(mode="test")
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
    pass
