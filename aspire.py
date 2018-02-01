import sys
from pathlib import Path
import subprocess as sp

import numpy as np

from tqdm import tqdm
import torch

from utils.audio import AudioDataset, AudioDataLoader
from utils.kaldi_io import smart_open, read_string, read_vec_int
from utils.logger import logger


"""
This recipe requires Kaldi's egs/aspire/s5 recipe directory containing the result
of its own scripts, especially the data/ and the exp/
"""

KALDI_ROOT = Path("/home/jbaik/kaldi").resolve()
ASPIRE_ROOT = Path(KALDI_ROOT, "egs/aspire/ics").resolve()
DATA_ROOT = (Path(__file__).parent / "data" / "aspire").resolve()


assert KALDI_ROOT.exists(), \
    f"no such path \"{str(KALDI_ROOT)}\" not found"
assert ASPIRE_ROOT.exists(), \
    f"no such path \"{str(ASPIRE_ROOT)}\" not found"


def get_num_lines(filename):
    import mmap

    with open(filename, "r+") as f:
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
    return lines


def strip_text(text):
    mask = "abcdefghijklmnopqrstuvwxyz'-"
    stripped = [x for x in text.lower() if x in mask]


def split_wav(mode, target_dir):
    import io
    import wave

    data_dir = Path(ASPIRE_ROOT, "data", mode).resolve()
    logger.info(f"processing {data_dir}/segments file ...")
    segments = dict()
    with smart_open(Path(data_dir, "segments"), "r") as f:
        for line in f:
            split = line.strip().split()
            uttid, wavid, start, end = split[0], split[1], float(split[2]), float(split[3])
            if wavid in segments:
                segments[wavid].append((uttid, start, end))
            else:
                segments[wavid] = [(uttid, start, end)]

    logger.info(f"processing {data_dir}/wav.scp file ...")
    manifest = dict()
    wav_scp = Path(data_dir, "wav.scp")
    with smart_open(wav_scp, "r") as f:
        for line in tqdm(f, total=get_num_lines(wav_scp)):
            wavid, cmd = line.strip().split(" ", 1)
            if not wavid in segments:
                continue
            cmd = cmd.strip().rstrip(' |').split()
            p = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            fp = io.BytesIO(p.stdout)
            with wave.openfp(fp, "rb") as wav:
                #signal = np.fromstring(wav.readframes(-1), dtype='int16')
                fr = wav.getframerate()
                for uttid, start, end in segments[wavid]:
                    fs, fe = int(fr * start), int(fr * end)
                    wav.rewind()
                    wav.setpos(fs)
                    signal = wav.readframes(fe - fs)
                    tar_dir = Path(target_dir) / uttid[6:9]
                    Path(tar_dir).mkdir(mode=0o755, parents=True, exist_ok=True)
                    wav_file = str(Path(tar_dir, uttid + ".wav"))
                    with wave.open(wav_file, "wb") as split_wav:
                        split_wav.setparams(wav.getparams())
                        split_wav.writeframes(signal)
                    manifest[uttid] = (wav_file, fe - fs)
    return manifest


def get_transcripts(mode, target_dir):
    data_dir = Path(ASPIRE_ROOT, "data", mode).resolve()
    logger.info(f"processing {data_dir}/text file ...")
    manifest = dict()
    with smart_open(Path(data_dir, "text"), "r") as f:
        for line in f:
            uttid, text = line.strip().split(" ", 1)
            text = strip_text(text)
            tar_dir = Path(target_dir) / uttid[6:9]
            Path(tar_dir).mkdir(mode=0o755, parents=True, exist_ok=True)
            txt_file = str(Path(tar_dir, uttid + ".txt"))
            with open(txt_file, "w") as txt:
                txt.write(text + "\n")
            manifest[uttid] = (txt_file, text)
    return manifest


def get_alignments(target_dir):
    import io
    import pipes
    import gzip

    exp_dir = Path(ASPIRE_ROOT, "exp", "tri5a").resolve()
    models = exp_dir.glob("*.mdl")
    model = sorted(models, key=lambda x: x.stat().st_mtime)[-1]

    logger.info("processing alignment files ...")
    manifest = dict()
    alis = [x for x in exp_dir.glob("ali.*.gz")]
    for ali in tqdm(alis):
        cmd = [
            str(Path(KALDI_ROOT, "src", "bin", "ali-to-phones")),
            "--per-frame",
            f"{model}",
            f"ark:-",
            f"ark,f:-",
        ]
        with gzip.GzipFile(ali, "rb") as a:
            p = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, input=a.read())
            with io.BytesIO(p.stdout) as f:
                while True:
                    try:
                        uttid = read_string(f)
                    except ValueError:
                        break
                    phones = read_vec_int(f)
                    num_phns = len(phones)
                    tar_dir = Path(target_dir) / uttid[6:9]
                    Path(tar_dir).mkdir(mode=0o755, parents=True, exist_ok=True)
                    phn_file = str(Path(tar_dir, uttid + ".phn"))
                    np.savetxt(phn_file, phones, "%d")
                    manifest[uttid] = (phn_file, num_phns, phones)
    return manifest


def prepare_data(target_dir):
    """
    since the target time-alignment exists only on the train set,
    we split the train set into train and dev set
    """
    train_wav_manifest = split_wav("train", DATA_ROOT)
    train_txt_manifest = get_transcripts("train", DATA_ROOT)
    phn_manifest = get_alignments(DATA_ROOT)

    logger.info("generating manifest files ...")
    with open(Path(target_dir, "train.csv"), "w") as f1:
        with open(Path(target_dir, "dev.csv"), "w") as f2:
            for k, v in train_wav_manifest.items():
                if not k in train_txt_manifest:
                    continue
                if not k in phn_manifest:
                    continue
                wav_file, samples = v
                txt_file, _ = train_txt_manifest[k]
                phn_file, num_phns, _ = phn_manifest[k]
                if 0 < int(k[6:11]) < 60:
                    f2.write(f"{k},{wav_file},{samples},{phn_file},{num_phns},{txt_file}\n")
                else:
                    f1.write(f"{k},{wav_file},{samples},{phn_file},{num_phns},{txt_file}\n")
    logger.info("data preparation finished.")


class Aspire(AudioDataset):
    """Kaldi's ASpIRE recipe (LDC Fisher dataset)
       loading Kaldi's frame-aligned phones target and the corresponding audio files

    Args:
        mode (str): either one of "train", "dev", or "test"
        data_dir (path): dir containing the processed data and manifests
    """
    root = DATA_ROOT
    manifest = list()
    size = 0

    def __init__(self, root=None, mode=None, unit_frames=9, *args, **kwargs):
        assert mode in ["train", "dev", "test"], \
            "invalid mode options: either one of \"train\", \"dev\", or \"test\""
        self.mode = mode
        if root is not None:
            self.root = Path(root).resolve()

        self._load_manifest()
        self._split_frame()
        super().__init__(max_samples=int(max_len_entry[2]), unit_frames=9, *args, **kwargs)

    def __getitem__(self, index):
        uttid, wav_file, samples, txt_file, phn_file = self.manifest[index]
        # read and transform wav file
        if self.transform is not None:
            audio = self.transform(wav_file)
        # read phn file
        target = torch.IntTensor(np.loadtxt(phn_file, dtype="int"))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return uttid, audio, target

    def __len__(self):
        return self.size

    def _load_manifest(self):
        manifest_file = self.root / f"{self.mode}.csv"
        if not self.root.exists() or not manifest_file.exists():
            logger.error(f"no such path {self.root} or manifest file {manifest_file} found. "
                         f"need to run 'python {__file__}' first")
            sys.exit(1)

        logger.info(f"loading dataset manifest {manifest_file} ...")
        with open(manifest_file, "r") as f:
            manifest = f.readlines()
        self.manifest = [tuple(x.strip().split(',')) for x in manifest]
        self.size = len(self.manifest)
        logger.info(f"{self.size} entries are loaded.")

    def _split_frames(self):



if __name__ == "__main__":
    #prepare_data(DATA_ROOT)
    train_dataset = Aspire(mode="dev")
    loader = AudioDataLoader(train_dataset, batch_size=10, shuffle=True)

    temp = iter(loader)
    tensors, targets, sizes = next(temp)

    import matplotlib
    matplotlib.use('TkAgg')
    matplotlib.interactive(True)
    import matplotlib.pyplot as plt

    for i in range(tensors.size(0)):
        fig = plt.figure(1)
        t = np.arange(0, tensors.size(3)) / 8000
        f = np.linspace(0, 4000, tensors.size(2))
        p = plt.pcolormesh(t, f, np.log10(10 ** tensors[i][0] - 1), cmap='plasma')
        plt.colorbar(p)
        plt.show(block=True)

    plt.close('all')
