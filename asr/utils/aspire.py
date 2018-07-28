import sys
import argparse
from pathlib import Path
import subprocess as sp
import random

import numpy as np

from tqdm import tqdm
import torch

from ..utils.kaldi_io import smart_open, read_string, read_vec_int
from ..utils.logger import logger
from ..utils import params as p
from ..kaldi._path import KALDI_ROOT


"""
This recipe requires Kaldi's egs/aspire/s5 recipe directory containing the result
of its own scripts, especially the data/ and the exp/
"""

KALDI_PATH = Path(KALDI_ROOT).resolve()
RECIPE_PATH = Path(KALDI_PATH, "egs/aspire/ics").resolve()
DATA_PATH = Path(__file__).parents[2].joinpath("data", "aspire").resolve()

assert KALDI_PATH.exists(), f"no such path \"{str(KALDI_PATH)}\" not found"
assert RECIPE_PATH.exists(), f"no such path \"{str(RECIPE_PATH)}\" not found"

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

    data_dir = Path(RECIPE_PATH, "data", mode).resolve()
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
    data_dir = Path(RECIPE_PATH, "data", mode).resolve()
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

    exp_dir = Path(RECIPE_PATH, "exp", "tri5a").resolve()
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
    # load labels.txt
    labels = dict()
    with open('asr/kaldi/graph/labels.txt', 'r') as f:
        for line in f:
            splits = line.strip().split()
            label = splits[0]
            labels[label] = splits[1]
    blank = labels['<blk>']
    # find *.phn files
    logger.info(f"finding *.phn files under {target_dir}")
    phn_files = [str(x) for x in Path(target_dir).rglob("*.phn")]
    # convert
    ctcs_count = [0] * len(labels)
    for phn_file in tqdm(phn_files):
        phns = np.loadtxt(phn_file, dtype="int", ndmin=1)
        # make ctc labelings by removing duplications
        ctcs = np.array([x for x in remove_duplicates(phns)])
        # write ctc file
        # blank labels will be inserted in warp-ctc loss module,
        # so here the target labels have not to contain the blanks interleaved
        ctc_file = phn_file.replace("phn", "ctc")
        np.savetxt(str(ctc_file), ctcs, "%d")
        # add blanks and count labels for priors
        # assume no blank label in the original ctcs
        for c in ctcs:
            ctcs_count[int(c)] += 1
        ctcs_count[int(blank)] += len(ctcs) + 1
    # write ctc count file
    ctcs_count_file = Path(target_dir).joinpath("ctc_count.txt")
    np.savetxt(str(ctcs_count_file), ctcs_count, "%d")


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
    parser.add_argument('--ctc-only', default=False, action='store_true', help="generate ctc symbols only instead of the overall processing")
    parser.add_argument('--path', default=None, type=str, help="path to store the processed data")
    args = parser.parse_args(argv)

    assert args.path is not None

    if args.manifest_only:
        reconstruct_manifest(args.path)
    elif args.ctc_only:
        make_ctc_labels(args.path)
    else:
        process(args.path)


if __name__ == "__main__":
    pass
