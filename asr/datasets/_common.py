import sys
import argparse
from pathlib import Path
import subprocess as sp
import random

import numpy as np

from tqdm import tqdm
import torch

from asr.utils.kaldi_io import smart_open, read_string, read_vec_int
from asr.utils.logger import logger
from asr.utils.misc import get_num_lines, remove_duplicates
from asr.utils import params
from asr.kaldi._path import KALDI_ROOT


"""
This recipe requires Kaldi's recipe directory containing the result
of its own scripts, especially the data/ directory
"""

KALDI_PATH = Path(KALDI_ROOT).resolve()
SPH2PIPE_PATH = KALDI_PATH.joinpath("tools", "sph2pipe_v2.5", "sph2pipe")

assert KALDI_PATH.exists(), f"no such path \"{str(KALDI_PATH)}\" found"
assert SPH2PIPE_PATH.exists(), f"no such path \"{str(SPH2PIPE_PATH)}\" found"

WIN_SAMP_SIZE = params.SAMPLE_RATE * params.WINDOW_SIZE
WIN_SAMP_SHIFT = params.SAMPLE_RATE * params.WINDOW_SHIFT
#SAMPLE_MARGIN = WIN_SAMP_SHIFT * params.FRAME_MARGIN  # samples
SAMPLE_MARGIN = 0


class KaldiDataImporter:

    def __init__(self, recipe_dir, target_dir):
        self.recipe_path = Path(recipe_dir).resolve()
        self.target_path = Path(target_dir).resolve()

    def split_wav(self, mode):
        import io
        import wave
        segments_file = self.recipe_path.joinpath("data", mode, "segments")
        logger.info(f"processing {str(segments_file)} file ...")
        segments = dict()
        with smart_open(segments_file, "r") as f:
            for line in tqdm(f, total=get_num_lines(segments_file), ncols=params.NCOLS):
                split = line.strip().split()
                uttid, wavid, start, end = split[0], split[1], float(split[2]), float(split[3])
                if wavid in segments:
                    segments[wavid].append((uttid, start, end))
                else:
                    segments[wavid] = [(uttid, start, end)]

        wav_scp = self.recipe_path.joinpath("data", mode, "wav.scp")
        logger.info(f"processing {str(wav_scp)} file ...")
        manifest = dict()
        with smart_open(wav_scp, "r") as rf:
            for line in tqdm(rf, total=get_num_lines(wav_scp), ncols=params.NCOLS):
                wavid, cmd = line.strip().split(" ", 1)
                if not wavid in segments:
                    continue
                cmd = cmd.strip().rstrip(' |').split()
                if cmd[0] == 'sph2pipe':
                    cmd[0] = str(SPH2PIPE_PATH)
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
                        p = uttid.find('-')
                        if p != -1:
                            tar_path = self.target_path.joinpath(mode, uttid[:p])
                        else:
                            tar_path = self.target_path.joinpath(mode)
                        tar_path.mkdir(mode=0o755, parents=True, exist_ok=True)
                        wav_file = tar_path.joinpath(uttid + ".wav")
                        with wave.open(str(wav_file), "wb") as wf:
                            wf.setparams(wav.getparams())
                            wf.writeframes(signal)
                        manifest[uttid] = (str(wav_file), fe - fs)
        return manifest

    def strip_text(self, text):
        """ default is dump function. will be overrided by each dataset """
        return text

    def get_transcripts(self, mode):
        texts_file = self.recipe_path.joinpath("data", mode, "text")
        logger.info(f"processing {str(texts_file)} file ...")
        manifest = dict()
        with smart_open(texts_file, "r") as f:
            with open(self.target_path.joinpath(f"{mode}_convert.txt"), "w") as wf:
                for line in tqdm(f, total=get_num_lines(texts_file), ncols=params.NCOLS):
                    try:
                        uttid, text = line.strip().split(" ", 1)
                        managed_text = self.strip_text(text)
                        if len(managed_text) == 0:
                            continue
                        if text != managed_text:
                            wf.write(f"{uttid} 0: {text}\n")
                            wf.write(f"{uttid} 1: {managed_text}\n\n")
                    except:
                        continue
                    p = uttid.find('-')
                    if p != -1:
                        tar_path = self.target_path.joinpath(mode, uttid[:p])
                    else:
                        tar_path = self.target_path.joinpath(mode)
                    tar_path.mkdir(mode=0o755, parents=True, exist_ok=True)
                    txt_file = tar_path.joinpath(uttid + ".txt")
                    with open(str(txt_file), "w") as txt:
                        txt.write(managed_text + "\n")
                    manifest[uttid] = (str(txt_file), managed_text)
        return manifest


    def get_alignments(self):
        import io
        import pipes
        import gzip

        exp_dir = self.recipe_path.joinpath("exp", "tri5a").resolve()
        models = exp_dir.glob("*.mdl")
        model = sorted(models, key=lambda x: x.stat().st_mtime)[-1]

        logger.info("processing alignment files ...")
        logger.info(f"using the trained kaldi model: {model}")
        manifest = dict()
        alis = [x for x in exp_dir.glob("ali.*.gz")]
        for ali in tqdm(alis, ncols=params.NCOLS):
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
                        p = uttid.find('-')
                        if p != -1:
                            tar_path = self.target_path.joinpath(mode, uttid[:p])
                        else:
                            tar_path = self.target_path.joinpath(mode)
                        tar_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
                        # store phn file
                        phn_file = tar_dir.joinpath(uttid + ".phn")
                        phones = read_vec_int(f)
                        np.savetxt(str(phn_file), phones, "%d")
                        # prepare manifest elements
                        num_frms = len(phones)
                        manifest[uttid] = (str(phn_file), num_frms)
        return manifest


    def make_ctc_labels(self):
        # find *.phn files
        logger.info(f"finding *.phn files under {str(self.target_path)}")
        phn_files = [str(x) for x in self.target_path.rglob("*.phn")]
        # convert
        for phn_file in tqdm(phn_files, ncols=params.NCOLS):
            phns = np.loadtxt(phn_file, dtype="int", ndmin=1)
            # make ctc labelings by removing duplications
            ctcs = np.array([x for x in remove_duplicates(phns)])
            # write ctc file
            # blank labels will be inserted in warp-ctc loss module,
            # so here the target labels have not to contain the blanks interleaved
            ctc_file = phn_file.replace("phn", "ctc")
            np.savetxt(str(ctc_file), ctcs, "%d")
        count_priors(phn_files)


    def count_priors(self, phn_files=None):
        # load labels.txt
        labels = dict()
        with open('asr/kaldi/graph/labels.txt', 'r') as f:
            for line in f:
                splits = line.strip().split()
                label = splits[0]
                labels[label] = splits[1]
        blank = labels['<blk>']
        if phn_files is None:
            # find *.phn files
            logger.info(f"finding *.phn files under {str(self.target_path)}")
            phn_files = [str(x) for x in self.target_path.rglob("*.phn")]
        # count
        counts = [0] * len(labels)
        for phn_file in tqdm(phn_files, ncols=params.NCOLS):
            phns = np.loadtxt(phn_file, dtype="int", ndmin=1)
            # count labels for priors
            for c in phns:
                counts[int(c)] += 1
            counts[int(blank)] += len(phns) + 1
        # write count file
        count_file = self.target_path.joinpath("priors_count.txt")
        np.savetxt(str(count_file), counts, "%d")


    def rebuild(self, mode):
        import wave
        logger.info(f"rebuilding \"{mode}\" ...")
        wav_manifest, txt_manifest = dict(), dict()
        for wav_file in self.target_path.joinpath(mode).rglob("*.wav"):
            uttid = wav_file.stem
            with wave.openfp(str(wav_file), "rb") as wav:
                samples = wav.getnframes()
            wav_manifest[uttid] = (str(wav_file), samples)
            txt_file = str(wav_file).replace('wav', 'txt')
            if Path(txt_file).exists():
                txt_manifest[uttid] = (str(txt_file), '-')
        self.make_manifest(mode, wav_manifest, txt_manifest)

    def process_text_only(self, mode):
        import wave
        logger.info(f"processing text only from \"{mode}\" ...")
        wav_manifest = dict()
        for wav_file in self.target_path.joinpath(mode).rglob("*.wav"):
            uttid = wav_file.stem
            with wave.openfp(str(wav_file), "rb") as wav:
                samples = wav.getnframes()
            wav_manifest[uttid] = (str(wav_file), samples)
        txt_manifest = self.get_transcripts(mode)
        self.make_manifest(mode, wav_manifest, txt_manifest)

    def process(self, mode):
        logger.info(f"processing \"{mode}\" ...")
        wav_manifest = self.split_wav(mode)
        txt_manifest = self.get_transcripts(mode)
        self.make_manifest(mode, wav_manifest, txt_manifest)

    def make_manifest(self, mode, wav_manifest, txt_manifest):
        logger.info(f"generating manifest to \"{mode}.csv\" ...")
        min_len, max_len = 1e30, 0
        histo = [0] * 31
        total = 0
        with open(self.target_path.joinpath(f"{mode}.csv"), "w") as f:
            for k, v in tqdm(wav_manifest.items(), ncols=params.NCOLS):
                if not k in txt_manifest:
                    continue
                wav_file, samples = v
                txt_file, _ = txt_manifest[k]
                f.write(f"{k},{wav_file},{samples},{txt_file}\n")
                total += 1
                sec = float(samples) / params.SAMPLE_RATE
                if sec < min_len:
                    min_len = sec
                if sec > max_len:
                    max_len = sec
                if sec < 30.:
                    histo[int(np.ceil(sec))] += 1
        logger.info(f"total {total} entries listed in the manifest file.")
        cum_histo = np.cumsum(histo) / total * 100.
        logger.info(f"min: {min_len:.2f} sec  max: {max_len:.2f} sec")
        logger.info(f"<5 secs: {cum_histo[5]:.2f} %  "
                    f"<10 secs: {cum_histo[10]:.2f} %  "
                    f"<15 secs: {cum_histo[15]:.2f} %  "
                    f"<20 secs: {cum_histo[20]:.2f} %  "
                    f"<25 secs: {cum_histo[25]:.2f} %  "
                    f"<30 secs: {cum_histo[30]:.2f} %")


if __name__ == "__main__":
    pass
