import sys
import argparse
from pathlib import Path

from ..utils.logger import logger, set_logfile
from ._common import KALDI_PATH, KaldiDataImporter


"""
Importing the data from Kaldi's egs/swbd/s5c recipe directory
"""

CHAR_MASK = "abcdefghijklmnopqrstuvwxyz'-._<>[] "

CORRECT_TABLE = {
    "\[vocalized-noise\]":  "<unk>",
    "\(\%hesitation\)":     "",
    "p h d":                "p._h._d.",
    "u c l a":              "u._c._l._a.",
    "h p":                  "h._p.",
    "l l":                  "l._l.",
    "p b":                  "p._b.",
    "t w a":                "t._w._a.",
    "a d j":                "a._d._j.",
    "p a e":                "p._a._e.",
    "c r e":                "c._r._e.",
    "y w c a":              "y._w._c._a.",
    "r p m":                "r._p._m.",
    "s m u":                "s._m._u.",
    "m t v":                "m._t._v.",
    "t v":                  "t._v.",
    "u s":                  "u._s.",
    "s l":                  "s._l.",
    "e x":                  "e._x.",
    "x. ray":               "x-ray",
}


class KaldiSwbdImporter(KaldiDataImporter):

    def __init__(self, target_dir):
        recipe_path = Path(KALDI_PATH, "egs", "swbd", "ics").resolve()
        assert recipe_path.exists(), f"no such path \"{str(recipe_path)}\" found"
        super().__init__(recipe_path, target_dir)

    def strip_text(self, text):
        import re
        text = text.lower()
        for k in sorted(CORRECT_TABLE, key=len, reverse=True):
            v = CORRECT_TABLE[k]
            while True:
                out = re.sub(fr"(^|\s){k}($|\s)", fr"\1{v}\2", text)
                if out == text:
                    break
                text = out
        # match abbreviated words
        p = re.compile(r'([a-z]\.\s)+[a-z]\.')
        matches = p.findall(text)
        if matches:
            for m in matches:
                s = m.group().replace(' ', '_')
                text = text.replace(m.group(), s)
        text = ' '.join([w.strip() for w in text.strip().split()])
        text = ''.join([c for c in text if c in CHAR_MASK])
        return text


def prepare(argv):
    parser = argparse.ArgumentParser(description="Prepare dataset by importing from Kaldi recipe")
    parser.add_argument('--text-only', default=False, action='store_true', help="if you want to process text only when wavs are already stored")
    parser.add_argument('--rebuild', default=False, action='store_true', help="if you want to rebuild manifest only instead of the overall processing")
    parser.add_argument('target_dir', type=str, help="path to store the processed data")
    args = parser.parse_args(argv)

    assert args.target_dir is not None
    assert not (args.text_only and args.rebuild), "options --text-only and --rebuild cannot together. choose either of them."

    log_file = Path(args.target_dir, 'prepare.log').resolve()
    init_logger(log_file="prepare.log")

    target_path = Path(args.target_dir).resolve()
    logger.info(f"target data path : {target_path}")

    importer = KaldiSwbdImporter(target_path)

    if args.rebuild:
        importer.rebuild("train")
        importer.rebuild("eval2000")
        importer.rebuild("rt03")
    elif args.text_only:
        importer.process_text_only("train")
        importer.process_text_only("eval2000")
        importer.process_text_only("rt03")
    else:
        importer.process("train")
        importer.process("eval2000")
        importer.process("rt03")

    logger.info("data preparation finished.")



if __name__ == "__main__":
    pass
