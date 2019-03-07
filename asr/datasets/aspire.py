import sys
import argparse
from pathlib import Path

from asr.utils.logger import logger, init_logger
from ._common import KALDI_PATH, KaldiDataImporter


"""
Importing the data from Kaldi's egs/aspire/s5c recipe directory
"""

CHAR_MASK = "abcdefghijklmnopqrstuvwxyz'-._<>[] "


class KaldiAspireImporter(KaldiDataImporter):

    def __init__(self, target_dir):
        recipe_path = Path(KALDI_PATH, "egs", "aspire", "mgh").resolve()
        assert recipe_path.exists(), f"no such path \"{str(recipe_path)}\" found"
        super().__init__(recipe_path, target_dir)

    def strip_text(self, text):
        text = text.lower()
        text = ''.join([x for x in text if x in CHAR_MASK])
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

    importer = KaldiAspireImporter(target_path)

    if args.rebuild:
        importer.rebuild("train")
        importer.rebuild("dev")
        importer.rebuild("test")
    elif args.text_only:
        importer.process_text_only("train")
        importer.process_text_only("dev")
        importer.process_text_only("test")
    else:
        importer.process("train")
        importer.process("dev")
        importer.process("test")

    logger.info("data preparation finished.")



if __name__ == "__main__":
    pass
