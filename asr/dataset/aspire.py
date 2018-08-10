import sys
import argparse
from pathlib import Path

from ..utils.logger import logger, set_logfile
from ._common import KALDI_PATH, KaldiDataImporter


"""
Importing the data from Kaldi's egs/aspire/s5c recipe directory
"""

CHAR_MASK = "abcdefghijklmnopqrstuvwxyz'-._<>[] "


class KaldiAspireImporter(KaldiDataImporter):

    def __init__(self, target_dir):
        recipe_path = Path(KALDI_PATH, "egs", "aspire", "ics").resolve()
        assert recipe_path.exists(), f"no such path \"{str(recipe_path)}\" found"
        super().__init__(recipe_path, target_dir)

    def strip_text(self, text):
        text = text.lower()
        text = ''.join([x for x in text if x in CHAR_MASK])
        return text


def prepare(argv):
    parser = argparse.ArgumentParser(description="Prepare dataset by importing from Kaldi recipe")
    parser.add_argument('--rebuild', default=False, action='store_true', help="if you want to rebuild manifest only instead of the overall processing")
    parser.add_argument('target_dir', type=str, help="path to store the processed data")
    args = parser.parse_args(argv)

    assert args.target_dir is not None

    log_file = Path(args.target_dir, 'prepare.log').resolve()
    print(f"begins logging to file: {str(log_file)}")
    set_logfile(log_file)

    target_path = Path(args.target_dir).resolve()
    logger.info(f"target data path : {target_path}")

    importer = KaldiAspireImporter(target_path)

    if not args.rebuild:
        importer.process("train")
        importer.process("dev")
        importer.process("test")
    else:
        importer.rebuild("train")
        importer.rebuild("dev")
        importer.rebuild("test")

    logger.info("data preparation finished.")



if __name__ == "__main__":
    pass
