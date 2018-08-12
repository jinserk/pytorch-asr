import sys
import argparse
from pathlib import Path

from ..utils.logger import logger, set_logfile
from ._common import KALDI_PATH, KaldiDataImporter


"""
Importing the data from Kaldi's egs/tedlium/s5c recipe directory
"""

CHAR_MASK = "abcdefghijklmnopqrstuvwxyz'-._<>[] "

CORRECT_TABLE = {
    "@"         : "at",
    "&"         : "and",
    "\="        : "equals",
    "\+"        : "plus",
    "1"         : "one",
    "2"         : "two",
    "3"         : "three",
    "4"         : "four",
    "5"         : "five",
    "6"         : "six",
    "7"         : "seven",
    "8"         : "eight",
    "9"         : "nine",
    "10"        : "ten",
    "11"        : "eleven",
    "12"        : "twelve",
    "13"        : "thirteen",
    "14"        : "fourteen",
    "15"        : "fifteen",
    "16"        : "sixteen",
    "17"        : "seventeen",
    "18"        : "eighteen",
    "19"        : "nineteen",
    "20"        : "twenty",
    "24"        : "twenty four",
    "25"        : "twenty five",
    "30"        : "thirty",
    "38"        : "thirty eight",
    "40"        : "fourty",
    "45"        : "fourty five",
    "50"        : "fifty",
    "60"        : "sixty",
    "70"        : "seventy",
    "80"        : "eighty",
    "90"        : "ninety",
    "100"       : "a hundred",
    "140"       : "one forty",
    "200"       : "two hundreds",
    "360"       : "three sixty",
    "400"       : "four hundreds",
    "5 000"     : "five thousand",
    "9 000"     : "nine thousand",
    "20s"       : "twenties",
    "30s"       : "thirties",
    "50s"       : "fifties",
    "60s"       : "sixties",
    "70s"       : "seventies",
    "80s"       : "eighties",
    "90s"       : "nineties",
    "1700s"     : "seventeen hundreds",
    "1800s"     : "eighteen hundreds",
    "1900s"     : "nineteen hundreds",
    "1920s"     : "nineteen twenties",
    "1930s"     : "nineteen thirties",
    "1950s"     : "nineteen fifties",
    "1960s"     : "nineteen fifties",
    "1970s"     : "nineteen seventies",
    "1980s"     : "nineteen eighties",
    "1990s"     : "nineteen nineties",
    "2000s"     : "two thousands",
    "2040s"     : "twenty forties",
    "5th"       : "fifth",
    "7th"       : "seventh",
    "11th"      : "eleventh",
    "12th"      : "twelfth",
    "14th"      : "fourteenth",
    "15th"      : "fifteenth",
    "17th"      : "seventeenth",
    "18th"      : "eighteenth",
    "19th"      : "ninteenth",
    "20th"      : "twentieth",
    "21st"      : "twenty first",
    "26th"      : "twenty sixth",
    "50th"      : "fiftieth",
    "61st"      : "sixty first",
    "2d"        : "two d",
    "3d"        : "three d",
    "3m"        : "three m",
    "3s"        : "three s",
    "5am"       : "five a._m.",
    "a4"        : "a four",
    "g2g"       : "g two g",
    "g8"        : "g eight",
    "g20"       : "g twenty",
    "h30"       : "h thirty",
    "m13"       : "m thirteen",
    "mr"        : "mister",
    "co2"       : "c._o. two",
    "p2p"       : "p two p",
    "sio2"      : "s i o two",
    "disc1"     : "disc one",
    "and'"      : "and",
    "the'"      : "the",
    "early'"    : "early",
    "a c"       : "a._c.",
    "a d"       : "a._d.",
    "a i"       : "a._i.",
    "a k a"     : "a._k._a.",
    "a m"       : "a._m.",
    "a p"       : "a._p.",
    "a r"       : "a._r.",
    "b c"       : "b._c.",
    "b s"       : "b._s.",
    "b t"       : "b._t.",
    "c d"       : "c._d.",
    "c d s"     : "c._d.s",
    "c g"       : "c._g.",
    "c t"       : "c._t.",
    "c v"       : "c._v.",
    "d a"       : "d._a.",
    "d c"       : "d._c.",
    "d j"       : "d._j.",
    "e p"       : "e._p.",
    "e q"       : "e._q.",
    "e r"       : "e._r.",
    "e t"       : "e._t.",
    "e u"       : "e._u.",
    "e x"       : "e._x.",
    "f a"       : "f._a.",
    "f m"       : "f._m.",
    "g e"       : "g._e.",
    "g i"       : "g._i.",
    "g m"       : "g._m.",
    "g p"       : "g._p.",
    "g s"       : "g._s.",
    "h g"       : "h._g.",
    "h p"       : "h._p.",
    "h r"       : "h._r.",
    "i d"       : "i._d.",
    "i e"       : "i._e.",
    "i p"       : "i._p.",
    "i q s"     : "i._q.s",
    "i q"       : "i._q.",
    "i t"       : "i._t.",
    "l a"       : "l._a.",
    "l g b t"   : "l._g._b._t.",
    "l l"       : "l._l.",
    "m d"       : "m._d.",
    "m i t"     : "m._i._t.",
    "m p"       : "m._p.",
    "m r"       : "m._r.",
    "m s"       : "m._s.",
    "n g"       : "n._g.",
    "o k"       : "o._k.",
    "o r"       : "o._r.",
    "o s"       : "o._s.",
    "p b"       : "p._b.",
    "p c"       : "p._c.",
    "p g"       : "p._g.",
    "ph d s"    : "p._h._d.'s",
    "ph d"      : "p._h._d.",
    "p h d"     : "p._h._d.",
    "p m"       : "p._m.",
    "p o box"   : "p._o. box",
    "p r"       : "p._r.",
    "p s"       : "p._s.",
    "r f"       : "r._f.",
    "s l"       : "s._l.",
    "t v"       : "t._v.",
    "u c"       : "u._c.",
    "u c l a"   : "u._c._l._a.",
    "u k"       : "u._k.",
    "u n"       : "u._n.",
    "u s c"     : "u._s._c.",
    "u s"       : "u._s.",
    "u t"       : "u._t.",
    "u v"       : "u._v.",
    "v c"       : "v._c.",
    "v r"       : "v._r.",
    "v s"       : "v._s.",
    "x ray"     : "x-ray",
    "\[ o k \]" : "o._k.",
    "at & t"    : "a t and t",
    "at & t's"  : "a t and t's",
    "\$ zero point zero one <unk>": "one",
    "r d engineers": "r and d engineers",
    "a c t scanner": "a c._t. scanner",
    "i 35w bridge": "i thirty five bridge",
    "html5"     : "h._t._m._l. five",
    "two \^ five": "two to the power of five",
    "e \= mc"    : "e equals m c squared",
    "\$ ten"     : "ten dollar",
    "\$ forty"   : "forty dollars",
    "\$ fifty"   : "fifty dollars",
    "\$ one hundred": "one hundred dollar",
    "\$ one hundred and forty": "one hundred and forty dollars",
    "\$ one thousand": "one thousand dollars",
    "\$ two thousand": "two thousand dollars",
    "\$ three thousand": "three thousand dollars",
    "\$ six thousand": "six thousand dollars",
    "\$ eight thousand": "eight thousand dollars",
    "\$ ten thousand": "ten thousand dollar",
    "\$ twelve thousand": "twelve thousand dollar",
    "\$ two hundred thousand": "two hundred thousand dollars",
    "\$ one point six billion": "one point six billion dollars",
    "\$ four trilion": "four tillion dollars",
    "\$ five trilion": "five tillion dollar",
    "\# gamergate": "<unk> gamergate",
}


class KaldiTedliumImporter(KaldiDataImporter):

    def __init__(self, target_dir):
        recipe_path = Path(KALDI_PATH, "egs", "tedlium", "ics").resolve()
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
        #p = re.compile(r'([a-z]\.\s)+[a-z]\.')
        #matches = p.findall(text)
        #if matches:
        #    for m in matches:
        #        s = m.group().replace(' ', '_')
        #        text = text.replace(m.group(), s)
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
    print(f"begins logging to file: {str(log_file)}")
    set_logfile(log_file)

    target_path = Path(args.target_dir).resolve()
    logger.info(f"target data path : {target_path}")

    importer = KaldiTedliumImporter(target_path)

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
