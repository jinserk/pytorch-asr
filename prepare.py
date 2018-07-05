#!python

import sys
import argparse
from pathlib import Path


data_root = Path.cwd().resolve() / "data"

argv = sys.argv[1:]
datasets = set(["aspire",])
dataset = None
for opt in argv:
    if opt in datasets:
        dataset = opt
        argv.remove(opt)
        break

if dataset == "aspire":
    from asr.dataset import aspire
    aspire.prepare(argv)
else:
    print(f"Error: choose one of datasets in {datasets}")
    sys.exit(1)
