#!python

import sys
import argparse
from pathlib import Path

from asr.dataset import aspire

data_root = Path.cwd().resolve() / "data"

argv = sys.argv[1:]
datasets = set(["aspire",])
dataset = None
for opt in argv:
    if opt in datasets:
        dataset = opt
        break

if dataset == "aspire":
    aspire.prepare_data(data_root / dataset)
else:
    print(f"Error: choose one of datasets in {datasets}")
    sys.exit(1)
