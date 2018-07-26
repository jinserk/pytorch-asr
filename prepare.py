#!python

import sys
import argparse
from pathlib import Path


argv = sys.argv[1:]
datasets = set([
    "aspire",
    "tedlium",
])

dataset = None
for opt in argv:
    if opt in datasets:
        dataset = opt
        argv.remove(opt)
        break

if dataset == "aspire":
    from asr.utils import aspire
    aspire.prepare(argv)
if dataset == "tedlium":
    from asr.utils import tedlium
    tedlium.prepare(argv)
else:
    print(f"Error: choose one of datasets in {datasets}")
    sys.exit(1)
