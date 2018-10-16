#!python

import sys
import importlib
from pathlib import Path

datasets = set([
    x.stem for x in Path('asr/datasets').glob("*.py")
    if x.is_file() and x.name[0] not in ['_', '.']
])

try:
    dataset, argv = sys.argv[1], sys.argv[2:]
    if dataset not in datasets:
        raise
except:
    print(f"Error: choose one of datasets in {datasets}")
    sys.exit(1)

try:
    d = importlib.import_module(f"asr.datasets.{dataset}")
    d.prepare(argv)
except:
    raise

