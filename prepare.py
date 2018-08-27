#!python

import sys
import importlib

datasets = set([
    "aspire",
    "swbd",
    "tedlium",
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

