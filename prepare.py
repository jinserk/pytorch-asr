#!python

import sys


argv = sys.argv[1:]
datasets = set([
    "aspire",
    "tedlium",
])

if argv[0] not in datasets:
    print(f"Error: choose one of datasets in {datasets}")
    sys.exit(1)

dataset = argv[0]
argv.remove(dataset)

if dataset == "aspire":
    from asr.utils import aspire
    aspire.prepare(argv)
if dataset == "tedlium":
    from asr.utils import tedlium
    tedlium.prepare(argv)
