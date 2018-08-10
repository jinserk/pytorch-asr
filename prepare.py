#!python

import sys


argv = sys.argv[1:]
datasets = set([
    "aspire",
    "swbd",
    "tedlium",
])

if argv[0] not in datasets:
    print(f"Error: choose one of datasets in {datasets}")
    sys.exit(1)

dataset = argv[0]
argv.remove(dataset)

if dataset == "aspire":
    from asr.dataset import aspire
    aspire.prepare(argv)
elif dataset == "swbd":
    from asr.dataset import swbd
    swbd.prepare(argv)
elif dataset == "tedlium":
    from asr.dataset import tedlium
    tedlium.prepare(argv)
