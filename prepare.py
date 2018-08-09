#!python

import sys


argv = sys.argv[1:]
datasets = set([
    "aspire",
    "swbd",
])

if argv[0] not in datasets:
    print(f"Error: choose one of datasets in {datasets}")
    sys.exit(1)

dataset = argv[0]
argv.remove(dataset)

if dataset == "aspire":
    from asr.utils import aspire
    aspire.prepare(argv)
if dataset == "swbd":
    from asr.utils import swbd
    swbd.prepare(argv)
