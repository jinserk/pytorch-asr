#!python

import sys


argv = sys.argv[1:]
models = set([
    "densenet_ctc",
    "deepspeech_ctc",
    "resnet_ctc",
])

if argv[0] not in models:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)

model = argv[0]
argv.remove(model)

if model == "densenet_ctc":
    from asr.models import densenet_ctc
    densenet_ctc.test(argv)
elif model == "deepspeech_ctc":
    from asr.models import deepspeech_ctc
    deepspeech_ctc.test(argv)
elif model == "resnet_ctc":
    from asr.models import resnet_ctc
    resnet_ctc.test(argv)

