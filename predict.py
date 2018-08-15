#!python

import sys

argv = sys.argv[1:]
models = set([
    "convnet",
    "densenet",
    "densenet_ctc",
    "deepspeech_ctc",
    "deepspeech_ce",
    "resnet_ctc",
    "resnet_ce",
    "resnet_split",
    "capsule1",
])

if argv[0] not in models:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)

model = argv[0]
argv.remove(model)

if model == "convnet":
    from asr.models import convnet
    convnet.predict(argv)
elif model == "densenet":
    from asr.models import densenet
    densenet.predict(argv)
elif model == "densenet_ctc":
    from asr.models import densenet_ctc
    densenet_ctc.predict(argv)
elif model == "deepspeech_ctc":
    from asr.models import deepspeech_ctc
    deepspeech_ctc.predict(argv)
elif model == "deepspeech_ce":
    from asr.models import deepspeech_ce
    deepspeech_ce.predict(argv)
elif model == "resnet_ctc":
    from asr.models import resnet_ctc
    resnet_ctc.predict(argv)
elif model == "resnet_ce":
    from asr.models import resnet_ce
    resnet_ce.predict(argv)
elif model == "resnet_split":
    from asr.models import resnet_split
    resnet_split.predict(argv)
elif model == "capsule1":
    from asr.models import capsule1
    capsule1.predict(argv)
