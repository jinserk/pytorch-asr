#!python

import sys

argv = sys.argv[1:]
models = set([
    "convnet",
    "densenet",
    "deepspeech_ctc",
    "deepspeech_ce",
    "resnet_ctc",
    "resnet_ce",
    "capsule1",
])

if argv[0] not in models:
    print(f"Error: choose one of models in {models}")
    sys.exit(1)

model = argv[0]
argv.remove(model)

if model == "convnet":
    from asr import convnet
    convnet.predict(argv)
elif model == "densenet":
    from asr import densenet
    densenet.predict(argv)
elif model == "deepspeech_ctc":
    from asr import deepspeech_ctc
    deepspeech_ctc.predict(argv)
elif model == "deepspeech_ce":
    from asr import deepspeech_ce
    deepspeech_ce.predict(argv)
elif model == "resnet_ctc":
    from asr import resnet_ctc
    resnet_ctc.predict(argv)
elif model == "resnet_ce":
    from asr import resnet_ce
    resnet_ce.predict(argv)
elif model == "capsule1":
    from asr import capsule1
    capsule1.predict(argv)
