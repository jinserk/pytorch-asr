#!python
from scipy.signal import tukey

# global constants
MODEL_SUFFIX = "pth.tar"

# feature frames params
SAMPLE_RATE = 8000
WINDOW_SHIFT = 0.010  # sec
WINDOW_SIZE = 0.025   # sec

# spectrogram
NFFT = 256

# augmentation
TEMPO_RANGE = (0.85, 1.15)
PITCH_RANGE = (-1., 1.)
NOISE_RANGE = (-20., -3.)

# images
CHANNEL = 2
HEIGHT = 129
WIDTH = 31
NUM_PIXELS = CHANNEL * HEIGHT * WIDTH
NUM_LABELS = 187
NUM_CTC_LABELS = 177
NUM_HIDDEN = [256, 256]
NUM_STYLE = 256
EPS = 1e-9

# for capsule network
NUM_ROUTING_ITERATIONS = 5
