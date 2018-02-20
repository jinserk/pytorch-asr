import scipy as sp

# feature frames params
SAMPLE_RATE = 8000
WINDOW_SHIFT = 0.010  # sec
WINDOW_SIZE = 0.025   # sec
WINDOW = sp.signal.tukey

# spectrogram
NFFT = 256
FRAME_MARGIN = 10

# augmentation
TEMPO_RANGE = (0.85, 1.15)
GAIN_RANGE = (-6., 8.)
NOISE_RANGE = (-30., -10.)

# images
CHANNEL = 2
WIDTH = 129
HEIGHT = 21
NUM_PIXELS = CHANNEL * WIDTH * HEIGHT
NUM_LABELS = 187
NUM_HIDDEN = [256, 256]
NUM_STYLE = 256
EPS = 1e-9

