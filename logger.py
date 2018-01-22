#!python

from pathlib import Path
import logging


# handler
logger = logging.getLogger('ss_vae.pytorch')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# stdout handler
chdr = logging.StreamHandler()
chdr.setLevel(logging.DEBUG)
chdr.setFormatter(formatter)
logger.addHandler(chdr)


def set_logfile(filename):
    filepath = Path(filename)
    try:
        Path.mkdir(filepath.parent, parents=True, exist_ok=True)
    except OSError:
        raise
    # file handler
    fhdr = logging.FileHandler(filepath)
    fhdr.setLevel(logging.DEBUG)
    fhdr.setFormatter(formatter)
    logger.addHandler(fhdr)

