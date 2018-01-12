#!python

from pathlib import Path, PurePath
import logging


# handler
log = logging.getLogger('ss_vae.pytorch')
log.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# screen
chdr = logging.StreamHandler()
chdr.setLevel(logging.DEBUG)
chdr.setFormatter(fmt)
log.addHandler(chdr)


def set_logfile(filename):
    filepath = PurePath(filename)
    try:
        Path.mkdir(filepath.parent, parents=True, exist_ok=True)
    except OSError:
        raise
    fhdr = logging.FileHandler(filepath)
    fhdr.setLevel(logging.DEBUG)
    fhdr.setFormatter(fmt)
    log.addHandler(fhdr)

