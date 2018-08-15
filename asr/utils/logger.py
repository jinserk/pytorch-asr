#!python
import sys
from pathlib import Path
import logging
import torch

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
    filepath = Path(filename).resolve()
    try:
        Path.mkdir(filepath.parent, parents=True, exist_ok=True)
    except OSError:
        raise
    # file handler
    fhdr = logging.FileHandler(filepath)
    fhdr.setLevel(logging.DEBUG)
    fhdr.setFormatter(formatter)
    logger.addHandler(fhdr)
    print(f"begins logging to file: {str(filepath)}")


def version_log(args):
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"command-line options: {' '.join(sys.argv)}")
    args_str = [f"{k}={v}" for (k, v) in vars(args).items()]
    logger.info(f"args: {' '.join(args_str)}")


class VisdomLogger:

    def __init__(self, host='127.0.0.1', port=8097, env='main'):
        from visdom import Visdom
        logger.info(f"using visdom on http://{host}:{port} env={env}")
        self.viz = Visdom(server=f"http://{host}", port=port, env=env)
        self.windows = dict()

    def add_plot(self, title, **kwargs):
        self.windows[title] = {
            'win': None,
            'opts': { 'title': title, },
        }
        self.windows[title]['opts'].update(kwargs)

    def add_point(self, title, x, y):
        X, Y = torch.FloatTensor([x,]), torch.FloatTensor([y,])
        if title not in self.windows:
            self.add_plot(title)
        if self.windows[title]['win'] is None:
            w = self.viz.line(Y=Y, X=X, opts=self.windows[title]['opts'])
            self.windows[title]['win'] = w
        else:
            self.viz.line(Y=Y, X=X, update='append', win=self.windows[title]['win'])


class TensorboardLogger:

    def __init__(self, log_dir):
        logger.info("using tensorboard on --logdir {log_dir}")
        log_path = Path(log_dir)
        try:
            Path.mkdir(log_path, parents=True, exist_ok=True)
        except OSError as e:
            if e.errno == errno.EEXIST:
                log.warning(f'Tensorboard log directory already exists: {log_dir}')
                for f in log_path.rglob("*"):
                    f.unlink()
            else:
                raise

        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(str(log_dir))

    def add_graph(self, model, xs):
        self.writer.add_graph(model, (xs, ))

    def add_text(self, title, x, txt):
        self.writer.add_text(title, txt, x)

    def add_image(self, title, x, img):
        self.writer.add_image(title, img, x)

    def add_scalars(self, title, x, y):
        self.writer.add_scalars(title, y, x)
