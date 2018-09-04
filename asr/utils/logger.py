#!python
import sys
from pathlib import Path
import logging
import torch


logger = logging.getLogger("pytorch-asr")
logger.setLevel(logging.DEBUG)


def init_logger(**kwargs):
    args_str = ' '.join([f"{k}={v}" for (k, v) in kwargs.items()])
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    # stream handler
    chdr = logging.StreamHandler()
    chdr.setLevel(logging.DEBUG)
    chdr.setFormatter(formatter)
    logger.addHandler(chdr)

    log_dir = kwargs.pop("log_dir", ".")
    # file handler
    if "log_file" in kwargs:
        log_file = kwargs.pop("log_file")
        log_path = Path(log_dir, log_file).resolve()
        Path.mkdir(log_path.parent, parents=True, exist_ok=True)
        fhdr = logging.FileHandler(log_path)
        fhdr.setLevel(logging.DEBUG)
        fhdr.setFormatter(formatter)
        logger.addHandler(fhdr)

    logger.info(f"begins logging to file: {str(log_path)}")

    # prepare visdom
    logger.visdom = None
    if "visdom" in kwargs and kwargs["visdom"]:
        env = str(Path(log_dir).name)
        visdom_host = kwargs.pop("visdom_host", "127.0.0.1")
        visdom_port = kwargs.pop("visdom_port", "8097")
        try:
            logger.visdom = VisdomLogger(host=args.visdom_host, port=args.visdom_port, env=env)
        except:
            logger.info("error to use visdom")

    # prepare tensorboard
    logger.tensorboard = None
    if "tensorboard" in kwargs and kwargs["tensorboard"]:
        env = str(Path(log_dir, 'tensorboard').resolve)
        try:
            logger.tensorboard = TensorboardLogger(env)
        except:
            logger.info("error to use tensorboard")

    # print version and args
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"command-line options: {' '.join(sys.argv)}")
    logger.info(f"args: {args_str}")


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
