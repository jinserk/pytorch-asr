#!python

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


class VisdomLog:
    windows = dict()
    data = dict()
    opts = dict()

    def __init__(self, viz):
        self.viz = viz

    def add_plot(self, title, xlabel, ylabel=''):
        self.opts[title] = dict(
            title = title,
            xlabel = xlabel,
            ylabel = ylabel,
        )
        self.data[title] = dict(
            x = torch.FloatTensor([]),
            y = torch.FloatTensor([]),
        )

    def add_point(self, title, x, y):
        if not torch.is_tensor(x):
            x = torch.FloatTensor((x,))
        if not torch.is_tensor(y):
            y = torch.FloatTensor((y,))
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        self.data[title]['x'] = torch.cat((self.data[title]['x'], x))
        self.data[title]['y'] = torch.cat((self.data[title]['y'], y))
        if title not in self.windows:
            self.windows[title] = self.viz.line(
                Y = self.data[title]['y'],
                X = self.data[title]['x'],
                opts = self.opts[title],
            )
        else:
            self.viz.line(
                Y = self.data[title]['y'],
                X = self.data[title]['x'],
                win = self.windows[title],
                update = 'replace',
            )

class TensorboardLog:

    def __init__(self, log_dir):
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
