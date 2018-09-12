#!python
import os
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

    log_dir = kwargs.pop("log_dir", "./logs")
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

    if "slack" in kwargs and kwargs["slack"]:
        try:
            env = str(Path(log_dir).name)
            shdr = SlackClientHandler(env=env)
            shdr.setLevel(logging.INFO)
            shdr.setFormatter(formatter)
            logger.addHandler(shdr)
        except:
            logger.error("error to setup slackclient")
            raise

    # prepare visdom
    logger.visdom = None
    if "visdom" in kwargs and kwargs["visdom"]:
        env = str(Path(log_dir).name)
        log_path = Path(log_dir, "visdom.log").resolve()
        visdom_host = kwargs.pop("visdom_host", "127.0.0.1")
        visdom_port = kwargs.pop("visdom_port", 8097)
        try:
            logger.visdom = VisdomLogger(host=visdom_host, port=visdom_port, env=env, log_path=log_path)
        except:
            logger.error("error to use visdom")
            raise

    # prepare tensorboard
    logger.tensorboard = None
    if "tensorboard" in kwargs and kwargs["tensorboard"]:
        env = str(Path(log_dir, 'tensorboard').resolve)
        try:
            logger.tensorboard = TensorboardLogger(env)
        except:
            logger.error("error to use tensorboard")

    # print version and args
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.debug(f"command-line options: {' '.join(sys.argv)}")
    logger.info(f"args: {args_str}")


class SlackClientHandler(logging.Handler):

    def __init__(self, env="main"):
        super().__init__()
        self.env = env
        self.slack_token = os.getenv("SLACK_API_TOKEN")
        self.slack_user = os.getenv("SLACK_API_USER")
        if self.slack_token is None or self.slack_user is None:
            raise Exception

        from slackclient import SlackClient
        self.sc = SlackClient(self.slack_token)

        # getting user id
        ans = self.sc.api_call("users.list")
        users = [u['id'] for u in ans['members'] if u['name'] == self.slack_user]
        # open DM channel to the users
        ans = self.sc.api_call("conversations.open", users=users)
        self.channel = ans['channel']['id']

    def emit(self, record):
        try:
            msg = self.format(record)
            self.sc.api_call("chat.postMessage", channel=self.channel, text=f"`{self.env}`\n{msg}")
        except:
            self.handleError(record)


class VisdomLogger:

    def __init__(self, host='127.0.0.1', port=8097, env='main', log_path=None):
        from visdom import Visdom
        import json
        logger.info(f"using visdom on http://{host}:{port} env={env}")
        self.env = env
        self.viz = Visdom(server=f"http://{host}", port=port, env=env, log_to_filename=log_path)
        self.windows = dict()
        # if prev log exists
        if log_path.exists():
            self.viz.replay_log(log_path)
            wins = json.loads(self.viz.get_window_data(win=None, env=env))
            for k, v in wins.items():
                names = [int(x['name']) for x in v['content']['data']]
                name = str(max(names) + 1)
                self.windows[v['title']] = { 'win': v['id'], 'name': name }

    def add_plot(self, title, **kwargs):
        if title not in self.windows:
            self.windows[title] = {
                'win': None,
                'name': '1',
            }
        self.windows[title]['opts'] = { 'title': title, }
        self.windows[title]['opts'].update(kwargs)

    def add_point(self, title, x, y):
        X, Y = torch.FloatTensor([x,]), torch.FloatTensor([y,])
        if title not in self.windows:
            self.add_plot(title)
        if self.windows[title]['win'] is None:
            w = self.viz.line(Y=Y, X=X, opts=self.windows[title]['opts'], name=self.windows[title]['name'])
            self.windows[title]['win'] = w
        else:
            self.viz.line(Y=Y, X=X, update='append', win=self.windows[title]['win'], name=self.windows[title]['name'])


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
