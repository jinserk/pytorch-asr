#!python
import os
import sys
from pathlib import Path
import logging

import torch
import git


logger = logging.getLogger("pytorch-asr")
logger.setLevel(logging.DEBUG)


def init_logger(**kwargs):
    args_str = ' '.join([f"{k}={v}" for (k, v) in kwargs.items()])
    formatter = logging.Formatter('%(asctime)s [%(levelname)-5s] %(message)s')
    # stream handler
    chdr = logging.StreamHandler()
    chdr.setLevel(logging.DEBUG)
    chdr.setFormatter(formatter)
    logger.addHandler(chdr)

    log_dir = kwargs.pop("log_dir", "./logs")
    rank = kwargs.pop("rank", None)

    # file handler
    if "log_file" in kwargs:
        log_file = kwargs.pop("log_file")
        log_path = Path(log_dir, log_file).resolve()
        if rank is not None:
            log_path = log_path.with_suffix(f".{rank}{log_path.suffix}")
        Path.mkdir(log_path.parent, parents=True, exist_ok=True)
        fhdr = logging.FileHandler(log_path)
        fhdr.setLevel(logging.DEBUG)
        fhdr.setFormatter(formatter)
        logger.addHandler(fhdr)

    logger.info(f"begins logging to file: {str(log_path)}")

    if "slack" in kwargs and kwargs["slack"]:
        try:
            env = str(Path(log_dir).name)
            if rank is not None:
                env += f":rank{rank}"
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
            logger.visdom = VisdomLogger(host=visdom_host, port=visdom_port, env=env,
                                         log_path=log_path, rank=rank)
        except:
            logger.error("error to use visdom")

    # prepare tensorboard
    logger.tensorboard = None
    if "tensorboard" in kwargs and kwargs["tensorboard"]:
        env = str(Path(log_dir, 'tensorboard').resolve)
        try:
            logger.tensorboard = TensorboardLogger(env)
        except:
            logger.error("error to use tensorboard")

    # print version and args
    logger.info(f"command-line options: {' '.join(sys.argv)}")
    logger.debug(f"args: {args_str}")
    logger.debug(f"pytorch version: {torch.__version__}")

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    dttm = repo.head.object.committed_datetime
    logger.debug(f"pytorch-asr version: {sha} ({dttm})")


class SlackClientHandler(logging.Handler):

    def __init__(self, env="main"):
        super().__init__()
        self.env = env
        self.slack_token = os.getenv("SLACK_API_TOKEN")
        self.slack_user = os.getenv("SLACK_API_USER")
        if self.slack_token is None or self.slack_user is None:
            raise KeyError

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

    def __init__(self, host='127.0.0.1', port=8097, env='main', log_path=None, rank=None):
        from visdom import Visdom
        logger.debug(f"using visdom on http://{host}:{port} env={env}")
        self.env = env
        self.rank = rank
        self.viz = Visdom(server=f"http://{host}", port=port, env=env, log_to_filename=log_path)
        self.windows = dict()
        # if prev log exists
        if log_path is not None and log_path.exists():
            self.viz.replay_log(log_path)

    def _get_win(self, title):
        import json
        win_data = json.loads(self.viz.get_window_data(win=None, env=self.env))
        wins = [(w, v) for w, v in win_data.items() if v['title'] == title]
        if wins:
            handle, value = sorted(wins, key=lambda x: x[0])[0]
            names = [int(x['name'].partition('_')[0]) for x in value['content']['data']]
            return handle, max(names)
        else:
            return None, 1

    def _new_window(self, title, X, Y, name, opts):
        if self.rank is not None and self.rank > 0:
            # wait and fetch the window handle until rank=0 client generates new window
            import time
            for _ in range(100):
                handle, name = self._get_win(title)
                if handle is not None:
                    break
                time.sleep(0.5)
            else:
                logger.error("could't get a proper window handle from the visdom server")
                raise RuntimeError
            name = f"{name}_{self.rank}"
            self.viz.line(Y=Y, X=X, update='append', win=handle, name=name, opts=opts)
        else:
            name = f"{name}_{self.rank}" if self.rank is not None else f"{name}"
            handle = self.viz.line(Y=Y, X=X, name=name, opts=opts)
        return handle, name, opts

    def add_point(self, title, x, y, **kwargs):
        X, Y = torch.FloatTensor([x,]), torch.FloatTensor([y,])
        if title not in self.windows:
            handle, name = self._get_win(title)
            opts = { 'title': title, }
            opts.update(kwargs)
            if handle is None:
                handle, name, opts = self._new_window(title, X, Y, name, opts)
            else:
                name = f"{name + 1}_{self.rank}" if self.rank is not None else f"{name}"
                self.viz.line(Y=Y, X=X, update='append', win=handle, name=name, opts=opts)
            self.windows[title] = { 'handle': handle, 'name': name, 'opts': opts, }
        else:
            handle = self.windows[title]['handle']
            name = self.windows[title]['name']
            opts = kwargs
            self.viz.line(Y=Y, X=X, update='append', win=handle, name=name, opts=opts)
            self.windows[title]['opts'].update(opts)


class TensorboardLogger:

    def __init__(self, log_dir):
        logger.debug("using tensorboard on --logdir {log_dir}")
        log_path = Path(log_dir)
        try:
            Path.mkdir(log_path, parents=True, exist_ok=True)
        except OSError as e:
            if e.errno == errno.EEXIST:
                logger.warning(f'Tensorboard log directory already exists: {log_dir}')
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
