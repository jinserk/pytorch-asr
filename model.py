#!python
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.nn import ClippedSoftmax, ClippedSigmoid

from utils.logger import logger

from network import *

NUM_PIXEL = 784
NUM_DIGITS = 10


class SsVae(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset

    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param h_dims: a tuple (or list) of MLP layers to be used in the neural networks
                   representing the parameters of the distributions in our model
    :param eps: a small float value used to scale down the output of Softmax and Sigmoid
                opertations in pytorch for numerical stability
    :param enum_discrete: if True, sum out the discrete latent variables to reduce variance of
                          the ELBO gradient
    :param aux_loss: use the auxiliary loss as the model variant 3
                     (http://pyro.ai/examples/ss-vae.html#Third-Variant:-Adding-a-Term-to-the-Objective)
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    :param use_cuda: use GPUs for faster training
    :param batch_size: batch size of calculation
    :param init_lr: initial learning rate to setup the optimizer
    :param continue_from: model file path to load the model states
    """
    def __init__(self, x_dim=NUM_PIXEL, y_dim=NUM_DIGITS, z_dim=50, h_dims=[256,],
                 eps=1e-9, enum_discrete=True, aux_loss=True, aux_loss_multiplier=300,
                 use_cuda=False, batch_size=100, init_lr=0.001, continue_from=None,
                 *args, **kwargs):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.x_dim = NUM_PIXEL
        self.y_dim = NUM_DIGITS

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.init_lr = init_lr

        if continue_from is None:
            self.z_dim = z_dim
            self.h_dims = h_dims
            self.eps = eps
            self.enum_discrete = enum_discrete
            self.aux_loss = aux_loss
            self.aux_loss_multiplier = aux_loss_multiplier
            # define and instantiate the neural networks representing
            # the paramters of various distributions in the model
            self.__setup_networks()
        else:
            self.load(continue_from)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def __setup_networks(self):
        # define the neural networks used later in the model and the guide.
        self.encoder_y = ConvEncoderY(x_dim=self.x_dim, y_dim=self.y_dim, eps=self.eps)
        self.encoder_z = MlpEncoderZ(x_dim=self.x_dim, y_dim=self.y_dim,
                                     z_dim=self.z_dim, h_dims=self.h_dims, eps=self.eps)
        self.decoder = ConvDecoder(x_dim=self.x_dim, y_dim=self.y_dim,
                                   z_dim=self.z_dim, eps=self.eps)

        # setup the optimizer
        params = {"lr": self.init_lr, "betas": (0.9, 0.999)}
        self.optimizer = Adam(params)

        # set up the loss(es) for inference setting the enum_discrete parameter builds the loss as a sum
        # by enumerating each class label for the sampled discrete categorical distribution in the model
        loss_basic = SVI(self.model, self.guide, self.optimizer, loss="ELBO",
                         enum_discrete=self.enum_discrete)
        self.losses = [loss_basic]

        # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
        if self.aux_loss:
            loss_aux = SVI(self.model_classify, self.guide_classify, self.optimizer, loss="ELBO")
            self.losses.append(loss_aux)

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # handwriting style (latent)
        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
        p(x|y,z) = bernoulli(mu(y,z))   # an image
        mu is given by a neural network  `decoder`

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        batch_size = xs.size(0)
        with pyro.iarange("independent"):
            # sample the handwriting style from the constant prior distribution
            prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
            prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
            zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma, extra_event_dims=1)

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = Variable(torch.ones([batch_size, self.y_dim]) / (1.0 * self.y_dim))
            if ys is None:
                ys = pyro.sample("y", dist.one_hot_categorical, alpha_prior)
            else:
                pyro.sample("y", dist.one_hot_categorical, alpha_prior, obs=ys)

            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            mu = self.decoder.forward(zs, ys)
            pyro.sample("x", dist.bernoulli, mu, extra_event_dims=1, obs=xs)

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(mu(x,y),sigma(x,y))       # infer handwriting style from an image and the digit
        mu, sigma are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.iarange("independent"):
            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.one_hot_categorical, alpha)

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(mu(x,y),sigma(x,y))
            mu, sigma = self.encoder_z.forward(xs, ys)
            zs = pyro.sample("z", dist.normal, mu, sigma, extra_event_dims=1)

    def classifier(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        alpha = self.encoder_y.forward(xs)

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        NIPS 2014 paper by Kingma et al titled
        "Semi-Supervised Learning with Deep Generative Models"
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.iarange("independent"):
            # this here is the extra Term to yield an auxiliary loss that we do gradient descend on
            # similar to the NIPS 14 paper (Kingma et al).
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(None, self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.one_hot_categorical, alpha, obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass

    def model_sample(self, ys, batch_size=1):
        with torch.no_grad():
            # sample the handwriting style from the constant prior distribution
            prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
            prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
            zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma, extra_event_dims=1)

            # sample an image using the decoder
            mu = self.decoder.forward(zs, ys)
            xs = pyro.sample("sample", dist.bernoulli, mu.cpu(), extra_event_dims=1)
            return xs, mu

    def guide_sample(self, xs, ys, batch_size=1):
        with torch.no_grad():
            # obtain z using `encoder_z`
            xs, ys = Variable(xs), Variable(ys)
            z_mu, z_sigma = self.encoder_z(xs, ys)
            return z_mu, z_sigma

    def train_epoch(self, epoch, data_loaders, periodic_interval_batches):
        """
        runs the inference algorithm for an epoch
        returns the values of all losses separately on supervised and unsupervised parts
        """
        # initialize variables to store loss values
        num_losses = len(self.losses)

        # compute number of batches for an epoch
        sup_batches = len(data_loaders["sup"])
        unsup_batches = len(data_loaders["unsup"])
        batches_per_epoch = sup_batches + unsup_batches

        # initialize variables to store loss values
        epoch_losses_sup = [0.] * num_losses
        epoch_losses_unsup = [0.] * num_losses

        # setup the iterators for training data loaders
        sup_iter = iter(data_loaders["sup"])
        unsup_iter = iter(data_loaders["unsup"])

        # count the number of supervised batches seen in this epoch
        ctr_sup = 0
        for i in tqdm(range(batches_per_epoch), desc="Traning"):
            # whether this batch is supervised or not
            is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

            # extract the corresponding batch
            if is_supervised:
                (xs, ys) = next(sup_iter)
                ctr_sup += 1
            else:
                (xs, ys) = next(unsup_iter)
            xs, ys = Variable(xs), Variable(ys)

            # run the inference for each loss with supervised or un-supervised
            # data as arguments
            for loss_id in range(num_losses):
                if is_supervised:
                    new_loss = self.losses[loss_id].step(xs, ys)
                    epoch_losses_sup[loss_id] += new_loss
                else:
                    new_loss = self.losses[loss_id].step(xs)
                    epoch_losses_unsup[loss_id] += new_loss

        # return the values of all losses
        return epoch_losses_sup, epoch_losses_unsup

    def get_accuracy(self, data_loader):
        """
        compute the accuracy over the supervised training set or the testing set
        """
        with torch.no_grad():
            predictions, actuals = [], []

            # use the appropriate data loader
            for (xs, ys) in data_loader:
                # use classification function to compute all predictions for each batch
                xs, ys = Variable(xs), Variable(ys)
                predictions.append(self.classifier(xs))
                actuals.append(ys)

            # compute the number of accurate predictions
            accurate_preds = 0
            for pred, act in zip(predictions, actuals):
                for i in range(pred.size(0)):
                    v = torch.sum(pred[i] == act[i])
                    accurate_preds += (v.data[0] == 10)

            # calculate the accuracy between 0 and 1
            accuracy = (accurate_preds * 1.0) / (len(predictions) * self.batch_size)
            return accuracy

    def save(self, file_path, **kwargs):
        Path(file_path).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        logger.info(f"saving the model to {file_path}")
        states = kwargs
        states["ss_vae"] = self.state_dict()
        states.update({
            "z_dim": self.z_dim,
            "h_dims": self.h_dims,
            "eps": self.eps,
            "enum_discrete": self.enum_discrete,
            "aux_loss": self.aux_loss,
            "aux_loss_multiplier": self.aux_loss_multiplier,
            "optimizer": self.optimizer.get_state(),
        })
        torch.save(states, file_path)

    def load(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"no such file {file_path} exists")
            sys.exit(1)
        logger.info(f"loading the model from {file_path}")
        states = torch.load(file_path)

        self.z_dim = states["z_dim"]
        self.h_dims = states["h_dims"]
        self.eps = states["eps"]
        self.enum_discrete = states["enum_discrete"]
        self.aux_loss = states["aux_loss"]
        self.aux_loss_multiplier = states["aux_loss_multiplier"]

        self.__setup_networks()
        self.load_state_dict(states["ss_vae"])
        self.optimizer.set_state(states["optimizer"])
        return states
