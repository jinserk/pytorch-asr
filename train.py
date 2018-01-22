#!python
from pathlib import Path

from pyro.infer import SVI
from pyro.optim import Adam
from pyro.shim import parse_torch_version

from logger import logger, set_logfile
from mnist_cached import MNISTCached, setup_data_loaders

from model import SsVae
from plot import visualize_setup, plot_samples, plot_tsne

MODEL_SUFFIX = "pth.tar"


def train(args):
    """
    train SS-VAE model
    :param args: arguments for SS-VAE
    :return: None
    """
    def get_model_file_path(desc):
        return Path(args.log_dir, f"{args.model_prefix}_{desc}.{MODEL_SUFFIX}")

    if args.visualize:
        visualize_setup(args.log_dir)

    # batch_size: number of images (and labels) to be considered in a batch
    ss_vae = SsVae(args)

    if args.continue_from is not None:
        parameters = ss_vae.load(args.continue_from)
        start_epoch = parameters("epoch")
    else:
        start_epoch = 0

    # prepare data loaders
    data_loaders = setup_data_loaders(MNISTCached, args.use_cuda, args.batch_size,
                                      sup_num=args.sup_num, drop_last=True)

    # how often would a supervised batch be encountered during inference
    # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
    # until we have traversed through the all supervised batches
    periodic_interval_batches = int(MNISTCached.train_data_size / (1.0 * args.sup_num))

    # number of unsupervised examples
    unsup_num = MNISTCached.train_data_size - args.sup_num

    # initializing local variables to maintain the best validation accuracy
    # seen across epochs over the supervised training set
    # and the corresponding testing set and the state of the networks
    best_valid_acc, corresponding_test_acc = 0.0, 0.0

    # run inference for a certain number of epochs
    for i in range(start_epoch, args.num_epochs):
        # get the losses for an epoch
        epoch = i + 1
        losses_sup, losses_unsup = ss_vae.train_epoch(epoch, data_loaders, periodic_interval_batches)

        # compute average epoch losses i.e. losses per example
        avg_losses_sup = map(lambda x: f"{x/args.sup_num:7.3f}", losses_sup)
        avg_losses_unsup = map(lambda x: f"{x/unsup_num:7.3f}", losses_unsup)

        validation_accuracy = ss_vae.get_accuracy(data_loaders["valid"])

        # this test accuracy is only for logging, this is not used
        # to make any decisions during training
        test_accuracy = ss_vae.get_accuracy(data_loaders["test"])

        str_avg_losses_sup = ' '.join([x for x in avg_losses_sup])
        str_avg_losses_unsup = ' '.join([x for x in avg_losses_unsup])

        logger.info(f"epoch {epoch:03d}: "
                    f"avg_loss_sup {str_avg_losses_sup} "
                    f"avg_loss_unsup {str_avg_losses_unsup} "
                    f"val_accuracy {validation_accuracy:5.3f} "
                    f"test_accuracy {test_accuracy:5.3f}")

        # update the best validation accuracy and the corresponding
        # testing accuracy and the state of the parent module (including the networks)
        if best_valid_acc < validation_accuracy:
            best_valid_acc = validation_accuracy
            corresponding_test_acc = test_accuracy

        # save
        ss_vae.save(get_model_file_path(f"epoch_{epoch:04d}"), epoch=epoch)

        # visualize the conditional samples
        if args.visualize:
            plot_samples(ss_vae)
            plot_tsne(ss_vae, data_loaders["test"])

    final_test_accuracy = ss_vae.get_accuracy(data_loaders["test"])
    logger.info(f"best validation accuracy {best_valid_acc:5.3f} "
                f"corresponding testing accuracy {corresponding_test_acc:5.3f} "
                f"last testing accuracy {final_test_accuracy:5.3f}")

    #save final model
    ss_vae.save(get_model_file_path("final"), epoch=epoch)

    if args.visualize:
        plot_tsne(ss_vae, data_loaders["test"])


if __name__ == "__main__":
    import sys
    import argparse
    import torch
    import numpy as np

    parser = argparse.ArgumentParser(description="SS-VAE model training")

    parser.add_argument('-ne', '--num-epochs', default=1000, type=int, help="number of epochs to run")
    parser.add_argument('-al', '--aux-loss', default=True, action="store_true", help="whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)")
    parser.add_argument('-alm', '--aux-loss-multiplier', default=300, type=float, help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum-discrete', default=True, action="store_true", help="whether to enumerate the discrete support of the categorical distribution while computing the ELBO loss")
    parser.add_argument('-sup', '--sup-num', default=3000, type=float, help="supervised amount of the data i.e. how many of the images have supervised labels")
    parser.add_argument('-zd', '--z-dim', default=50, type=int, help="size of the tensor representing the latent variable z variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-hd', '--h-dims', nargs='+', default=[256,], type=int, help="a tuple (or list) of MLP layers to be used in the neural networks representing the parameters of the distributions in our model")
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=100, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('-eps', '--epsilon-scale', default=1e-9, type=float, help="a small float value used to scale down the output of Softmax and Sigmoid opertations in pytorch for numerical stability")

    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--visualize', default=True, action="store_true", help="use a visdom server to visualize the embeddings")
    parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='ss_vae_mnist', type=str, help="model file prefix to store")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")

    args = parser.parse_args()

    # some assertions to make sure that batching math assumptions are met
    assert args.sup_num % args.batch_size == 0, "assuming simplicity of batching math"
    torch_version = parse_torch_version()
    assert torch_version >= (0, 2, 1), "you need pytorch 0.2.1 or later"

    set_logfile(Path(args.log_dir, "train.log"))

    logger.info(f"Training started with command: {' '.join(sys.argv)}")
    args_str = [f"{k}={v}" for (k, v) in vars(args).items()]
    logger.info(f"args: {' '.join(args_str)}")

    if args.use_cuda:
        x = torch.randn(1).cuda() # to initialize cuda in torch
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)

    # run training
    train(args)

