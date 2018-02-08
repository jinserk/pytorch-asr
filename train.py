#!python
from pathlib import Path

from utils.logger import logger, set_logfile
from aspire import setup_data_loaders, NUM_PIXELS, NUM_LABELS

from model import SsVae

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
        from plot import visualize_setup, plot_samples, plot_tsne
        visualize_setup(args.log_dir)

    # batch_size: number of images (and labels) to be considered in a batch
    ss_vae = SsVae(x_dim=NUM_PIXELS, y_dim=NUM_LABELS, **vars(args))

    if args.continue_from is not None:
        parameters = ss_vae.load(args.continue_from)
        start_epoch = parameters["epoch"]
    else:
        start_epoch = 0

    # prepare data loaders
    data_loaders = setup_data_loaders(batch_size=args.batch_size, use_cuda=args.use_cuda,
                                      num_workers=args.num_workers, drop_last=True)

    # number of supervised and unsupervised examples
    unsup_num = 10  # len(data_loaders["train_unsup"])
    sup_num = 1  # len(data_loaders["train_sup"])
    val_num = 1  # len(data_loaders["dev"])

    # initializing local variables to maintain the best validation accuracy
    # seen across epochs over the supervised training set
    # and the corresponding testing set and the state of the networks
    best_valid_acc, corresponding_test_acc = 0.0, 0.0

    # run inference for a certain number of epochs
    for i in range(start_epoch, args.num_epochs):
        epoch = i + 1
        # get the losses for an epoch
        avg_losses_sup, avg_losses_unsup = ss_vae.train_epoch(epoch, data_loaders, unsup_num, sup_num)
        # validate
        validation_accuracy = ss_vae.get_accuracy(data_loaders["dev"], val_num, desc="validating")

        str_avg_loss_sup = ' '.join([f"{x:7.3f}" for x in avg_losses_sup])
        str_avg_loss_unsup = ' '.join([f"{x:7.3f}" for x in avg_losses_unsup])
        logger.info(f"epoch {epoch:03d}: "
                    f"avg_loss_sup {str_avg_loss_sup} "
                    f"avg_loss_unsup {str_avg_loss_unsup} "
                    f"val_accuracy {validation_accuracy:5.3f}")

        # update the best validation accuracy and the corresponding
        # testing accuracy and the state of the parent module (including the networks)
        if best_valid_acc < validation_accuracy:
            best_valid_acc = validation_accuracy
        # save
        ss_vae.save(get_model_file_path(f"epoch_{epoch:04d}"), epoch=epoch)
        # visualize the conditional samples
        if args.visualize:
            from plot import visualize_setup, plot_samples, plot_tsne
            plot_samples(ss_vae)
            #if epoch % 100 == 0:
            #    plot_tsne(ss_vae, data_loaders["test"], use_cuda=args.use_cuda)

    # test
    test_accuracy = ss_vae.get_accuracy(data_loaders["test"])

    logger.info(f"best validation accuracy {best_valid_acc:5.3f} "
                f"test accuracy {test_accuracy:5.3f}")

    #save final model
    ss_vae.save(get_model_file_path("final"), epoch=epoch)
    #if args.visualize:
    #    plot_tsne(ss_vae, data_loaders["test"], use_cuda=args.use_cuda)


if __name__ == "__main__":
    import sys
    import argparse

    import numpy as np

    import torch
    from pyro.shim import parse_torch_version

    parser = argparse.ArgumentParser(description="SS-VAE model training")
    # for model network
    parser.add_argument('-zd', '--z-dim', default=200, type=int, help="size of the tensor representing the latent variable z variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-hd', '--h-dims', nargs='+', default=[256, 256], type=int, help="a tuple (or list) of MLP layers to be used in the neural networks representing the parameters of the distributions in our model")
    parser.add_argument('-eps', '--eps', default=1e-9, type=float, help="a small float value used to scale down the output of Softmax and Sigmoid opertations in pytorch for numerical stability")
    # for SVI model
    parser.add_argument('-al', '--aux-loss', default=True, action="store_true", help="whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)")
    parser.add_argument('-alm', '--aux-loss-multiplier', default=500, type=float, help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum-discrete', default=True, action="store_true", help="whether to enumerate the discrete support of the categorical distribution while computing the ELBO loss")
    # for training
    parser.add_argument('--num-workers', default=8, type=int, help="number of dataloader workers")
    parser.add_argument('--num-epochs', default=1000, type=int, help="number of epochs to run")
    parser.add_argument('--batch-size', default=1024, type=int, help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--init-lr', default=0.0001, type=float, help="initial learning rate for Adam optimizer")
    # optional
    parser.add_argument('--use-cuda', default=False, action='store_true', help="use cuda")
    parser.add_argument('--seed', default=None, type=int, help="seed for controlling randomness in this example")
    parser.add_argument('--visualize', default=False, action="store_true", help="use a visdom server to visualize the embeddings")
    parser.add_argument('--log-dir', default='./logs', type=str, help="filename for logging the outputs")
    parser.add_argument('--model-prefix', default='ss_vae_aspire', type=str, help="model file prefix to store")
    parser.add_argument('--continue-from', default=None, type=str, help="model file path to make continued from")

    args = parser.parse_args()

    # some assertions to make sure that batching math assumptions are met
    assert parse_torch_version() >= (0, 2, 1), "you need pytorch 0.2.1 or later"

    set_logfile(Path(args.log_dir, "train.log"))

    logger.info(f"Training started with command: {' '.join(sys.argv)}")
    args_str = [f"{k}={v}" for (k, v) in vars(args).items()]
    logger.info(f"args: {' '.join(args_str)}")

    if args.use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)

    # run training
    train(args)
