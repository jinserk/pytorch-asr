#!python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import params as p
from ..utils.misc import View, Swish


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):

    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=p.NUM_ROUTING_ITERATIONS, use_cuda=False):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

        self.use_cuda = use_cuda

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size()))
            if self.use_cuda:
                logits = logits.cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):

    def __init__(self, use_cuda=False, num_iterations=p.NUM_ROUTING_ITERATIONS):
        super(CapsuleNet, self).__init__()
        self.use_cuda = use_cuda
        self.encoder = nn.Sequential(
            View(dim=(-1, 2, 129, 21)),
            # conv layer
            nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(21, 3), stride=(2, 1)),
            Swish(),
            # primary capsule
            CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=16,
                         kernel_size=(21, 3), stride=(4, 2), num_iterations=num_iterations),
            # class capsule
            CapsuleLayer(num_capsules=p.NUM_LABELS, num_route_nodes=16 * 9 * 9, in_channels=8,
                         out_channels=16, num_iterations=num_iterations)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16 * p.NUM_LABELS, 1024),
            Swish(),
            nn.Linear(1024, 4096),
            Swish(),
            nn.Linear(4096, 5418),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        h = self.encoder(x).squeeze().transpose(0, 1)

        y_hat = (h ** 2).sum(dim=-1) ** 0.5
        y_hat = F.softmax(y_hat, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = y_hat.max(dim=1)
            y = Variable(torch.eye(p.NUM_LABELS))
            if self.use_cuda:
                y = y.cuda()
            y = y.index_select(dim=0, index=max_length_indices.data)

        x_hat = self.decoder((h * y[:, :, None]).view(x.size(0), -1))

        return y_hat, x_hat


class CapsuleLoss(nn.Module):

    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, x, y, y_hat, x_hat):
        left = F.relu(0.9 - y_hat, inplace=True) ** 2
        right = F.relu(y_hat - 0.1, inplace=True) ** 2

        margin_loss = y * left + 0.5 * (1. - y) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(x) == torch.numel(x_hat)
        x = x.view(x_hat.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(x_hat, x)

        return (margin_loss + 0.0005 * reconstruction_loss) / x.size(0)


if __name__ == "__main__":
    pass

