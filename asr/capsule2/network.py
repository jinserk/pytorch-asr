#!python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import params as p

COORDINATE_SCALE = 10.


class ConvCapsule(nn.Module):

    def __init__(self, in_channel, in_dim, out_channel, out_dim, kernel_size, stride, routing = 0):
        super(ConvCapsule, self).__init__()

        self.in_channel = in_channel
        self.in_dim = in_dim
        self.out_channel = out_channel
        self.out_dim = out_dim
        self.routing = routing
        self.kernel_size = kernel_size
        self.stride = stride

        if self.routing:
            self.routing_capsule = nn.Conv2d(in_channels=(kernel_size * kernel_size * in_dim * in_channel),
                                             out_channels=(kernel_size * kernel_size * out_dim * in_channel * out_channel),
                                             kernel_size=1,
                                             stride=1,
                                             groups=(kernel_size * kernel_size * in_channel))
        else:
            self.no_routing_capsule = nn.Conv2d(in_channels=(in_channel * (in_dim + 1)),
                                                out_channels=(out_channel * (out_dim + 1)),
                                                kernel_size=kernel_size,
                                                stride=stride)

    def squash(self, tensor):
        # no sure about this operation, may cause error
        size = tensor.size()
        if (len(tensor.size()) < 5):
            # [batch, channel, h, w] --> [batch, cap_channel, cap_dim, h, w]
            tensor = torch.stack(tensor.split(self.out_dim, dim=1), dim = 1)
        squared_norm = (tensor ** 2).sum(dim=2, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        outputs = scale * tensor / torch.sqrt(squared_norm)
        return outputs.view(size)

    def down_h(self, h):
        return range(h*self.stride, h*self.stride+self.kernel_size)

    def down_w(self, w):
        return range(w*self.stride, w*self.stride+self.kernel_size)

    def EM_routing(self, votes, activations):
        # routing coefficient
        R = (1. / self.out_channel) * Variable(torch.ones(self.batches, self.in_channel, self.kernel_size, self.kernel_size,
                                                          self.out_channel, self.out_h, self.out_w), requires_grad=False).cuda()
        votes_reshape = votes.view(self.batches, self.in_channel, self.kernel_size, self.kernel_size,
                                   self.out_channel, self.out_dim, self.out_h, self.out_w)
        activations = activations.squeeze(dim=2)

        a_reshape = [activations[:, :, :, self.down_w(w)][:,:,self.down_h(h),:] for h in range(self.out_h) for w in range(self.out_w)]
        a_stack = torch.stack(a_reshape, dim=4).view(self.batches, self.in_channel, self.kernel_size, self.kernel_size, self.out_h, self.out_w)
        for _ in range(self.routing):
            # M-STEP
            # r_hat.size = [b, in_c, k, k, out_c, out_h, out_w]
            r_hat = R * a_stack[:,:,:,:,None,:,:]
            # sum_r_hat.size = [b, out_c, out_h, out_w]
            sum_r_hat = r_hat.sum(3).sum(2).sum(1)
            # u_h.size = [b, out_c, out_d, out_h, out_w]
            u_h = torch.sum(r_hat[:,:,:,:,:,None,:,:] * votes_reshape, dim=3).sum(2).sum(1) / sum_r_hat[:,:,None,:,:]
            # sigma_h_square.size = [b, out_c, out_d, out_h, out_w]
            sigma_h_square = torch.sum(r_hat[:,:,:,:,:,None,:,:] * (votes_reshape - u_h[:,None,None,None,:,:,:,:]) ** 2, dim=3).sum(2).sum(1) / sum_r_hat[:,:,None,:,:]
            # cost_h.size = [b, out_c, out_d, out_h, out_w]
            cost_h = (self.beta_v[None,:,None,:,:] + torch.log(torch.sqrt(sigma_h_square))) * sum_r_hat[:, :, None, :, :]
            # a_hat.size = [b, out_c, out_h, out_w]
            a_hat = torch.sigmoid(self.lamda * (self.beta_a[None,:,:,:] - cost_h.sum(2)))

            # E-STEP
            # sigma_product.size = [b, out_c, out_h, out_w]
            sigma_product = Variable(torch.ones(self.batches, self.out_channel, self.out_h, self.out_w), requires_grad=False).cuda()
            for dm in range(self.out_dim):
                sigma_product = sigma_product * 2 * 3.1416 *sigma_h_square[:,:,dm,:,:]
            # p_c.size = [b, in_c, k, k, out_c, out_h, out_w]
            p_c = torch.exp(-torch.sum((votes_reshape - u_h[:,None,None,None,:,:,:,:]) ** 2 / (2 * sigma_h_square[:,None,None,None,:,:,:,:]), dim=5) / torch.sqrt(sigma_product[:,None,None,None,:,:,:]))
            # R,size = [b,in_c, k, k, out_c, out_h, out_w]
            R = a_hat[:,None,None,None,:,:,:] * p_c / torch.sum(a_hat[:,None,None,None,:,:,:] * p_c, dim=6, keepdim=True).sum(dim=5, keepdim=True).sum(dim=4, keepdim=True)
        return a_hat, u_h

    def forward(self, x, lamda=0):
        if self.routing:
            size = x.size()
            self.batches = size[0]
            out_h = int((size[2] - self.kernel_size) / self.stride) + 1
            out_w = int((size[3] - self.kernel_size) / self.stride) + 1
            self.out_h = out_h
            self.out_w = out_w
            try:
                self.beta_v
            except AttributeError:
                self.beta_v = Variable(torch.randn(self.out_channel, self.out_h, self.out_w)).cuda()
                self.beta_a = Variable(torch.randn(self.out_channel, self.out_h, self.out_w)).cuda()
            self.lamda = lamda
            x_reshape = x.view(size[0], self.in_channel, 1+self.in_dim, size[2], size[3])
            activations = x_reshape[:,:,0,:,:]
            vector = x_reshape[:,:,1:,:,:].contiguous().view(size[0], -1, size[2], size[3])
            # sampling
            # z[batch, k*k*vhannel, out_h, out_w]
            maps = []
            for k_h in range(self.kernel_size):
                for k_w in range(self.kernel_size):
                    onemap = [vector[:, :, k_h+i, k_w+j] for i in range(0, out_h*self.stride, self.stride) for j in range(0, out_w*self.stride, self.stride)]
                    onemap = torch.stack(onemap, dim=2)
                    onemap = onemap.view(size[0], onemap.size(1), out_h, out_w)
                    maps.append(onemap)
            # maps channel is kernal_size**2 * in_channel * in_dim
            map_ = torch.cat(maps, dim=1)

            # votes.size: (out_h * out_w) * k * k * in_channel * out_channel( * D)
            votes = self.routing_capsule(map_)
            # output_a.size = [b, out_c, out_h, out_w]
            # output_v.size = [b, out_c, out_d, out_h, out_w]
            output_a, output_v = self.EM_routing(votes, activations)
            outputs = torch.cat([output_a[:,:,None,:,:], output_v], dim=2)
            return outputs.view(self.batches, self.out_channel * (self.out_dim + 1), self.out_h, self.out_w)
        else:
            # outputs [batch, channel, out_h, out_w]
            outputs = self.no_routing_capsule(x)
            return outputs


class ClassCapsule(nn.Module):

    def __init__(self, in_channel, in_dim, classes, out_dim, routing):
        super(ClassCapsule, self).__init__()
        self.in_channel = in_channel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.classes = classes
        self.routing = routing

        self.beta_v = Variable(torch.randn(self.classes)).cuda()
        self.beta_a = Variable(torch.randn(self.classes)).cuda()
        self.capsules = nn.Conv2d(in_channels=in_channel * in_dim,
                                  out_channels=in_channel * out_dim * classes,
                                  kernel_size=1,
                                  stride=1,
                                  groups=in_channel)

    def EM_routing(self, votes, activations):
        # routing coefficient
        R = (1. / self.classes) * Variable(torch.ones(self.batches, self.in_channel, self.classes, self.h, self.w), requires_grad=False).cuda()
        # activations.size = [b, in_channel, h, w]
        activations = activations.squeeze(dim=2)
        votes_reshape = votes.view(self.batches, self.in_channel, self.classes, self.out_dim, self.h, self.w)
        for _ in range(self.routing):
            # r_hat.size = [b, in_c, classes, h, w]
            r_hat = R * activations[:,:,None,:,:]
            # sum_r_hat.size = [b, classes]
            sum_r_hat = r_hat.sum(4).sum(3).sum(1)
            # votes_reshape.size = [b, in_channel, classes, out_dim, h, w]
            # u_h.size = [b, classes, out_dim]
            u_h = torch.sum(r_hat[:,:,:,None,:,:] * votes_reshape, dim=5).sum(4).sum(1) / sum_r_hat[:,:,None]
            sigma_h_square = torch.sum(r_hat[:,:,:,None,:,:]*(votes_reshape - u_h[:,None,:,:,None,None])**2, dim=5).sum(4).sum(1) / sum_r_hat[:,:,None]
            # cost_h.size = [b, classes, out_dim]
            cost_h = (self.beta_v[None,:,None] + torch.log(sigma_h_square)) * sum_r_hat[:,:,None]
            # a_hat.size = [b,classes]
            a_hat = torch.sigmoid(self.lamda * (self.beta_a[None,:] - torch.sum(cost_h, dim=2)))

            sigma_product = Variable(torch.ones(self.batches, self.classes), requires_grad=False).cuda()
            for dm in range(self.out_dim):
                sigma_product = 2 * 3.1416 * sigma_product * sigma_h_square[:,:,dm]
            # p_c.size = [b, in_channel, classes, h, w]
            p_c = torch.exp(-torch.sum((votes_reshape - u_h[:,None,:,:,None,None])**2 / (2 * sigma_h_square[:,None,:,:,None,None]), dim=3)) / torch.sqrt(sigma_product[:,None,:,None,None])
            R = a_hat[:,None,:,None,None] * p_c / torch.sum(a_hat[:,None,:,None,None] * p_c, dim=2, keepdim=True)
        return a_hat, u_h

    def CoordinateAddition(self, vector):
        output = Variable(torch.zeros(vector.size())).cuda()
        coordinate_x = Variable(torch.FloatTensor(torch.arange(0, self.h))/COORDINATE_SCALE, requires_grad=False).cuda()
        coordinate_y = Variable(torch.FloatTensor(torch.arange(0, self.w))/COORDINATE_SCALE, requires_grad=False).cuda()
        output[:,:,0,:,:] = vector[:,:,0,:,:] + coordinate_x[None,None,:,None]
        output[:,:,1,:,:] = vector[:,:,1,:,:] + coordinate_y[None,None,None,:]
        if output.size(2) >2:
            output[:,:,2:,:,:] = vector[:,:,2:,:,:]
        return output

    def forward(self, x, lamda=0):
        self.lamda = lamda
        size = x.size()
        self.batches = size[0]
        self.h = size[2]
        self.w = size[3]
        x_reshape = x.view(size[0], self.in_channel, 1+self.in_dim, size[2], size[3])
        activations = x_reshape[:,:,0,:,:]
        vector = x_reshape[:,:,1:,:,:]
        vec = self.CoordinateAddition(vector)
#        for i in range(self.h):
#            for j in range(self.w):
#                vector[:,:,0,i,j] += i/10.
#                vector[:,:,1,i,j] += j/10.
        # vector.size = [b,in_channel*in_dim, h,w]
        vec = vec.view(size[0], -1, size[2], size[3])
        # votes.size = [b, in_channel*out_dum*classes, h,w]
        votes = self.capsules(vec)
        # output_a.size = [b, classes]
        # output_v.size = [b, classes, out_dim]
        output_a, output_v = self.EM_routing(votes, activations)
        return output_a


class CapsuleNet(nn.Module):

    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.Conv1 = ConvCapsule(1, 0, 32, 0, 5, 2)
        self.PrimaryCaps = ConvCapsule(32, 0, 32, 16, 1, 1)
        self.ConvCaps1 = ConvCapsule(32, 16, 32, 16, 3, 2, routing=3)
        self.ConvCaps2 = ConvCapsule(32, 16, 32, 16, 3, 1, routing=3)
        self.ClassCaps = ClassCapsule(32, 16, 10, 16, routing=3)

    def forward(self, x, lamda):
        x = F.relu(self.Conv1(x, lamda))
        x = F.relu(self.PrimaryCaps(x, lamda))
        x = F.sigmoid(self.ConvCaps1(x, lamda))
        x = self.ConvCaps2(x, lamda)
        x = self.ClassCaps(x, lamda)
        return x


def SpreadLoss(self, output, target, m):
    one_shot_target = torch.eye(p.NUM_LABELS).index_select(dim=0, index=target)
    a_t = torch.sum(output * one_shot_target, dim=1)
    loss = torch.sum(max(m - (a_t -output)) ** 2, dim=1) - m ** 2
    return loss


if __name__ == "__main__":
    # test ConvCapsule
    torch.cuda.manual_seed(1)
    layer = ConvCapsule(2, 2, 2, 2, 3, 2, routing=3, lamda=Variable(torch.rand(1)).cuda())
    layer.cuda()
    x = Variable(torch.rand(2,6,5,5))
    y = layer(x.cuda())

    # test ClassCapsule
    layer = ClassCapsule(2, 2, 2, 2, 3)
    layer.cuda()
    x = Variable(torch.rand(3,6,5,5))
    lamda=Variable(torch.rand(1))
    y = layer(x.cuda(), lamda.cuda())
    print(y.size())

