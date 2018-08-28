import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.modules import Module

'''
This version of DistributedDataParallel is designed to be used in conjunction with the DistributedSampler
You will be able to enable the distributed MPI-backend PyTorch Training with only 2 lines:
    1. add DistributedSampler in your DataLoader
    2. pass your model to DistributedDataParallel
This usage is exactly the same as the torch.nn.parallel.DistributedDataParallel()
See imagenet example here: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L88

Parameters are broadcasted to the other processes on initialization of DistributedDataParallel,
and will be allreduced at the finish of the backward pass.
'''


class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        self.first_call = True

        def allreduce_params():
            if (self.needs_reduction):
                self.needs_reduction = False
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)

                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)

            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def weight_broadcast(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, 0)
    """
        for p in self.module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            dist.broadcast(p, 0)
    """

    def forward(self, *inputs, **kwargs):
        if self.first_call:
            print("first broadcast start")
            self.weight_broadcast()
            self.first_call = False
            print("first broadcast done")
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)

