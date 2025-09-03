"""
Integrates EG coding with a DDP communication hook.
"""

import torch
import torch.distributed as dist

from eg_coding import encode, decode

QUANT_FAC = 1000


def eg_coding_cpp(state, bucket):
    """
    Use C++ implementation of EG coding to compress gradients.

    state: None.
    bucket: DDP gradient bucket.
    """
    # Encode tensor with EG.
    grad = bucket.buffer()
    grad = (grad * QUANT_FAC).long()
    grad_eg = encode(grad)

    # All gather.
    world_size = dist.get_world_size()
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, grad_eg)

    # Decode result.
    grad_list = [decode(data).long() for data in gather_list]
    grad = torch.stack(grad_list).sum(dim=0).float() / QUANT_FAC

    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut
