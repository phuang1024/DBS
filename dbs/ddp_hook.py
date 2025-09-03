"""
Integrates EG coding with a DDP communication hook.
"""

import torch
import torch.distributed as dist

from eg_coding import encode, decode

QUANT_FAC = 500


class EGHookState:
    def __init__(self):
        # Number of hook calls.
        self.calls = 0
        # Total number of parameters transferred.
        self.params = 0
        # Total number of bytes transferred.
        self.bytes = 0


def ddp_eg_coding(state: None | EGHookState, bucket):
    """
    Use C++ implementation of EG coding to compress gradients.

    state: Optional, EGHookState instance for recording stats.
    bucket: DDP gradient bucket.
    """
    # Encode tensor with EG.
    grad = bucket.buffer()
    grad = (grad * QUANT_FAC).int()
    grad_eg = encode(grad)

    if state is not None:
        # Update stats.
        state.calls += 1
        state.params += grad.numel()
        state.bytes += grad_eg.numel()

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
