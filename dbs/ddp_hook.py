"""
Integrates EG coding with a DDP communication hook.
"""

import time

import torch
import torch.distributed as dist

from eg_coding import encode, decode

QUANT_FAC = 1000


class EGHookState:
    def __init__(self):
        # Number of hook calls.
        self.calls = 0
        # Total number of parameters transferred.
        self.params = 0
        # Total number of bytes transferred.
        self.bytes = 0

        # 0: to_cpu
        # 1: quantize
        # 2: encode
        # 3: all_gather
        # 4: decode
        # 5: dequantize
        # 6: to_device
        self.profiling = [0] * 100


def ddp_eg_coding(state: None | EGHookState, bucket):
    """
    Use C++ implementation of EG coding to compress gradients.

    state: Optional, EGHookState instance for recording stats.
    bucket: DDP gradient bucket.
    """
    # Encode tensor with EG.
    grad = bucket.buffer()
    orig_device = grad.device
    #t1 = time.time()
    grad = grad.cpu()
    #t2 = time.time()
    grad = (grad * QUANT_FAC).int()
    #t3 = time.time()
    grad_eg = encode(grad)
    #t4 = time.time()
    #state.profiling[0] += t2 - t1
    #state.profiling[1] += t3 - t2
    #state.profiling[2] += t4 - t3

    if state is not None:
        # Update stats.
        state.calls += 1
        state.params += grad.numel()
        state.bytes += grad_eg.numel()

    # All gather.
    world_size = dist.get_world_size()
    gather_list = [None for _ in range(world_size)]
    #t1 = time.time()
    dist.all_gather_object(gather_list, grad_eg)
    #t2 = time.time()
    #state.profiling[3] += t2 - t1

    # Decode result.
    #t1 = time.time()
    grad_list = [decode(data).long() for data in gather_list]
    #t2 = time.time()
    grad = torch.stack(grad_list).sum(dim=0).float() / QUANT_FAC
    #t3 = time.time()
    grad = grad.to(orig_device)
    #t4 = time.time()
    #state.profiling[4] += t2 - t1
    #state.profiling[5] += t3 - t2
    #state.profiling[6] += t4 - t3

    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut


def _noop(state: EGHookState, bucket):
    """
    No-op DDP hook. Used to records stats.
    """
    data = bucket.buffer()
    if state is not None:
        state.calls += 1
        state.params += data.numel()
        state.bytes += data.numel() * data.element_size()

    dist.all_reduce(data)

    fut = torch.futures.Future()
    fut.set_result(data)

    return fut