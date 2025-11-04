"""
Implement the all reduce operation in a custom DDP communication hook.
"""

import torch
import torch.distributed as dist


class EGHookState:
    def __init__(self):
        # Number of hook calls.
        self.calls = 0
        # Total number of parameters transferred.
        self.params = 0
        # Total number of bytes transferred.
        self.bytes = 0


def send_comp(tensor, rank, state: EGHookState):
    """
    Send compressed tensor to rank.

    Stats tracking is done here.
    """
    state.calls += 1
    state.params += tensor.numel()
    state.bytes += tensor.element_size() * tensor.numel()

    req = dist.isend(tensor, rank, tag=0)
    return req


def recv_comp(tensor, rank):
    """
    Receive compressed tensor from rank.
    """
    req = dist.irecv(tensor, rank, tag=0)
    return req


def custom_hook(state: EGHookState, bucket):
    """
    Ring implementation of all reduce.
    """
    tensor = bucket.buffer()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    chunk_size = tensor.numel() // world_size

    # Phase 1
    # After phase 1, rank i has the sum of chunk i + 1
    for i in range(world_size - 1):
        send_chunk_i = (rank - i) % world_size
        recv_chunk_i = (rank - i - 1) % world_size
        send_offset = send_chunk_i * chunk_size
        recv_offset = recv_chunk_i * chunk_size
        send_tensor = tensor[send_offset : send_offset + chunk_size]
        recv_tensor = torch.zeros_like(send_tensor)

        send_req = send_comp(send_tensor, (rank + 1) % world_size, state)
        recv_req = recv_comp(recv_tensor, (rank - 1) % world_size)
        send_req.wait()
        recv_req.wait()

        tensor[recv_offset : recv_offset + chunk_size] += recv_tensor

    # Phase 2
    for i in range(world_size - 1):
        send_chunk_i = (rank - i + 1) % world_size
        recv_chunk_i = (rank - i) % world_size
        send_offset = send_chunk_i * chunk_size
        recv_offset = recv_chunk_i * chunk_size
        send_tensor = tensor[send_offset : send_offset + chunk_size]
        recv_tensor = torch.zeros_like(send_tensor)

        send_req = send_comp(send_tensor, (rank + 1) % world_size, state)
        recv_req = recv_comp(recv_tensor, (rank - 1) % world_size)
        send_req.wait()
        recv_req.wait()

        tensor[recv_offset : recv_offset + chunk_size] = recv_tensor

    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut
