"""
Implement the all reduce operation in a custom DDP communication hook.
"""

import os
import time

import torch
import torch.distributed as dist


def custom_hook(state: None, bucket):
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

        send_req = dist.isend(send_tensor, (rank + 1) % world_size, tag=0)
        recv_req = dist.irecv(recv_tensor, (rank - 1) % world_size, tag=0)
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

        send_req = dist.isend(send_tensor, (rank + 1) % world_size, tag=0)
        recv_req = dist.irecv(recv_tensor, (rank - 1) % world_size, tag=0)
        send_req.wait()
        recv_req.wait()

        tensor[recv_offset : recv_offset + chunk_size] = recv_tensor

    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut
