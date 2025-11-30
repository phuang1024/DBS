"""
Implement the all reduce operation in a custom DDP communication hook.
"""

import torch
import torch.distributed as dist

from eg_coding import encode_tensor, decode_tensor

QUANT_FAC = 5000
# Mean abs value of grads.
MEAN = 7e-4


class EGHookState:
    def __init__(self):
        # Number of hook calls.
        self.calls = 0
        # Total number of parameters transferred.
        self.params = 0
        # Total number of bytes transferred.
        self.bytes = 0


def compress_tensor(tensor):
    """
    Quantization and encoding.

    tensor: float32
    -> quantization: int8
    -> EG coding: uint64
    -> view as int8: int8
    """
    # Sign quantization
    #tensor = torch.sign(tensor).to(torch.int8)

    # Standard scale quantization
    tensor = torch.clamp(tensor * QUANT_FAC, -127, 127).to(torch.int8)

    tensor = encode_tensor(tensor).view(torch.int8)
    return tensor


def decompress_tensor(tensor):
    tensor = decode_tensor(tensor.view(torch.uint64))

    # Sign quantization
    #tensor = tensor.to(torch.float32) * MEAN

    # Standard scale quantization
    tensor = tensor.to(torch.float32) / QUANT_FAC

    return tensor


def send_and_recv_compressed(send_tensor, send_rank, recv_rank, state: EGHookState):
    """
    Quantize and compress the tensor.
    Send to send_rank.
    Expect to receive the same format from recv_rank.
    Decompress and dequantize received tensor.
    Update stats of send tensor in state.
    """
    state.calls += 1
    state.params += send_tensor.numel()

    # Encode data tensor.
    # send_len_tensor is (original_numel, after_encoding_numel)
    send_len_tensor = torch.zeros((2,), dtype=torch.int32)
    send_len_tensor[0] = send_tensor.numel()
    send_tensor = compress_tensor(send_tensor)
    send_len_tensor[1] = send_tensor.numel()

    state.bytes += send_tensor.element_size() * send_tensor.numel()

    # Send and receive length.
    send_req = dist.isend(send_len_tensor, send_rank, tag=0)
    recv_len_tensor = torch.zeros_like(send_len_tensor)
    recv_req = dist.irecv(recv_len_tensor, recv_rank, tag=0)
    send_req.wait()
    recv_req.wait()

    # Send and receive data tensor.
    recv_tensor = torch.zeros(recv_len_tensor[1].item(), dtype=torch.int8)
    send_req = dist.isend(send_tensor, send_rank, tag=1)
    recv_req = dist.irecv(recv_tensor, recv_rank, tag=1)
    send_req.wait()
    recv_req.wait()

    # Decode received tensor.
    recv_tensor = decompress_tensor(recv_tensor)
    recv_tensor = recv_tensor[:send_len_tensor[0].item()]

    return recv_tensor


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
        # Determine send and recv chunks.
        send_chunk_i = (rank - i) % world_size
        recv_chunk_i = (rank - i - 1) % world_size
        send_offset = send_chunk_i * chunk_size
        recv_offset = recv_chunk_i * chunk_size

        send_tensor = tensor[send_offset : send_offset + chunk_size]
        recv_tensor = send_and_recv_compressed(send_tensor, (rank + 1) % world_size, (rank - 1) % world_size, state)

        tensor[recv_offset : recv_offset + chunk_size] += recv_tensor

    # Phase 2
    for i in range(world_size - 1):
        send_chunk_i = (rank - i + 1) % world_size
        recv_chunk_i = (rank - i) % world_size
        send_offset = send_chunk_i * chunk_size
        recv_offset = recv_chunk_i * chunk_size

        send_tensor = tensor[send_offset : send_offset + chunk_size]
        recv_tensor = send_and_recv_compressed(send_tensor, (rank + 1) % world_size, (rank - 1) % world_size, state)

        tensor[recv_offset : recv_offset + chunk_size] = recv_tensor

    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut
