"""
Integrates EG coding with a DDP communication hook.
"""

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


def encode_4le(n: int, device) -> torch.Tensor:
    """
    Encode 4 byte little endian integer as a 4 element uint8 tensor.
    """
    t = torch.tensor((
        n & 0xFF,
        (n >> 8) & 0xFF,
        (n >> 16) & 0xFF,
        (n >> 24) & 0xFF,
    ), dtype=torch.uint8, device=device)
    return t


def decode_4le(t: torch.Tensor) -> int:
    n = t[0].item() | (t[1].item() << 8) | (t[2].item() << 16) | (t[3].item() << 24)
    return n


def ddp_eg_coding(state: None | EGHookState, bucket):
    """
    Use C++ implementation of EG coding to compress gradients.

    Procedure:
    Two all gathers are performed.
    The first is to share the length of each rank's tensor.
    The second shares the actual encoded tensors.

    1. The gradient tensor is quantized and encoded using EG coding.

    2. A 4 byte little endian header is created as a 4 element uint8 tensor.
      It represents the length of this rank's encoded tensor.
      Call this length n_i.
    The 4 element tensor is all gathered and decoded back to an integer.
      Take the maximum length N = max(n_i) among all ranks.

    3. Prepend the 4 element header to the encoded gradient tensor.
      Pad with garbage values to length N + 4.
      This ensures all ranks have the same length tensor,
      and that the tensor is long enough to fit the data of all ranks.

    4. All gather the padded encoded tensors.
      Decode the data, and sum the results.

    state: Optional, EGHookState instance for recording stats.
    bucket: DDP gradient bucket.
    """
    # Encode tensor with EG.
    grad = bucket.buffer()
    grad = (grad * QUANT_FAC).int()
    grad_eg = encode(grad)

    # Update stats.
    if state is not None:
        state.calls += 1
        state.params += grad.numel()
        state.bytes += grad_eg.numel()

    # Create 4 byte header.
    header = encode_4le(grad_eg.numel(), grad_eg.device)

    # All gather header.
    headers = [torch.empty_like(header) for _ in range(dist.get_world_size())]
    dist.all_gather(headers, header)
    lengths = [decode_4le(h) for h in headers]
    max_len = max(lengths)

    # Pad encoded tensor.
    grad_padded = torch.empty(max_len + 4, dtype=torch.uint8, device=grad_eg.device)
    grad_padded[0:4] = header
    grad_padded[4:4 + grad_eg.numel()] = grad_eg

    # All gather gradients.
    world_size = dist.get_world_size()
    gather_list = [torch.empty_like(grad_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, grad_padded)

    # Decode result.
    lengths = [decode_4le(data[0:4]) for data in gather_list]
    datas = [data[4:4 + n] for data, n in zip(gather_list, lengths)]

    grads = [decode(data).long() for data in datas]
    grad = torch.stack(grads).sum(dim=0).float() / QUANT_FAC

    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut
