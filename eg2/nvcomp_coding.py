"""
Wrapper around nvcomp compress and decompress.
"""

import torch
from nvidia import nvcomp

codec = nvcomp.Codec(algorithm="LZ4")


def encode_tensor(tensor):
    """
    torch cuda int8 -> torch cpu int8
    """
    tensor = nvcomp.as_array(tensor.cuda())
    tensor = codec.encode(tensor)
    tensor = torch.frombuffer(bytes(tensor.cpu()), dtype=torch.int8)
    return tensor


def decode_tensor(tensor):
    """
    torch cuda int8 -> torch cpu int8
    """
    tensor = nvcomp.as_array(tensor.cuda())
    tensor = codec.decode(tensor)
    tensor = torch.frombuffer(bytes(tensor.cpu()), dtype=torch.int8)
    return tensor