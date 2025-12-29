"""
Script to test nvcomp api.
"""

import torch
from nvidia import nvcomp


def nvcomp_to_torch(array):
    """
    Convert nvcomp array to torch tensor via bytes.

    Converts by creating a bytes object on CPU, and then creating a torch
    tensor from byte data. Returned tensor will be CPU.

    There appears to be a bug when converting via dlpack.
    """
    return torch.frombuffer(bytes(array.cpu()), dtype=torch.int8)


tensor = torch.randn(1000, dtype=torch.float32, device="cuda")
tensor = (tensor * 1000).to(torch.int8)

codec = nvcomp.Codec(algorithm="LZ4")

# Compress
comp = nvcomp.as_array(tensor)
comp = codec.encode(comp)
#comp = torch.from_dlpack(comp.to_dlpack())
comp = nvcomp_to_torch(comp).cuda()
print("Compressed:", comp.shape, comp.dtype, comp.device)

# Decompress
decomp = nvcomp.as_array(comp)
decomp = codec.decode(comp)
#decomp = torch.from_dlpack(decomp.to_dlpack())
decomp = nvcomp_to_torch(decomp).cuda()
print("Decompressed:", decomp.shape, decomp.dtype, decomp.device)

# Check correctness
print("Correct:", torch.allclose(decomp, tensor))
