"""
Apply various compression and quantization algorithms to gradients.
"""

import torch
from nvidia import nvcomp

from grad_analysis import load_gradients

import os
import sys
sys.path.append("../eg2")
os.chdir("../eg2")
from eg_coding import encode_tensor, decode_tensor
os.chdir("../model_benchmark")

QUANT_FACS = (
    100,
    300,
    1000,
    5000,
)

COMP_ALGS = (
    "EG",
    "LZ4",
    "Cascaded",
    "GDeflate",
    #"CRC32",
)


def nvcomp_to_torch(array):
    return torch.frombuffer(bytes(array.cpu()), dtype=torch.int8).cuda()


def torch_to_nvcomp(tensor):
    return nvcomp.as_array(tensor)


def tensor_size(tensor) -> str:
    size = tensor.numel() * tensor.element_size()
    return f"{size / 1e6:.2f}MB"


def main():
    grads = load_gradients().cuda()
    print(f"Gradients: shape={grads.shape}, dtype={grads.dtype}, numel={grads.numel()}, total_size={tensor_size(grads)}")

    for quant_fac in QUANT_FACS:
        print(f"Quant fac: {quant_fac}")
        quant_tensor = torch.clamp(grads * quant_fac, -128, 127).to(torch.int8)

        for alg in COMP_ALGS:
            print(f"  Alg: {alg}", end="\t")

            if alg == "EG":
                comp_tensor = encode_tensor(quant_tensor.cpu()).cuda()
            else:
                codec = nvcomp.Codec(algorithm=alg)
                comp = codec.encode(torch_to_nvcomp(quant_tensor))
                comp_tensor = nvcomp_to_torch(comp).cuda()

            print(tensor_size(comp_tensor), end="\t")

            if alg == "EG":
                decomp_tensor = decode_tensor(comp_tensor.cpu()).cuda()
            else:
                decomp_tensor = nvcomp_to_torch(codec.decode(comp))
                #decomp_tensor = decomp_tensor.float() / quant_fac

            # Check correctness
            if decomp_tensor.shape != quant_tensor.shape:
                print("Shape mismatch.")
            else:
                max_diff = torch.max(torch.abs(decomp_tensor - quant_tensor))
                print(f"Max diff: {max_diff}")


if __name__ == "__main__":
    main()
