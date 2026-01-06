"""
Apply various compression and quantization algorithms to gradients.
"""

import torch
from nvidia import nvcomp

from grad_analysis import load_gradients

QUANT_FACS = (
    100,
    300,
    1000,
    5000,
)

COMP_ALGS = (
    "LZ4",
    #"Cascaded",
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

    print("Codecs:", end="\t")
    print("\t".join(COMP_ALGS))

    for quant_fac in QUANT_FACS:
        print(f"Quant fac: {quant_fac}", end="\t")
        quant_tensor = torch.clamp(grads * quant_fac, -128, 127).to(torch.int8)

        for alg in COMP_ALGS:
            codec = nvcomp.Codec(algorithm=alg)
            comp = codec.encode(torch_to_nvcomp(quant_tensor))
            comp_tensor = nvcomp_to_torch(comp).cuda()
            print(tensor_size(comp_tensor), end="\t")

            # Check correctness
            decomp_tensor = nvcomp_to_torch(codec.decode(comp))
            decomp_tensor = decomp_tensor.float() / quant_fac
            max_diff = torch.max(torch.abs(decomp_tensor - grads))
            print(f"max_diff={max_diff:.4f}", end="\t")

        print()


if __name__ == "__main__":
    main()
