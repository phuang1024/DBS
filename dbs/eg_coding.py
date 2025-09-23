import os
import re

import torch
from torch.utils.cpp_extension import load


def _load_extension():
    os.environ.setdefault("FORCE_CUDA", "1")
    sources = [
        "exp_golomb_extension.cpp",
        "expgolomb_cpu.cpp",
        "expgolomb_cuda.cu",
    ]
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        raw_archs = torch.cuda.get_arch_list()
        def _normalize(arch: str) -> str:
            match = re.search(r"(\d+)$", arch)
            if not match:
                return arch
            digits = match.group(1)
            if len(digits) == 1:
                return f"{digits}.0"
            major = digits[:-1]
            minor = digits[-1]
            return f"{int(major)}.{minor}"

        if raw_archs:
            normalized = [_normalize(a) for a in raw_archs]
        else:
            normalized = ["5.0", "6.0", "7.0", "8.0", "8.6"]
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(normalized)
    return load(
        name="exp_golomb_ext",
        sources=[str(src) for src in sources],
        extra_include_paths=["include"],
        extra_cflags=["-std=c++17"],
        extra_cuda_cflags=["-std=c++17"],
        verbose=True,
    )


def _pack_streams(ext, values: torch.Tensor, n_streams: int):
    concat_words: list[int] = []
    offsets_bits: list[int] = []
    bitlen = 0
    for _ in range(n_streams):
        cpu_words, cpu_bits = ext.encode_ue_cpu(values)
        cpu_words = cpu_words.to(torch.uint64).cpu()
        words_list = cpu_words.tolist()
        pad = (64 - (bitlen % 64)) % 64
        bitlen += pad
        offsets_bits.append(bitlen)
        concat_words.extend(int(w) for w in words_list)
        bitlen += int(cpu_bits)
    return (
        torch.tensor(concat_words, dtype=torch.uint64),
        torch.tensor(offsets_bits, dtype=torch.uint64),
        bitlen,
    )


ext = _load_extension()


def main():
    torch.manual_seed(123)
    values = torch.randint(0, 1 << 20, (1 << 18,), dtype=torch.int32)

    # CPU encode/decode
    cpu_words, cpu_bits = ext.encode_ue_cpu(values)
    decoded_cpu = ext.decode_ue_cpu(cpu_words, cpu_bits, values.numel())
    assert torch.equal(values, decoded_cpu), "CPU round-trip failed"

    # GPU encode/decode-many demo
    gpu_values = values.cuda()
    gpu_words, gpu_bits = ext.encode_ue_gpu(gpu_values)
    n_streams = 4
    packed_words, offsets, _ = _pack_streams(ext, values, n_streams)
    decoded_many = ext.decode_ue_many_gpu(packed_words.cuda(), offsets.cuda(), values.numel())
    assert torch.equal(
        decoded_many.cpu(), values.repeat(n_streams).view(n_streams, -1)
    ), "GPU multi-stream decode mismatch"
    print("GPU round-trip check passed.")

    print("PyTorch custom op demo passed: CPU round-trip OK.")


def encode(values):
    words, bits = ext.encode_ue_gpu(values)
    return words, bits


def decode(words, bits, n_values):
    return ext.decode_ue_cpu(words, bits, n_values)


if __name__ == "__main__":
    main()
