import torch
from torch.utils.cpp_extension import load

_eg_coding = load(
    name="eg_coding",
    sources=["eg_coding.cpp"],
    extra_cflags=["-std=c++20"],
    verbose=True,
)

encode_tensor = _eg_coding.encode_tensor
decode_tensor = _eg_coding.decode_tensor

# Warm up
encode_tensor(torch.randn(1000).long())


def speed_test():
    tensor = (torch.randn(int(1e7)) * 100).long()

    import time

    t1 = time.time()
    encoded = encode_tensor(tensor)
    t2 = time.time()
    decoded = decode_tensor(encoded)
    t3 = time.time()

    correct = torch.equal(tensor, decoded)

    print("Encode time", t2 - t1)
    print("Decode time", t3 - t2)
    print("Correct?", correct)


if __name__ == "__main__":
    speed_test()
