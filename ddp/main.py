import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.cpp_extension import load

cpp = load(
    name="cpp",
    sources=["../eg/cpp.cpp"],
    verbose=True,
)

torch.manual_seed(0)

mnist = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
train_loader = torch.utils.data.DataLoader(
    mnist,
    batch_size=32,
    shuffle=True,
)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


class HookState:
    def __init__(self):
        # Number of hook calls.
        self.calls = 0
        # Total number of parameters transferred.
        self.params = 0
        # Total number of bytes transferred.
        self.bytes = 0


def noop(state, bucket):
    """
    No-Op (sanity check): Returns original tensor.
    """
    data = bucket.buffer()
    fut = torch.futures.Future()
    fut.set_result(bucket.buffer())

    state.calls += 1
    state.params += data.numel()
    state.bytes += data.numel() * data.element_size()

    return fut


def vanilla(state, bucket):
    """
    Vanilla (sanity check): Uses default all-reduce.
    """
    data = bucket.buffer()
    fut = dist.all_reduce(data, async_op=True).get_future()
    def callback(fut):
        return fut.value()[0]

    state.calls += 1
    state.params += data.numel()
    state.bytes += data.numel() * data.element_size()

    return fut.then(callback)


def fp16(state, bucket):
    """
    Casts data to fp16 during comm.
    """
    data = bucket.buffer()
    data = data.half()
    fut = dist.all_reduce(data, async_op=True).get_future()
    def callback(fut):
        return fut.value()[0].float()

    state.calls += 1
    state.params += data.numel()
    state.bytes += data.numel() * data.element_size()

    return fut.then(callback)


def int8(state, bucket):
    """
    Casts and scales data to int8.
    """
    scaling = 127
    data = bucket.buffer()
    data = (data * scaling).char()
    fut = dist.all_reduce(data, async_op=True).get_future()
    def callback(fut):
        return fut.value()[0].float() / scaling

    state.calls += 1
    state.params += data.numel()
    state.bytes += data.numel() * data.element_size()

    return fut.then(callback)


def gather_sanity_check(state, bucket):
    """
    Use all-gather, and then manually sum.
    """
    tensor = bucket.buffer()
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Run all_gather asynchronously
    fut = dist.all_gather(gather_list, tensor, async_op=True).get_future()

    def after_gather(fut):
        gathered = fut.value()  # list of tensors from all ranks
        reduced = torch.stack(gathered).sum(dim=0)
        return reduced

    state.calls += 1
    state.params += tensor.numel() * world_size
    state.bytes += tensor.numel() * tensor.element_size() * world_size

    return fut.then(after_gather)


def gather_object_sanity_check(state, bucket):
    """
    Use all_gather_object, and then manually sum.
    """
    tensor = bucket.buffer()
    world_size = dist.get_world_size()
    gather_list = [None for _ in range(world_size)]

    # TODO: This is using all_gather_object. Very inefficient.
    # Required for variable-length data
    dist.all_gather_object(gather_list, tensor)

    result = torch.stack(gather_list).sum(dim=0)

    fut = torch.futures.Future()
    fut.set_result(result)
    return fut


class ExpGolombCode:
    def __init__(self, k=0):
        self.k = k

    def encode(self, nums):
        codes = [None] * len(nums)
        for i, num in enumerate(nums):
            code = num + (1 << self.k)
            codes[i] = "0" * (int(code).bit_length() - self.k - 1) + bin(code)[2:]
        return codes

    def decode(self, codes):
        nums = torch.zeros(len(codes), dtype=torch.long)
        for i, code in enumerate(codes):
            num = int("0b" + code, base=2)
            nums[i] = num - (1 << self.k)
        return nums

    def streamEncode(self, nums):
        codes = self.encode(nums)
        return "".join(codes)

    def streamDecode(self, streamStr):
        codes = []
        start = 0
        while start < len(streamStr):
            cnt = 0
            while streamStr[start + cnt] == "0":
                cnt += 1
            end = start + 2 * cnt + self.k + 1
            codes.append(streamStr[start:end])
            start = end
        nums = self.decode(codes)
        return nums

EGCode = ExpGolombCode(k=0)
quant_fac = 10000


def eg_coding(state, bucket):
    """
    Use EG codes during comm.
    """
    # Encode tensor with EG.
    state.calls += 1
    grad = bucket.buffer()
    state.params += grad.numel()
    grad = (grad * quant_fac).long()

    sign = torch.sign(grad)
    grad = torch.abs(grad)
    grad_eg = (EGCode.encode(grad.tolist()), sign)
    state.bytes += sum(len(code) for code in grad_eg[0]) // 8 + 1

    # All gather.
    world_size = dist.get_world_size()
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, grad_eg)

    # Decode result.
    grad_list = [EGCode.decode(codes) * sign for codes, sign in gather_list]
    grad_list = [g.float() / quant_fac for g in grad_list]
    reduced = torch.stack(grad_list).sum(dim=0)

    fut = torch.futures.Future()
    fut.set_result(reduced)
    return fut


def eg_coding_cpp(state, bucket):
    """
    Use C++ implementation of EG coding.
    """
    # Encode tensor with EG.
    state.calls += 1
    grad = bucket.buffer()
    state.params += grad.numel()
    grad = (grad * quant_fac).long()

    sign = torch.sign(grad)
    grad = torch.abs(grad).to(torch.uint32)
    grad_eg = (cpp.encode(grad), sign)
    state.bytes += grad_eg[0].numel()

    # All gather.
    world_size = dist.get_world_size()
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, grad_eg)

    # Decode result.
    grad_list = [cpp.decode(codes).long() * sign for codes, sign in gather_list]
    grad_list = [g.float() / quant_fac for g in grad_list]
    reduced = torch.stack(grad_list).sum(dim=0)

    fut = torch.futures.Future()
    fut.set_result(reduced)
    return fut


def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = TestModel()
    ddp_model = DDP(model)
    hook_state = HookState()
    ddp_model.register_comm_hook(state=hook_state, hook=eg_coding_cpp)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    data_iter = iter(train_loader)
    time_start = time.time()
    for step in range(100):
        x, y = next(data_iter)

        optim.zero_grad()
        outputs = ddp_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optim.step()

        if rank == 0 and step % 20 == 0 or True:
            print(f"rank={rank}, step={step}, loss={loss.item()}, batch_size={x.size(0)}")

    elapse = time.time() - time_start

    print(f"Rank {rank} finished training")
    if rank == 0:
        print(f"Elapsed time: {elapse:.2f} seconds")
        print("Hook stats:")
        print(f"  calls: {hook_state.calls}")
        print(f"  params transferred: {hook_state.params}")
        print(f"  bytes transferred: {hook_state.bytes}")


def main():
    model = TestModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Number of parameters: {num_params}")

    print("Begin DDP training")
    world_size = 2
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
