import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP

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


def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = TestModel()
    ddp_model = DDP(model)
    hook_state = HookState()
    ddp_model.register_comm_hook(state=hook_state, hook=int8)

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

        if rank == 0 and step % 20 == 0:
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
