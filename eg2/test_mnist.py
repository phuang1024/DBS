"""
Train an MNIST model using DDP with EG compression communication hook.
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP

from ddp_hook import custom_hook, EGHookState

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORLD_SIZE = 2
BATCH_SIZE = 32
STEPS = 1000

torch.manual_seed(0)

mnist = torchvision.datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
train_loader = torch.utils.data.DataLoader(
    mnist,
    batch_size=BATCH_SIZE,
    shuffle=False,
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


def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = TestModel().to(DEVICE)
    model.train()

    ddp_model = DDP(model)
    print("Registering custom hook")
    hook_state = EGHookState()
    ddp_model.register_comm_hook(state=hook_state, hook=custom_hook)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    data_iter = iter(train_loader)
    time_start = time.time()
    for step in range(STEPS):
        x, y = next(data_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)

        optim.zero_grad()
        outputs = ddp_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optim.step()

        if rank == 0 and (step % (STEPS // 10) == 0 or step == STEPS - 1):
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
    print("Using device", DEVICE)

    model = TestModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Number of parameters: {num_params}")

    print("Begin DDP training")
    mp.spawn(
        train,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
