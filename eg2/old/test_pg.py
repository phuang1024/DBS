"""
Test creating a custom process group.
"""

import os

import torch
import torch.distributed as dist


class CustomPG(dist.ProcessGroup):
    def __init__(self, pg):
        rank = pg.rank()
        size = pg.size()
        super().__init__(rank, size)
        self.pg = pg

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        #compressed = [compress(t) for t in tensors]
        work = self.pg.allreduce(tensors, opts)
        work.wait()
        #for i, t in enumerate(tensors):
            #tensors[i].copy_(decompress(compressed[i]))
        return work


def init_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="gloo",  # use Gloo for simplicity here
        rank=rank,
        world_size=world_size,
    )

    # Wrap the current backend
    default_pg = dist.distributed_c10d._get_default_group()
    pg_compressed = CustomPG(default_pg)

    tensor = torch.ones(2, 2) * (rank + 1)
    # Change this to test all dtypes
    tensor = tensor.to(torch.int64)
    print(f"[{rank}] before allreduce: {tensor}")

    pg_compressed.allreduce([tensor])
    print(f"[{rank}] after allreduce: {tensor}")

    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    world_size = 2
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)
