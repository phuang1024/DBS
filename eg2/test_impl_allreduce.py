"""
Test implementing the all reduce operation in a custom process group.
"""

import os
import time

import torch
import torch.distributed as dist


class CustomPG(dist.ProcessGroup):
    def __init__(self, pg):
        super().__init__(pg.rank(), pg.size())
        self.pg = pg

    def allreduce(self, tensor, opts=dist.AllreduceOptions()):
        """
        Ring implementation of all reduce.
        """
        world_size = self.pg.size()
        rank = self.pg.rank()
        chunk_size = tensor.numel() // world_size

        # Phase 1
        # After phase 1, rank i has the sum of chunk i + 1
        for i in range(world_size - 1):
            send_chunk_i = (rank - i) % world_size
            recv_chunk_i = (rank - i - 1) % world_size
            send_offset = send_chunk_i * chunk_size
            recv_offset = recv_chunk_i * chunk_size
            send_tensor = tensor[send_offset : send_offset + chunk_size]
            recv_tensor = torch.zeros_like(send_tensor)

            print(rank, send_tensor)
            send_req = dist.isend(send_tensor, (rank + 1) % world_size, tag=0)
            recv_req = dist.irecv(recv_tensor, (rank - 1) % world_size, tag=0)
            send_req.wait()
            recv_req.wait()

            tensor[recv_offset : recv_offset + chunk_size] += recv_tensor

        print("after phase 1", rank, tensor)

        # Phase 2
        for i in range(world_size - 1):
            send_chunk_i = (rank - i + 1) % world_size
            recv_chunk_i = (rank - i) % world_size
            send_offset = send_chunk_i * chunk_size
            recv_offset = recv_chunk_i * chunk_size
            send_tensor = tensor[send_offset : send_offset + chunk_size]
            recv_tensor = torch.zeros_like(send_tensor)

            print(rank, send_tensor)
            send_req = dist.isend(send_tensor, (rank + 1) % world_size, tag=0)
            recv_req = dist.irecv(recv_tensor, (rank - 1) % world_size, tag=0)
            send_req.wait()
            recv_req.wait()

            tensor[recv_offset : recv_offset + chunk_size] = recv_tensor

        print("after phase 2", rank, tensor)
        return None


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

    tensor = torch.ones(256) * (rank + 1)
    # Change this to test all dtypes
    tensor = tensor.to(torch.int64)
    print(f"[{rank}] before allreduce: {tensor}")

    pg_compressed.allreduce(tensor)
    print(f"[{rank}] after allreduce: {tensor}")

    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    world_size = 8
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)
