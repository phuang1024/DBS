"""
Implementation of compression backend, and integration with PyTorch.
"""

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


class ProcessGroupComp(dist.ProcessGroupGloo):
    def __init__(self, store, rank, world_size, timeout):
        super().__init__(store, rank, world_size, timeout)

    def allreduce(self, tensor, opts=dist.AllreduceOptions()):
        """
        Ring implementation of all reduce.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
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


def backend_creator(store, rank, world_size, timeout, **kwargs):
    return ProcessGroupComp(store, rank, world_size, timeout)


def register_backend():
    print("Register comp backend")
    dist.Backend.register_backend("comp", backend_creator, devices=["cpu", "cuda"])
