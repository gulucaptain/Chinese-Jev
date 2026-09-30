"""Process-group setup, sharding, and metric reduction.

The backend is chosen from the run's layout: NCCL when every rank can hold its own GPU,
gloo otherwise. gloo is what makes a CPU-only two-process run possible, which is how the
distributed path gets exercised without a second card.
"""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def distributed_context():
    """(rank, local_rank, world_size, is_distributed) for this process."""
    if not dist.is_available() or not dist.is_initialized():
        return 0, 0, 1, False
    return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size(), True


def init_process_group(device_count=None):
    """Start the group with the backend the hardware allows."""
    if dist.is_initialized():
        return distributed_context()
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1, False
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size == 1:
        return 0, 0, 1, False
    backend = os.environ.get("CHINESE_JEV_BACKEND")
    if backend is None:
        available = device_count if device_count is not None else torch.cuda.device_count()
        # torchrun reports processes per node separately from the global group size.
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        backend = "nccl" if available >= local_world_size else "gloo"
    # A generous timeout: in `chinese-jev run` the other ranks wait at a barrier while
    # rank 0 tokenizes, which on a large corpus takes far longer than the default 10 min.
    dist.init_process_group(backend, timeout=datetime.timedelta(hours=12))
    return distributed_context()


def setup_device(local_rank, world_size):
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if torch.cuda.is_available() and torch.cuda.device_count() >= local_world_size:
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if torch.cuda.is_available() and world_size == 1:
        torch.cuda.set_device(0)
        return torch.device("cuda", 0)
    return torch.device("cpu")


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src=0):
    """Share one rank's value with the rest, so every rank trains the same plan."""
    if not dist.is_initialized():
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_gather_object(obj):
    """Every rank's `obj`, in rank order. The single-process answer is just `[obj]`.

    Used by distributed evaluation to reassemble per-item predictions: each rank scores
    its shard of the split and the metrics are computed over the concatenation, so the
    reported numbers are identical to a single-process run over the same items.
    """
    if not dist.is_initialized():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def reduce_metric(value, device, world_size):
    """Mean a scalar across ranks so every rank logs the global loss, not its own shard's."""
    if not dist.is_initialized() or world_size == 1:
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


def truncate_to_shortest(n_batches):
    """Cut every rank to the shortest shard, given each rank's own batch count.

    A rank that ran out of batches early would leave the others' all-reduce waiting
    forever, so the epoch length is the minimum across ranks rather than each rank's own
    count.
    """
    if not dist.is_initialized():
        return n_batches
    # NCCL only accepts CUDA tensors; setup_device has bound this rank's GPU.
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    n = torch.tensor([int(n_batches)], dtype=torch.long, device=device)
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    return int(n.item())
