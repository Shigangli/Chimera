import os
import torch.distributed as dist
from torch.utils.data import DataLoader

DEFAULT_MASTER_ADDR = '127.0.0.1'
DEFAULT_MASTER_PORT = '1234'


def init_dist_process_group(backend='nccl', is_high_priority=True):
    if os.environ.get('LOCAL_RANK', None) is not None:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_size = int(os.environ['LOCAL_SIZE'])
    elif os.environ.get('SLURM_JOBID', None) is not None:
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1
        local_size = 1

    if world_size > 1:
        assert dist.is_available()
        master_addr = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)
        master_port = os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT)
        init_method = 'tcp://' + master_addr + ':' + master_port
        if backend == 'nccl' and is_high_priority:
            pg_options = dist.ProcessGroupNCCL.Options(
                is_high_priority_stream=True)
        else:
            pg_options = None
        print(world_rank)
        dist.init_process_group(backend,
                                init_method=init_method,
                                rank=world_rank,
                                world_size=world_size,
                                pg_options=pg_options)
        assert dist.get_rank() == world_rank
        assert dist.get_world_size() == world_size
    return local_rank, local_size, world_rank, world_size


def get_data_fetch_fn(loader: DataLoader):
    fetcher = iter(loader)

    def next_batch():
        try:
            return next(fetcher)
        except StopIteration:
            return None

    return next_batch
