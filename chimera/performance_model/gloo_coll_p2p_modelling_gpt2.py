from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import time
import os
import os.path as osp
import shlex
import signal
import subprocess
import threading
from typing import Any, Optional, Tuple

import ifcfg
import torch.distributed as distrib

EXIT = threading.Event()
EXIT.clear()
REQUEUE = threading.Event()
REQUEUE.clear()

# Default port to initialized the TCP store on
DEFAULT_PORT = 8738
# Default address of world rank 0
DEFAULT_MASTER_ADDR = "127.0.0.1"

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
INTERRUPTED_STATE_FILE = osp.join(
    os.environ["HOME"], ".interrupted_states", f"{SLURM_JOBID}.pth"
)
os.environ['MASTER_PORT'] = "1234"

def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


def _requeue_handler(signal, frame):
    EXIT.set()
    REQUEUE.set()


def add_signal_handlers():
    signal.signal(signal.SIGINT, _clean_exit_handler)
    signal.signal(signal.SIGTERM, _clean_exit_handler)

    # SIGUSR2 can be sent to all processes to have them cleanup
    # and exit nicely.  This is nice to use with SLURM as scancel <job_id>
    # sets a 30 second timer for the job to exit, and it can take more than
    # 30 seconds for the job to cleanup and exit nicely.  When using NCCL,
    # forcing the job to exit without cleaning up can be bad.
    # scancel --signal SIGUSR2 <job_id> will set no such timer and will give
    # the job ample time to cleanup and exit.
    signal.signal(signal.SIGUSR2, _clean_exit_handler)

    signal.signal(signal.SIGUSR1, _requeue_handler)


def save_interrupted_state(state: Any, filename: str = None):
    r"""Saves the interrupted job state to the specified filename.
        This is useful when working with preemptable job partitions.

    This method will do nothing if SLURM is not currently being used and the filename is the default

    :param state: The state to save
    :param filename: The filename.  Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"
    """
    if SLURM_JOBID is None and filename is None:
        logger.warn("SLURM_JOBID is none, not saving interrupted state")
        return

    if filename is None:
        filename = INTERRUPTED_STATE_FILE

    torch.save(state, filename)


def load_interrupted_state(filename: str = None) -> Optional[Any]:
    r"""Loads the saved interrupted state

    :param filename: The filename of the saved state.
        Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"

    :return: The saved state if the file exists, else none
    """
    if SLURM_JOBID is None and filename is None:
        return None

    if filename is None:
        filename = INTERRUPTED_STATE_FILE

    if not osp.exists(filename):
        return None

    return torch.load(filename, map_location="cpu")


def requeue_job():
    r"""Requeues the job by calling `scontrol requeue ${SLURM_JOBID}`
    """
    if SLURM_JOBID is None:
        return

    if not REQUEUE.is_set():
        return

    distrib.barrier()

    if distrib.get_rank() == 0:
        logger.info(f"Requeueing job {SLURM_JOBID}")
        subprocess.check_call(shlex.split("scontrol requeue {SLURM_JOBID}"))


def get_ifname():
    return ifcfg.default_interface()["device"]

def init_distrib_slurm(backend: str = "gloo"):
    r"""Initializes torch.distributed by parsing environment variables set
        by SLURM when `srun` is used or by parsing environment variables set
        by torch.distributed.launch

    :param backend: Which torch.distributed backend to use

    :returns: Tuple of the local_rank (aka which GPU to use for this process)
        and the TCPStore used for the rendezvous
    """
    assert (
        torch.distributed.is_available()
    ), "torch.distributed must be available"

    #if "GLOO_SOCKET_IFNAME" not in os.environ:
    #    os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    master_port = int(os.environ.get("MASTER_PORT", DEFAULT_PORT))
    master_addr = os.environ.get("MASTER_ADDR", DEFAULT_MASTER_ADDR)

    # Check to see if we should parse from torch.distributed.launch
    if os.environ.get("LOCAL_RANK", None) is not None:
        print("local_rank")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    # Else parse from SLURM is using SLURM
    elif os.environ.get("SLURM_JOBID", None) is not None:
        print("slurm_jobid")
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    # Otherwise setup for just 1 process, this is nice for testing
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1

    distrib.init_process_group(
        backend, rank=world_rank, world_size=world_size
    )
    return world_rank




def main():
    #use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = torch.cuda.is_available()
    print("use_cuda: ", use_cuda)


    device = torch.device("cuda" if use_cuda else "cpu")
    world_rank = init_distrib_slurm("gloo")
    print("world_rank: ", world_rank, "device", device)


    #bert48 16d 
    #allreduce_size = 69571584
    #bert48 8d 
    #allreduce_size = 107360256
    #bert48 4d 
    #allreduce_size = 182937600
    #bert48 2d 
    #allreduce_size = 334092288



    #gpt2 64d
    #allreduce_size = 85317120
    #gpt2 32d
    #allreduce_size = 104994560

    #gpt2 16d
    allreduce_size = 144349440

    #gpt2 8d
    #allreduce_size = 223059200
     

    torch.manual_seed(1 + world_rank)
    if world_rank == 1:
        iten = torch.ones([allreduce_size], dtype=torch.float32, device=device)
    else:
        iten = torch.ones([allreduce_size], dtype=torch.float32, device=device)
        iten += 2.2

    distrib.all_reduce(iten, op=distrib.ReduceOp.SUM)
    start_time = time.time()
    for i in range(8):
        distrib.all_reduce(iten, op=distrib.ReduceOp.SUM)
    print("tensor: ", iten)
    print("Rank: ", world_rank, "mszie: ", allreduce_size, "float32, allreduce time time: ", (time.time()-start_time)/8.0)
   
    #bert 16b
    #p2psize = 2099200

    #bert 8b
    #p2psize = 1049600

    #bert 4b
    #p2psize = 524800

    #gpt2 1b
    p2psize = 808960

    bsend = torch.ones([p2psize], dtype=torch.float32, device=device)

    distrib.barrier()
    start_time = time.time()

    itenrecv = torch.zeros([p2psize], dtype=torch.float32)
    sack = torch.ones([1], dtype=torch.float32)
    rack = torch.zeros([1], dtype=torch.float32)
    itensend = bsend.cpu()
    
    if world_rank == 0: 
        distrib.send(tensor=itensend, dst=1, tag=1)
        distrib.send(tensor=sack, dst=1, tag=2)
        distrib.recv(tensor=itenrecv, src=1, tag=3)
    if world_rank == 1: 
        distrib.recv(tensor=itenrecv, src=0, tag=1)
        distrib.recv(tensor=rack, src=0, tag=2)
        if float(rack[0]) == 1.0:
            distrib.send(tensor=itensend, dst=0, tag=3)

    itenrecv_res = itenrecv.cuda()
    print("recv tensor: ", itenrecv_res)

    if world_rank == 0:
        print("Rank: ", world_rank, "msize: ", p2psize, "float32, ptp time: ", (time.time()-start_time)/2.0)


if __name__ == '__main__':
    main()
