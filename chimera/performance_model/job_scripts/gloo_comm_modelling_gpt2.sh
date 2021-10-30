#!/bin/bash -l
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --account=g34
#SBATCH --output=gpt2_dpp_1b_1steps_32wx16d_comm_time.txt


module load daint-gpu
module load PyTorch
module load cudatoolkit

which nvcc
nvidia-smi
which python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ipogif0


export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
echo $MASTER_ADDR

srun python gloo_coll_p2p_modelling_gpt2.py
