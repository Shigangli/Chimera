#!/bin/bash -l
#SBATCH --nodes=512
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=gpt2_512nodes_gems_1b_4steps_64w8d.txt


module load daint-gpu
module load PyTorch


which nvcc
nvidia-smi

which python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ipogif0

#export CUDA_LAUNCH_BLOCKING=1

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
echo $MASTER_ADDR

cd ..
srun python main_with_runtime_gems.py \
        --module models.depth=8 \
        --train_batch_size 1 \
        --train_data_file ./bert_dataset/gpt2/data/wikitext-2-raw/wiki.train.raw \
        --do_train \
        --num_minibatches 12 \
        --gradient_accumulation_steps 4 \
        --config_path tests/depth=8/conf_512nodes.json \
        --reverse_config_path tests/depth=8/conf_reverse_pipe_512nodes.json --gems
