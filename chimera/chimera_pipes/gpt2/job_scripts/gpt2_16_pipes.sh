#!/bin/bash -l
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=gpt2_32nodes_chimera_1b_1s_16pipes.txt


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
srun python main_with_runtime_16pipes.py \
        --module models.depth=32 \
        --train_batch_size 1 \
        --train_data_file ./bert_dataset/gpt2/data/wikitext-2-raw/wiki.train.raw \
        --do_train \
        --num_minibatches 32 \
        --gradient_accumulation_steps 1 \
        --config_path tests/depth=32/conf_32nodes.json \
        --config_path_1 tests/depth=32/conf_32nodes_1.json \
        --config_path_2 tests/depth=32/conf_32nodes_2.json \
        --config_path_3 tests/depth=32/conf_32nodes_3.json \
        --config_path_4 tests/depth=32/conf_32nodes_4.json \
        --config_path_5 tests/depth=32/conf_32nodes_5.json \
        --config_path_6 tests/depth=32/conf_32nodes_6.json \
        --config_path_7 tests/depth=32/conf_32nodes_7.json \
        --reverse_config_path tests/depth=32/conf_reverse_pipe_32nodes.json \
        --reverse_config_path_1 tests/depth=32/conf_reverse_pipe_32nodes_1.json \
        --reverse_config_path_2 tests/depth=32/conf_reverse_pipe_32nodes_2.json \
        --reverse_config_path_3 tests/depth=32/conf_reverse_pipe_32nodes_3.json \
        --reverse_config_path_4 tests/depth=32/conf_reverse_pipe_32nodes_4.json \
        --reverse_config_path_5 tests/depth=32/conf_reverse_pipe_32nodes_5.json \
        --reverse_config_path_6 tests/depth=32/conf_reverse_pipe_32nodes_6.json \
        --reverse_config_path_7 tests/depth=32/conf_reverse_pipe_32nodes_7.json \
        --chimera
