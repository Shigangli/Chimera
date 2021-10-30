#!/bin/bash -l
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=1-bert48_32nodes_pipedream2bw_16b_1steps_recompute_4wx8d.txt


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
#update interval = 10

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
echo $MASTER_ADDR

cd ..
srun python main_with_runtime.py \
        --module models.bert48.depth=8 \
        --max_seq_length 128 \
        --train_batch_size 16 \
        --train_path ./bert_dataset/bert_raw/wikipedia.segmented.nltk.txt \
        --bert_config_path configs/bert_config_bert-large-uncased.json \
        --vocab_path ./bert_dataset/bert_raw/bert-large-uncased-vocab.txt \
        --do_train \
        --do_lower_case \
        --num_minibatches 512 \
        --gradient_accumulation_steps 1 --recompute_step --config_path tests/depth=8/conf_32nodes.json
