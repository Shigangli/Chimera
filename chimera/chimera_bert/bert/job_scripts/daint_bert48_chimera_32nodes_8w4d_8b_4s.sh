#!/bin/bash -l
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=bert48_32nodes_chimera_8b_4s_8wx4d_direct.txt


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
srun python main_with_runtime_direct.py \
        --module models.bert48.depth=4 \
        --max_seq_length 128 \
        --train_batch_size 8 \
        --train_path ./bert_dataset/bert_raw/wikipedia.segmented.nltk.txt \
        --bert_config_path configs/bert_config_bert-large-uncased.json \
        --vocab_path ./bert_dataset/bert_raw/bert-large-uncased-vocab.txt \
        --do_train \
        --do_lower_case \
        --num_minibatches 256 \
        --gradient_accumulation_steps 4 --chimera --config_path tests/depth=4/conf_32nodes.json --reverse_config_path tests/depth=4/conf_reverse_pipe_32nodes.json
