# PipeFisher

The implementation of pipeline-parallel training with Chimera in PyTorch used in [PipeFisher: Efficient Training of Large Language Models Using Pipelining and Fisher Information Matrices](https://arxiv.org/abs/2211.14133) (to appear at MLSys 2023).

## Setup

### Data preparation
https://github.com/microsoft/AzureML-BERT/blob/master/docs/dataprep.md

Please store `wikipedia.segmented.nltk.txt` file under the `bert_data/` directory.

### Installation
```
pip install -r requirements.txt
```
For training, we use `apex.optimizers.FusedLAMB` of [NVIDIA's Apex library](https://github.com/NVIDIA/apex). Please follow the [instruction](https://github.com/NVIDIA/apex#installation) for installing `apex`. 

For profiling, we use [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems). Please make sure you can execute `nsys` command.

Our scripts are intended to run through the SLURM workload manager on a GPU cluster with 1 GPU per node.

## Profiling

### Profiling **Chimera** with 8 stages for BERT-Large on 8 GPUs 
```
sbatch scripts/prof_steps.sh
```
```
sh scripts/plot_cuda_timeline.sh
```
output: `bert_prof/bert-large_chimera_8stages_8gpus_microbs32_acc1.pdf`


By changing the settings of each script, you can run training/profiling on other BERT models, pipeline methods, number of pipeline stages, number of GPUs, etc.