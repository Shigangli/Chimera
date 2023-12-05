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

### Profiling **Chimera** with 8 stages for BERT-Large on 8 GPUs 
```
sbatch scripts/prof_steps.sh
```
```
sh scripts/plot_cuda_timeline.sh
```
output: `bert_prof/bert-large_chimera_8stages_8gpus_microbs32_acc1.pdf`



### Publication

Chimera is pulished in SC'21, **Best Paper Finalist**. See the [paper](https://dl.acm.org/doi/abs/10.1145/3458817.3476145) and the [video talk](https://dl.acm.org/doi/abs/10.1145/3458817.3476145#sec-supp) for more details. To cite our work:
```bibtex
@inproceedings{li143,
  author = {Li, Shigang and Hoefler, Torsten},
  title = {Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines},
  year = {2021},
  isbn = {9781450384421},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3458817.3476145},
  doi = {10.1145/3458817.3476145},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  articleno = {27},
  numpages = {14},
  location = {St. Louis, Missouri},
  series = {SC '21}
}

```

### License

See [LICENSE](LICENSE).
