
## Chimera: efficiently training large-scale neural networks with bidirectional pipelines

Chimera is novel pipeline parallelism approach, which is proposed for efficiently training large-scale neural network models (e.g., BERT, GPT-2/3) on parallel machines (e.g., GPU clusters). The key idea of Chimera is to reduce the number of bubbles in the pipeline, **without** introducing staleness in the training process.
Our implementations are based on PyTorch and adapted from the PipeDream (https://github.com/msr-fiddle/pipedream). We use GLOO as the distributed backend.

```diff
**A concise and also fully-fledged verion of Chimera will be added in the [Chimera-BERT](https://github.com/Shigangli/Chimera/tree/Chimera-BERT) branch.**
```

## Directory Structure

`chimera/chimera_bert`
Bert in Chimera.

`chimera/chimera_gpt2` 
GPT-2 in Chimera.

`chimera/chimera_pipes` 
Chimera generalized to more than two pipelines.

`chimera/performance_model`
Performance modelling for communications.

## Run the Experiments

To install the required Python modules: 

`conda create --name py37 python=3.7`

`source activate py37`

`pip install -r requirements.txt`

We run experiments on GPU clusters with SLURM job scheduler. For example, one can submit a job to the job queue by

`cd ./job_scripts`

`sbatch daint_bert48_32nodes_chimera_4w8d.sh`


## Publication

Chimera is pulished in SC'21, **Best Paper Finalist**. See the [paper](https://dl.acm.org/doi/abs/10.1145/3458817.3476145) for details. To cite our work:
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

## License

See [LICENSE](LICENSE).
