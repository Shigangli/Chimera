
## Chimera: efficiently training large-scale neural networks with bidirectional pipelines

This is a concise and also fully-fledged verion of Chimera, using BERT pre-training as a case study. We also have 1F1B and GPipe implemented for comparison. We use NCCL as the distributed backend for Allreduces and GLOO as the distributed backend for point-to-point communication between pipeline stages. Activation recomputation is also supported. 


## Publication

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

## License

See [LICENSE](LICENSE).
