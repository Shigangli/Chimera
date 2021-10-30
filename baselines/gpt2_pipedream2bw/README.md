# PipeDream-2BW Runtime

This directory contains implementation for the distributed runtime that integrates
model parallelism, pipelining, and data parallelism into PyTorch.

`runtime.py`: Contains the main `StageRuntime` class.

`communication.py`: Simple communication library that sends PyTorch tensors between
a single sender and receiver.

`tests`: Contains a simple test harness for the `send_tensor` and `receive_tensor`
functions in `communication.py`.

`bert/models` and `gpt2/models`: Contains implementations of BERT and GPT-2
models that can be run with the runtime.

## Running throughput experiments comparing PipeDream-2BW with baselines

`bert/main_with_runtime.py` and `gpt2/main_with_runtime.py` are driver programs
for BERT and GPT-2 implementations that use  `StageRuntime`.

`main_with_runtime.py` can be run using driver scripts provided in `runtime/scripts`.
The main driver script has the following command line arguments,

```bash
usage: driver_sweep.py [-h] --docker_image_name DOCKER_IMAGE_NAME
                       [--mount_directories MOUNT_DIRECTORIES [MOUNT_DIRECTORIES ...]]
                       --num_gpus_per_worker NUM_GPUS_PER_WORKER --code_dir
                       CODE_DIR --data_dir DATA_DIR [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --docker_image_name DOCKER_IMAGE_NAME
                        Docker image name
  --mount_directories MOUNT_DIRECTORIES [MOUNT_DIRECTORIES ...]
                        List of directories to mount
  --num_gpus_per_worker NUM_GPUS_PER_WORKER
                        Number of GPUs per worker
  --code_dir CODE_DIR   Location of code on workers
  --data_dir DATA_DIR   Location of data on workers
  --quiet               Quiet execution
```

This sweeps a number of different settings and can also run baselines.
To use this script, a `workers.txt` file is required. This file has the following format,

```bash
PUBLIC_IP1:22:PRIVATE_IP1
PUBLIC_IP2:22:PRIVATE_IP2
...
```

The script `scripts/generate_worker_file.py` shows how to generate this file
automatically if using `p3.16xlarge` instances on Amazon AWS.
