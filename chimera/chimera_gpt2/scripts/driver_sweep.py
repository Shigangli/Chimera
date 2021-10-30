# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import subprocess
import threading


class WorkerInfo(object):
    def __init__(self, ip, port=22, internal_ip=None):
        self.ip = ip
        self.port = port
        self.internal_ip = internal_ip

    def __repr__(self):
        return '%s:%s' % (self.ip, self.port)


def kill_all(workers):
    for worker in workers:
        node_ip = worker.ip
        node_port = worker.port
        subprocess.call(
            "ssh -n %s -p %s -o StrictHostKeyChecking=no \"sudo pkill -9 python*\"" % (
                node_ip, node_port),
            shell=True)

def run(commands, workers, log_file_paths):
    kill_all(workers)
    def run_helper(command, worker, log_file_path):
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        with open(log_file_path, 'w') as f:
            for line in proc.stdout:
                if line.strip() == b"Exception" or b"RuntimeError" in line:
                    print("Command ran into an exception; cleaning up processes...")
                    kill_all(workers)
                    return
                f.write(line.decode())

    threads = []
    for i, (command, worker, log_file_path) in \
        enumerate(zip(commands, workers, log_file_paths)):
        thread = threading.Thread(target=run_helper,
                                  args=(command, worker, log_file_path))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    kill_all(workers)

def run_remote(runtime_cmds, workers, docker_image_name,
               output_dir, mount_directories):
    launch_cmds = []
    log_file_paths = []
    for i, runtime_cmd in enumerate(runtime_cmds):
        # Put IP addresses in a list, then use them.
        if workers[0].internal_ip is not None:
            runtime_cmd = runtime_cmd.format(workers[0].internal_ip)
        else:
            runtime_cmd = runtime_cmd.format(workers[0].ip)
        docker_cmd = 'nvidia-docker run %(mount_directories)s ' \
                     '--net=host ' \
                     '--ipc=host %(docker_image_name)s /bin/bash -c' % {
            "docker_image_name": docker_image_name,
            "mount_directories":
                " ".join(["-v %s:%s" % (x, x)
                          for x in mount_directories])
        }

        log_file_path = '%s/output.log.%d' % (output_dir, i)
        log_file_paths.append(log_file_path)

        node_ip = workers[i].ip
        node_port = workers[i].port
        launch_cmd = '%s \'cd %s; %s\'' % (docker_cmd, command_line_args.code_dir,
                                           runtime_cmd)
        if node_ip != 'localhost' and node_ip != '127.0.0.1':
            launch_cmd = 'ssh -n %s -p %s -o StrictHostKeyChecking=no \"%s\"' % (node_ip,
                                                                                 node_port,
                                                                                 launch_cmd)
            copy_cmd = 'scp -P %s conf.json %s:~; ' % (node_port, node_ip)
            launch_cmd = copy_cmd + launch_cmd
            launch_cmds.append(launch_cmd)
            print(launch_cmd)
        print(log_file_path)
    run(launch_cmds, workers, log_file_paths)

def run_sweep(models, widths_and_depths, per_gpu_batch_sizes,
              technique_to_command_suffix_mapping,
              all_gradient_accumulation_steps, max_seq_lengths):

    for gradient_accumulation_steps in all_gradient_accumulation_steps:
        for max_seq_length in max_seq_lengths:
            for model in models:
                if model.startswith("bert"):
                    template = """mv /home/ubuntu/conf.json .; GLOO_SOCKET_IFNAME=ens3 python -m launch --nnodes %(nnodes)d \\
        --node_rank %(node_rank)d \\
        --nproc_per_node=%(nproc_per_node)d \\
        main_with_runtime.py \\
        --master_addr {0} \\
        --module %(module)s \\
        --max_seq_length %(max_seq_length)d \\
        --train_batch_size %(per_gpu_batch_size)d \\
        --train_path %(data_dir)s/bert/enwiki_corpus_for_bert.200K.postprocess.txt \\
        --bert_config_path configs/bert_config_bert-large-uncased.json \\
        --vocab_path %(data_dir)s/bert/vocab.txt \\
        --do_train \\
        --on_memory \\
        --do_lower_case \\
        --num_minibatches 256 \\
        --gradient_accumulation_steps %(gradient_accumulation_steps)d %(command_suffix)s \\
        --config_path conf.json
    """
                elif model == "gpt2":
                    template = """mv /home/ubuntu/conf.json .; GLOO_SOCKET_IFNAME=ens3 python -m launch --nnodes %(nnodes)d \\
        --node_rank %(node_rank)d \\
        --nproc_per_node=%(nproc_per_node)d \\
        main_with_runtime.py \\
        --master_addr {0} \\
        --module %(module)s \\
        --train_batch_size %(per_gpu_batch_size)d \\
        --train_data_file %(data_dir)s/gpt2/wiki.train.raw \\
        --do_train \\
        --num_minibatches 256 \\
        --gradient_accumulation_steps %(gradient_accumulation_steps)d %(command_suffix)s \\
        --config_path conf.json
    """
                else:
                    raise Exception("Invalid model!")

                for (width, depth) in widths_and_depths:
                    for technique in technique_to_command_suffix_mapping:
                        for per_gpu_batch_size in per_gpu_batch_sizes:
                            num_gpus_per_worker = command_line_args.num_gpus_per_worker
                            if model == "gpt2":
                                module = "models.depth=%d" % depth
                            else:
                                module = "models.%s.depth=%d" % (model, depth)
                            command_suffix = technique_to_command_suffix_mapping[technique]
                            nproc_per_node = min(num_gpus_per_worker, width * depth)
                            nnodes = (width * depth) // num_gpus_per_worker
                            if (width * depth) % num_gpus_per_worker != 0:
                                nnodes += 1
                            if technique == 'dp':
                                module_to_stage_map = [0] * (depth+1)
                                stage_to_rank_map = {"0": list(range(width * depth))}
                            else:
                                module_to_stage_map = list(range(depth)) + [depth-1]
                                stage_to_rank_map = {}
                                offset = 0
                                for i in range(depth):
                                    stage_to_rank_map[str(i)] = list(range(offset, offset + width))
                                    offset += width
                            conf = {
                                "module_to_stage_map": module_to_stage_map,
                                "stage_to_rank_map": stage_to_rank_map,
                            }
                            print(conf)
                            import json
                            with open('conf.json', 'w') as f:
                                json.dump(conf, f)
                            runtime_cmds = []
                            for node_rank in range(nnodes):
                                args = {
                                    'module': module,
                                    'gradient_accumulation_steps': gradient_accumulation_steps,
                                    'nproc_per_node': nproc_per_node, 'nnodes': nnodes,
                                    'node_rank': node_rank,
                                    'per_gpu_batch_size': per_gpu_batch_size,
                                    'command_suffix': command_suffix,
                                    'data_dir': command_line_args.data_dir,
                                }
                                if model.startswith("bert"):
                                    args['max_seq_length'] = max_seq_length
                                runtime_cmd = template % args
                                runtime_cmds.append(runtime_cmd)
                            if model.startswith("bert"):
                            	output_dir = "logs/width=%d/max_seq_length=%d/model=%s/depth=%d/" \
                               		     "gradient_accumulation_steps=%d/" \
                                             "%s_bs=%d" % (width, max_seq_length, model, depth,
                                                           gradient_accumulation_steps,
                                                           technique, per_gpu_batch_size)
                            else:
                                output_dir = "logs/width=%d/model=%s/depth=%d/" \
                                             "gradient_accumulation_steps=%d/" \
                                             "%s_bs=%d" % (width, model, depth,
                                                           gradient_accumulation_steps,
                                                           technique, per_gpu_batch_size)
                            subprocess.call("mkdir -p %s" % output_dir, shell=True)
                            run_remote(runtime_cmds, workers,
                                       command_line_args.docker_image_name,
                                       output_dir=output_dir,
                                       mount_directories=command_line_args.mount_directories)


def read_workers_file(filename):
    workers = []
    with open(filename, 'r') as f:
        for line in f:
            worker = line.strip()
            worker_info = worker.split(":")
            assert len(worker_info) == 2 or len(worker_info) == 3, worker
            internal_ip = None
            if len(worker_info) == 3:
                internal_ip = worker_info[2]
            workers.append(WorkerInfo(ip=worker_info[0],
                                      port=worker_info[1],
                                      internal_ip=internal_ip))
    return workers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docker_image_name', type=str, required=True,
                        help='Docker image name')
    parser.add_argument('--mount_directories', type=str, nargs='+',
                        help='List of directories to mount')
    parser.add_argument('--num_gpus_per_worker', type=int, required=True,
                        help='Number of GPUs per worker')
    parser.add_argument('--code_dir', type=str, required=True,
                        help='Location of code on workers')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Location of data on workers')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet execution')
    command_line_args = parser.parse_args()

    workers = read_workers_file('workers.txt')

    models = ['gpt2']
    all_gradient_accumulation_steps = [1]
    widths_and_depths = [(8, 8)]
    per_gpu_batch_sizes = [2]
    max_seq_lengths = [128]
    technique_to_command_suffix_mapping = {
        'pipedream': '--pipedream',
        'bagpipe+recomputation': '--recompute_step',
        'bagpipe': '',
        'gpipe+recomputation': '--gpipe --recompute_step',
        'gpipe': '--gpipe',
        'dp': '',
        'mp': '--no_input_pipelining',
    }
    run_sweep(models, widths_and_depths, per_gpu_batch_sizes,
              technique_to_command_suffix_mapping,
              all_gradient_accumulation_steps, max_seq_lengths)
