import argparse
import os
import random
import math
import pickle
from contextlib import nullcontext
import yaml

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda import nvtx

from transformers import BertTokenizer, BertConfig, BertLayer

from pipeline import PipelineStage, PIPELINE_1F1B, PIPELINE_GPIPE, PIPELINE_CHIMERA, PIPELINE_INTER
from utils import init_dist_process_group
from bert_optim import BertAdam
from bert_dataset import BERTDataset
from bert_model import get_stage_bert_for_pretraining
import auto_schedule
from chimera_pipeline_rank import AutoGeneratePipelineRank, MyPipeLine

#import sys
# sys.stdout.flush()

try:
    import wandb
except ImportError:
    wandb = None


parser = argparse.ArgumentParser()
# Dataset & BERT
parser.add_argument("--corpus_path", default=None, type=str, required=True,
                    help="The input train corpus.")
parser.add_argument('--corpus_lines', default=None, type=int)
parser.add_argument("--vocab_path", type=str, required=True)
parser.add_argument("--on_memory", action='store_true',
                    help="Whether to load train samples into memory or use disk")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--bert_config_path", type=str, required=True,
                    help="config to use.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
# Training
parser.add_argument("--micro_batch_size", default=32, type=int,
                    help="Micro-batch size for training.")
parser.add_argument('--num_optimization_steps', default=None, type=int,
                    help="Total number of optimization steps to perform.")
parser.add_argument("--num_epochs", default=None, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--adam_max_grad_norm", type=float, default=1.)
parser.add_argument("--beta1", default=0.9, type=float,
                    help="beta1 for Adam.")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for.")
parser.add_argument("--damping", type=float, default=0.01)
# Pipeline
parser.add_argument('--pipeline_method', choices=[
                    PIPELINE_1F1B, PIPELINE_GPIPE, PIPELINE_CHIMERA, PIPELINE_INTER], default=PIPELINE_1F1B)
parser.add_argument("--chunks", default=2, type=int,
                    help="Number of chunks for interleaved 1f1b.")
parser.add_argument('--recompute', action='store_true',
                    help='Recompute activations in backward pass')
parser.add_argument('--num_stages', type=int, default=4,
                    help='number of stages in configurable BERT model')
parser.add_argument('--num_pipelines', type=int, default=2,
                    help='number of pipeline')
# Others
parser.add_argument('--checkpoint_dir', default=None, type=str,
                    help='path to directory to save checkpoints')
parser.add_argument('--save_checkpoint_steps', type=int, default=200)
parser.add_argument('--seed', type=int, default=1,
                    help="random seed for initialization")
parser.add_argument('--p2p_backend', default=dist.Backend.GLOO, type=str)
parser.add_argument('--collective_backend',
                    default=dist.Backend.NCCL, type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--profile', action='store_true')

parser.add_argument('--observe_norm', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--wandb', action='store_true')


def main():
    total_steps = 0
    for epoch in range(num_epochs):
        dist.barrier()
        if num_replicas > 1:
            # deterministically shuffle based on epoch
            train_loader.sampler.set_epoch(epoch)

        steps_for_this_epoch = min(
            num_steps - total_steps, max_steps_per_epoch)
        train_one_epoch(epoch, total_steps, steps_for_this_epoch)
        total_steps += steps_for_this_epoch

    if is_master:
        print('Finished.')


def train_one_epoch(epoch, step, num_steps_for_this_epoch):

    num_p2p_comm = num_steps_for_this_epoch * num_micro_batches_per_step
    if interleaved_pipelines:
        stage.start_interleaved_pipeline_comm_threads(num_p2p_comm)
    elif not dual_pipelines:
        stage.start_comm_threads(num_p2p_comm)
        stage.stage_module.train()
    else:
        for index, s in enumerate(stages):
            s.start_comm_threads(
                num_p2p_comm)
            s.stage_module.train()
    if interleaved_pipelines:
        for inter_stage in stage.interleaved_stages:
            inter_stage.start_interleaved_pipeline_comm_threads(num_p2p_comm)
            inter_stage.stage_module.train()

    train_iterator = iter(train_loader)
    train_iterator_for_up_pipe = iter(
        train_loader_for_up_pipe) if dual_pipelines else None

    save_cxt = nullcontext()
    save_cxt_up_pipe = nullcontext()
    with save_cxt as cxt:
        with save_cxt_up_pipe as cxt_up_pipe:

            for i in range(num_steps_for_this_epoch):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                dist.barrier()
                loss = stage.call_pipeline(train_iterator,
                                           num_micro_batches=num_micro_batches_per_step,
                                           data_iterator_for_up_pipe=train_iterator_for_up_pipe,
                                           iteration=step+i)
                for optimizer in optimizers:
                    optimizer.step()

                if (step+i) % args.log_interval == 0:
                    loss = torch.tensor(loss, device=stage.device)
                    dist.reduce(loss, dst=0)
                    loss /= total_num_micro_batches_per_step
                    if dual_pipelines:
                        loss *= 2
                    if is_master:
                        print(
                            f'epoch{epoch+1} step{step+i+1} loss = {float(loss)}')
                        if args.wandb:
                            log = {'epoch': epoch+1, 'step': step+i+1, 'loss': float(loss),
                                   'adam_learning_rate': optimizers[0].get_lr()[0]}
                            if args.observe_norm:
                                log['p_norm'] = np.sqrt(
                                    sum([float(p.data.norm()) ** 2 for p in stage.stage_module.parameters()]))
                                log['g_norm'] = np.sqrt(
                                    sum([float(p.grad.norm()) ** 2 for p in stage.stage_module.parameters()]))
                            wandb.log(log)

                if args.checkpoint_dir is not None and (step+i+1) % args.save_checkpoint_steps == 0 and is_stage_master:
                    state = {
                        'epoch': epoch + 1,
                        'model': stage.stage_module.state_dict(),
                        'optimizer': optimizers[0].state_dict()
                    }
                    assert os.path.isdir(args.checkpoint_dir)
                    ckpt_file_path = os.path.join(
                        args.checkpoint_dir, f'epoch{epoch+1}_step{step+i+1}_stage{rank_to_stage(rank)}.pt')
                    torch.save(state, ckpt_file_path)
                    print(f'Saved checkpoint to {ckpt_file_path}')


if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)
    if args.config is not None:
        dict_args.update(yaml.safe_load(open(args.config, 'r')))

    # Setup rank and device
    local_rank, local_size, rank, world_size = init_dist_process_group(
        backend=args.p2p_backend)
    assert local_size <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    assert world_size > 1
    is_master = rank == 0

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_stages = args.num_stages
    recompute = args.recompute
    chunks = args.chunks
    num_pipelines = args.num_pipelines
    dual_pipelines = args.pipeline_method == PIPELINE_CHIMERA
    interleaved_pipelines = args.pipeline_method == PIPELINE_INTER
    if interleaved_pipelines:
        assert chunks > 1
        assert num_stages % chunks == 0
        assert world_size % (num_stages // chunks) == 0
    else:
        assert world_size % num_stages == 0

    num_ranks_per_stage = int(world_size / num_stages)
    if interleaved_pipelines:
        num_ranks_per_stage = world_size // (num_stages // chunks)
    num_replicas = num_ranks_per_stage

    if dual_pipelines:
        num_replicas *= 2
    is_distributed = num_replicas > 1

    def rank_to_stage(_rank, down_pipe=True):
        if down_pipe:
            return _rank // num_ranks_per_stage
        else:
            return (world_size - 1 - _rank) // num_ranks_per_stage

    def rank_to_stages(_rank, down_pipe=True):
        stages_per_chunk = num_stages // chunks
        stages = []
        for _chunk in range(chunks):
            stages.append(_rank // num_ranks_per_stage +
                          stages_per_chunk * _chunk)
        return stages

    stage_to_ranks = {_stage_id: [] for _stage_id in range(num_stages)}
    num_micro_batches_per_step = num_stages * args.gradient_accumulation_steps
    if dual_pipelines:
        num_micro_batches_per_step //= 2  # each pipeline handles half micro_batches

    for _rank in range(world_size):
        if interleaved_pipelines:
            stages_per_chunk = num_stages // chunks
            for _chunk in range(chunks):
                stage_to_ranks[_rank // num_ranks_per_stage +
                               _chunk * stages_per_chunk].append(_rank)
        elif dual_pipelines:
            # Chimera的stage
            pipeline = AutoGeneratePipelineRank(
                num_stages, num_pipelines, num_micro_batches_per_step*2)
            pipeline.generate_pipeline()
            for pipe in pipeline.up_pipline_list:
                for k, v in pipe.stage_to_rank_map.items():
                    stage_to_ranks[int(k)].append(*v)
            for pipe in pipeline.down_pipeline_list:
                for k, v in pipe.stage_to_rank_map.items():
                    stage_to_ranks[int(k)].append(*v)
            break
        else:
            stage_to_ranks[rank_to_stage(_rank)].append(_rank)
    grad_sync_groups = []
    for _stage_id in range(num_stages):
        grad_sync_groups.append(dist.new_group(ranks=stage_to_ranks[_stage_id],
                                               backend=args.collective_backend))

    # Prepare BERT pipeline stages
    bert_config = BertConfig.from_json_file(args.bert_config_path)
    micro_batch_size = args.micro_batch_size
    max_seq_length = args.max_seq_length

    def get_pipeline_stage(down_pipe=True):
        stage_id = rank_to_stage(rank, down_pipe=down_pipe)
        stage_module = get_stage_bert_for_pretraining(stage_id,
                                                      num_stages,
                                                      bert_config).to(device)
        rank_interval = num_ranks_per_stage if down_pipe else -num_ranks_per_stage
        return PipelineStage(stage_id=stage_id,
                             num_stages=num_stages,
                             stage_module=stage_module,
                             batch_sizes=(micro_batch_size, max_seq_length),
                             pipeline_method=args.pipeline_method,
                             recompute=recompute,
                             prev_rank=rank-rank_interval if stage_id > 0 else None,
                             next_rank=rank+rank_interval if stage_id < num_stages-1 else None,
                             rank=rank,
                             grad_sync_group=grad_sync_groups[stage_id],
                             is_up_pipe=not down_pipe,
                             pipe_stage=[] if down_pipe and dual_pipelines else None,
                             interleaved_stages=[],
                             chunks=chunks,
                             nvtx_tag='' if down_pipe else auto_schedule.TAG_UP_PIPE)

    def get_interleaved_pipeline_stages(down_pipe=True):
        stage_ids = rank_to_stages(rank, down_pipe=down_pipe)
        rank_interval = num_ranks_per_stage if down_pipe else -num_ranks_per_stage
        stages = []
        for i, stage_id in enumerate(stage_ids):
            if i > 0:
                stage_module = get_stage_bert_for_pretraining(stage_id,
                                                              num_stages,
                                                              bert_config).to(device)
                inter_stage = PipelineStage(stage_id=stage_id,
                                            num_stages=num_stages,
                                            stage_module=stage_module,
                                            batch_sizes=(
                                                micro_batch_size, max_seq_length),
                                            pipeline_method=args.pipeline_method,
                                            recompute=recompute,
                                            prev_rank=(
                                                rank-rank_interval+world_size) % world_size if stage_id > 0 else None,
                                            next_rank=(
                                                rank+rank_interval) % world_size if stage_id < num_stages-1 else None,
                                            rank=rank,
                                            grad_sync_group=grad_sync_groups[stage_id],
                                            is_up_pipe=not down_pipe,
                                            pipe_stage=None,
                                            interleaved_stages=[],
                                            chunks=chunks,
                                            nvtx_tag='' if down_pipe else auto_schedule.TAG_UP_PIPE)
                stages.append(inter_stage)

        first_stage_id = stage_ids[0]
        stage_module = get_stage_bert_for_pretraining(first_stage_id,
                                                      num_stages,
                                                      bert_config).to(device)

        return PipelineStage(stage_id=first_stage_id,
                             num_stages=num_stages,
                             stage_module=stage_module,
                             batch_sizes=(micro_batch_size, max_seq_length),
                             pipeline_method=args.pipeline_method,
                             recompute=recompute,
                             prev_rank=(
                                 rank-rank_interval+world_size) % world_size if first_stage_id > 0 else None,
                             next_rank=(
                                 rank+rank_interval) % world_size if first_stage_id < num_stages-1 else None,
                             rank=rank,
                             grad_sync_group=grad_sync_groups[first_stage_id],
                             is_up_pipe=not down_pipe,
                             pipe_stage=None,
                             interleaved_stages=stages,
                             chunks=chunks,
                             nvtx_tag='' if down_pipe else auto_schedule.TAG_UP_PIPE)

    stages = []

    if interleaved_pipelines:
        stage = get_interleaved_pipeline_stages()
    else:
        for i in range(num_pipelines//2):
            stages.append(get_pipeline_stage(False))
        for i in range(num_pipelines//2):
            stages.append(get_pipeline_stage(True))
        for s in stages:
            s.pipe_stage = stages
        stage = stages[0]

    is_stage_master = rank % num_ranks_per_stage == 0

    # Prepare BERT dataset
    tokenizer = BertTokenizer(
        args.vocab_path, do_lower_case=args.do_lower_case)
    train_dataset = BERTDataset(args.corpus_path,
                                tokenizer,
                                seq_len=max_seq_length,
                                corpus_lines=args.corpus_lines,
                                encoding='latin-1',
                                on_memory=args.on_memory)

    def get_train_loader(down_pipe=True):
        sampler = None
        if num_replicas > 1:
            rank_in_replicas = rank_in_stage = rank % num_ranks_per_stage
            if dual_pipelines:
                rank_in_replicas = 2 * rank_in_stage + int(not down_pipe)
            sampler = DistributedSampler(
                train_dataset, num_replicas=num_replicas, rank=rank_in_replicas)
        return DataLoader(train_dataset,
                          sampler=sampler,
                          batch_size=micro_batch_size,
                          drop_last=True,
                          num_workers=args.num_workers)

    train_loader = get_train_loader()
    train_loader_for_up_pipe = get_train_loader(
        down_pipe=False) if dual_pipelines else None

    # Set the number of optimization steps and epochs
    total_num_micro_batches_per_step = num_replicas * num_micro_batches_per_step
    total_num_samples_per_step = total_num_micro_batches_per_step * micro_batch_size
    max_steps_per_epoch = len(train_dataset) // total_num_samples_per_step
    num_steps = args.num_optimization_steps
    if num_steps is None:
        assert args.num_epochs, 'num_optimization_steps or num_epochs needs to be specified.'
        num_epochs = args.num_epochs
        num_steps = max_steps_per_epoch * args.num_epochs
    else:
        total_num_samples = num_steps * total_num_samples_per_step
        num_epochs = math.ceil(total_num_samples / len(train_dataset))

    first_half = rank_to_stage(rank) // (num_stages // 2) == 0

    # Prepare natural gradient preconditioners

    # Prepare optimizers
    def get_optimizer(module):
        decay_param_group = {'params': [], 'weight_decay': args.weight_decay}
        no_decay_param_group = {'params': [], 'weight_decay': 0.}
        for m in module.modules():
            if isinstance(m, nn.LayerNorm):
                no_decay_param_group['params'] += list(m.parameters())
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                if hasattr(m, 'bias') and m.bias is not None:

                    no_decay_param_group['params'].append(m.bias)
                decay_param_group['params'].append(m.weight)
        params = [decay_param_group, no_decay_param_group]

        return BertAdam(params,
                        lr=args.adam_learning_rate,
                        b1=args.beta1,
                        warmup=args.warmup_proportion,
                        t_total=num_steps,
                        max_grad_norm=args.adam_max_grad_norm)

    if dual_pipelines:
        # chimera 需要优化多个pipeline
        optimizers = []
        for s in stages:
            optimizers.append(get_optimizer(s.stage_module))
    else:
        optimizers = [get_optimizer(stage.stage_module)]

    if interleaved_pipelines:
        for inter_stage in stage.interleaved_stages:
            optimizers.append(get_optimizer(inter_stage.stage_module))

    dist.barrier()
    if is_master:
        if args.wandb:
            wandb.init(entity=os.getenv('WANDB_ENTITY'),
                       project=os.getenv('WANDB_PROJECT'))
            wandb.config.update(dict_args)
        print('============================')
        print(f'pipeline_method: {args.pipeline_method}')
        print(f'num_epochs: {num_epochs}')
        print(f'num_optimization_steps: {num_steps}')
        print(f'world_size: {world_size}')
        print(f'num_replica: {num_replicas}')
        print(f'num_pipeline: {num_pipelines}')
        print(f'num_micro_batches_per_step: {num_micro_batches_per_step}')
        print(f'recompute: {recompute}')
        for _stage_id in range(num_stages):
            print(f'stage{_stage_id}: ranks {stage_to_ranks[_stage_id]}')
        print('----------------------------')
        for key, value in dict_args.items():
            print(f'{key}: {value}')
        print('============================')

    if args.profile:
        with torch.cuda.profiler.profile():
            main()
    else:
        main()
