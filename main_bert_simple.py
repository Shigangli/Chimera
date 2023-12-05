import argparse
import os
import random
import math
from contextlib import nullcontext

import yaml

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from transformers import BertTokenizer, BertConfig, BertLayer

from utils import init_dist_process_group
from bert_optim import BertAdam
from bert_dataset import BERTDataset
from bert_optim import PolyWarmUpScheduler
from bert_model import BertForPreTrainingEx

from apex.optimizers import FusedLAMB
import transformers

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
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training.")
parser.add_argument('--num_optimization_steps', default=None, type=int,
                    help="Total number of optimization steps to perform.")
parser.add_argument("--num_epochs", default=None, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--momentum", default=0.9, type=float)


parser.add_argument("--max_grad_norm", type=float, default=1.)

parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for.")
parser.add_argument("--damping", type=float, default=0.01)
parser.add_argument("--inv_interval", type=int, default=1)
parser.add_argument('--weight_scaling', action='store_true')
parser.add_argument('--lars', action='store_true')
parser.add_argument('--adam', action='store_true')
parser.add_argument('--sgd', action='store_true')
# Others
parser.add_argument('--checkpoint_dir', default=None, type=str,
                    help='path to directory to save checkpoints')
parser.add_argument('--save_checkpoint_steps', type=int, default=200)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--seed', type=int, default=1,
                    help="random seed for initialization")
parser.add_argument('--collective_backend',
                    default=dist.Backend.NCCL, type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--profile', action='store_true')
parser.add_argument('--observe_norm', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--subset_size', type=int, default=None)
parser.add_argument('--wandb', action='store_true')


def main():
    total_steps = 0
    for epoch in range(num_epochs):
        if is_distributed:
            dist.barrier()
            # deterministically shuffle based on epoch
            train_loader.sampler.set_epoch(epoch)

        steps_for_this_epoch = min(
            num_steps - total_steps, max_steps_per_epoch)
        train_one_epoch(epoch, total_steps, steps_for_this_epoch)
        total_steps += steps_for_this_epoch

    if is_master:
        if args.checkpoint_dir is not None:
            save_checkpoint(num_epochs, num_steps)
        print('Finished.')


def train_one_epoch(epoch, step, num_steps_for_this_epoch):
    train_iterator = iter(train_loader)
    after_reset = False

    for i in range(num_steps_for_this_epoch):
        if not args.adam:
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
        for optim in optimizers:
            optim.zero_grad()

        total_loss = 0
        total_masked_lm_loss = 0
        total_next_sentence_loss = 0
        for j in range(grad_acc_steps):
            inputs = next(train_iterator)
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            cov_cxt = nullcontext()
            with cov_cxt as cxt:
                outputs = model(**inputs)
                total_loss += float(outputs['loss']) / \
                    num_micro_batches_per_step
                total_masked_lm_loss += float(
                    outputs['masked_lm_loss']) / num_micro_batches_per_step
                total_next_sentence_loss += float(
                    outputs['next_sentence_loss']) / num_micro_batches_per_step
                loss = outputs['loss']
                loss /= num_micro_batches_per_step
                no_sync_if_needed = model.no_sync() \
                    if isinstance(model, DDP) and j < grad_acc_steps - 1 \
                    else nullcontext()
                with no_sync_if_needed:
                    loss.backward()

        for optim in optimizers:
            optim.step()
            if not isinstance(optim, FusedLAMB):
                for pg in optim.param_groups:
                    pg['step'] += 1

        if is_distributed:
            total_loss = torch.tensor(total_loss).to(device)
            total_masked_lm_loss = torch.tensor(
                total_masked_lm_loss).to(device)
            total_next_sentence_loss = torch.tensor(
                total_next_sentence_loss).to(device)
            dist.reduce(total_loss, dst=0)
            dist.reduce(total_masked_lm_loss, dst=0)
            dist.reduce(total_next_sentence_loss, dst=0)

        if (step+i) % args.log_interval == 0:
            if is_master:
                print(f"epoch{epoch+1} step{step+i+1} loss = {float(total_loss)} "
                      f"({float(total_masked_lm_loss)} + {float(total_next_sentence_loss)})", flush=True)
                if args.wandb:
                    lr = optimizers[0].param_groups[0]['lr']
                    log = {'epoch': epoch+1, 'step': step+i+1,
                           'loss': float(total_loss),
                           'masked_lm_loss': float(total_masked_lm_loss),
                           'next_sentence_loss': float(total_next_sentence_loss),
                           'learning_rate': lr}
                    if args.observe_norm:
                        log['p_norm'] = np.sqrt(
                            sum([float(p.data.norm()) ** 2 for p in model.parameters()]))
                        log['g_norm'] = np.sqrt(sum(
                            [float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None]))
                        for pname, p in model.named_parameters():
                            log[f'{pname}_p_norm'] = p.norm()
                            log[f'{pname}_g_norm'] = p.grad.norm()
                    wandb.log(log)

        if args.checkpoint_dir is not None and (step+i+1) % args.save_checkpoint_steps == 0 and is_master:
            save_checkpoint(epoch, step+i+1)


def save_checkpoint(epoch, step):
    state = {
        'epoch': epoch + 1,
        'step': step,
        'model': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer': optimizers[0].state_dict()
    }

    assert os.path.isdir(args.checkpoint_dir)
    ckpt_file_path = os.path.join(
        args.checkpoint_dir, f'epoch{epoch+1}_step{step}.pt')
    torch.save(state, ckpt_file_path)
    print(f'Saved checkpoint to {ckpt_file_path}', flush=True)
    global prev_prev_checkpoint_path, prev_checkpoint_path
    prev_prev_checkpoint_path = prev_checkpoint_path
    prev_checkpoint_path = ckpt_file_path


if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)
    if args.config is not None:
        dict_args.update(yaml.safe_load(open(args.config, 'r')))

    # Setup rank and device
    local_rank, local_size, rank, world_size = init_dist_process_group(
        backend=args.collective_backend)
    assert local_size <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    is_master = rank == 0

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    is_distributed = world_size > 1

    # Prepare BERT model
    bert_config = BertConfig.from_json_file(args.bert_config_path)
    model = BertForPreTrainingEx(config=bert_config).to(device)
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
    elif is_distributed:
        packed_tensor = parameters_to_vector(model.parameters())
        dist.broadcast(packed_tensor, src=0)
        vector_to_parameters(packed_tensor, model.parameters())
    if is_distributed:
        model = DDP(model)
    
    # Prepare BERT dataset
    batch_size = args.batch_size
    max_seq_length = args.max_seq_length
    num_tokens = batch_size * max_seq_length
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    grad_acc_steps = args.gradient_accumulation_steps
    assert local_batch_size % grad_acc_steps == 0
    micro_batch_size = local_batch_size // grad_acc_steps
    tokenizer = BertTokenizer(
        args.vocab_path, do_lower_case=args.do_lower_case)
    train_dataset = BERTDataset(args.corpus_path,
                                tokenizer,
                                seq_len=max_seq_length,
                                corpus_lines=args.corpus_lines,
                                encoding='latin-1',
                                on_memory=args.on_memory)

    if args.subset_size is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(args.subset_size))

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset,
                              sampler=sampler,
                              batch_size=micro_batch_size,
                              drop_last=True,
                              num_workers=args.num_workers)

    # Set the number of optimization steps and epochs
    num_micro_batches_per_step = world_size * grad_acc_steps
    max_steps_per_epoch = len(train_dataset) // batch_size
    num_steps = args.num_optimization_steps
    if num_steps is None:
        assert args.num_epochs, 'num_optimization_steps or num_epochs needs to be specified.'
        num_epochs = args.num_epochs
        num_steps = max_steps_per_epoch * args.num_epochs
    else:
        total_num_samples = num_steps * batch_size
        num_epochs = math.ceil(total_num_samples / len(train_dataset))

    decay_param_group = {'params': [], 'weight_decay': args.weight_decay}
    no_decay_param_group = {'params': [], 'weight_decay': 0.}
    if args.weight_scaling:
        no_decay_param_group['weight_scaling'] = False
    for name, m in model.named_modules():
        if 'word_embeddings' in name:
            continue
        if isinstance(m, nn.LayerNorm):

            no_decay_param_group['params'] += list(m.parameters())
        elif isinstance(m, (nn.Linear, nn.Embedding)):
            if hasattr(m, 'bias') and m.bias is not None:

                no_decay_param_group['params'].append(m.bias)

            decay_param_group['params'].append(m.weight)
    optimizers = []
    lr_schedulers = []
    if args.adam:
        params = [decay_param_group, no_decay_param_group]

        optimizer = BertAdam(params,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_steps,
                             b1=args.beta1,
                             b2=args.beta2,
                             max_grad_norm=args.max_grad_norm,
                             weight_scaling=args.weight_scaling,
                             lars=args.lars)
        for pg in optimizer.param_groups:
            pg['step'] = 0
        lr_schedulers.append(PolyWarmUpScheduler(optimizer,
                                                 warmup=args.warmup_proportion,
                                                 total_steps=num_steps,
                                                 base_lr=args.learning_rate,
                                                 device=device))
        optimizers.append(optimizer)
    else:
        if args.sgd:
            optimizer = torch.optim.SGD([decay_param_group, no_decay_param_group],
                                        lr=args.learning_rate,
                                        momentum=args.momentum)
            for pg in optimizer.param_groups:
                pg['step'] = 0
        else:
            optimizer = FusedLAMB(
                [decay_param_group, no_decay_param_group], lr=args.learning_rate)
        lr_schedulers.append(PolyWarmUpScheduler(optimizer,
                                                 warmup=args.warmup_proportion,
                                                 total_steps=num_steps,
                                                 base_lr=args.learning_rate,
                                                 device=device))
        optimizers.append(optimizer)

    if checkpoint is not None:
        for group in checkpoint['optimizer']['param_groups']:
            group['step'] = 0
            group['lr'] = args.learning_rate
        optimizers[0].load_state_dict(checkpoint['optimizer'])

    unused_keys = []

    unused_keys.extend(['damping', 'inv_interval'])
    if not args.adam:
        unused_keys.extend(['beta1', 'beta2'])
    for key in unused_keys:
        dict_args.pop(key)

    prev_prev_checkpoint_path = prev_checkpoint_path = None
    prev_checkpoint_loss = None
    if is_distributed:
        dist.barrier()
    if is_master:
        if args.wandb:
            wandb.init(entity=os.getenv('WANDB_ENTITY'),
                       project=os.getenv('WANDB_PROJECT'),
                       settings=wandb.Settings(start_method="thread"))
            wandb.config.update(dict_args)
        print('============================')
        print(f'num_epochs: {num_epochs}')
        print(f'num_optimization_steps: {num_steps}')
        print(f'world_size: {world_size}')
        print('----------------------------')
        for key, value in dict_args.items():
            print(f'{key}: {value}')
        print('============================')

    if args.profile:
        with torch.cuda.profiler.profile():
            main()
    else:
        main()
