# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import logging
import os
import random
import re
from io import open
import time

import apex.amp as amp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers.modeling import BertForPreTraining, BertConfig, BertPreTrainingHeads
from transformers.tokenization import BertTokenizer
from transformers.optimization import BertAdam, warmup_linear

from torch.utils.data import Dataset
import random

import sys
sys.path.append("..")
import runtime
from torch.optim.optimizer import required
from optimizer_with_stashing import OptimizerWithStashing
from optimizer_with_stashing_and_aggregation import OptimizerWithStashingAndAggregation
from ptflops import get_model_complexity_info

import os.path as osp
import shlex
import signal
import subprocess
import threading
from typing import Any, Optional, Tuple

import ifcfg
import contextlib
import torch.distributed as distrib

EXIT = threading.Event()
EXIT.clear()
REQUEUE = threading.Event()
REQUEUE.clear()
# Default port to initialized the TCP store on
DEFAULT_PORT = 12345
# Default address of world rank 0
DEFAULT_MASTER_ADDR = "127.0.0.1"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
INTERRUPTED_STATE_FILE = osp.join(
    os.environ["HOME"], ".interrupted_states", f"{SLURM_JOBID}.pth"
)
os.environ['MASTER_PORT'] = "12345"

# Helper methods.

def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


def _requeue_handler(signal, frame):
    EXIT.set()
    REQUEUE.set()


def add_signal_handlers():
    signal.signal(signal.SIGINT, _clean_exit_handler)
    signal.signal(signal.SIGTERM, _clean_exit_handler)

    # SIGUSR2 can be sent to all processes to have them cleanup
    # and exit nicely.  This is nice to use with SLURM as scancel <job_id>
    # sets a 30 second timer for the job to exit, and it can take more than
    # 30 seconds for the job to cleanup and exit nicely.  When using NCCL,
    # forcing the job to exit without cleaning up can be bad.
    # scancel --signal SIGUSR2 <job_id> will set no such timer and will give
    # the job ample time to cleanup and exit.
    signal.signal(signal.SIGUSR2, _clean_exit_handler)

    signal.signal(signal.SIGUSR1, _requeue_handler)


def save_interrupted_state(state: Any, filename: str = None):
    r"""Saves the interrupted job state to the specified filename.
        This is useful when working with preemptable job partitions.

    This method will do nothing if SLURM is not currently being used and the filename is the default

    :param state: The state to save
    :param filename: The filename.  Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"
    """
    if SLURM_JOBID is None and filename is None:
        logger.warn("SLURM_JOBID is none, not saving interrupted state")
        return

    if filename is None:
        filename = INTERRUPTED_STATE_FILE

    torch.save(state, filename)


def load_interrupted_state(filename: str = None) -> Optional[Any]:
    r"""Loads the saved interrupted state

    :param filename: The filename of the saved state.
        Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"

    :return: The saved state if the file exists, else none
    """
    if SLURM_JOBID is None and filename is None:
        return None

    if filename is None:
        filename = INTERRUPTED_STATE_FILE

    if not osp.exists(filename):
        return None

    return torch.load(filename, map_location="cpu")


def requeue_job():
    r"""Requeues the job by calling `scontrol requeue ${SLURM_JOBID}`
    """
    if SLURM_JOBID is None:
        return

    if not REQUEUE.is_set():
        return

    distrib.barrier()

    if distrib.get_rank() == 0:
        logger.info(f"Requeueing job {SLURM_JOBID}")
        subprocess.check_call(shlex.split("scontrol requeue {SLURM_JOBID}"))


def get_ifname():
    return ifcfg.default_interface()["device"]

def init_distrib_slurm():
    r"""Initializes torch.distributed by parsing environment variables set
        by SLURM when `srun` is used or by parsing environment variables set
        by torch.distributed.launch

    :param backend: Which torch.distributed backend to use

    :returns: Tuple of the local_rank (aka which GPU to use for this process)
        and the TCPStore used for the rendezvous
    """
    assert (
        torch.distributed.is_available()
    ), "torch.distributed must be available"

    #if "GLOO_SOCKET_IFNAME" not in os.environ:
    #    os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    #if "NCCL_SOCKET_IFNAME" not in os.environ:
    #    os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    master_port = int(os.environ.get("MASTER_PORT", DEFAULT_PORT))
    master_addr = os.environ.get("MASTER_ADDR", DEFAULT_MASTER_ADDR)

    # Check to see if we should parse from torch.distributed.launch
    if os.environ.get("LOCAL_RANK", None) is not None:
        print("local_rank")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_size = int(os.environ["LOCAL_SIZE"])
    # Else parse from SLURM is using SLURM
    elif os.environ.get("SLURM_JOBID", None) is not None:
        print("slurm_jobid")
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    # Otherwise setup for just 1 process, this is nice for testing
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1
        local_size = 1

    return local_rank, local_size, world_rank, world_size, master_addr

# Helper methods.

def save_checkpoint(state, checkpoint_dir, stage, epoch):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir,
                                        "checkpoint.%d.pth.tar.epoch.%d" % (stage, epoch))
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)

class BertAdamWithStashingAndAggregation(OptimizerWithStashingAndAggregation):
    """
    BERT Adam optimizer with weight stashing and aggregation (to reduce memory overhead).
    """
    def __init__(self, modules, master_parameters, num_stages, update_interval,
                 lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0,
                 verbose_freq=0):
        super(BertAdamWithStashingAndAggregation, self).__init__(
            optim_name='BertAdam', modules=modules,
            master_parameters=master_parameters, num_stages=num_stages,
            update_interval=update_interval,
            verbose_freq=verbose_freq, lr=lr, warmup=warmup, t_total=t_total,
            schedule=schedule, b1=b1, b2=b2, e=e, weight_decay=weight_decay,
            base_optimizer_cls=BertAdam
        )

class BertAdamWithStashing(OptimizerWithStashing):
    """
    BERT Adam optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, num_versions,
                 lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0,
                 verbose_freq=0):
        super(BertAdamWithStashing, self).__init__(
            optim_name='BertAdam',
            modules=modules, master_parameters=master_parameters,
            model_parameters=None, loss_scale=1.0,
            num_versions=num_versions, lr=lr, warmup=warmup, t_total=t_total,
            schedule=schedule, b1=b1, b2=b2, e=e, weight_decay=weight_decay,
            verbose_freq=verbose_freq,
            base_optimizer_cls=BertAdam
        )


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=False):
    #def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        self.all_docs.append(doc)
                        doc = []
                        #remove last added sample because there won't be a subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        #store as one sample
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        t1, t2, is_next_label = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = self.tokenizer.tokenize(t2)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next))

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"]+1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t2 = next(self.file).strip()
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t2 == "" or t1 == "":
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = t2

        assert t1 != ""
        assert t2 != ""
        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                #pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            #check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features

class InputSource(runtime.InputSourceBase):
    def __init__(self, dataloader, parameters):
        self.loader_iter = iter(dataloader)
        self.dtype = next(parameters).dtype

    def get_inputs(self):
        input_tensors = {}

        input = next(self.loader_iter)
        batch = tuple(t.to("cuda") for t in input)
        
        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch

        input_mask = input_mask.unsqueeze(1).unsqueeze(2)
        input_mask = input_mask.to(dtype=self.dtype)
        input_mask = (1.0 - input_mask) * -10000.0

        input_tensors["input0"] = input_ids
        input_tensors["input1"] = segment_ids
        input_tensors["input2"] = input_mask
        input_tensors["target_lm"] = lm_label_ids
        input_tensors["target_sentence"] = is_next.view(-1)

        return input_tensors

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")
    parser.add_argument("--partitioned_dataset",
                        action='store_true',
                        default=False,
                        help="Is the dataset partitioned into a number of files.")
    parser.add_argument("--bert_config_path", type=str, required=True,
                        help="config to use.")
    parser.add_argument("--vocab_path", type=str, required=True)

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--num_minibatches', default=None, type=int,
                        help="Number of minibatches to run.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # PipeDream runtime parameters
    parser.add_argument('--module', '-m', required=True,
                        help='name of module that contains model and tensor_shapes definition')
    parser.add_argument('--configurable_model', action='store_true',
                        help='configurable model')
    parser.add_argument('--num_stages', type=int, default=4,
                        help='number of stages in configurable BERT model')
    parser.add_argument('--num_layers_per_stage', type=int, default=1,
                        help='number of layers per stage in configurable BERT model')
    parser.add_argument('--master_addr', default=None, type=str,
                        help="IP address of master (machine with rank 0)")
    parser.add_argument('--config_path', default=None, type=str,
                        help="Path of configuration file")
    parser.add_argument('--reverse_config_path', default=None, type=str,
                        help="Path of configuration file for the reverse pipeline")
    parser.add_argument('--no_input_pipelining', action='store_true',
                        help="No pipelining of inputs")
    parser.add_argument('--rank', default=None, type=int,
                        help="Rank of worker")
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='path to directory to save checkpoints')
    parser.add_argument('--num_ranks_in_server', default=1, type=int,
                        help="number of gpus per machine")
    parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                        help="Log verbose information")
    # Recompute tensors from forward pass, instead of saving them.
    parser.add_argument('--recompute_step', action='store_true',
                        help='Recompute tensors in backward pass')
    # PipeDream-style execution.
    parser.add_argument('--pipedream', action='store_true',
                        help='Use PipeDream-style weight updates with worse memory efficiency')
    # GPipe-style execution.
    parser.add_argument('--gpipe', action='store_true',
                        help='Use GPipe-style weight updates')

    parser.add_argument('--gems', action='store_true',
                        help='Use gems-style weight updates')

    parser.add_argument('--seed',
                        type=int,
                        default=12,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--use_apex',
                        action='store_true',
                        help="Use Apex for data-parallel communication among stages")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--load', type=str, default=None)

    global args
    args = parser.parse_args()

    assert (args.pipedream and args.gpipe) is False

    if args.gems:
        assert (args.gems and (args.reverse_config_path is not None)) is True

    local_rank, local_size, world_rank, world_size, master_addr = init_distrib_slurm()

    n_gpu = torch.cuda.device_count()
    assert local_size <= n_gpu
    args.local_rank = local_rank
    args.rank = world_rank
    args.master_addr = master_addr

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    logger.info("device: {} local_size: {}, distributed training: {}, 16-bits training: {}".format(
        device, local_size, bool(world_size != 1), args.fp16))

    #if args.local_rank == -1 or args.no_cuda:
    #    torch.cuda.set_device(0)
    #    device = torch.device("cuda")
    #    n_gpu = 1
    #else:
    #    n_gpu = torch.cuda.device_count()
    #    logger.info('local_rank: {}, device_count: {}'.format(
    #        args.local_rank,
    #        torch.cuda.device_count()))

    #    torch.cuda.set_device(args.local_rank)
    #    device = torch.device("cuda", args.local_rank)
    #logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #    device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #if n_gpu > 0:
    if local_size > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=args.do_lower_case)
    print("after tokenizer")
    # Prepare model.
    config = BertConfig.from_json_file(args.bert_config_path)
    m = re.match(r'.*bert(\d+).*', args.module)
    if m is not None:
        num_hidden_layers = int(m.group(1))
        config.num_hidden_layers = num_hidden_layers
        args.module = args.module.replace(m.group(1), "")
        print("num_hidden_layers: ", num_hidden_layers, "module name: ", args.module)
    else:
        raise Exception("Invalid --module argument!")
    import importlib
    module = importlib.import_module(args.module)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    arch = None
    if args.configurable_model:
        arch = module.arch(args.num_stages, args.num_layers_per_stage)
        model = module.model(args.num_stages, args.num_layers_per_stage, config, criterion)
    else:
        arch = module.arch()
        model = module.model(config, criterion)

    print("after model prepare")
    input_size = [args.train_batch_size, args.max_seq_length]
    training_tensor_shapes = {
        "input0": input_size, "input1": input_size,
        "input2": [args.train_batch_size, 1, 1, args.max_seq_length],
        "target_lm": input_size, "target_sentence": [args.train_batch_size]}
    dtypes = {"input0": torch.int64, "input1": torch.int64, "input2": torch.float32,
              "target_lm": torch.int64, "target_sentence": torch.int64}
    inputs_module_destinations = {"input0": 0, "input1": 0, "input2": 0}
    target_tensor_names = {"target_lm", "target_sentence"}

    total_flops = 0
    #total_params = 0
    total_trainable_params = 0
    for module_id, (stage_module_fn, inputs, outputs) in enumerate(model[:-1]):  # Skip last layer (loss).
        input_tensors = []
        total_params = 0
        for module_input in inputs:
            if module_input in inputs_module_destinations:
                inputs_module_destinations[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes[module_input]),
                                      dtype=dtypes[module_input]).cuda()
            input_tensors.append(input_tensor)
        stage_module = stage_module_fn()
        stage_module.cuda()
        # PyTorch should not maintain metadata for a backward pass on
        # synthetic inputs. Without the following line, the runtime is
        # as much as 1.5x slower in a full DP configuration.

        module_start_time = None
        module_end_time = None
        with torch.no_grad():
            output_tensors = stage_module(*tuple(input_tensors))
            module_start_time = time.time()
            for i in range(32):
                output_tensors = stage_module(*tuple(input_tensors))
            module_end_time = time.time()
            #print("Average runtime, module id [%d]: , runtime: [%.6f] seconds" % (module_id, (module_end_time-module_start_time)/32.0))

        output_size = 0
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            output_size += torch.numel(output_tensor)
            #print("module id: ", module_id, "output_name: ", output, "output_tensor: ", output_tensor)
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype
        model_complexity_info = get_model_complexity_info(stage_module, tuple(input_tensors), print_per_layer_stat=False)
        current_module_flops = float(model_complexity_info[0].split(" ")[0])
        print("module id [%d], train_batch_size = [%d], max_seq_length = [%d], flops = [%.3f], output_size = [%d] elements" % (module_id, args.train_batch_size, args.max_seq_length, current_module_flops, output_size))

        total_flops += float(model_complexity_info[0].split(" ")[0])
        total_params += sum(p.numel() for p in stage_module.parameters())
        #total_trainable_params += sum(p.numel() for p in stage_module.parameters() if p.requires_grad)
        for name, p in stage_module.named_parameters():
            #if(args.rank == 0):
            #    print("module id: ", module_id, "rank: ", args.rank, "param name: ", name, "param size: ", p.numel())
            if p.requires_grad:
                total_trainable_params += p.numel()
        print("module id: ", module_id, "total params size: ", total_params)
        del stage_module
    print("Total number of floating point operations: %.2f * 10**9" % (
        total_flops * args.train_batch_size * 6))
    print("Total number of parameters: %d" % total_params)
    print("Total number of trainable parameters: %d" % total_trainable_params)
    
    print("after total params")
    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)


    if args.reverse_config_path is not None:
        configuration_maps_reverse = {
            'module_to_stage_map': None,
            'stage_to_rank_map': None,
            'stage_to_depth_map': None
        }
        json_config_file = json.load(open(args.reverse_config_path, 'r'))
        configuration_maps_reverse['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps_reverse['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps_reverse['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps_reverse['stage_to_rank_map'].items()}
        configuration_maps_reverse['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    print("before stageruntime")
    r = runtime.StageRuntime(
        model=model,
        fp16=args.fp16, loss_scale=1.0,  # fp32 training, so disable loss scaling.
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.BERT,
        use_apex=args.use_apex,
        reverse=False)

    if args.reverse_config_path is not None:
        r_reverse = runtime.StageRuntime(
                    model=model,
                    fp16=args.fp16, loss_scale=1.0,  # fp32 training, so disable loss scaling.
                    training_tensor_shapes=training_tensor_shapes,
                    eval_tensor_shapes=eval_tensor_shapes,
                    training_tensor_dtypes=dtypes,
                    inputs_module_destinations=inputs_module_destinations,
                    target_tensor_names=target_tensor_names,
                    configuration_maps=configuration_maps_reverse,
                    master_addr=args.master_addr, rank=args.rank,
                    local_rank=args.local_rank,
                    num_ranks_in_server=args.num_ranks_in_server,
                    verbose_freq=args.verbose_frequency,
                    model_type=runtime.BERT,
                    use_apex=args.use_apex,
                    reverse=True)
    print("after stageruntime")

    update_interval = 1
    if args.pipedream:
        update_interval = 1
    update_interval *= args.gradient_accumulation_steps
    r.update_interval = update_interval
    r.vocab_size = config.vocab_size
    r.cuda()

    if args.reverse_config_path is not None:
        r_reverse.update_interval = update_interval
        r_reverse.vocab_size = config.vocab_size
        r_reverse.cuda()

    flush_groups = []
    flush_group_sizes = []
    stage2rank_map = r.stage_to_rank_map
    stage2rank_map_reverse = r_reverse.stage_to_rank_map
    nstages = r.num_stages
    #stageID_reverse = r_reverse.stage
    for stageID in range(nstages):
        flush_group_ranks = stage2rank_map[stageID] + stage2rank_map_reverse[stageID]
        flush_group_size = float(len(flush_group_ranks))
        #flush_group = distrib.new_group(ranks=flush_group_ranks, backend='nccl')
        flush_group = distrib.new_group(ranks=flush_group_ranks)
        flush_groups.append(flush_group)
        flush_group_sizes.append(flush_group_size)
    

    print("after stage init")
    # Stage needed to determine if current stage is the first stage.
    # num_stages needed to determine if current stage is the last stage.
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining.
    args.stage = r.stage
    if args.reverse_config_path is not None:
        args.stage_reverse = r_reverse.stage

    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not r.is_first_stage():
        args.synthetic_data = True

    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_path)
        if args.partitioned_dataset:
            from dataset import BERTDatasetPartitioned
            train_dataset = BERTDatasetPartitioned(tokenizer=tokenizer, folder=args.train_path,
                                                   max_seq_length=args.max_seq_length)
        else:
            train_dataset = BERTDataset(args.train_path, tokenizer, seq_len=args.max_seq_length,
                                        corpus_lines=None, on_memory=args.on_memory)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // r.num_ranks_in_stage
    print("Number of optimization steps: %d" % num_train_optimization_steps)

    if args.load is not None:
        model.module = torch.load(args.load)

    if args.load is not None:
        if hasattr(model, 'module'):
            model.module = torch.load(args.load)
        else:
            model = torch.load(args.load)

    # Prepare optimizer.
    param_optimizer = list(r.named_parameters())

    if args.reverse_config_path is not None:
        param_optimizer_reverse = list(r_reverse.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.reverse_config_path is not None:
        optimizer_grouped_parameters_reverse = [
            {'params': [p for n, p in param_optimizer_reverse if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer_reverse if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

    if args.pipedream:
        optimizer = BertAdamWithStashing(r.modules(),
                                         optimizer_grouped_parameters,
                                         num_versions=r.num_warmup_minibatches+1,
                                         lr=args.learning_rate,
                                         warmup=args.warmup_proportion,
                                         t_total=num_train_optimization_steps,
                                         verbose_freq=args.verbose_frequency)

    elif args.gpipe or args.no_input_pipelining:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    elif args.gems:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps,
                             flush_group=flush_groups[r.stage],
                             flush_group_size=flush_group_sizes[r.stage],
                             stage_id=r.stage,
                             num_stages=r.num_stages)

        if args.reverse_config_path is not None:
            optimizer_reverse = BertAdam(optimizer_grouped_parameters_reverse,
                                         lr=args.learning_rate,
                                         warmup=args.warmup_proportion,
                                         t_total=num_train_optimization_steps,
                                         flush_group=flush_groups[r_reverse.stage],
                                         flush_group_size=flush_group_sizes[r_reverse.stage],
                                         stage_id=r_reverse.stage,
                                         num_stages=r_reverse.num_stages)
                                     
    else:
        optimizer = BertAdamWithStashingAndAggregation(r.modules(),
                                                       optimizer_grouped_parameters,
                                                       num_stages=r.num_stages,
                                                       update_interval=update_interval,
                                                       lr=args.learning_rate,
                                                       warmup=args.warmup_proportion,
                                                       t_total=num_train_optimization_steps,
                                                       verbose_freq=args.verbose_frequency)
        if args.reverse_config_path is not None:
            optimizer_reverse = BertAdamWithStashingAndAggregation(r_reverse.modules(),
                                                                   optimizer_grouped_parameters_reverse,
                                                                   num_stages=r_reverse.num_stages,
                                                                   update_interval=update_interval,
                                                                   lr=args.learning_rate,
                                                                   warmup=args.warmup_proportion,
                                                                   t_total=num_train_optimization_steps,
                                                                   verbose_freq=args.verbose_frequency)
    modules = r.modules()
    if args.reverse_config_path is not None:
        modules_reverse = r_reverse.modules()

    if args.fp16:
        from apex.optimizers import FP16_Optimizer

        if args.gpipe or args.no_input_pipelining:
            new_modules, optimizers = amp.initialize(
                models=r.modules(), optimizers=[optimizer], opt_level='O3')
            optimizer = optimizers[0]

        elif args.gems:
            new_modules, optimizers = amp.initialize(
                models=r.modules(), optimizers=[optimizer], opt_level='O3')
            optimizer = optimizers[0]

            if args.reverse_config_path is not None:
                new_modules_reverse, base_optimizers_reverse = amp.initialize(
                    models=r_reverse.modules(), optimizers=[optimizer_reverse], opt_level='O3')
                optimizer_reverse = base_optimizers_reverse[0]

        else:
            new_modules, base_optimizers = amp.initialize(
                models=r.modules(), optimizers=[optimizer.base_optimizer],
                opt_level='O3')
            optimizer.base_optimizer = base_optimizers[0]
            if args.reverse_config_path is not None:
                new_modules_reverse, base_optimizers_reverse = amp.initialize(
                    models=r_reverse.modules(), optimizers=[optimizer_reverse.base_optimizer],
                    opt_level='O3')
                optimizer_reverse.base_optimizer = base_optimizers_reverse[0]

        for i in range(len(modules)):
            modules[i] = new_modules[i]
        if args.reverse_config_path is not None:
            for i in range(len(modules_reverse)):
                modules_reverse[i] = new_modules_reverse[i]

    r.initialize_distributed_backend()
    if args.reverse_config_path is not None:
        r_reverse.initialize_distributed_backend()

    if not args.gpipe and not args.no_input_pipelining and not args.gems:
        optimizer.initialize_queue()

        if args.reverse_config_path is not None:
            optimizer_reverse.initialize_queue()

    print("before train")
    torch.cuda.reset_peak_memory_stats()
    global_step = 0
    if args.do_train:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if args.reverse_config_path is not None:
            num_ranks_in_first_stage_reverse = len(configuration_maps_reverse['stage_to_rank_map'][0])

        # The number of processes in each stage is the same in double pipelines
        if args.reverse_config_path is not None:
            assert num_ranks_in_first_stage == num_ranks_in_first_stage_reverse
            num_replicas = num_ranks_in_first_stage + num_ranks_in_first_stage_reverse
        else:
            num_replicas = num_ranks_in_first_stage

        # ddataset_rank is used to partition the dataset
        ddataset_rank = -1
        if args.rank in configuration_maps['stage_to_rank_map'][0]:
            ddataset_rank = args.rank

        if args.reverse_config_path is not None:
            if args.rank in configuration_maps_reverse['stage_to_rank_map'][0]:     
                ddataset_rank = num_ranks_in_first_stage + num_ranks_in_first_stage_reverse - world_size + args.rank

        train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=ddataset_rank)

        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        losses = AverageMeter()

        #if args.local_rank != -1:
        #    import torch.distributed as dist; dist.barrier()
        distrib.barrier()

        if args.num_minibatches is not None:
            args.num_train_epochs = 32
        for epoch in range(int(args.num_train_epochs)):
            epoch_start_time_1 = time.time()
            
            n = len(train_loader)
            if args.num_minibatches is not None:
                n = args.num_minibatches
            # Number of iterations should be multiple of update_interval.
            n = ((n // update_interval)) * update_interval
            print("Number of iterations in one epoch: ", n)

            if r.is_first_stage():
                input_source = InputSource(train_loader, r.parameters())
                r.set_input_source(input_source)

            if args.reverse_config_path is not None:
                if r_reverse.is_first_stage():
                    input_source_reverse = InputSource(train_loader, r_reverse.parameters())
                    r_reverse.set_input_source(input_source_reverse)

            if args.gpipe:
                r.run_training_loop_with_flushes(n, optimizer, args.recompute_step)

            elif args.gems:
                recompute_step = args.recompute_step and r.stage is not None and \
                    (r.stage != (r.num_stages - 1))

                recompute_step_reverse = args.recompute_step and r_reverse.stage is not None and \
                    (r_reverse.stage != (r_reverse.num_stages - 1))

                print("rank = ", args.rank, ", stage = ", r.stage, "reverse_stage = ", r_reverse.stage, "update iv: ", update_interval, "r update iv: ", r.update_interval, " flush_group_ranks: ", flush_group_ranks)

                cumulative_loss = []
                loss = None

                cumulative_loss_reverse = []
                loss_reverse = None
        
                r.train(n)
                r_reverse.train(n)

                #warmup
                start_time = time.time()
                epoch_start_time = time.time()
                base_step = 0
                for i in range(r.update_interval):
                    r.run_forward(recompute_step=recompute_step)
                    if r.is_last_stage():
                        loss = r.loss.item()
                        cumulative_loss.append(loss)
                    r.run_backward(recompute_step=recompute_step)

                    r_reverse.run_forward(recompute_step=recompute_step_reverse)
                    if r_reverse.is_last_stage():
                        loss_reverse = r_reverse.loss.item()
                        cumulative_loss_reverse.append(loss_reverse)
                    r_reverse.run_backward(recompute_step=recompute_step_reverse)

                if r.stage < r_reverse.stage:
                    optimizer.step()
                    optimizer.zero_grad()
                    r._print_training_progress(base_step+r.update_interval-1, n, start_time, epoch_start_time,
                                                  loss, cumulative_loss, args.rank)
                    optimizer_reverse.step()
                    optimizer_reverse.zero_grad()        
                    r_reverse._print_training_progress(base_step+r_reverse.update_interval-1, n, start_time, 
                                                          epoch_start_time, loss_reverse, cumulative_loss_reverse, args.rank)
                else:
                    optimizer_reverse.step()
                    optimizer_reverse.zero_grad()        
                    r_reverse._print_training_progress(base_step+r_reverse.update_interval-1, n, start_time, 
                                                          epoch_start_time, loss_reverse, cumulative_loss_reverse, args.rank)
                    optimizer.step()
                    optimizer.zero_grad()
                    r._print_training_progress(base_step+r.update_interval-1, n, start_time, epoch_start_time,
                                                  loss, cumulative_loss, args.rank)

                print("finish warmup")
        
                epoch_start_time = time.time()
                for base_step in range(r.update_interval, n, r.update_interval):
                    start_time = time.time()

                    for i in range(r.update_interval):
                        r.run_forward(recompute_step=recompute_step)
                        if r.is_last_stage():
                            loss = r.loss.item()
                            cumulative_loss.append(loss)
                        r.run_backward(recompute_step=recompute_step)

                        r_reverse.run_forward(recompute_step=recompute_step_reverse)
                        if r_reverse.is_last_stage():
                            loss_reverse = r_reverse.loss.item()
                            cumulative_loss_reverse.append(loss_reverse)
                        r_reverse.run_backward(recompute_step=recompute_step_reverse)

                    if r.stage < r_reverse.stage:
                        optimizer.step()
                        optimizer.zero_grad()
                        r._print_training_progress(base_step-1, n, start_time, epoch_start_time,
                                                      loss, cumulative_loss, args.rank)
                        optimizer_reverse.step()
                        optimizer_reverse.zero_grad()        
                        r_reverse._print_training_progress(base_step-1, n, start_time, 
                                                              epoch_start_time, loss_reverse, cumulative_loss_reverse, args.rank)
                    else:
                        optimizer_reverse.step()
                        optimizer_reverse.zero_grad()        
                        r_reverse._print_training_progress(base_step-1, n, start_time, 
                                                              epoch_start_time, loss_reverse, cumulative_loss_reverse, args.rank)
                        optimizer.step()
                        optimizer.zero_grad()
                        r._print_training_progress(base_step-1, n, start_time, epoch_start_time,
                                                      loss, cumulative_loss, args.rank)

                #distrib.barrier()
                if epoch > 0:
                    print("Rank = ", args.rank, "epoch: ", epoch, ", Finish one epoch by GEMS.", "Number of update steps: ", (n/r.update_interval-1), " Average time per update step: ", (time.time()-epoch_start_time)/(n/r.update_interval-1))

            else:
                recompute_step = args.recompute_step and r.stage is not None and \
                    (r.stage != (r.num_stages - 1))
                r.run_training_loop(n, optimizer, recompute_step,
                                    args.no_input_pipelining)

                if args.reverse_config_path is not None:
                    recompute_step_reverse = args.recompute_step and r_reverse.stage is not None and \
                        (r_reverse.stage != (r_reverse.num_stages - 1))
                    r_reverse.run_training_loop(n, optimizer_reverse, recompute_step_reverse,
                                                args.no_input_pipelining)

            print("Epoch %d (%d iterations): %.3f seconds" % (
                epoch, n, time.time() - epoch_start_time_1))
            print("Epoch start time: %.3f, epoch end time: %.3f" % (
                epoch_start_time_1, time.time()))

            # Barrier after completing iterations to wait for other ranks to finish.
            if args.local_rank != -1:
                import torch.distributed as dist; dist.barrier()

            should_save_checkpoint = r.rank_in_stage == 0
            #if args.checkpoint_dir and should_save_checkpoint:
            #    save_checkpoint({
            #        'epoch': epoch + 1,
            #        'arch': arch,
            #        'state_dict': r.state_dict(),
            #        'optimizer' : optimizer.state_dict(),
            #    }, args.checkpoint_dir, r.stage, epoch)
    print("after train")
    return


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    #main()
    try:
        main()
    except Exception as e:
        print("Exception in main")
        print(e.args)
        print(str(e))
        print(repr(e))
        print(e)
        exit(-1)
