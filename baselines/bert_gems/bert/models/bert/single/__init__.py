# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stage0 import Stage0
from .stage1 import Stage1
from transformers.modeling import BertEmbeddings

def arch():
    return "bert"

def model(config, criterion):
    return [
        (lambda: Stage0(config), ["input0", "input1", "input2"], ["out1", "out0"]),
        (lambda: Stage1(config, BertEmbeddings(config).word_embeddings.weight), ["out1", "out0"], ["out2"]),
        (lambda: criterion, ["out2"], ["loss"])
    ]
