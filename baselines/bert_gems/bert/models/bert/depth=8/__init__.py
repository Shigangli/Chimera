# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage
from transformers.modeling import BertEmbeddings

def arch():
    return "bert"

def model(config, criterion):
    return [
        (lambda: StartingStage(config, 0), ["input0", "input1", "input2"], ["out1", "out0"]),
        (lambda: IntermediateStage(config, 1), ["out1", "out0"], ["out3", "out2"]),
        (lambda: IntermediateStage(config, 2), ["out3", "out2"], ["out5", "out4"]),
        (lambda: IntermediateStage(config, 3), ["out5", "out4"], ["out7", "out6"]),
        (lambda: IntermediateStage(config, 4), ["out7", "out6"], ["out9", "out8"]),
        (lambda: IntermediateStage(config, 5), ["out9", "out8"], ["out11", "out10"]),
        (lambda: IntermediateStage(config, 6), ["out11", "out10"], ["out13", "out12"]),
        (lambda: EndingStage(config, BertEmbeddings(config).word_embeddings.weight, 7), ["out13", "out12"], ["out14"]),
        (lambda: criterion, ["out14"], ["loss"])
    ]
