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
        (lambda: IntermediateStage(config, 7), ["out13", "out12"], ["out15", "out14"]),
        (lambda: IntermediateStage(config, 8), ["out15", "out14"], ["out17", "out16"]),
        (lambda: IntermediateStage(config, 9), ["out17", "out16"], ["out19", "out18"]),
        (lambda: IntermediateStage(config, 10), ["out19", "out18"], ["out21", "out20"]),
        (lambda: IntermediateStage(config, 11), ["out21", "out20"], ["out23", "out22"]),
        (lambda: IntermediateStage(config, 12), ["out23", "out22"], ["out25", "out24"]),
        (lambda: IntermediateStage(config, 13), ["out25", "out24"], ["out27", "out26"]),
        (lambda: IntermediateStage(config, 14), ["out27", "out26"], ["out29", "out28"]),
        (lambda: IntermediateStage(config, 15), ["out29", "out28"], ["out31", "out30"]),
        (lambda: IntermediateStage(config, 16), ["out31", "out30"], ["out33", "out32"]),
        (lambda: IntermediateStage(config, 17), ["out33", "out32"], ["out35", "out34"]),
        (lambda: IntermediateStage(config, 18), ["out35", "out34"], ["out37", "out36"]),
        (lambda: IntermediateStage(config, 19), ["out37", "out36"], ["out39", "out38"]),
        (lambda: IntermediateStage(config, 20), ["out39", "out38"], ["out41", "out40"]),
        (lambda: IntermediateStage(config, 21), ["out41", "out40"], ["out43", "out42"]),
        (lambda: IntermediateStage(config, 22), ["out43", "out42"], ["out45", "out44"]),
        (lambda: EndingStage(config, BertEmbeddings(config).word_embeddings.weight, 23), ["out45", "out44"], ["out46"]),
        (lambda: criterion, ["out46"], ["loss"])
    ]
