# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage

def arch():
    return "gpt2"

def model(config, criterion):
    return [
        (lambda: StartingStage(config), ["input_ids"], ["out0"]),
        (lambda: IntermediateStage(config), ["out0"], ["out1"]),
        (lambda: IntermediateStage(config), ["out1"], ["out2"]),
        (lambda: IntermediateStage(config), ["out2"], ["out3"]),
        (lambda: IntermediateStage(config), ["out3"], ["out4"]),
        (lambda: IntermediateStage(config), ["out4"], ["out5"]),
        (lambda: IntermediateStage(config), ["out5"], ["out6"]),
        (lambda: IntermediateStage(config), ["out6"], ["out7"]),
        (lambda: IntermediateStage(config), ["out7"], ["out8"]),
        (lambda: IntermediateStage(config), ["out8"], ["out9"]),
        (lambda: IntermediateStage(config), ["out9"], ["out10"]),
        (lambda: IntermediateStage(config), ["out10"], ["out11"]),
        (lambda: IntermediateStage(config), ["out11"], ["out12"]),
        (lambda: IntermediateStage(config), ["out12"], ["out13"]),
        (lambda: IntermediateStage(config), ["out13"], ["out14"]),
        (lambda: IntermediateStage(config), ["out14"], ["out15"]),
        (lambda: IntermediateStage(config), ["out15"], ["out16"]),
        (lambda: IntermediateStage(config), ["out16"], ["out17"]),
        (lambda: IntermediateStage(config), ["out17"], ["out18"]),
        (lambda: IntermediateStage(config), ["out18"], ["out19"]),
        (lambda: IntermediateStage(config), ["out19"], ["out20"]),
        (lambda: IntermediateStage(config), ["out20"], ["out21"]),
        (lambda: IntermediateStage(config), ["out21"], ["out22"]),
        (lambda: IntermediateStage(config), ["out22"], ["out23"]),
        (lambda: IntermediateStage(config), ["out23"], ["out24"]),
        (lambda: IntermediateStage(config), ["out24"], ["out25"]),
        (lambda: IntermediateStage(config), ["out25"], ["out26"]),
        (lambda: IntermediateStage(config), ["out26"], ["out27"]),
        (lambda: IntermediateStage(config), ["out27"], ["out28"]),
        (lambda: IntermediateStage(config), ["out28"], ["out29"]),
        (lambda: IntermediateStage(config), ["out29"], ["out30"]),
        (lambda: EndingStage(config), ["out30"], ["out31"]),
        (lambda: criterion, ["out31"], ["loss"])
    ]
