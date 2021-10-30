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
        (lambda: EndingStage(config), ["out6"], ["out7"]),
        (lambda: criterion, ["out7"], ["loss"])
    ]
