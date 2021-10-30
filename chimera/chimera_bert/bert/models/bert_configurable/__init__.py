# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage
from transformers.modeling import BertEmbeddings

def arch(num_stages, num_layers_per_stage):
    return "bert%d" % (num_stages * num_layers_per_stage)

def model(num_stages, num_layers_per_stage, config):
    intermediate_stages = []
    starting_stage = lambda: StartingStage(num_layers_per_stage, config)
    for i in range(num_stages-2):
        intermediate_stages.append(lambda: IntermediateStage(num_layers_per_stage, config))
    ending_stage = lambda: EndingStage(num_layers_per_stage, config,
                                       BertEmbeddings(config).word_embeddings.weight)

    outputs = ["out1", "out0"]
    counter = 2
    model_with_inputs_and_outputs = [
        (starting_stage, ["input0", "input1", "input2"], outputs)]
    for intermediate_stage in intermediate_stages:
        new_outputs = ["out%d" % (counter+1), "out%d" % counter]
        model_with_inputs_and_outputs.append(
            (intermediate_stage, outputs, new_outputs))
        outputs = new_outputs
        counter += 2
    model_with_inputs_and_outputs.append(
        (ending_stage, outputs, ["out%d" % counter]))
    return model_with_inputs_and_outputs
