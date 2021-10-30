# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage

class BertPartitioned(torch.nn.Module):
    def __init__(self, config):
        super(BertPartitioned, self).__init__()
        self.stage0 = StartingStage(config)
        self.intermediate_stages = [IntermediateStage(config)
                                    for i in range(6)]
        self.stage7 = EndingStage(config, self.stage0.embedding_layer.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input0, input1, input2):
        (out1, out0) = self.stage0(input0, input1, input2)
        for intermediate_stage in self.intermediate_stages:
            (out1, out0) = self.stage1(out1, out0)
        out2 = self.stage3(out1, out0)
        return out2
