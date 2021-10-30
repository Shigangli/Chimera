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
        self.stage1 = IntermediateStage(config)
        self.stage2 = IntermediateStage(config)
        self.stage3 = EndingStage(config, self.stage0.embedding_layer.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input0, input1, input2):
        (out1, out0) = self.stage0(input0, input1, input2)
        (out3, out2) = self.stage1(out1, out0)
        (out5, out4) = self.stage2(out3, out2)
        out6 = self.stage3(out5, out4)
        return out6
