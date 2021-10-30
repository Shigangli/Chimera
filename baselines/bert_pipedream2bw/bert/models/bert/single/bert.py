# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .stage0 import Stage0
from .stage1 import Stage1

class BertPartitioned(torch.nn.Module):
    def __init__(self, config):
        super(BertPartitioned, self).__init__()
        self.stage0 = Stage0(config)
        self.stage1 = Stage1(config, self.stage0.embedding_layer.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input0, input1, input2):
        (out1, out0) = self.stage0(input0, input1, input2)
        out2 = self.stage1(out1, out0)
        return out2
