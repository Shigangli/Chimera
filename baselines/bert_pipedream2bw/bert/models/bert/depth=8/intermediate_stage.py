# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.modeling import BertLayer
from transformers.modeling import BertPreTrainingHeads
from transformers.modeling import BertPooler
from transformers.modeling import BertLayerNorm

class IntermediateStage(torch.nn.Module):
    def __init__(self, config):
        super(IntermediateStage, self).__init__()
        self.layers = []
        for i in range(config.num_hidden_layers // 8):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.config=config; self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization.
            # cf https://github.com/pytorch/pytorch/pull/5617.
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input1, input0):
        out0 = input0
        out1 = input1
        out = out0
        for layer in self.layers:
            out = layer(out, out1)
        return (out1, out)
