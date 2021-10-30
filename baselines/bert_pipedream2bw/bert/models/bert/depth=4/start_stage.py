# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.modeling import BertLayer
from transformers.modeling import BertEmbeddings
from transformers.modeling import BertLayerNorm

class StartingStage(torch.nn.Module):
    def __init__(self, config, module_id):
        super(StartingStage, self).__init__()
        torch.manual_seed(module_id)
        self.embedding_layer = BertEmbeddings(config)
        self.layers = []
        for i in range(config.num_hidden_layers // 4):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.module_size = 0
        self.config=config; self.apply(self.init_bert_weights)
        #print("module: ", module_id, ", size: ", self.module_size)
        

    def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization.
            # cf https://github.com/pytorch/pytorch/pull/5617.
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            #print("weights: ", module.weight.data, "module name: ", module.__class__)
            #print("stage 0 weight size: ", module.weight.data.size(), "module name: ", module.__class__)
            #self.module_size += module.weight.data.numel()
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            #self.module_size += module.bias.data.numel()
            #self.module_size += module.weight.data.numel()
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            #self.module_size += module.bias.data.numel()

        self.module_size += sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(self, input0, input1, input2):
        out0 = input0
        out1 = input1
        out2 = input2
        out = self.embedding_layer(out0, out1)
        for layer in self.layers:
            out = layer(out, out2)
        return (out2, out)
