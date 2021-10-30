# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.modeling import BertLayer
from transformers.modeling import BertPreTrainingHeads
from transformers.modeling import BertPooler
from transformers.modeling import BertLayerNorm

class EndingStage(torch.nn.Module):
    def __init__(self, config, bert_model_embedding_weights, module_id):
        super(EndingStage, self).__init__()
        torch.manual_seed(module_id)
        self.layers = []
        for i in range(config.num_hidden_layers // 4):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.pooling_layer = BertPooler(config)
        self.pre_training_heads_layer = BertPreTrainingHeads(config, bert_model_embedding_weights)
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
        #for p in module.parameters():
        #    print("p name: ", p.__class__, "size: ", p.numel(), "module name: ", module.__class__)

    def forward(self, input1, input0):
        out0 = input0
        out1 = input1
        out = out0
        for layer in self.layers:
            out = layer(out, out1)
        out2 = self.pooling_layer(out)
        out3 = self.pre_training_heads_layer(out, out2)
        return out3
