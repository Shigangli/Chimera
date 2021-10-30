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
        for i in range(config.num_hidden_layers // 24):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.pooling_layer = BertPooler(config)
        self.pre_training_heads_layer = BertPreTrainingHeads(config, bert_model_embedding_weights)
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
        out0 = input0.clone()
        out1 = input1.clone()
        out = out0
        for layer in self.layers:
            out = layer(out, out1)
        out2 = self.pooling_layer(out)
        out3 = self.pre_training_heads_layer(out, out2)
        return out3
