# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from transformers.modeling_gpt2 import Block

class IntermediateStage(torch.nn.Module):
    def __init__(self, config):
        super(IntermediateStage, self).__init__()

        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.h = torch.nn.ModuleList([Block(config.n_ctx, config, scale=True)
                                      for _ in range(config.n_layer // 4)])

        self.config=config

    def forward(self, hidden_states):
        head_mask = [None] * len(self.h)

        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states,
                layer_past=None,
                attention_mask=None,
                head_mask=head_mask[i],
                use_cache=True,
            )

            hidden_states, present = outputs[:2]

        return hidden_states
