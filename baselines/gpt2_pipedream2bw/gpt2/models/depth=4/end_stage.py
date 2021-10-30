# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from transformers.modeling_gpt2 import Block

class EndingStage(torch.nn.Module):
    def __init__(self, config):
        super(EndingStage, self).__init__()

        self.h = torch.nn.ModuleList([Block(config.n_ctx, config, scale=True)
                                      for _ in range(config.n_layer // 4)])
        self.ln_f = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

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

        hidden_states = self.ln_f(hidden_states)

        # hidden_states = hidden_states.view(*output_shape)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits
