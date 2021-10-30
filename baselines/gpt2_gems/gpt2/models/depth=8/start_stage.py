# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from transformers.modeling_gpt2 import Block

class StartingStage(torch.nn.Module):
    def __init__(self, config):
        super(StartingStage, self).__init__()

        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(config.n_positions, config.n_embd)
        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.h = torch.nn.ModuleList([Block(config.n_ctx, config, scale=True)
                                      for _ in range(config.n_layer // 8)])
        self.config = config

    def forward(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        use_cache = True

        past_length = 0
        past = [None] * len(self.h)

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = [None] * len(self.h)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_hidden_states = ()
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
