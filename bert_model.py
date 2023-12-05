from typing import List
from collections import OrderedDict
import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import ModuleUtilsMixin, ModelOutput
from transformers.models.bert import BertConfig, BertForPreTraining, BertModel, BertPreTrainedModel
from pipeline import StageModule

# prepare a minimum size dummy model for extracting Module classes
dummy_config = BertConfig.from_dict({
    'hidden_size': 1,
    'num_attention_heads': 1,
    'num_hidden_layers': 1,
    'vocab_size': 1,
    'intermediate_size': 1,
    'max_position_embeddings': 1,
})
dummy_model = BertForPreTraining(dummy_config)
BertEncoder = dummy_model.bert.encoder.__class__
BertPooler = dummy_model.bert.pooler.__class__
BertPreTrainingHeads = dummy_model.cls.__class__


def get_stage_bert_for_pretraining(stage_id: int,
                                   num_stages: int,
                                   config: BertConfig,
                                   hidden_layers_assignments: List[int] = None,
                                   loss_reduction='mean') -> StageModule:
    """
    start_stage (stage_id == 0): BertEmbeddings + [BertLayer] * N_s
    intermediate_stage (0 < stage_id < num_stages - 1): [BertLayer] * N_i
    end_stage (stage_id == num_stages - 1): [BertLayer] * N_e + BertPreTrainingHeads

    N_s, N_i, N_e: the number of hidden layers (BertLayers) for each stage
    """
    assert num_stages > 1, 'At least 2 stages are required.'
    if hidden_layers_assignments is None:
        """
        Assign the number of hidden layers (BertLayers) so that
        the following are satisfied: 
            N_e <= N_s <= N_i
        """
        hidden_layers_assignments = [0] * num_stages
        for i in range(config.num_hidden_layers):
            hidden_layers_assignments[-((i + 2) % num_stages)] += 1
    assert len(hidden_layers_assignments) == num_stages
    assert stage_id in range(num_stages)
    # overwrite num_hidden_layers with the number for this stage
    config = copy.deepcopy(config)
    config.num_hidden_layers = hidden_layers_assignments[stage_id]

    if stage_id == 0:
        return StartingStage(config)
    elif stage_id == num_stages - 1:
        return EndingStage(config, loss_reduction=loss_reduction)
    else:
        return IntermediateStage(config)


class StartingStage(BertModel, StageModule):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        return OrderedDict(hidden_states=outputs.last_hidden_state)

    @property
    def keys_from_source(self):
        return ['input_ids', 'attention_mask', 'token_type_ids']

    @property
    def sizes_from_prev_stage(self):
        return {}

    @property
    def sizes_for_next_stage(self):
        return {'hidden_states': (self.config.hidden_size,)}


class IntermediateStage(BertPreTrainedModel, StageModule, ModuleUtilsMixin):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.post_init()

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask,
                                                                   hidden_states.size()[:-1],
                                                                   hidden_states.device)
        outputs = self.encoder.forward(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        return OrderedDict(hidden_states=outputs.last_hidden_state)

    @property
    def keys_from_source(self):
        return ['attention_mask']

    @property
    def sizes_from_prev_stage(self):
        return {'hidden_states': (self.config.hidden_size,)}

    @property
    def sizes_for_next_stage(self):
        return {'hidden_states': (self.config.hidden_size,)}


class EndingStage(BertPreTrainedModel, StageModule, ModuleUtilsMixin):
    def __init__(self, config, loss_reduction='mean'):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.cls = BertPreTrainingHeads(config)
        self.post_init()
        self.loss_reduction = loss_reduction

    def forward(self, hidden_states, attention_mask, labels, next_sentence_label):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask,
                                                                   hidden_states.size()[:-1],
                                                                   hidden_states.device)
        encoder_outputs = self.encoder(hidden_states, extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        loss_fct = CrossEntropyLoss(reduction=self.loss_reduction)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return OrderedDict(loss=total_loss)

    @property
    def keys_from_source(self):
        return ['attention_mask', 'labels', 'next_sentence_label']

    @property
    def sizes_from_prev_stage(self):
        return {'hidden_states': (self.config.hidden_size,)}

    @property
    def sizes_for_next_stage(self):
        return {}


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    next_sentence_loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertForPreTrainingEx(BertForPreTraining):

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        masked_lm_loss = None
        next_sentence_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            masked_lm_loss=masked_lm_loss,
            next_sentence_loss=next_sentence_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


