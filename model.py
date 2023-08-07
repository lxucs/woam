import random
import torch
import losses
from model_base import EncoderBaseModel
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Model(EncoderBaseModel):
    def __init__(self, config, num_attrs=None):
        super(Model, self).__init__(config)
        self.num_attrs = num_attrs
        self.attr_classifier = self.make_linear(self.seq_hidden_size, num_attrs or 0)

        self.debug = True
        self.steps = 0

    def freeze_layers(self, freeze_layers):
        freeze_layers = freeze_layers if freeze_layers > 0 else (self.seq_config.num_hidden_layers + freeze_layers)

        seq_encoder_param, _ = self.get_params(named=True)
        for name, p in seq_encoder_param:
            freeze = False
            if 'embeddings' in name:
                freeze = True
            elif 'layer.' in name:
                char_s = name.find('layer.') + len('layer.')
                char_e = name[char_s:].find('.') + char_s
                layer = int(name[char_s: char_e])
                if layer < freeze_layers:
                    freeze = True
            if freeze:
                p.requires_grad = False
        logger.info(f'Froze encoder embeddings and < {freeze_layers} layers')

    def get_seq_hidden(self, input_ids=None, attention_mask=None, layer=-1, **kwargs):
        conf, (num_seq, seq_len) = self.config, input_ids.size()[:2]
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                  'output_attentions': False, 'output_hidden_states': True, 'return_dict': True}
        encoded = self.seq_encoder(**inputs)
        seq_hidden = encoded['hidden_states'][layer]
        return seq_hidden

    @classmethod
    def get_span_hidden(cls, hidden, span_starts, span_ends, normalized=True):
        """ Only used in inference. """
        span_hidden = [hidden[span_start: span_end + 1].mean(dim=0)
                       for span_start, span_end in zip(span_starts, span_ends)]
        if normalized and span_hidden:
            span_hidden = torch.stack(span_hidden, dim=0)  # [num_spans, hidden]
            span_hidden = F.normalize(span_hidden, p=2, dim=-1, eps=1e-8)
            span_hidden = torch.unbind(span_hidden, dim=0)
        return span_hidden

    @classmethod
    def _get_batch_span_hidden(cls, seq_hidden, batch_span_starts, batch_span_ends, batch_span_clusters):
        assert seq_hidden.size()[0] == len(batch_span_starts)

        all_span_hidden, all_span_clusters = [], []
        for seq_i, (span_starts, span_ends, span_clusters) in enumerate(zip(
                batch_span_starts, batch_span_ends, batch_span_clusters)):
            all_span_hidden += (cls.get_span_hidden(seq_hidden[seq_i], span_starts, span_ends, normalized=False))
            all_span_clusters += span_clusters
        all_span_hidden = torch.stack(all_span_hidden, dim=0)
        all_span_clusters = torch.tensor(all_span_clusters, dtype=torch.long, device=seq_hidden.device)

        # Normalize all in the end
        all_span_hidden = F.normalize(all_span_hidden, p=2, dim=-1, eps=1e-8)  # [num_spans, hidden]

        return all_span_hidden, all_span_clusters

    def forward(self, input_ids=None, attention_mask=None,
                text_asin=None, text_indices=None, text_starts=None, text_lengths=None,
                selected_span_starts=None, selected_span_ends=None, selected_clusters=None,
                ngram_span_starts=None, ngram_span_ends=None, **kwargs):
        conf, device = self.config, input_ids.device

        # Obtain normalized seed representation and labels
        seq_hidden = self.get_seq_hidden(input_ids, attention_mask)  # [bsz, seq_len, hidden]
        span_hidden, span_clusters = self._get_batch_span_hidden(
            seq_hidden, selected_span_starts, selected_span_ends, selected_clusters)  # [num_spans, hidden]

        # Get sup loss using seeds
        loss = losses.get_contrastive_loss(span_hidden.unsqueeze(0), span_clusters, temp=conf['contrastive_temp'])

        if self.debug:
            if self.steps % (conf['report_frequency'] * 2) == 0:
                logger.info(f'---------debug step: {self.steps}---------')
                logger.info(f'contrastive loss: {loss_contrastive.item():.4f}')
                if conf['attr_cls_coef']:
                    logger.info(f'attr classification loss: {loss_attr_cls.item():.4f}')
        self.steps += 1

        return loss
