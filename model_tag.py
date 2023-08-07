import torch.nn as nn
import logging
from model import Model
from module_crf import CRF
import torch
from util_tag import get_entities

logger = logging.getLogger(__name__)


class ModelTag(Model):
    def __init__(self, config, num_tags=None, do_crf=False):
        super(Model, self).__init__(config)
        self.num_tags = num_tags
        self.num_attrs = (num_tags - 1) // 2
        self.do_crf = do_crf

        self.tag_classifier = self.make_linear(self.seq_hidden_size, self.num_tags)
        self.crf = CRF(self.num_tags, batch_first=True)

    def decode(self, logits, attention_mask):
        if self.do_crf:
            all_tags = self.crf.decode(logits, attention_mask)
        else:
            _, all_tags = torch.max(logits, dim=-1)  # [bsz, seq_len]
            all_tags *= attention_mask.type_as(all_tags)  # O tag has id 0
            all_tags = all_tags.tolist()

        all_seq_entities = []
        for tag_seq in all_tags:
            tag_seq = [('O' if tag == 0 else f'B-{tag - 1}' if tag <= self.num_attrs else f'I-{tag - self.num_attrs - 1}')
                       for tag in tag_seq]
            entities = get_entities(tag_seq)
            all_seq_entities.append(entities)
        return all_seq_entities

    def forward(self, input_ids=None, attention_mask=None, token_tags=None, **kwargs):
        conf, device = self.config, input_ids.device
        seq_hidden = self.get_seq_hidden(input_ids, attention_mask)  # [bsz, seq_len, hidden]
        logits = self.tag_classifier(seq_hidden)  # [bsz, seq_len, num_tags]

        if token_tags is None:
            return logits

        if self.do_crf:
            loss = self.crf(logits, token_tags, attention_mask)
        else:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_tags), token_tags.view(-1))
        return loss
