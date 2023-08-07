# Refactored from https://github.com/kmkurn/pytorch-crf/blob/master/torchcrf/__init__.py
from typing import List, Optional
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first=False):
        if num_tags <= 0:
            raise ValueError(f'Invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = bool(batch_first)

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def _validate(self, emissions, tags=None, mask=None):
        assert emissions.dim() == 3  # [bsz, seq_len, num_tags] if batch_first
        assert emissions.size(2) == self.num_tags
        if tags is not None:
            assert tags.size() == emissions.size()[:2]
        if mask is not None:
            assert mask.size() == emissions.size()[:2]

    def forward(
            self,
            emissions: torch.Tensor,  # [bsz, seq_len, num_tags] if batch_first
            tags: torch.LongTensor,  # [bsz, seq_len]
            mask: Optional[torch.Tensor] = None,  # [bsz, seq_len]
            reduction: str = 'mean'):
        self._validate(emissions, tags=tags, mask=mask)
        assert reduction in ('none', 'sum', 'mean', 'token_mean')
        if mask is None:
            mask = torch.ones_like(tags, dtype=emissions.dtype)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)  # [bsz]
        denominator = self._compute_normalizer(emissions, mask)
        nll = denominator - numerator  # [bsz]

        if reduction == 'none':
            return nll
        elif reduction == 'sum':
            return nll.sum()
        elif reduction == 'mean':
            return nll.mean()
        else:
            return nll.sum() / mask.sum()

    def _compute_score(self, emissions, tags, mask):
        seq_len, bsz = emissions.size()[:2]
        mask = mask.type_as(emissions)
        tags[tags < 0] = 0

        score = self.start_transitions[tags[0]] + emissions[0, torch.arange(bsz), tags[0]]
        for seq_i in range(1, seq_len):
            score += self.transitions[tags[seq_i - 1], tags[seq_i]] * mask[seq_i]
            score += emissions[seq_i, torch.arange(bsz), tags[seq_i]] * mask[seq_i]

        last_seq_i = mask.long().sum(dim=0) - 1
        last_tags = tags[last_seq_i, torch.arange(bsz)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        seq_len, bsz = emissions.size()[:2]
        mask = mask.bool()

        score = self.start_transitions + emissions[0]  # [bsz, num_tags]
        for seq_i in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions + emissions[seq_i].unsqueeze(1)  # [bsz, num_tags, num_tags]
            next_score = next_score.logsumexp(dim=1)
            score = torch.where(mask[seq_i].unsqueeze(1), next_score, score)
        score += self.end_transitions

        return torch.logsumexp(score, dim=1)  # [bsz]

    def decode(self, emissions, mask) -> List[List[int]]:
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.size()[:2], dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)  # [seq_len, bsz, num_tags]
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        seq_len, bsz = emissions.size()[:2]
        mask = mask.bool()

        score = self.start_transitions + emissions[0]  # [bsz, num_tags]
        best_prev_tags = []  # [seq_len, bsz, num_tags]

        for seq_i in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions + emissions[seq_i].unsqueeze(1)  # [bsz, num_tags, num_tags]
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[seq_i].unsqueeze(1), next_score, score)
            best_prev_tags.append(indices)
        score += self.end_transitions

        last_seq_i = mask.long().sum(dim=0) - 1  # [bsz]
        batch_best_seq = []
        for b_i in range(bsz):
            _, last_tag = score[b_i].max(dim=0)
            best_seq = [last_tag.item()]
            for best_prev_tag in reversed(best_prev_tags[:last_seq_i[b_i]]):
                last_tag = best_prev_tag[b_i][best_seq[-1]]
                best_seq.append(last_tag.item())

            best_seq.reverse()
            batch_best_seq.append(best_seq)

        return batch_best_seq
