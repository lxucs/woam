from module_vi import ViMixin
from model import Model
import logging
import torch
import torch.nn.functional as F
import losses
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelAsin(Model, ViMixin):
    def __init__(self, config, num_attrs=None):
        super(ModelAsin, self).__init__(config, num_attrs=num_attrs)

        self.vocab_size = self.seq_config.vocab_size
        self.post_bow = self.make_linear(self.vocab_size, self.seq_hidden_size)
        self.post_mu = self.make_linear(self.seq_hidden_size, config['num_topics'])
        self.post_logvar = self.make_linear(self.seq_hidden_size, config['num_topics'])
        self.emb_topics = self.make_embedding(config['num_topics'], self.seq_hidden_size)

        self.debug = True

    def _get_batch_span_hidden(self, input_ids, seq_hidden, batch_asins, batch_indices, batch_bow_ids,
                               batch_span_starts, batch_span_ends, batch_span_clusters,
                               batch_ngram_starts, batch_ngram_ends, vi_only_ngram=False):
        assert seq_hidden.size()[0] == len(batch_span_starts)

        # Get span hidden
        all_span_hidden, all_span_clusters, all_ngram_hidden = [], [], []
        all_span_tokens, all_ngram_tokens, max_phrase_tokens = [], [], 16
        all_seq_cls, asin2lastseqi, asin2lastspani, asin2lastngrami = [], {}, {}, {}
        asin2firstbulletngrami, asin2bulletngramidx = {}, defaultdict(list)
        all_seq_bow = torch.zeros(seq_hidden.size()[0], self.vocab_size, dtype=torch.float, device=seq_hidden.device)
        for seq_i, (asin, asin_idx, bow_ids, span_starts, span_ends, span_clusters, ngram_starts, ngram_ends) in enumerate(zip(
                batch_asins, batch_indices, batch_bow_ids, batch_span_starts, batch_span_ends, batch_span_clusters, batch_ngram_starts, batch_ngram_ends)):
            if asin_idx > 0:
                if asin not in asin2firstbulletngrami:
                    asin2firstbulletngrami[asin] = len(all_ngram_hidden)
                asin2bulletngramidx[asin] += [asin_idx] * len(ngram_starts)

            all_span_hidden += self.get_span_hidden(seq_hidden[seq_i], span_starts, span_ends, normalized=False)
            all_span_clusters += span_clusters
            all_span_tokens += [F.pad(input_ids[seq_i][s: e+1], (0, max_phrase_tokens - e + s - 1), value=0)
                                for s, e in zip(span_starts, span_ends)]
            all_ngram_hidden += self.get_span_hidden(seq_hidden[seq_i], ngram_starts, ngram_ends, normalized=False)
            all_ngram_tokens += [F.pad(input_ids[seq_i][s: e + 1], (0, max_phrase_tokens - e + s - 1), value=0)
                                 for s, e in zip(ngram_starts, ngram_ends)]
            all_seq_cls.append(seq_hidden[seq_i][0])
            for i, cnt in bow_ids.items():
                all_seq_bow[seq_i, i] = cnt
            asin2lastseqi[asin] = seq_i + 1  # Exclusive
            asin2lastspani[asin] = len(all_span_hidden)  # Exclusive
            asin2lastngrami[asin] = len(all_ngram_hidden)
        all_span_hidden = torch.stack(all_span_hidden, dim=0)
        all_span_clusters = torch.tensor(all_span_clusters, dtype=torch.long, device=seq_hidden.device)
        all_ngram_hidden = torch.stack(all_ngram_hidden, dim=0)
        all_seq_cls = torch.stack(all_seq_cls, dim=0)
        all_span_tokens = torch.stack(all_span_tokens, dim=0)
        all_ngram_tokens = torch.stack(all_ngram_tokens, dim=0)

        # Normalize span representation
        all_span_hidden = F.normalize(all_span_hidden, p=2, dim=-1, eps=1e-8)  # [num_spans, hidden]
        all_ngram_hidden = F.normalize(all_ngram_hidden, p=2, dim=-1, eps=1e-8)  # [num_ngrams, hidden]

        # Obtain phrase lexical pairwise for this batch
        num_phrases = (0 if vi_only_ngram else all_span_hidden.size()[0]) + all_ngram_hidden.size()[0]
        all_phrase_tokens = all_ngram_tokens if vi_only_ngram else torch.cat([all_span_tokens, all_ngram_tokens], dim=0)
        _, unique_map = all_phrase_tokens.unique(dim=0, return_inverse=True, return_counts=False)  # [num_phrases]
        phrase_lexical_pairwise = (unique_map.unsqueeze(1) == unique_map.unsqueeze(0)).to(torch.float)

        # Obtain asin-level
        max_asin = max(asin2lastspani.keys())
        all_asin_input_hidden, all_asin_input_bow, all_asin_phrase_hidden, all_asin_phrase_labels = [], [], [], []
        asin2bulletngramhidden = {}
        for asin in range(max_asin + 1):
            asin_startseqi = 0 if asin == 0 else asin2lastseqi[asin - 1]
            asin_endseqi = asin2lastseqi[asin]  # Exclusive
            asin_startspani = 0 if asin == 0 else asin2lastspani[asin - 1]
            asin_lastspani = asin2lastspani[asin]  # Exclusive
            asin_startngrami = 0 if asin == 0 else asin2lastngrami[asin - 1]
            asin_endngrami = asin2lastngrami[asin]  # Exclusive
            if asin_endseqi - asin_startseqi < 1:
                a = 1
            assert asin_endseqi - asin_startseqi >= 1
            if asin_lastspani - asin_startspani + asin_endngrami - asin_startngrami == 0:
                continue
            if vi_only_ngram and asin_endngrami - asin_startngrami == 0:
                continue
            # Obtain asin input hidden: averaged CLS among asin text sequences
            asin_input_hidden = all_seq_cls[asin_startseqi: asin_endseqi].mean(dim=0)  # [hidden]
            all_asin_input_hidden.append(asin_input_hidden)
            # Obtain asin input BOW
            asin_input_bow = all_seq_bow[asin_startseqi: asin_endseqi].sum(dim=0)  # [vocab_size]
            all_asin_input_bow.append(asin_input_bow)
            # Obtain asin phrase hidden: should be the same order as phrase lexicons
            asin_startphrasei = len(all_asin_phrase_hidden)
            if not vi_only_ngram:
                all_asin_phrase_hidden.append(all_span_hidden[asin_startspani: asin_lastspani])
            all_asin_phrase_hidden.append(all_ngram_hidden[asin_startngrami: asin_endngrami])
            asin_lastphrasei = len(all_asin_phrase_hidden)  # Exclusive
            # Obtain asin span labels
            asin_phrase_labels = phrase_lexical_pairwise[asin_startphrasei: asin_lastphrasei].sum(dim=0)  # [num_phrases]
            all_asin_phrase_labels.append(asin_phrase_labels)
            # Obtain bullet ngram
            asin_startbulletngrami = asin2firstbulletngrami.get(asin, asin_endngrami)  # Inclusive
            if asin_endngrami - asin_startbulletngrami > 4:
                if len(asin2bulletngramidx[asin]) == len(set(asin2bulletngramidx[asin])):
                    continue  # Make sure positive pairs exist (at least two ngram from same bullet)
                asin2bulletngramhidden[asin] = all_ngram_hidden[asin_startbulletngrami: asin_endngrami]
                asin2bulletngramidx[asin] = torch.tensor(asin2bulletngramidx[asin], dtype=torch.long, device=seq_hidden.device)
        all_asin_input_hidden = torch.stack(all_asin_input_hidden, dim=0)
        all_asin_input_bow = torch.stack(all_asin_input_bow, dim=0)
        all_asin_phrase_hidden = torch.cat(all_asin_phrase_hidden, dim=0)
        all_asin_phrase_labels = torch.stack(all_asin_phrase_labels, dim=0)
        assert all_asin_phrase_hidden.size()[0] == num_phrases

        return all_span_hidden, all_span_clusters, all_asin_input_hidden, all_asin_input_bow, all_asin_phrase_hidden, all_asin_phrase_labels,\
               asin2bulletngramhidden, asin2bulletngramidx

    def forward(self, input_ids=None, attention_mask=None,
                text_asin=None, text_indices=None, text_starts=None, text_lengths=None, bow_ids=None,
                selected_span_starts=None, selected_span_ends=None, selected_clusters=None,
                ngram_span_starts=None, ngram_span_ends=None, **kwargs):
        conf, device = self.config, input_ids.device

        # Obtain normalized span representation and labels
        seq_hidden = self.get_seq_hidden(input_ids, attention_mask)  # [bst, seq_len, hidden]
        span_hidden, span_clusters, asin_input_hidden, asin_input_bow, asin_phrase_hidden, asin_phrase_labels,\
        asin2bulletngramhidden, asin2bulletngramidx = self._get_batch_span_hidden(
            input_ids, seq_hidden, text_asin, text_indices, bow_ids, selected_span_starts, selected_span_ends, selected_clusters,
            ngram_span_starts, ngram_span_ends, vi_only_ngram=conf['vi_only_ngram'])

        loss_contrastive = losses.get_contrastive_loss(span_hidden.unsqueeze(0), span_clusters, temp=conf['contrastive_temp'])
        loss = loss_contrastive

        if conf['usp_bullet_coef']:
            loss_bullet = []
            for asin, bullet_ngram_hidden in asin2bulletngramhidden.items():
                bullet_ngram_indices = asin2bulletngramidx[asin]
                assert bullet_ngram_hidden.size()[0] == bullet_ngram_indices.size()[0]
                loss_bullet.append(losses.get_contrastive_loss(bullet_ngram_hidden.unsqueeze(0), bullet_ngram_indices,
                                                               temp=conf['usp_bullet_temp'], reduction=False))
            if loss_bullet:
                loss_bullet = torch.cat(loss_bullet).mean() * conf['usp_bullet_coef']
                loss = loss + loss_bullet
            else:
                loss_bullet = 0

        if conf['vi_coef']:
            loss_vi = self.vi_forward(asin_input_hidden, asin_input_bow, asin_phrase_hidden, asin_phrase_labels, sampling=conf['vi_sampling'])
            loss_vi *= conf['vi_coef']
            loss = loss + loss_vi

        if self.debug:
            if self.steps % (conf['report_frequency'] * 2) == 0:
                logger.info(f'---------debug step: {self.steps}---------')
                logger.info(f'contrastive: {loss_contrastive.item():.4f}')
                if conf['usp_bullet_coef']:
                    logger.info(f'unsupervised bullet loss: {loss_bullet.item():.4f}')
                if conf['vi_coef']:
                    logger.info(f'VI loss: {loss_vi.item():.4f}')
                if conf['vi_diverse']:
                    logger.info(f'VI diversity loss: {loss_vi_diverse.item():.4f}')
        self.steps += 1

        return loss
