import random
from dataclasses import dataclass
import torch
from collections import defaultdict
import util
from torch.nn import CrossEntropyLoss


@dataclass
class FeatureCollator:

    tokenizer: None
    device: torch.device('cpu')
    pad_ce_label_id = CrossEntropyLoss().ignore_index

    def __post_init__(self):
        assert self.tokenizer.padding_side == 'right'
        self.pad_to_multiple_of = 8

    @classmethod
    def _right_pad_batched_attr(cls, attr_segs, pad_to_len, pad_val):
        return [(attr_segs[seg_i] + [pad_val] * (pad_to_len - len(attr_segs[seg_i])))
                for seg_i in range(len(attr_segs))]

    def __call__(self, features):
        all_keys = set(features[0].keys())

        collated = {
            'input_ids': [f['input_ids'] for f in features]
        }
        collated = self.tokenizer.pad(collated, padding=True, pad_to_multiple_of=8)
        padded_seq_len = len(collated['input_ids'][0])
        if 'token_tags' in all_keys:
            collated['token_tags'] = [(f['token_tags'] + [self.pad_ce_label_id] * (padded_seq_len - len(f['token_tags'])))
                                      for f in features]
        collated = {k: torch.tensor(v, device=self.device) for k, v in collated.items()}
        if 'opentag_types' in all_keys:
            collated['opentag_types'] = [torch.tensor(f['opentag_types'], dtype=torch.long, device=self.device)
                                         for f in features]
            if 'opentag_typed_token_tags' in all_keys:
                opentag_typed_token_tags = [(typed_token_tags + [self.pad_ce_label_id] * (padded_seq_len - len(typed_token_tags)))
                                            for f in features for typed_token_tags in f['opentag_typed_token_tags']]
                collated['opentag_typed_token_tags'] = torch.tensor(opentag_typed_token_tags, device=self.device)
        collated['ngram_span_starts'] = [f['ngram_span_starts'] for f in features]
        collated['ngram_span_ends'] = [f['ngram_span_ends'] for f in features]
        if 'selected_span_starts' in all_keys:
            collated['selected_span_starts'] = [
                [s for s, c in zip(f['selected_span_starts'], f['selected_clusters']) if c > 0]
                for f in features]
            collated['selected_span_ends'] = [
                [e for e, c in zip(f['selected_span_ends'], f['selected_clusters']) if c > 0]
                for f in features]
            collated['selected_clusters'] = [[c for c in f['selected_clusters'] if c > 0] for f in features]

        collated['text_indices'] = [int(f['id'][f['id'].rfind('_') + 1:]) for f in features]
        collated['text_lengths'] = [len(f['input_ids']) for f in features]
        collated['text_starts'] = [(f['input_ids'].index(self.tokenizer.sep_token_id) + 1) for f in features]
        collated['bow_ids'] = [f['bow_ids'] for f in features]

        asin2i = {}
        collated['text_asin'] = []
        collated['text_indices'] = []
        collated['text_lengths'] = []
        collated['text_starts'] = []

        for f in features:
            asin, idx = f['id'].split('_')
            asin_i = asin2i.get(asin, -1)
            if asin_i == -1:
                asin_i = len(asin2i)
                asin2i[asin] = asin_i

            collated['text_asin'].append(asin_i)  # index id based on asin (for asin-level processing)
            collated['text_indices'].append(int(idx))  # text index (0 is title, 1 .. is bullet point)
            collated['text_lengths'].append(len(f['input_ids']))  # text length
            collated['text_starts'].append(f['input_ids'].index(self.tokenizer.sep_token_id) + 1)  # text start (after first SEP)

        return collated


def iterate_asins(features, max_batch_size, shuffle=True):
    asin2features = defaultdict(list)
    for f in features:
        assert f['asin'] == f['id'].split('_')[0]
        asin2features[f['asin']].append(f)
    asin_features = list(asin2features.values())

    if shuffle:
        random.shuffle(asin_features)

    batches, asin_i = [], 0
    while asin_i < len(asin_features):
        batch_features, batch_size = [asin_features[asin_i]], len(asin_features[asin_i])
        asin_i += 1
        while asin_i < len(asin_features) and batch_size + len(asin_features[asin_i]) <= max_batch_size:
            batch_features.append(asin_features[asin_i])
            batch_size += len(asin_features[asin_i])
            asin_i += 1
        batches.append(util.flatten(batch_features))

    return batches
