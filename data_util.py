import io_util
import logging
from tqdm import tqdm
import span_util
import util
from preprocess.dataset import core_stop_words
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

pt2prefix = {'TEA': 'tea', 'VITAMIN': 'vitamin', 'SOFA': 'sofa', 'CELLULAR_PHONE_CASE': 'phone case'}


def convert_spans(tokenizer, text, subtoks, charstart2i, charend2i, orig_spans, orig_span_counts=None, orig_span_clusters=None):
    """ From char idx to subtok idx. """
    spans, span_counts, span_clusters = [], [], []
    for orig_i, span_by_charidx in enumerate(orig_spans):
        charstart, charend = span_by_charidx[:2]
        subtok_si, subtok_ei = charstart2i.get(charstart, -1), charend2i.get(charend, -1)  # Inclusive
        if subtok_ei == -1:
            if charend - charstart > 6:
                subtok_ei = charend2i.get(charend + 1, -1)

        if 0 <= subtok_si <= subtok_ei and tokenizer.unk_token not in subtoks[subtok_si: subtok_ei + 1]:
            spans.append((subtok_si, subtok_ei))
            if orig_span_counts:
                span_counts.append(orig_span_counts[orig_i])
            if orig_span_clusters:
                span_clusters.append(orig_span_clusters[orig_i])

    indices = util.argsort(spans)
    spans = [spans[span_i] for span_i in indices]
    if span_counts:
        span_counts = [span_counts[span_i] for span_i in indices]
    if span_clusters:
        span_clusters = [span_clusters[span_i] for span_i in indices]
    return spans, span_counts, span_clusters


def get_all_docs(dataset_name, file_path, meta, tokenizer, only_title=False, is_training=False):
    records = io_util.read_jsonlines(file_path)
    instances = []
    for record in records:
        # Title
        instances.append({
            'id': f'{record["asin"]}_0',
            'asin': record['asin'],
            'text': record['title'],
            'pt': record['pt'],
            'gv': record['gv'],
            'ngram': record['title_ngram'],
            'ngram_count': record['title_ngram_count'],
            'attr': record['title_attr'],
            'attr_clusters': record['title_attr_clusters'],
            'is_dev': record['is_dev'],
            'is_test': record['is_test']
        })
        # Bullet points
        for bullet_i in range(len(record['bullet_point'])):
            instances.append({
                'id': f'{record["asin"]}_{bullet_i + 1}',
                'asin': record['asin'],
                'text': record['bullet_point'][bullet_i],
                'pt': record['pt'],
                'gv': record['gv'],
                'ngram': record['bp_ngram'][bullet_i],
                'ngram_count': record['bp_ngram_count'][bullet_i],
                'attr': record['bp_attr'][bullet_i],
                'attr_clusters': record['bp_attr_clusters'][bullet_i],
                'is_dev': record['is_dev'],
                'is_test': record['is_test']
            })
    print(f'Split {len(records)} records into {len(instances)} docs (w/{"o" if only_title else ""} bullet points)')

    total_ngrams_before, total_ngrams_after = 0, 0
    total_attrs_before, total_attrs_after = 0, 0

    def tokenize(text):
        encoded = tokenizer(text, add_special_tokens=False, padding=False, truncation=False,
                            return_token_type_ids=True, return_attention_mask=True, return_offsets_mapping=True)
        subtoks = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        charstart2i = {char_idx: tok_i for tok_i, (char_idx, _) in enumerate(encoded['offset_mapping'])}
        charend2i = {char_idx: tok_i for tok_i, (_, char_idx) in enumerate(encoded['offset_mapping'])}
        return subtoks, (charstart2i, charend2i)

    new_instances = []
    for inst in tqdm(instances, desc='Docs'):
        text_prefix = f'{pt2prefix[inst["pt"]].lower()} {tokenizer.sep_token} '
        char_offset = len(text_prefix)
        text = text_prefix + inst['text']
        text_ngram = [(i_s + char_offset, i_e + char_offset) for i_s, i_e in inst['ngram']]
        text_ngram_count = inst['ngram_count']
        text_attr = [(i_s + char_offset, i_e + char_offset) for i_s, i_e in inst['attr']]
        text_attr_clusters = inst['attr_clusters']

        subtoks, (charstart2i, charend2i) = tokenize(text)
        ngram_spans, ngram_counts, _ = convert_spans(tokenizer, text, subtoks, charstart2i, charend2i,
                                                     orig_spans=text_ngram, orig_span_counts=text_ngram_count)
        attr_spans, _, attr_clusters = convert_spans(tokenizer, text, subtoks, charstart2i, charend2i,
                                                     orig_spans=text_attr, orig_span_clusters=text_attr_clusters)

        total_ngrams_before += len(text_ngram)
        total_ngrams_after += len(ngram_spans)
        total_attrs_before += len(text_attr)
        total_attrs_after += len(attr_spans)

        bow_tokens = [subtok for subtok in subtoks[subtoks.index(tokenizer.sep_token) + 1:]
                      if not (len(subtok) == 1 and not subtok.isdigit()) and subtok not in core_stop_words]
        bow_ids = tokenizer.convert_tokens_to_ids(bow_tokens)

        inst = {
            'id': inst['id'],  # _0 is title; _[1..] is bullet
            'asin': inst['asin'],
            'text': text,
            'bow_ids': Counter(bow_ids),
            'text_subtoks': subtoks,
            'charstart2i': charstart2i,
            'charend2i': charend2i,
            'text_ngram_spans': ngram_spans,
            'text_ngram_counts': ngram_counts,
            'text_attr_spans': attr_spans,
            'text_attr_clusters': attr_clusters,
            'pt': inst['pt'],
            'gv': inst['gv'],
            'is_dev': inst['is_dev'],
            'is_test': inst['is_test']
        }
        new_instances.append(inst)

    logger.info(f'Raw # ngrams: {total_ngrams_before}; # matched attribute substrings: {total_attrs_before}')
    logger.info(f'Processed # ngrams: {total_ngrams_after}; # attrs: {total_attrs_after}')
    miss_ratio = 100 - (total_ngrams_after + total_attrs_after) / (total_ngrams_before + total_attrs_before) * 100
    logger.info(f'{miss_ratio:.2f}% spans are discarded due to char idx mismatch')
    return new_instances


def convert_docs_to_features(dataset_name, docs, tokenizer, max_seq_len, is_training, show_example=False):
    """ Assume BERT-like encoding. """
    features = []
    example_shown = 0
    for doc_i, doc in enumerate(tqdm(docs, desc='Features')):
        num_text_subtoks = min(len(doc['text_subtoks']), max_seq_len - 2)
        input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(doc['text_subtoks']) + [tokenizer.sep_token_id]

        text_ngram_spans = [span for span in doc['text_ngram_spans'] if span[1] < num_text_subtoks]
        text_ngram_counts = [cnt for span, cnt in zip(doc['text_ngram_spans'], doc['text_ngram_counts'])
                             if span[1] < num_text_subtoks]

        keep_attr = [(1 if span[1] < num_text_subtoks else 0) for span in doc['text_attr_spans']]
        text_attr_spans = [span for span, keep in zip(doc['text_attr_spans'], keep_attr) if keep]
        text_attr_clusters = [cluster for cluster, keep in zip(doc['text_attr_clusters'], keep_attr) if keep]

        span_offset = 1  # CLS
        text_ngram_spans = [(i_s + span_offset, i_e + span_offset) for i_s, i_e in text_ngram_spans]
        text_attr_spans = [(i_s + span_offset, i_e + span_offset) for i_s, i_e in text_attr_spans]

        feature = {
            'id': doc['id'],
            'asin': doc['asin'],
            'bow_ids': {int(i): cnt for i, cnt in doc['bow_ids'].items()},
            'input_ids': input_ids,
            'ngram_span_starts': [i_s for i_s, i_e in text_ngram_spans],
            'ngram_span_ends': [i_e for i_s, i_e in text_ngram_spans],
            'ngram_counts': {ngram: count for ngram, count in zip(text_ngram_spans, text_ngram_counts)},
            'attr_span_starts': [i_s for i_s, i_e in text_attr_spans],
            'attr_span_ends': [i_e for i_s, i_e in text_attr_spans],
            'attr_clusters': text_attr_clusters,
            'pt': doc['pt'],
            'is_dev': doc['is_dev'],
            'is_test': doc['is_test']
        }
        features.append(feature)

        if show_example and example_shown < 1:
            show_feature(tokenizer, feature)
            example_shown += 1
    return features


def show_feature(tokenizer, feat):
    print()
    print(feat['id'])
    subtoks = tokenizer.convert_ids_to_tokens(feat['input_ids'])
    text = ' '.join(subtoks).replace(' ##', '')
    print(text)

    if 'selected_span_starts' not in feat:
        spans = span_util.show_span_text(tokenizer, feat['ngram_span_starts'], feat['ngram_span_ends'], subtoks=subtoks)
        print(f'NGRAM: {" || ".join(spans)}')

        spans = span_util.show_span_text(tokenizer, feat['attr_span_starts'], feat['attr_span_ends'],
                                         span_clusters=feat['attr_clusters'], subtoks=subtoks)
        print(f'ATTR: {" || ".join(spans)}')
    else:
        spans = span_util.show_span_text(tokenizer, feat['ngram_span_starts'], feat['ngram_span_ends'], subtoks=subtoks)
        print(f'NGRAM: {" || ".join(spans)}')

        spans = span_util.show_span_text(tokenizer, feat['selected_span_starts'], feat['selected_span_ends'],
                                         span_clusters=feat['selected_clusters'], span_properties=feat['selected_properties'],
                                         subtoks=subtoks)
        print(f'SELECTED: {" || ".join(spans)}')

        if 'token_tags' in feat:
            assert 'num_attrs' in feat
            tags = [('O' if tag == 0 else f'B-{tag - 1}' if tag <= feat['num_attrs'] else f'I-{tag - feat["num_attrs"] - 1}')
                    for tag in feat['token_tags']]
            print(f'TAGS: {" ".join(tags)}')
        if 'opentag_types' in feat and 'opentag_typed_token_tags' in feat:
            for attr, seq_tags in zip(feat['opentag_types'], feat['opentag_typed_token_tags']):
                print(f'Attr {attr}: {seq_tags}')
