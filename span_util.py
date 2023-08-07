import logging

logger = logging.getLogger(__name__)


def show_span_text(tokenizer, span_starts, span_ends, span_clusters=None, span_properties=None,
                   subtoks=None, input_ids=None):
    if subtoks is None:
        assert input_ids is not None
        subtoks = tokenizer.convert_ids_to_tokens(input_ids)

    spans = []
    for span_i in range(len(span_starts)):
        span_text = ' '.join(subtoks[span_starts[span_i]: span_ends[span_i] + 1]).replace(' ##', '')
        if span_clusters:
            span_text += f' ({span_clusters[span_i]})'
        if span_properties:
            span_text += f' : {span_properties[span_i]}'
        spans.append(span_text)
    return spans


def get_sel_text(c2text, subtoks, span_starts, span_ends, span_clusters, span_properties, p_th=1.01):
    """ Update dict of selected span text whose property is within a threshold. """
    for s, e, c, p in zip(span_starts, span_ends, span_clusters, span_properties):
        if p < p_th:
            span_text = ' '.join(subtoks[s: e + 1]).replace(' ##', '')
            c2text[c][span_text] += 1
    return c2text


def get_span_text(subtoks, span_starts, span_ends):
    span_text = [' '.join(subtoks[i_s: i_e + 1]).replace(' ##', '')
                 for i_s, i_e in zip(span_starts, span_ends)]
    return span_text


def remove_overlap_batch(candidate_starts, candidate_ends, selected_starts, selected_ends, allow_nested=False):
    """ Remove overlap candidates against selected spans (>=); assuming no overlapping in selected. """
    if not candidate_starts or not selected_starts:
        return candidate_starts, candidate_ends

    selected_token_idx = set()  # For any overlap
    start_to_max_end, end_to_min_start = {}, {}  # For cross-overlap
    toki2selectedi = [-1] * (max(max(candidate_ends), max(selected_ends)) + 1)  # For cross-overlap
    for selected_i, (i_s, i_e) in enumerate(zip(selected_starts, selected_ends)):
        if allow_nested:
            max_end = start_to_max_end.get(i_s, -1)
            if i_e > max_end:
                start_to_max_end[i_s] = i_e
            min_start = end_to_min_start.get(i_e, -1)
            if min_start == -1 or i_s < min_start:
                end_to_min_start[i_e] = i_s
            for tok_i in range(i_s, i_e + 1):
                assert toki2selectedi[tok_i] == -1, 'Initial seeds have overlaps'
                toki2selectedi[tok_i] = selected_i
        else:
            selected_token_idx.update(range(i_s, i_e + 1))

    filtered_starts, filtered_ends = [], []
    for i_s, i_e in zip(candidate_starts, candidate_ends):
        if allow_nested:
            if start_to_max_end.get(i_s, -1) == i_e and end_to_min_start.get(i_e, -1) == i_s:  # Allow exact
                pass
            cross_overlap = False
            for tok_i in range(i_s, i_e + 1):
                max_end = start_to_max_end.get(tok_i, -1)
                if tok_i > i_s and max_end > i_e:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(tok_i, -1)
                if tok_i < i_e and 0 <= min_start < i_s:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Do not allow candidate < selected
                unique_selected_i = set(toki2selectedi[i_s: i_e + 1])
                if len(unique_selected_i) == 1 and toki2selectedi[i_s] != -1:
                    if (i_s > 0 and toki2selectedi[i_s - 1] == toki2selectedi[i_s]) or \
                            (i_e < len(toki2selectedi) - 1 and toki2selectedi[i_e + 1] == toki2selectedi[i_e]):
                        cross_overlap = True
            if not cross_overlap:
                filtered_starts.append(i_s)
                filtered_ends.append(i_e)
        else:
            overlap = False
            for tok_i in range(i_s, i_e + 1):
                if tok_i in selected_token_idx:
                    overlap = True
                    break
            if not overlap:
                filtered_starts.append(i_s)
                filtered_ends.append(i_e)

    return filtered_starts, filtered_ends


def remove_overlap_sequential(span_starts, span_ends, allow_nested=False):
    """ Remove overlap candidates sequentially based on scores; assuming already sorted. """
    selected_span_indices = []
    selected_token_idx = set()  # For any overlap
    start_to_max_end, end_to_min_start = {}, {}  # For cross-overlap
    for span_i, (span_start, span_end) in enumerate(zip(span_starts, span_ends)):
        if allow_nested:
            if start_to_max_end.get(span_start, -1) == span_end and end_to_min_start.get(span_end, -1) == span_start:  # Remove exact
                continue
            cross_overlap = False
            for token_idx in range(span_start, span_end + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start and max_end > span_end:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end and 0 <= min_start < span_start:
                    cross_overlap = True
                    break
            if not cross_overlap:
                selected_span_indices.append(span_i)
                max_end = start_to_max_end.get(span_start, -1)
                if span_end > max_end:
                    start_to_max_end[span_start] = span_end
                min_start = end_to_min_start.get(span_end, -1)
                if min_start == -1 or span_start < min_start:
                    end_to_min_start[span_end] = span_start
        else:
            overlap = False
            for token_idx in range(span_start, span_end + 1):
                if token_idx in selected_token_idx:
                    overlap = True
                    break
            if not overlap:
                selected_span_indices.append(span_i)
                selected_token_idx.update(range(span_start, span_end + 1))
    return selected_span_indices
