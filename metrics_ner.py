from collections import defaultdict
from metrics import get_prf


class PartialMeEvaluator:

    @classmethod
    def _get_best_candidate(cls, span, candidate_starts, candidate_ends):
        """ Get the best overlapping candidate (could be exact). """
        overlaps = []
        for i, (s, e) in enumerate(zip(candidate_starts, candidate_ends)):
            if e >= span[0] and s <= span[1]:  # Exist any overlap
                if min(span[1], e) - max(span[0], s) >= (span[1] - span[0]) // 2:  # Partial: cover at least half characters
                    overlaps.append(i)
        if overlaps:
            best_i = min(overlaps, key=lambda el: abs(span[0] - candidate_starts[el]) + abs(span[1] - candidate_ends[el]))
            return candidate_starts[best_i], candidate_ends[best_i]
        return None

    def _collect_stats(self, id2result):
        num_tp, num_ptp, num_gold, num_pred = 0, 0, 0, 0
        for test_id, result in id2result.items():
            num_gold += len(result['gold_starts'])
            num_pred += len(result['selected_starts'])

            result['all_tp_gold_spans'], result['all_tp_pred_spans'] = [], []
            all_tp_pred_spans = set()
            for gold_s, gold_e in zip(result['gold_starts'], result['gold_ends']):
                gold_span = (gold_s, gold_e)
                best_pred = self._get_best_candidate(gold_span, result['selected_starts'], result['selected_ends'])
                if best_pred and best_pred not in all_tp_pred_spans:
                    all_tp_pred_spans.add(best_pred)  # pred may serve more than once; ensure only once, so precision <= 1
                    result['all_tp_gold_spans'].append(gold_span)
                    result['all_tp_pred_spans'].append(best_pred)
                    if gold_span == best_pred:
                        num_tp += 1
                    else:
                        num_ptp += 1
        return num_tp, num_ptp, num_gold, num_pred

    def evaluate(self, id2result, partial_coef=1):
        num_tp, num_ptp, num_gold, num_pred = self._collect_stats(id2result)
        exact_p, exact_r, exact_f = get_prf(num_tp, num_pred, num_gold)
        partial_p, partial_r, partial_f = get_prf(num_tp + num_ptp * partial_coef, num_pred, num_gold)
        return {'p': exact_p, 'r': exact_r, 'f': exact_f}, {'p': partial_p, 'r': partial_r, 'f': partial_f}


class NerEvaluator:
    def __init__(self, ner2id):
        self.ner2id = ner2id

    def _collect_stats(self, predicted_spans, predicted_types, gold_spans, gold_types):
        all_preds, all_golds = set(), set()
        num_preds, num_golds = defaultdict(int), defaultdict(int)
        for span, span_type in zip(predicted_spans, predicted_types):
            all_preds.add((span, span_type))
            num_preds[span_type] += 1
        for span, span_type in zip(gold_spans, gold_types):
            all_golds.add((span, span_type))
            num_golds[span_type] += 1
        all_tps = all_preds & all_golds
        num_tps = defaultdict(int)
        for span, span_type in all_tps:
            num_tps[span_type] += 1
        return num_tps, num_preds, num_golds

    def evaluate(self, predicted_spans, predicted_types, gold_spans, gold_types):
        num_tps, num_preds, num_golds = self._collect_stats(predicted_spans, predicted_types, gold_spans, gold_types)
        type2prf = {ner_type: get_prf(num_tps[ner_id], num_preds[ner_id], num_golds[ner_id])
                    for ner_type, ner_id in self.ner2id.items()}
        total_prf = get_prf(sum(num_tps.values()), sum(num_preds.values()), sum(num_golds.values()))
        return total_prf, type2prf


class MeEvaluator:
    def __init__(self):
        pass

    def _collect_stats(self, predicted_spans, gold_spans):
        all_preds, all_golds = set(predicted_spans), set(gold_spans)
        num_preds, num_golds = len(all_preds), len(all_golds)
        all_tps = all_preds & all_golds
        num_tps = len(all_tps)
        return num_tps, num_preds, num_golds

    def evaluate(self, predicted_spans, gold_spans):
        num_tps, num_preds, num_golds = self._collect_stats(predicted_spans, gold_spans)
        total_prf = get_prf(num_tps, num_preds, num_golds)
        return total_prf
