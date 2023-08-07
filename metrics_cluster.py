import logging
from metrics import get_f1
from collections import defaultdict
import math
from sklearn.metrics import accuracy_score, adjusted_rand_score

logger = logging.getLogger(__name__)


def jaccard(pred_label_pairs):
    pl_count = defaultdict(int)
    p_count = defaultdict(int)
    l_count = defaultdict(int)
    n_total = len(pred_label_pairs)

    for pred, label in pred_label_pairs:
        p_count[pred] += 1
        l_count[label] += 1
        pl_count[(pred, label)] += 1

    n_pred = len(p_count)  # num of predicted clusters
    n_label = len(l_count)

    TP = 0
    FN = 0
    FP = 0
    for (p, l), count in pl_count.items():
        TP += count * (count - 1) / 2

    for p, count in p_count.items():
        FN += count * (count - 1) / 2
    FN -= TP

    for l, count in l_count.items():
        FP += count * (count - 1) / 2
    FP -= TP

    return TP / (TP + FN + FP)


def nmi(pred_label_pairs, debug=False):
    pl_count = defaultdict(int)
    p_count = defaultdict(int)
    l_count = defaultdict(int)
    n_total = len(pred_label_pairs)

    for pred, label in pred_label_pairs:
        p_count[pred] += 1
        l_count[label] += 1
        pl_count[(pred, label)] += 1

    # compute TP, I_c_t
    I_p_l = 0
    for (pred, label), count in pl_count.items():
        p_pred_label = count / n_total
        p_pred = p_count[pred] / n_total
        p_label = l_count[label] / n_total
        I_p_l += p_pred_label * math.log2(p_pred_label / p_pred / p_label)

    # compute FN, FP, H_C, H_T
    H_pred = 0
    H_label = 0
    for pred, count in p_count.items():
        p_pred = count / n_total
        H_pred += p_pred * math.log2(p_pred)

    for label, count in l_count.items():
        p_label = count / n_total
        H_label += p_label * math.log2(p_label)

    # generate output
    nmi = (I_p_l / math.sqrt(H_pred * H_label)) if I_p_l else 0
    return nmi


def matching_tp(pred_label_pairs, debug=False):
  pl_count = defaultdict(int)
  p_count = defaultdict(int)
  l_count = defaultdict(int)

  for pred, label in pred_label_pairs:
    p_count[pred] += 1
    l_count[label] += 1
    pl_count[(pred, label)] += 1

  TP = 0
  for p, count in p_count.items():
    # for each predicted cluster
    # find largest subset that belongs to the same true cluster
    p_max = 0
    match_cluster = 0
    for l in l_count:
      if (p, l) not in pl_count:
        continue
      if pl_count[(p, l)] > p_max:
        p_max = pl_count[(p, l)]
    TP += p_max

  return TP


class ClusterEvaluator:

    @classmethod
    def _get_best_candidate(cls, span, candidate_starts, candidate_ends, candidate_clusters):
        overlaps = []
        for i, (s, e) in enumerate(zip(candidate_starts, candidate_ends)):
            if e >= span[0] and s <= span[1]:  # Exist any overlap
                if min(span[1], e) - max(span[0], s) >= (span[1] - span[0]) // 2:  # Partial: cover at least half characters
                    overlaps.append(i)
        if overlaps:
            best_i = min(overlaps, key=lambda el: abs(span[0] - candidate_starts[el]) + abs(span[1] - candidate_ends[el]))
            return (candidate_starts[best_i], candidate_ends[best_i]), candidate_clusters[best_i]
        return None, None

    @classmethod
    def _get_cluster_assignment(cls, id2result):
        exact_pred_gold_pairs, partial_pred_gold_pairs = [], []

        for test_id, result in id2result.items():
            result['partial_pred_gold_pairs'], result['partial_gold_spans'], result['partial_pred_spans'] = [], [], []
            for gold_s, gold_e, gold_cluster in zip(result['gold_starts'], result['gold_ends'], result['gold_clusters']):
                gold_span = (gold_s, gold_e)
                best_candidate, best_candidate_cluster = cls._get_best_candidate(
                    gold_span, result['selected_starts'], result['selected_ends'], result['selected_clusters'])

                if best_candidate:
                    partial_pred_gold_pairs.append((best_candidate_cluster, gold_cluster))
                    result['partial_pred_gold_pairs'].append((best_candidate_cluster, gold_cluster))
                    result['partial_gold_spans'].append(gold_span)
                    result['partial_pred_spans'].append(best_candidate)
                    if best_candidate == gold_span:
                        exact_pred_gold_pairs.append((best_candidate_cluster, gold_cluster))

        return exact_pred_gold_pairs, partial_pred_gold_pairs

    @classmethod
    def _evaluate_jaccard(cls, pred_gold_pairs):
        return 0 if len(pred_gold_pairs) <= 1 else jaccard(pred_gold_pairs)

    @classmethod
    def _evaluate_nmi(cls, pred_gold_pairs):
        return 0 if not pred_gold_pairs else nmi(pred_gold_pairs)

    @classmethod
    def _evaluate_ari(cls, pred_gold_pairs):
        if not pred_gold_pairs:
            return 0
        all_clusters = set()
        for pair in pred_gold_pairs:
            all_clusters.add(pair[0])
            all_clusters.add(pair[1])
        cluster2i = {c: i for i, c in enumerate(all_clusters)}
        preds = [cluster2i[pair[0]] for pair in pred_gold_pairs]
        golds = [cluster2i[pair[1]] for pair in pred_gold_pairs]
        return adjusted_rand_score(golds, preds)

    @classmethod
    def _evaluate_recall(cls, pred_gold_pairs, id2result):
        tp = 0 if not pred_gold_pairs else matching_tp(pred_gold_pairs)
        all_p = sum(len(result['gold_starts']) for result in id2result.values())
        return 0 if not all_p else (tp / all_p)

    @classmethod
    def evaluate(cls, id2result):
        exact_pred_gold_pairs, partial_pred_gold_pairs = cls._get_cluster_assignment(id2result)

        metrics_exact, metrics_partial = {
            'jaccard': cls._evaluate_jaccard(exact_pred_gold_pairs) * 100,
            'nmi': cls._evaluate_nmi(exact_pred_gold_pairs) * 100,
            'ari': cls._evaluate_ari(exact_pred_gold_pairs) * 100,
            'recall': cls._evaluate_recall(exact_pred_gold_pairs, id2result) * 100
        }, {
            'jaccard': cls._evaluate_jaccard(partial_pred_gold_pairs) * 100,
            'nmi': cls._evaluate_nmi(partial_pred_gold_pairs) * 100,
            'ari': cls._evaluate_ari(partial_pred_gold_pairs) * 100,
            'recall': cls._evaluate_recall(partial_pred_gold_pairs, id2result) * 100
        }

        metrics_exact['meta'] = get_f1(p=(metrics_exact['jaccard'] + metrics_exact['nmi'] + metrics_exact['ari']) / 3,
                                       r=metrics_exact['recall'])
        metrics_partial['meta'] = get_f1(p=(metrics_partial['jaccard'] + metrics_partial['nmi'] + metrics_partial['ari']) / 3,
                                         r=metrics_partial['recall'])
        return metrics_exact, metrics_partial
