import util
import logging
from collections import defaultdict, Counter
import torch
from tqdm import tqdm
import span_util
import torch.nn.functional as F
from model import Model
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from cluster import antecedent_linking, dbscan_clustering

logger = logging.getLogger(__name__)


def filter_candidates_against_special(runner, features, special_clusters):
    """ Filter out overlapping candidates against special seed spans. """
    num_filtered, num_total, num_selected = 0, 0, 0
    for feat in features:
        feat['selected_span_starts'] = feat['attr_span_starts'][:]
        feat['selected_span_ends'] = feat['attr_span_ends'][:]
        feat['selected_clusters'] = feat['attr_clusters'][:]
        feat['selected_properties'] = [(runner.prop_orig_seed + e - s + 1)
                                       for s, e in zip(feat['attr_span_starts'], feat['attr_span_ends'])]

        filtered_starts, filtered_ends = span_util.remove_overlap_batch(
            feat['ngram_span_starts'], feat['ngram_span_ends'],
            [s for s, c in zip(feat['selected_span_starts'], feat['selected_clusters']) if c == 0 or c in special_clusters],
            [e for e, c in zip(feat['selected_span_ends'], feat['selected_clusters']) if c == 0 or c in special_clusters],
            allow_nested=False)
        filtered_starts += [s for s, c in zip(feat['selected_span_starts'], feat['selected_clusters']) if c in special_clusters]
        filtered_ends += [e for e, c in zip(feat['selected_span_ends'], feat['selected_clusters']) if c in special_clusters]
        num_filtered += (len(feat['ngram_span_starts']) - len(filtered_starts))
        num_total += len(feat['ngram_span_starts'])
        num_selected += len(feat['selected_span_starts'])

        feat['ngram_span_starts'] = filtered_starts
        feat['ngram_span_ends'] = filtered_ends
    logger.info(f'Filtered overlapping candidates against {num_selected} special seed spans: '
                f'{num_total} -> {num_total - num_filtered} ({num_filtered / num_total * 100:.2f}%)')
    return features


def filter_candidates_against_selected(features):
    """ Filter out cross-overlapping candidates against selected spans (can be nested against selected). """
    num_filtered, num_total, num_selected = 0, 0, 0
    for feat in features:
        filtered_starts, filtered_ends = span_util.remove_overlap_batch(
            feat['ngram_span_starts'], feat['ngram_span_ends'],
            feat['selected_span_starts'], feat['selected_span_ends'],
            allow_nested=True)
        num_filtered += (len(feat['ngram_span_starts']) - len(filtered_starts))
        num_total += len(feat['ngram_span_starts'])
        num_selected += len(feat['selected_span_starts'])

        feat['ngram_span_starts'] = filtered_starts
        feat['ngram_span_ends'] = filtered_ends
    logger.info(f'Filtered cross-overlapping candidates against {num_selected} selected: '
                f'{num_total} -> {num_total - num_filtered} ({num_filtered / num_total * 100:.2f}%)')
    return features


def filter_overlapping_candidates(features):
    """ Filter out overlapping candidates (no nested) sequentially based on length; called once in initialize_with_seed(). """
    num_ngram_before, num_ngram_after = 0, 0
    for feat in features:
        ngram_priority = [(e - s + 1, feat['ngram_counts'].get((s, e), 100000))
                          for s, e in zip(feat['ngram_span_starts'], feat['ngram_span_ends'])]
        indices = util.argsort(ngram_priority, reverse=True)
        feat['ngram_span_starts'] = [feat['ngram_span_starts'][i] for i in indices]
        feat['ngram_span_ends'] = [feat['ngram_span_ends'][i] for i in indices]
        num_ngram_before += len(feat['ngram_span_starts'])

        selected_i = span_util.remove_overlap_sequential(feat['ngram_span_starts'], feat['ngram_span_ends'],
                                                         allow_nested=False)
        feat['ngram_span_starts'] = [feat['ngram_span_starts'][i] for i in selected_i]
        feat['ngram_span_ends'] = [feat['ngram_span_ends'][i] for i in selected_i]
        num_ngram_after += len(feat['ngram_span_starts'])
    logger.info(f'Filtered overlapping candidates sequentially based on length: {num_ngram_before} -> {num_ngram_after}')
    return features


def get_selected_spans(feat, prop_threshold):
    """ Get spans with prop >= threshold. """
    orig_starts, orig_ends, orig_clusters = [], [], []
    for start, end, cluster, prop in zip(feat['selected_span_starts'], feat['selected_span_ends'],
                                         feat['selected_clusters'], feat['selected_properties']):
        if prop >= prop_threshold:
            if cluster == 0:
                continue
            orig_starts.append(start)
            orig_ends.append(end)
            orig_clusters.append(cluster)
    return orig_starts, orig_ends, orig_clusters


def _get_ngram_and_selected_hidden(runner, model, features, layer, selected_threshold):
    """ Get normalized hidden for ngram and selected (prop >= threshold). """
    conf = runner.config
    eval_dataloader = DataLoader(features, sampler=SequentialSampler(features),
                                 batch_size=conf['eval_batch_size'], collate_fn=runner.collator)
    model.eval()
    model.to(runner.device)
    feat_i, ngram_hidden, selected_hidden, selected_clusters = 0, [], [], []
    for batch_i, batch in enumerate(tqdm(eval_dataloader, desc='Obtaining hidden')):
        seq_hidden = model.get_seq_hidden(**batch, layer=layer)
        for row_i in range(seq_hidden.size()[0]):
            feat, hidden = features[feat_i], seq_hidden[row_i]  # [seq_len, hidden]
            ngram_hidden.append(
                Model.get_span_hidden(hidden, feat['ngram_span_starts'], feat['ngram_span_ends']))

            sel_starts, sel_ends, clusters = get_selected_spans(feat, prop_threshold=selected_threshold)
            selected_hidden += Model.get_span_hidden(hidden, sel_starts, sel_ends)
            selected_clusters += clusters
            feat_i += 1
    assert feat_i == len(features)

    selected_hidden = torch.stack(selected_hidden, dim=0)  # [num_selected, hidden]
    selected_clusters = torch.tensor(selected_clusters, dtype=torch.long, device=selected_hidden.device)
    return ngram_hidden, selected_hidden, selected_clusters


def _get_similarity_for_expansion(runner, model, features, layer):
    """ Get cosine similarity between ngram & seed.
    """
    model.eval()
    with torch.no_grad():
        ngram_hidden, seed_hidden, seed_clusters = _get_ngram_and_selected_hidden(runner, model, features, layer, selected_threshold=runner.prop_orig_seed)

        # Get cosine similarity between ngram & seed
        all_similarities, all_clusters = [], []
        for feat_i, (feat, feat_ngram_hidden) in enumerate(tqdm(zip(features, ngram_hidden), total=len(ngram_hidden))):
            if not feat_ngram_hidden:
                all_similarities.append(None)
                all_clusters.append(None)
                continue
            feat_ngram_hidden = torch.stack(feat_ngram_hidden, dim=0)  # [num_ngrams, hidden]
            similarity = torch.matmul(feat_ngram_hidden, seed_hidden.t())  # [num_ngrams, num_seeds]
            ngram_similarity, ngram_sel_idx = torch.max(similarity, dim=-1)

            all_similarities.append(ngram_similarity.cpu())
            all_clusters.append(seed_clusters[ngram_sel_idx].cpu())

    return all_similarities, all_clusters


def expand_seed_lexical(runner, features):
    """ Expand by lexical matching between ngram & seed.
    Expanded will have property = (prop_expanded_seed + lexicon_length).
    """
    seed_lexicons = defaultdict(lambda: Counter())
    count_th = 1
    # Collect seed lexicons
    for feat in features:
        sel_starts, sel_ends, clusters = get_selected_spans(feat, prop_threshold=runner.prop_orig_seed)
        for span_start, span_end, span_cluster in zip(sel_starts, sel_ends, clusters):
            lexicon = tuple(feat['input_ids'][span_start: span_end + 1])
            seed_lexicons[lexicon][span_cluster] += 1
    # Filter by count
    # Prioritize most common cluster for a lexicon
    filtered_lexicons = {}
    for lexicon, counter in seed_lexicons.items():
        if sum(counter.values()) < count_th:
            continue
        orig_most_common_cluster, _ = counter.most_common(1)[0]
        filtered_lexicons[lexicon] = orig_most_common_cluster
    seed_lexicons = filtered_lexicons

    # Expand and attach lexicon length as property
    ngram_expand, ngram_total = 0, 0
    for feat in tqdm(features):
        for ngram_start, ngram_end in zip(feat['ngram_span_starts'], feat['ngram_span_ends']):
            ngram_lexicon = tuple(feat['input_ids'][ngram_start: ngram_end + 1])
            if ngram_lexicon in seed_lexicons:
                ngram_cluster = seed_lexicons[ngram_lexicon]
                feat['selected_span_starts'].append(ngram_start)
                feat['selected_span_ends'].append(ngram_end)
                feat['selected_clusters'].append(ngram_cluster)
                feat['selected_properties'].append(runner.prop_expanded_seed + len(ngram_lexicon))
                ngram_expand += 1
            ngram_total += 1

    logger.info(f'Expanded seed upon {len(seed_lexicons)} lexicons (count >= {count_th}): '
                f'{ngram_expand}/{ngram_total} ({ngram_expand / ngram_total * 100:.2f}%) candidates')
    return features


def filter_expanded_seed(runner, features):
    """ Remove overlapping/duplicate seed sequentially. """
    sanitize_before, sanitize_after, sanitize_after_from_sim = 0, 0, 0
    for feat in features:
        # Sort selection based on scores
        indices = util.argsort(feat['selected_properties'], reverse=True)
        selected_starts = [feat['selected_span_starts'][i] for i in indices]
        selected_ends = [feat['selected_span_ends'][i] for i in indices]
        selected_clusters = [feat['selected_clusters'][i] for i in indices]
        selected_properties = [feat['selected_properties'][i] for i in indices]
        sanitize_before += sum([(p < runner.prop_orig_seed) for p in selected_properties])
        # Remove overlap/duplicate
        selected_span_indices = span_util.remove_overlap_sequential(selected_starts, selected_ends, allow_nested=False)
        feat['selected_span_starts'] = [selected_starts[i] for i in selected_span_indices]
        feat['selected_span_ends'] = [selected_ends[i] for i in selected_span_indices]
        feat['selected_clusters'] = [selected_clusters[i] for i in selected_span_indices]
        feat['selected_properties'] = [selected_properties[i] for i in selected_span_indices]
        sanitize_after += sum([(p < runner.prop_orig_seed) for p in feat['selected_properties']])
        sanitize_after_from_sim += sum([(p < runner.prop_expanded_seed + 1) for p in feat['selected_properties']])

    logger.info(f'Filtered expanded seeds: {sanitize_before} -> {sanitize_after} spans')
    logger.info(f'After filtering, {sanitize_after_from_sim} ({sanitize_after_from_sim / sanitize_after * 100:.2f}%)'
                f' expanded seeds are by similarity')
    return features


def _get_hidden_for_clustering(runner, model, features, exp_suffix):
    """ Get normalized hidden of existing spans & ngrams for clustering (cache results). """
    using_gold_ngram = features[0].get('gold_ngram', False)
    cache_identifier = 'cluster_cache' + ('_w_gold' if using_gold_ngram else '')
    cache = runner.load_results(runner.dataset_name, cache_identifier, suffix=exp_suffix, ext='bin')
    if True and cache is not None:
        all_ngrams, all_ngram_hidden, existing_hidden, existing_clusters, ngram_cls_clusters, ngram_cls_probs = cache
        all_ngram_hidden = all_ngram_hidden.to(runner.device)
        existing_hidden = existing_hidden.to(runner.device)
        existing_clusters = existing_clusters.to(runner.device)
        if ngram_cls_clusters is not None:
            ngram_cls_clusters = ngram_cls_clusters.to(runner.device)
            ngram_cls_probs = ngram_cls_probs.to(runner.device)
    else:
        model.eval()
        with torch.no_grad():
            ngram_hidden, existing_hidden, existing_clusters = _get_ngram_and_selected_hidden(
                runner, model, features, layer=-1, selected_threshold=runner.prop_clustered)
            all_ngrams, all_ngram_hidden = [], []
            for feat_i, (feat, feat_ngram_hidden) in enumerate(zip(features, ngram_hidden)):
                all_ngrams += [(feat_i, start, end) for start, end in
                               zip(feat['ngram_span_starts'], feat['ngram_span_ends'])]
                all_ngram_hidden += feat_ngram_hidden
            all_ngram_hidden = torch.stack(all_ngram_hidden, dim=0)  # [num_ngrams, hidden]
            assert len(all_ngrams) == all_ngram_hidden.size()[0]

            # Do cls
            if model.config['attr_cls_coef']:
                ngram_cls_clusters, ngram_cls_probs = get_attr_cls(model, all_ngram_hidden, step=256)
            else:
                ngram_cls_clusters, ngram_cls_probs = None, None
        runner.save_results(runner.dataset_name, cache_identifier, suffix=exp_suffix, ext='bin', results=(
            all_ngrams, all_ngram_hidden.cpu(), existing_hidden.cpu(), existing_clusters.cpu(),
            None if ngram_cls_clusters is None else ngram_cls_clusters, None if ngram_cls_probs is None else ngram_cls_probs))
    return all_ngrams, all_ngram_hidden, existing_hidden, existing_clusters, ngram_cls_clusters, ngram_cls_probs


def get_attr_cls(model, span_hidden, step=256):
    model.eval()
    with torch.no_grad():
        all_pred_attrs, all_pred_probs = [], []
        i = 0
        while i < span_hidden.size()[0]:
            hidden = span_hidden[i: i+step]
            pred_attrs, pred_probs = model.get_attr_classification(hidden)  # [step, hidden]
            all_pred_attrs.append(pred_attrs)
            all_pred_probs.append(pred_probs)
            i += step
        all_pred_attrs = torch.cat(all_pred_attrs, dim=0)  # [num_spans]
        all_pred_probs = torch.cat(all_pred_probs, dim=0)
        return all_pred_attrs, all_pred_probs


def gradual_cluster(runner, model, features, exp_suffix, use_attr_cls, new_cluster_start=1000):
    conf = runner.config
    all_ngrams, all_ngram_hidden, existing_hidden, existing_clusters, ngram_cls_clusters, ngram_cls_probs = _get_hidden_for_clustering(runner, model, features, exp_suffix)

    with torch.no_grad():
        # Gather existing clusters
        cluster_ids = existing_clusters.unique().tolist()
        assert 0 not in cluster_ids
        cluster2size, cluster_meanhidden, cluster_meanth = {}, [], []
        for cluster_i in cluster_ids:
            hidden = existing_hidden[existing_clusters == cluster_i]
            mean_hidden = hidden.mean(dim=0)
            cluster2size[cluster_i] = hidden.size()[0]
            cluster_meanhidden.append(mean_hidden)
            cluster_meanth.append(max(torch.matmul(hidden, mean_hidden.unsqueeze(-1)).mean() * conf['cluster_sim_relax'],
                                      torch.tensor(0.4, device=hidden.device)))
        cluster_meanhidden = torch.stack(cluster_meanhidden, dim=0)  # [num_clusters, hidden]
        cluster_meanth = torch.stack(cluster_meanth, dim=0)  # [num_clusters]

        # Expand existing
        logger.info(f'Expanding existing clusters by {conf["cluster_sim_relax"]} relaxation...')
        sim = torch.matmul(all_ngram_hidden, cluster_meanhidden.t())  # [num_ngrams, num_clusters]
        max_sim, max_cluster_i = (sim - cluster_meanth.unsqueeze(0)).max(dim=-1)  # [num_ngrams]
        ngram_sel_clusters = torch.tensor(cluster_ids, device=sim.device)[max_cluster_i]
        ngram_sel_scores = max_sim
        ngram_sel_invalid = max_sim < 0
        ngram_sel_clusters[ngram_sel_invalid] = -1  # Mark invalid selection: assign cluster -1
        ngram_sel_clusters, ngram_sel_scores = ngram_sel_clusters.tolist(), ngram_sel_scores.tolist()
        logger.info(f'Expanded existing clusters: {ngram_sel_invalid.size()[0] - ngram_sel_invalid.sum().item()} ngrams')

        # Add newly selected ngrams
        for ngram_i, (cluster_i, score) in enumerate(zip(ngram_sel_clusters, ngram_sel_scores)):
            if cluster_i == -1 and use_attr_cls:
                cluster_i, score = ngram_cls_clusters[ngram_i], ngram_cls_probs[ngram_i]
            if cluster_i != -1:
                feat_i, start, end = all_ngrams[ngram_i]
                feat = features[feat_i]
                feat['selected_span_starts'].append(start)
                feat['selected_span_ends'].append(end)
                feat['selected_clusters'].append(cluster_i)
                feat['selected_properties'].append(score + 0.02 * (end - start))

        # Gather remaining ngrams
        ngram_remaining_indices = torch.arange(0, len(all_ngrams), device=max_sim.device)[ngram_sel_invalid]
        remaining_ngram_hidden = all_ngram_hidden[ngram_remaining_indices]  # [num_remaining_ngrams, hidden]
        ngram_remaining_indices = ngram_remaining_indices.tolist()
        remaining_ngrams = [all_ngrams[i] for i in ngram_remaining_indices]
        if not remaining_ngrams:
            logger.info(f'No ngrams left for DBSCAN')
            return features

        # Cluster remaining ngrams by DBSCAN
        logger.info(f'Getting new clusters on {len(remaining_ngrams)} ngrams by DBSCAN...')
        remaining_ngram_cluster_ids = dbscan_clustering(remaining_ngram_hidden.cpu().numpy(), metric='cosine',
                                                        eps=conf['dbscan_eps'], min_samples=conf['dbscan_min_samples'], n_jobs=8)
        remaining_ngram_cluster_ids = remaining_ngram_cluster_ids.tolist()
        # Add newly selected ngrams
        num_existing_clusters = max(max(cluster_ids) + 1, new_cluster_start)
        for ngram, cluster_i in zip(remaining_ngrams, remaining_ngram_cluster_ids):
            if cluster_i >= 0:
                feat_i, start, end = ngram
                feat = features[feat_i]
                feat['selected_span_starts'].append(start)
                feat['selected_span_ends'].append(end)
                feat['selected_clusters'].append(cluster_i + num_existing_clusters)
                feat['selected_properties'].append(0.02 * (end - start))
    return features


def baseline_cluster(runner, model, features, exp_suffix, use_attr_cls, new_cluster_start=1000):
    """ In-place. (1) clustering by DBSCAN (2) combined with classification if needed.
    """
    conf = runner.config
    all_ngrams, all_ngram_hidden, existing_hidden, existing_clusters, ngram_cls_attrs, ngram_cls_probs = _get_hidden_for_clustering(runner, model, features, exp_suffix)

    assert 0 not in existing_clusters  # Make sure not expanding brand cluster
    num_existing_clusters = max(max(existing_clusters) + 1, new_cluster_start)

    # DBSCAN
    ngram_dbscan_clusters = dbscan_clustering(all_ngram_hidden.cpu().numpy(), metric='cosine',
                                              eps=conf['dbscan_eps'], min_samples=conf['dbscan_min_samples'], n_jobs=8)
    dbscan_clusters = defaultdict(lambda: set())
    for ngram_i, ngram_c in enumerate(ngram_dbscan_clusters.tolist()):
        if ngram_c >= 0:
            dbscan_clusters[ngram_c].add(ngram_i)
    all_dbscan_ngrami = set.union(*dbscan_clusters.values())
    logger.info(f'DBSCAN clustering: {len(all_dbscan_ngrami)} ngrams')

    # Add classification
    if not use_attr_cls:
        clusters = {(ci + num_existing_clusters): c for ci, c in dbscan_clusters.items()}
        ngrami2score = {}
    else:
        assert ngram_cls_attrs is not None
        cls_prob_threshold = min(1, 1 / len(existing_clusters.unique().tolist()) * conf['attr_cls_th'])
        ngram_cls_invalid = ngram_cls_probs < cls_prob_threshold
        ngram_cls_attrs[ngram_cls_invalid] = -1  # Mark invalid selection: assign cluster -1

        cls_clusters = defaultdict(lambda: set())
        for ngram_i, ngram_c in enumerate(ngram_cls_attrs.tolist()):
            if ngram_c >= 0:
                cls_clusters[ngram_c].add(ngram_i)
        all_cls_ngrami = set.union(*cls_clusters.values())
        logger.info(f'Attr cls (> {cls_prob_threshold} threshold): {len(all_cls_ngrami)} ngrams')

        # Align CLS to DBSCAN clusters
        cls2dbscan, dbscan2cls = {}, {}
        for cls_ci, cls_c in cls_clusters.items():
            num_overlapping = [(len(dbscan_c & cls_c), dbscan_ci) for dbscan_ci, dbscan_c in dbscan_clusters.items()]
            most_overlapping, matched_dbscan_ci = max(num_overlapping)
            cls2dbscan[cls_ci] = matched_dbscan_ci
            dbscan2cls[matched_dbscan_ci] = cls_ci

        # Merge
        for cls_ci, cls_c in cls_clusters.items():
            dbscan_c = dbscan_clusters[cls2dbscan[cls_ci]]
            dbscan_c |= (cls_c - all_dbscan_ngrami)  # Combine clusters

        # Align with existing cluster ids
        clusters = {dbscan2cls.get(ci, ci + num_existing_clusters): c for ci, c in dbscan_clusters.items()}
        ngrami2score = {ngram_i: prob for ngram_i, prob in enumerate(ngram_cls_probs.tolist()) if prob >= 0}

    # Add ngram attributes to features
    ngrami2ci = {}
    for ci, c in clusters.items():
        for ngram_i in c:
            ngrami2ci[ngram_i] = ci
    for ngram_i, ci in ngrami2ci.items():
        feat_i, start, end = all_ngrams[ngram_i]
        feat = features[feat_i]
        feat['selected_span_starts'].append(start)
        feat['selected_span_ends'].append(end)
        feat['selected_clusters'].append(ci)
        feat['selected_properties'].append(ngrami2score.get(ngram_i, 0))

    return features


def tag_cluster(runner, model, features, exp_suffix, use_attr_cls, new_cluster_start=1000):
    logger.info(f'Performing tagging for clustering ...')
    dataloader = DataLoader(features, sampler=SequentialSampler(features),
                            batch_size=runner.config['eval_batch_size'], collate_fn=runner.collator)
    model.eval()
    model.to(runner.device)
    all_seq_entities = []
    for batch_i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch.pop('token_tags', None)
            logits = model(**batch)
            all_seq_entities += model.decode(logits, attention_mask=batch['attention_mask'])
    assert len(all_seq_entities) == len(features)

    # Add entities to features
    for feat_i, entities in enumerate(all_seq_entities):
        feat = features[feat_i]
        for ci, s, e, _ in entities:
            feat['selected_span_starts'].append(s)
            feat['selected_span_ends'].append(e)
            feat['selected_clusters'].append(int(ci))
            feat['selected_properties'].append(0)

    return features


def opentag_cluster(runner, model, features, exp_suffix, use_attr_cls, new_cluster_start=1000):
    # Condition all training types for inference on each input sequence
    for feat in features:
        feat['opentag_types'] = model.training_types

    logger.info(f'Performing opentag for clustering ...')
    dataloader = DataLoader(features, sampler=SequentialSampler(features),
                            batch_size=runner.config['eval_batch_size'], collate_fn=runner.collator)
    model.eval()
    model.to(runner.device)
    all_seq_entities = []
    for batch_i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            flattened_seq_logits = model(**batch)
            flattened_typed_entities = model.decode(flattened_seq_logits, attention_mask=batch['attention_mask'].repeat_interleave(len(model.training_types), dim=0))
            assert len(flattened_typed_entities) == batch['input_ids'].size()[0] * len(model.training_types)

            flattened_i = 0
            while flattened_i < len(flattened_typed_entities):
                seq_entities = []
                for entity_type, entities in zip(model.training_types, flattened_typed_entities[flattened_i: flattened_i+len(model.training_types)]):
                    for _, s, e, _ in entities:
                        seq_entities.append((entity_type, s, e))
                flattened_i += len(model.training_types)
                all_seq_entities.append(seq_entities)
    assert len(all_seq_entities) == len(features)

    # Add entities to features
    for feat_i, entities in enumerate(all_seq_entities):
        feat = features[feat_i]
        for ci, s, e in entities:
            feat['selected_span_starts'].append(s)
            feat['selected_span_ends'].append(e)
            feat['selected_clusters'].append(ci)
            feat['selected_properties'].append(0)

    return features
