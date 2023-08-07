import io_util
from run_base import BaseRunner
import logging
from functools import cached_property
from data import DataProcessor
import sys
import random
import torch
from tqdm import tqdm
import span_util
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import selection_util
from selection_util import gradual_cluster, baseline_cluster, tag_cluster, opentag_cluster
import numpy as np
import time
import plotly.graph_objects as go
from cluster import get_precompute_dist
from data_util import show_feature, pt2prefix
from collator import FeatureCollator, iterate_asins
from model import Model
from model_asin import ModelAsin
from model_tag import ModelTag
from model_opentag import ModelOpenTag
from metrics_cluster import ClusterEvaluator
from metrics_ner import PartialMeEvaluator
from metrics import get_accuracy
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import torch.cuda.amp as amp
from os.path import join, exists
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Runner(BaseRunner):
    prop_orig_seed = 500000  # Indication of original seeds
    prop_expanded_seed = 100000  # Indication of expanded seeds
    prop_clustered = 1000

    def __init__(self, config_name, gpu_id=None, **kwargs):
        super(Runner, self).__init__(config_name, gpu_id, **kwargs)
        logger.info(self.config)

        self.asin_collation = not self.config['do_tag'] and (self.config['vi_coef'] or self.config['usp_bullet_coef'])
        self.num_attrs = None  # Set in initialize_model()

    @cached_property
    def data(self):
        return DataProcessor(self.config)

    @cached_property
    def collator(self):
        return FeatureCollator(self.data.tokenizer, device=self.device)

    def initialize_model(self, init_suffix=None):
        self.num_attrs = len(self.data.get_meta(self.dataset_name)['attr2cluster'])
        if self.config['do_tag'] and not self.config['opentag']:
            model = ModelTag(self.config, num_tags=(2 * self.num_attrs + 1), do_crf=self.config['use_crf'])
        elif self.asin_collation:
            model = ModelAsin(self.config, num_attrs=self.num_attrs)
        else:
            model = Model(self.config, num_attrs=self.num_attrs)

        if init_suffix:
            self.load_model_checkpoint(model, init_suffix)
        if self.config['freeze_layers']:
            model.freeze_layers(self.config['freeze_layers'])
        return model

    def start(self, model, eval_save_threshold=None):
        conf = self.config
        docs, features = self.data.get_data(self.dataset_name, partition='all')

        features = self.initialize_with_seed(model, features, layer=(conf['similarity_layer']))  # Saved
        if conf['do_tag']:
            features = self.initialize_for_tagging(features)

        # Start iterating
        all_training_eval_scores, all_selection_eval_scores = [], []
        for itr in range(1, conf['max_itr'] + 1):
            # Remove test set features from representation learning (they only involve in clustering)
            train_docs = [doc for doc in docs if not doc['is_test'] and not doc['is_dev']]
            train_features = [feat for feat in features if not feat['is_test'] and not feat['is_dev']]
            dev_docs = [doc for doc in docs if doc['is_dev']]
            dev_features = [feat for feat in features if feat['is_dev']]

            # Representation learning
            _, eval_scores = self.train(model, train_docs, train_features, dev_docs, dev_features,
                                        itr=itr, use_amp=conf['use_amp'], eval_save_threshold=eval_save_threshold,
                                        asin_collation=self.asin_collation)
            all_training_eval_scores += eval_scores

            # Expand clusters and save
            features = self.expand_clusters(model, itr=itr, exp_suffix=self.name_suffix)  # Saved
            all_selection_eval_scores.append(0)

        # Wrap up
        logger.info(f'Selection eval scores at each iteration: {all_selection_eval_scores}')

        return all_training_eval_scores, all_selection_eval_scores

    def get_itr_feat_path(self, partition, itr, exp_suffix, using_gold_ngram=False):
        conf = self.config
        partition = partition + ('_w_gold' if using_gold_ngram else '')
        if itr == 0:  # If initial expanded seed, use exp-agnostic path for shared usage
            path = self.data.get_data_feature_path(self.dataset_name, partition)
            path = path[:path.rfind('.')] + '_itr0'
            if conf['expand_by_sim']:
                path += f'_layer{conf["similarity_layer"]}_sim{conf["emb_similarity"]}'
        else:
            clustering = f'bs_eps{conf["dbscan_eps"]}_ms{conf["dbscan_min_samples"]}' if conf['baseline_clustering'] \
                else f'gradual_cr{conf["cluster_sim_relax"]}_eps{conf["dbscan_eps"]}_ms{conf["dbscan_min_samples"]}'
            if conf['attr_cls_coef']:
                clustering = f'{clustering}_w_cls{conf["attr_cls_th"]}'
            if conf['do_tag']:
                clustering = f'tag'
            path = join(self.config['log_dir'], 'results', f'feat_{self.dataset_name}_{partition}_{exp_suffix}_itr{itr}_{clustering}')
            os.makedirs(join(self.config['log_dir'], 'results'), exist_ok=True)
        path += '.bin'
        return path

    def initialize_with_seed(self, model, features, layer):
        """ Expand (per PT) seed selection and sanitize candidates in-place and save. """
        feat_save_path = self.get_itr_feat_path('all', itr=0, exp_suffix=self.name_suffix)
        if exists(feat_save_path):
            features = io_util.read_pickle(feat_save_path)
            logger.info(f'Loaded cached features at itr 0 (seed expansion) at {feat_save_path}')
            return features

        pt2feats = defaultdict(list)
        for feat in features:
            pt2feats[feat['pt']].append(feat)
        logger.info(f'Expanding seed: found PT {tuple(pt2feats.keys())}; expand within each PT')

        # All operations are in-place
        meta = self.data.get_meta(self.dataset_name)
        for pt, pt_features in pt2feats.items():
            logger.info(f'Expanding on {len(pt_features)} features of {pt}:')
            pt_features = selection_util.filter_candidates_against_special(self, pt_features, meta['special_attri'])
            pt_features = selection_util.filter_candidates_against_selected(pt_features)
            if self.config['expand_by_sim']:
                pt_features = selection_util.expand_seed_similar(self, model, pt_features, layer)  # Temporarily disabled
            pt_features = selection_util.expand_seed_lexical(self, pt_features)
            pt_features = selection_util.filter_expanded_seed(self, pt_features)
            pt_features = selection_util.filter_candidates_against_selected(pt_features)
            pt_features = selection_util.filter_overlapping_candidates(pt_features)

        io_util.write_pickle(feat_save_path, features)
        logger.info(f'Saved features at itr0 (seed expansion) to {feat_save_path}')
        return features

    def initialize_for_tagging(self, features):
        """ model.decode() should follow the same BIO tagging scheme. """
        if not self.config['opentag']:
            # For normal tagging
            for feat in features:
                tag_seq = [0] * len(feat['input_ids'])  # O tag
                for s, e, c in zip(feat['selected_span_starts'], feat['selected_span_ends'], feat['selected_clusters']):
                    tag_seq[s] = 1 + c  # B-xxx tag
                    for i in range(s + 1, e + 1):
                        tag_seq[i] = 1 + self.num_attrs + c  # I-xxx tag
                feat['token_tags'] = tag_seq
                feat['num_attrs'] = self.num_attrs
            logger.info(f'Finished converting BIO tags')
        else:
            # For opentag
            for feat in features:
                feat['opentag_types'], feat['opentag_typed_token_tags'] = [], []  # len = num clusters for this feat
                c2spans = defaultdict(list)
                for s, e, c in zip(feat['selected_span_starts'], feat['selected_span_ends'], feat['selected_clusters']):
                    c2spans[c].append((s, e))
                for c, spans in c2spans.items():
                    tag_seq = [0] * len(feat['input_ids'])  # O tag
                    for s, e in spans:
                        tag_seq[s] = 1  # B tag
                        for i in range(s + 1, e + 1):
                            tag_seq[i] = 2  # I tag
                    feat['opentag_types'].append(c)
                    feat['opentag_typed_token_tags'].append(tag_seq)
            logger.info(f'Finished converting for opentag')
        return features

    def expand_clusters(self, model, itr, exp_suffix, features=None):
        """ Expand test-feature clusters (existing and new) in-places and save. """
        conf = self.config
        if features is None:  # If features not provided, load from saved features
            saved_feat_path = self.get_itr_feat_path('all', itr=itr-1, exp_suffix=exp_suffix)
            features = io_util.read_pickle(saved_feat_path)
            if features is None:
                logger.info(f'Features do not exist for {exp_suffix} at itr{itr}')
                return None
            logger.info(f'Expanding clusters: using saved itr features from {saved_feat_path}')
        else:
            logger.info(f'Expanding clusters: using provided features')
        features = [feat for feat in features if feat['is_test']]

        # Mark any new selection as existing
        for feat in features:
            feat['selected_properties'] = [
                (p if p >= self.prop_expanded_seed else (self.prop_clustered + itr))
                for p in feat['selected_properties']]
        # Expand clusters
        clustering = (opentag_cluster if conf['opentag'] else tag_cluster) if conf['do_tag'] \
            else baseline_cluster if conf['baseline_clustering'] else gradual_cluster
        model.to(self.device)
        clustering(self, model, features, exp_suffix, use_attr_cls=bool(conf['attr_cls_coef']))  # In-place

        # Save features of current itr
        save_path = self.get_itr_feat_path('all', itr=itr, exp_suffix=exp_suffix,
                                           using_gold_ngram=features[0].get('gold_ngram', False))
        io_util.write_pickle(save_path, features)
        logger.info(f'Saved features at itr {itr} to {save_path}')
        return features

    def train(self, model, train_docs, train_features, dev_docs, dev_features,
              itr=0, use_amp=True, eval_save_threshold=None,
              asin_collation=False):
        """ Train a single iteration. """
        conf = self.config
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']
        logger.info(f'AMP: {"enabled" if use_amp else "disabled"}')

        model.to(self.device)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], f'{self.config_name}_{self.name_suffix}_itr{itr}')
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        if asin_collation:
            # Estimate total_update_steps
            train_batches = iterate_asins(train_features, max_batch_size=conf['batch_size'], shuffle=True)
            total_update_steps = len(train_batches) * epochs // grad_accum  # Approximated
        else:
            train_dataloader = DataLoader(train_features, sampler=RandomSampler(train_features),
                                          batch_size=conf['batch_size'], collate_fn=self.collator)
            total_update_steps = len(train_dataloader) * epochs // grad_accum
        eval_after_step = int(total_update_steps * conf['start_eval_after_ratio'])

        # Set up optimizer and scheduler
        optimizer = self.get_optimizer(model, bert_lr=conf['bert_learning_rate'], task_lr=conf['task_learning_rate'],
                                       bert_wd=conf['bert_wd'], task_wd=conf['task_wd'], eps=conf['adam_eps'])
        scheduler = self.get_scheduler(optimizer, total_update_steps, conf['warmup_ratio'])

        # Get model parameters for grad clipping
        clipping_param = [p for p in model.parameters() if p.requires_grad]

        # Get scaler for automatic mixed precision
        scaler = amp.GradScaler(enabled=use_amp)

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num features: %d' % len(train_features))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Batch size: %d' % conf['batch_size'])
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss for each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        eval_scores = []
        start_time = time.time()
        model.zero_grad()

        for epo in range(epochs):
            collation_source = iterate_asins(train_features, max_batch_size=conf['batch_size'], shuffle=True)\
                if asin_collation else train_dataloader
            for batch_i, batch in enumerate(collation_source):
                if asin_collation:
                    batch = self.collator(batch)

                # Forward
                model.train()

                with amp.autocast(enabled=use_amp):
                    loss = model(**batch)
                    loss /= grad_accum

                # Backward
                scaler.scale(loss).backward()
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf['max_grad_norm']:
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(clipping_param, conf['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()
                    scheduler.step()

                    # Compute effective loss
                    effective_loss = sum(loss_during_accum)
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.4f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', scheduler.get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', scheduler.get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > eval_after_step and len(loss_history) % conf['eval_frequency'] == 0:
                        eval_score, _, _ = self.evaluate(model, 'dev', dev_docs, dev_features,
                                                         tb_writer, len(loss_history))
                        if not eval_scores or eval_score > max(eval_scores) or True:
                            if eval_save_threshold is None or eval_score > eval_save_threshold or True:
                                self.save_model_checkpoint(model, len(loss_history))
                        eval_scores.append(eval_score)
                        logger.info(f'Best eval score: {max(eval_scores):.2f}')
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))
        model.zero_grad()

        # Eval at the end
        eval_score, _, _ = self.evaluate(model, 'dev', dev_docs, dev_features,
                                         tb_writer, len(loss_history))
        if not eval_scores or eval_score > max(eval_scores) or True:
            if eval_save_threshold is None or eval_score > eval_save_threshold or True:
                self.save_model_checkpoint(model, len(loss_history))
        eval_scores.append(eval_score)
        logger.info(f'All eval scores: {eval_scores}')

        # Wrap up
        tb_writer.close()
        return loss_history, eval_scores

    def evaluate(self, model, partition, docs, features, tb_writer=None, step=0, do_eval=True):
        return self.evaluate_sim(model, partition, docs, features, tb_writer, step, do_eval)

    def evaluate_sim(self, model, partition, docs, features, tb_writer=None, step=0, do_eval=True):
        """ Evaluate similarity of seeds.
        """
        conf = self.config
        features = random.sample(features, k=min(len(features), 5000))
        logger.info(f'Evaluating similarity on {len(features)} features...')
        eval_dataloader = DataLoader(features, sampler=SequentialSampler(features),
                                     batch_size=conf['eval_batch_size'], collate_fn=self.collator)
        model.to(self.device)
        model.eval()
        feat_i, seed_hidden, seed_clusters = 0, [], []
        for batch_i, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                seq_hidden = model.get_seq_hidden(**batch, layer=-1)
                for row_i in range(seq_hidden.size()[0]):
                    feat, hidden = features[feat_i], seq_hidden[row_i]  # [seq_len, hidden]
                    existing_starts, existing_ends, clusters = selection_util.get_selected_spans(feat, prop_threshold=self.prop_expanded_seed)
                    seed_hidden += Model.get_span_hidden(hidden, existing_starts, existing_ends)
                    seed_clusters += clusters
                    feat_i += 1
        assert feat_i == len(features)
        seed_hidden = torch.stack(seed_hidden, dim=0)  # [num_existing, hidden]
        logger.info(f'Evaluating on {seed_hidden.size()[0]} seeds')

        cluster2spans = defaultdict(list)
        for span_i, span_cluster in enumerate(seed_clusters):
            cluster2spans[span_cluster].append(seed_hidden[span_i])
        cluster2spans = {cluster: torch.stack(spans, dim=0) for cluster, spans in cluster2spans.items()}

        results = cluster2spans
        if not do_eval:
            return results

        with torch.no_grad():
            cluster2size = {cluster: spans.size()[0] for cluster, spans in cluster2spans.items()}
            cluster2sim = {cluster: torch.matmul(spans, spans.t()).mean().item() for cluster, spans in cluster2spans.items()}
            within_sim_weighted = sum((sim * cluster2size[cluster] / seed_hidden.size()[0])
                                      for cluster, sim in cluster2sim.items())
            global_sim = torch.matmul(seed_hidden, seed_hidden.t()).mean().item()

        eval_score = within_sim_weighted + 1 - global_sim
        metrics = {f'{partition}_within_sim': within_sim_weighted,
                   f'{partition}_global_sim': global_sim}

        self.log_metrics(metrics, tb_writer, step)
        return eval_score, metrics, results

    def evaluate_test_clustered(self, exp_suffix, itr, features=None, do_eval=True):
        """ Evaluate clustered on test set after certain training iteration. """
        docs, _ = self.data.get_data(self.dataset_name, 'all')
        id2doc = {doc['id']: doc for doc in docs if doc['is_test']}

        if features is None:  # If features not provided, load from saved features
            saved_feat_path = self.get_itr_feat_path('all', itr=itr, exp_suffix=exp_suffix)
            features = io_util.read_pickle(saved_feat_path)
            if features is None:
                logger.info(f'Features do not exist for {exp_suffix} at itr{itr}')
                return None
            logger.info(f'Evaluating test clustering: using saved itr features from {saved_feat_path}')
        else:
            logger.info(f'Evaluating test clustering: using provided features')
        id2feat = {feat['id']: feat for feat in features if feat['is_test']}
        pts = {feat['pt'] for feat in id2feat.values()}

        # Load gold attrs
        labels, test_labels_exist = [], set()
        for pt in pts:
            pt_label_path = join(self.config['dataset_dir'], f'{pt.lower()}_labels.jsonl')
            if exists(pt_label_path):
                logger.info(f'Found test labels for {pt}; use for evaluation')
                test_labels_exist.add(pt)
                pt_labels = io_util.read_jsonlines(pt_label_path)
                labels += pt_labels
            else:
                logger.info(f'Test labels do NOT exist for {pt}; evaluation will be dummy')
        id2label = {label['id']: label for label in labels}
        cluster2attr = self.data.get_meta(self.dataset_name)['cluster2attr']

        # Obtain results
        logger.info(f'Evaluating selection on {len(id2feat)} features...')
        id2result = {}
        for test_id, feat in id2feat.items():
            # Skip id that is not in test labels, unless test labels do not exist for this PT
            doc = id2doc[test_id]
            label = id2label.get(test_id, None)
            if not label and feat['pt'] in test_labels_exist:
                continue
            elif not label:
                continue
            else:
                gold_starts = [e['offset'][0] for e in label['entities']]
                gold_ends = [e['offset'][1] for e in label['entities']]
                gold_clusters = [e['label'] for e in label['entities']]

            # Get selected spans (excluding seeds)
            is_clustered = [(p < self.prop_expanded_seed) for p in feat['selected_properties']]
            selected_starts = [el for el, keep in zip(feat['selected_span_starts'], is_clustered) if keep]
            selected_ends = [el for el, keep in zip(feat['selected_span_ends'], is_clustered) if keep]
            selected_clusters = [cluster2attr.get(el, f'CLUSTER_{el}') for el, keep in zip(feat['selected_clusters'], is_clustered) if keep]
            selected_properties = [el for el, keep in zip(feat['selected_properties'], is_clustered) if keep]
            ngram_starts = feat['ngram_span_starts']
            ngram_ends = feat['ngram_span_ends']
            ngram_counts = [feat['ngram_counts'].get((s, e), 100000) for s, e in zip(ngram_starts, ngram_ends)]

            # Convert subtok idx to char idx
            i2charstart = {i: int(c_s) for c_s, i in doc['charstart2i'].items()}
            i2charend = {i: int(c_e) for c_e, i in doc['charend2i'].items()}
            text_prefix = f'{pt2prefix[doc["pt"]].lower()} {self.data.tokenizer.sep_token} '  # Consistent with data_util
            assert doc['text'].startswith(text_prefix), doc['text']
            selected_starts = [(i2charstart[s - 1] - len(text_prefix)) for s in selected_starts if (s-1) in i2charstart]  # Compensate CLS; s/e may be invalid if from tagging
            selected_ends = [(i2charend[e - 1] - len(text_prefix)) for e in selected_ends if (e-1) in i2charend]  # Exclusive
            ngram_starts = [(i2charstart[s - 1] - len(text_prefix)) for s in ngram_starts]
            ngram_ends = [(i2charend[e - 1] - len(text_prefix)) for e in ngram_ends]

            id2result[test_id] = {'id': test_id,
                                  'text': doc['text'][len(text_prefix):],
                                  'pt': doc['pt'],
                                  'selected_starts': selected_starts,
                                  'selected_ends': selected_ends,
                                  'selected_clusters': selected_clusters,
                                  'selected_properties': selected_properties,
                                  'ngram_starts': ngram_starts,
                                  'ngram_ends': ngram_ends,
                                  'ngram_counts': ngram_counts,
                                  'gold_starts': gold_starts,
                                  'gold_ends': gold_ends,
                                  'gold_clusters': gold_clusters}
        logger.info(f'Identified {len(id2result)} test features')
        results = id2result
        if not do_eval:
            return results

        # Evaluate per PT
        pt2score, pt2metrics = {}, {}
        for pt in pts:
            pt_id2result = {id_: result for id_, result in id2result.items() if result['pt'] == pt}
            eval_score_me, metrics_me = self.get_me_metrics('test', pt_id2result)
            eval_score_e2e, metrics_e2e = self.get_e2e_metrics('test', pt_id2result)
            pt2score[pt] = eval_score_e2e
            pt2metrics[pt] = {**metrics_me, **metrics_e2e}
        # Calculate macro across PTs
        all_metrics, macro_metrics, macro_score = {}, {}, 0
        for pt in pts:
            all_metrics.update({f'{pt}_{name}': score for name, score in pt2metrics[pt].items()})
            for name, score in pt2metrics[pt].items():
                name = f'macro_{name}'
                macro_metrics[name] = macro_metrics.get(name, 0) + (score / len(pts))
            macro_score += pt2score[pt] / len(pts)
        all_metrics.update(macro_metrics)

        self.log_metrics(all_metrics)
        logger.info(f'macro_eval_score: {macro_score}')
        return macro_score, all_metrics, results

    @classmethod
    def get_me_metrics(cls, partition, id2result):
        evaluator = PartialMeEvaluator()
        metrics_exact, metrics_partial = evaluator.evaluate(id2result)
        metrics = {f'{partition}_exact_me_{name}': score for name, score in metrics_exact.items()}
        metrics.update({f'{partition}_partial_me_{name}': score for name, score in metrics_partial.items()})
        eval_score = metrics_partial['f']
        return eval_score, metrics

    @classmethod
    def get_e2e_metrics(cls, partition, id2result):
        evaluator = ClusterEvaluator()
        metrics_exact, metrics_partial = evaluator.evaluate(id2result)
        metrics = {f'{partition}_exact_e2e_{name}': score for name, score in metrics_exact.items()}
        metrics.update({f'{partition}_partial_e2e_{name}': score for name, score in metrics_partial.items()})
        eval_score = metrics_partial['meta']
        return eval_score, metrics


if __name__ == '__main__':
    config_name, suffix, gpu_id = sys.argv[1], None, None
    if len(sys.argv) == 3:
        gpu_id = int(sys.argv[2])
    else:
        suffix = sys.argv[2]
        gpu_id = int(sys.argv[3])

    runner = Runner(config_name, gpu_id)

    if suffix:
        itr = 1
        model = runner.initialize_model(init_suffix=None if 'no_rpl' in config_name else suffix)
        runner.expand_clusters(model, itr=itr, exp_suffix=suffix)
        runner.evaluate_test_clustered(suffix, itr)
    else:
        model = runner.initialize_model()
        runner.start(model)
