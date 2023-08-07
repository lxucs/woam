import logging
import util
import io_util
from functools import cached_property, lru_cache
from os.path import join, exists
import os
from data_util import (
    get_all_docs,
    convert_docs_to_features,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config):
        self.config = config

    @cached_property
    def tokenizer(self):
        return util.get_transformer_tokenizer(self.config)

    @classmethod
    def is_training(cls, partition):
        return 'train' in partition or 'all' in partition

    @lru_cache()
    def get_meta(self, dataset_name):
        meta_path = join(self.get_data_dir(dataset_name), 'meta.json')
        meta = io_util.read_json(meta_path)
        assert meta['attr2cluster']['brand'] == 0, 'Brand should have cluster id 0'
        meta['cluster2attr'] = {cluster: attr for attr, cluster in meta['attr2cluster'].items()}
        meta['special_attri'], meta['seed_attri'], meta['new_attri'] = set(meta['special_attri']), set(meta['seed_attri']), set(meta['new_attri'])
        return meta

    def get_data(self, dataset_name, partition):
        doc_path = self.get_data_doc_path(dataset_name, partition)
        feat_path = self.get_data_feature_path(dataset_name, partition)
        conf, is_training = self.config, self.is_training(partition)

        # Get docs
        if exists(doc_path):
            docs = io_util.read_jsonlines(doc_path)
            logger.info(f'Loaded docs from {doc_path}')
        else:
            logger.info(f'Getting docs for {dataset_name}-{partition}...')
            raw_path = self.get_data_raw_path(dataset_name, partition)
            docs = get_all_docs(dataset_name, raw_path, self.get_meta(dataset_name),
                                self.tokenizer, only_title=conf['only_title'], is_training=is_training)
            # Save
            io_util.write_jsonlines(doc_path, docs)
            logger.info(f'Saved docs to {doc_path}')

        # Get features
        if exists(feat_path):
            features = io_util.read_pickle(feat_path)
            logger.info(f'Loaded features from {feat_path}')
        else:
            logger.info(f'Getting features for {dataset_name}-{partition}...')
            features = convert_docs_to_features(dataset_name, docs, self.tokenizer, max_seq_len=conf['max_seq_len'],
                                                is_training=is_training, show_example=True)
            # Save
            io_util.write_pickle(feat_path, features)
            logger.info(f'Saved features to {feat_path}')

        return docs, features

    def get_data_dir(self, dataset_name):
        return join(self.config['dataset_dir'], dataset_name)

    def get_data_raw_path(self, dataset_name, partition):
        file_path = join(self.get_data_dir(dataset_name), f'{partition}.jsonlines')
        return file_path

    def get_data_doc_path(self, dataset_name, partition):
        save_dir = join(self.config['data_dir'], 'processed')
        os.makedirs(save_dir, exist_ok=True)

        t = self.config['model_type']
        title_bp = 'title' if self.config['only_title'] else 'title+bp'
        save_path = join(save_dir, f'doc_{dataset_name}_{partition}_{title_bp}_{t}.jsonlines')
        return save_path

    def get_data_feature_path(self, dataset_name, partition):
        save_dir = join(self.config['data_dir'], 'processed')
        os.makedirs(save_dir, exist_ok=True)

        t = self.config['model_type']
        title_bp = 'title' if self.config['only_title'] else 'title+bp'
        msl = self.config['max_seq_len']
        save_path = join(save_dir, f'feat_{dataset_name}_{partition}_{title_bp}_{t}_max{msl}.bin')
        return save_path

    def get_results_path(self, dataset_name, partition, suffix, ext='json'):
        save_dir = join(self.config['log_dir'], 'results')
        os.makedirs(save_dir, exist_ok=True)
        return join(save_dir, f'results_{dataset_name}_{partition}_{suffix}.{ext}')
