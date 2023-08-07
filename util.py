from transformers import (AutoTokenizer,
                          RobertaTokenizer, RobertaTokenizerFast,
                          BartTokenizer, BartTokenizerFast,
                          BertTokenizer, BertTokenizerFast
                          )
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel
import torch
import operator
import random
import logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


def flatten(l):
    """ List. """
    # Equivalent to: list(chain(l))
    return [e for ll in l for e in ll]


def random_select(tensor, num_selection):
    """ Randomly select first dimension. """
    if tensor.size()[0] > num_selection:
        rand_idx = torch.randperm(tensor.size()[0])[:num_selection]
        return tensor[rand_idx]
    else:
        return tensor


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def get_transformer_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def is_bpe_tokenizer(tokenizer):
    bpe_tokenizers = (RobertaTokenizer, RobertaTokenizerFast, BartTokenizer, BartTokenizerFast)
    wordpiece_tokenizers = (BertTokenizer, BertTokenizerFast)
    if isinstance(tokenizer, bpe_tokenizers):
        return True
    elif isinstance(tokenizer, wordpiece_tokenizers):
        return False
    else:
        raise ValueError(f'Unsupported tokenizer: {tokenizer}')


def get_transformers(config):
    if config['model_type'] == 'bert':
        return BertModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'roberta':
        return RobertaModel.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])
