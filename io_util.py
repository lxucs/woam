import logging
import pyhocon
from os.path import join
import os
import ujson as json
import pickle

logger = logging.getLogger(__name__)


def parse_configs(config_path='experiments.conf'):
    return pyhocon.ConfigFactory.parse_file(config_path)


def get_config(config_name, create_dir=True, config_path='experiments.conf'):
    logger.info("Experiment: {}".format(config_name))

    config = parse_configs(config_path)[config_name]

    config['log_dir'] = join(config['log_root'], config_name)
    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    if create_dir:
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['tb_dir'], exist_ok=True)

    # logger.info(pyhocon.HOCONConverter.convert(config, 'hocon'))
    return config


def read_jsonlines(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonlines(file_path, instances):
    with open(file_path, 'w') as f:
        for inst in instances:
            f.write(f'{json.dumps(inst)}\n')


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def read_plain(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]
