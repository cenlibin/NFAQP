import os, sys
from os.path import abspath, dirname
from nflows import distributions, flows
import torch
import logging
import json
from time import time
import numpy as np
import pandas as pd
import random
from .transform import create_transform

PROJECT_DIR = '/home/clb/AQP'
DATA_PATH = PROJECT_DIR + '/data'
METADATA_DIR = PROJECT_DIR + '/meta_data.json'
OUTPUT_ROOT = PROJECT_DIR + '/output'


def make_flow(config):
    transform = create_transform(config)
    distribution = distributions.StandardNormal((config['num_features'],))
    return flows.Flow(transform, distribution)


def load_table(dataset_name, data_dir=None):
    data_path = os.path.join(DATA_PATH if data_dir is None else data_dir, '{}.csv'.format(dataset_name))
    heads = {
        'lineitem': ['l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_shipstruct',
                     'l_shipmode'],
        'movie_companies': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
        'catalog_sales': ['cs_warehouse_sk', 'cs_item_sk', 'cs_order_number', 'cs_quantity', 'cs_sales_price'],
        'power': ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1',
                  'Sub_metering_2', 'Sub_metering_3'],
        'flights': ["DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEPARTURE_TIME", "DISTANCE", "ARRIVAL_TIME"]
    }
    cate = {
        'lineitem': ['l_returnflag', 'l_shipstruct', 'l_shipmode'],
        'movie_companies': ['company_type_id'],
        'catalog_sales': ['cs_warehouse_sk', 'cs_item_sk'],
        'flights': ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", ],
        'power': [],
    }

    try:
        head = heads[dataset_name]
    except KeyError:
        head = None
    sep = '\t' if dataset_name in ['order', 'movie_companies'] else ','
    data = pd.read_csv(data_path, names=head, sep=sep)
    data = data.dropna().reset_index(drop=True)
    # threshold = 1000000
    # if data.shape[0] > threshold:
    #     data = data.sample(threshold).reset_index(drop=True)

    if dataset_name in cate:
        for col in cate[dataset_name]:
            data[col].astype('str')

    for col in data.columns:
        if data.dtypes[col] == 'object':
            data[col].fillna('NULL')
        else:
            data[col].fillna(0)
    return data


def clear_json():
    if os.path.exists(METADATA_DIR):
        os.remove(METADATA_DIR)


def read_from_json(key):
    if not os.path.exists(METADATA_DIR):
        return None
    with open(METADATA_DIR, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
        try:
            return json_dict[key]
        except KeyError:
            return None


def write_to_json(key, value):
    if os.path.exists(METADATA_DIR):
        with open(METADATA_DIR, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
    else:
        json_dict = {}

    json_dict[key] = value
    json_str = json.dumps(json_dict, ensure_ascii=False, indent=4)
    with open(METADATA_DIR, 'w', encoding='utf-8') as f:
        f.write(json_str)


def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


def test_forward_latency(model, input_dim, n_queries=1000, device="cuda"):
    queries = torch.rand(size=[n_queries, 1, input_dim])
    st = time()
    with torch.no_grad():
        for x in queries:
            _ = model.log_prob(x.to(device))
    et = time()
    return (et - st) / n_queries * 1000


def get_model_size_mb(model):
    size_mb = 0
    for p in model.parameters():
        ps = 1
        for s in p.shape:
            ps *= s
        size_mb += ps
    return size_mb * 4 / (1024 * 1024)


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        except TypeError:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch.record_stream(torch.cuda.current_stream())
        self.preload()
        return batch


def make_uniform_sample(dataset_name, sample_rate=0.01):
    full_table = load_table(dataset_name, '/home/clb/AQP/data/backup')
    if dataset_name == 'lineitem':
        full_table.columns = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity',
                              'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus',
                              'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode']
    output_path = DATA_PATH + f'/{dataset_name}.csv'
    print(f'saving to {output_path}')
    n = full_table.shape[0]
    idx = np.random.randint(0, n, size=int(sample_rate * n))
    sampled_table = full_table.iloc[idx]
    sep = '\t' if dataset_name in ['lineitem', 'order'] else ','
    sampled_table.to_csv(output_path, index=False, sep=sep)


def get_logger(out_dir, file_name):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    fh, ch = logging.FileHandler(os.path.join(out_dir, file_name), 'w', encoding='utf-8'), logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        """ no output """
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        """ output """
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


