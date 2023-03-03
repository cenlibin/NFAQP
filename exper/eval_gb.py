import os
import sys

sys.path.append('/home/clb/AQP')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import torch
import logging
from query_engine import QueryEngine
from table_wapper import TableWrapper
from utils import q_error, relative_error, seed_everything, OUTPUT_ROOT

SEED = 376899
DATASET_NAME = 'order'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'middle'

MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 100
N_SAMPLE_POINT = 100000
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def eval():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    fh, ch = logging.FileHandler(os.path.join(OUT_DIR, f'eval-gb.log'), 'w', encoding='utf-8'), logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
    query_engine = QueryEngine(
        model,
        n_sample_points=N_SAMPLE_POINT,
        integrator=INTEGRATOR,
        deqan_type=DEQUAN_TYPE, 
        dataset_name=DATASET_NAME,
        out_path=OUT_DIR,
        device=DEVICE
    )
    logger.info(f"full range integrator is {query_engine.full_domain_integrate()}")
    for i in range(N_QUERIES):
        query = table_wapper.generate_AQP_query(gb=True)
        # gb_real = table_wapper.query(query)
        gb_pred = query_engine.query(query)
        # print(gb_real)

        # cnt_real, ave_real, sum_real, var_real, std_real = data_wapper.query(query)
        # sel_real = cnt_real / data_wapper.n

        # cnt_pred, ave_pred, sum_pred, var_pred, std_pred = aqp_engine.query(query)

        # ms = aqp_engine.last_qeury_time * 1000

        pass


if __name__ == '__main__':
    eval()
