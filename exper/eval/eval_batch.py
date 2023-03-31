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
from utils import q_error, relative_error, seed_everything, OUTPUT_ROOT, get_logger, batch_relative_error, TimeTracker, log_metric
import numpy as np

SEED = 3407
DATASET_NAME = 'BJAQ'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'small'
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 300
BATCH_SIZE = 10
N_SAMPLE_POINT = 16000 * 20
MAX_ITERATION = 4
NUM_PREDICATES_RANGE = (1, 3)

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def batch_eval():
    global BATCH_SIZE
    logger = get_logger(OUT_DIR, 'eval-batch.log')
    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
    query_engine = QueryEngine(
        model,
        n_sample_points=N_SAMPLE_POINT,
        max_iteration=MAX_ITERATION,
        integrator=INTEGRATOR,
        deqan_type=DEQUAN_TYPE, 
        dataset_name=DATASET_NAME,
        out_path=OUT_DIR,
        device=DEVICE
    )
    logger.info(f"full range integrator is {query_engine.full_domain_integrate()}")
    logger.info(f"Integrator:{INTEGRATOR} N_sample_points:{N_SAMPLE_POINT}")
    metics = []
    latencies = []
    sta = 0
    while sta < N_QUERIES:
        if sta + BATCH_SIZE > N_QUERIES:
            # sta = 0
            BATCH_SIZE = N_QUERIES - sta
        sta += BATCH_SIZE
        query = [table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE) for i in range(BATCH_SIZE)]
        batch_pred = query_engine.query(query)
        batch_real = torch.FloatTensor([table_wapper.query(q) for q in query]).cpu()
        batch_err = batch_relative_error(batch_pred, batch_real)
        metics.append(batch_err)
        ms = query_engine.last_qeury_time * 1000 / N_QUERIES
        latencies.append(ms)
    metics = np.concatenate(metics, axis=0)
    metics = pd.DataFrame(metics, columns=['rsel', 'rcnt', 'rave', 'rsum', 'rvar', 'rstd'])
    # metics.to_csv(os.path.join(OUT_DIR, 'eval.csv'))
    ms = sum(latencies) / len(latencies)
    logger.info(f"Ave Query Lantency:{ms:.3f} ms")
    logger.info(f"[mean]\n{log_metric(metics.mean())}")
    logger.info(f"[.5  ]\n{log_metric(metics.quantile(0.5))}")
    logger.info(f"[.95 ]\n{log_metric(metics.quantile(0.95))}")
    logger.info(f"[.99 ]\n{log_metric(metics.quantile(0.99))}")
    logger.info(f"[max ]\n{log_metric(metics.max())}]")



if __name__ == '__main__':
    batch_eval()
