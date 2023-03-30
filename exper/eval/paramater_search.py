import os
import sys
sys.path.append('/home/clb/AQP')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
N_QUERIES = 100
BATCH_SIZE = 10
N_SAMPLE_POINT = 16000 * 10
MAX_ITERATION = 2
NUM_PREDICATES_RANGE = (1, 3)

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

APLHAS = [0.2, 0.4, 0.6, 0.8]
BETAS = [0.2, 0.4, 0.6, 0.8]
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
queries = [table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE) for i in range(N_QUERIES)]
reals = torch.FloatTensor([table_wapper.query(q) for q in queries]).cpu()
def eval(alpha, beta):
    batch_size = BATCH_SIZE
    logger = get_logger(OUT_DIR, 'eval-grid-search.log')
    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    if os.path.exists(os.path.join(OUT_DIR, "vegas.map")):
        os.remove(os.path.join(OUT_DIR,  "vegas.map"))
    query_engine = QueryEngine(
        model,
        n_sample_points=N_SAMPLE_POINT,
        max_iteration=MAX_ITERATION,
        integrator=INTEGRATOR,
        deqan_type=DEQUAN_TYPE, 
        dataset_name=DATASET_NAME,
        out_path=OUT_DIR,
        device=DEVICE, 
        alpha=alpha,
        beta=beta
    )
    logger.info(f"full range integrator is {query_engine.full_domain_integrate()} alpha:{alpha} beta:{beta}")
    logger.info(f"Integrator:{INTEGRATOR} N_sample_points:{N_SAMPLE_POINT}")
    metics = []
    latencies = []
    sta = 0
    while sta < N_QUERIES:
        if sta + batch_size > N_QUERIES:
            batch_size = N_QUERIES - sta
        query = queries[sta: sta + batch_size]
        batch_real = reals[sta: sta + batch_size]
        sta += batch_size
        batch_pred = query_engine.query(query)
        batch_err = batch_relative_error(batch_pred, batch_real)
        metics.append(batch_err)
        ms = query_engine.last_qeury_time * 1000 / N_QUERIES
        latencies.append(ms)
    metics = np.concatenate(metics, axis=0)
    metics = pd.DataFrame(metics, columns=['rsel', 'rcnt', 'rave', 'rsum', 'rvar', 'rstd'])
    # metics.to_csv(os.path.join(OUT_DIR, 'eval.csv'))
    ms = sum(latencies) / len(latencies)
    logger.info(f"Ave Query Lantency:{ms:.3f} ms")
    logger.info(f"alpha:{alpha} beta:{beta} [mean]\n{log_metric(metics.mean())}")
    return metics.mean()['rcnt']

def grid_search():
    min_err = 1e9
    best_para = (-1, -1)
    for alpha in APLHAS:
        for beta in BETAS:
            err = eval(alpha, beta)
            if err < min_err:
                min_err = err
                best_para = (alpha, beta)
    
    print(f"best para: alpha -> {best_para[0]} beta -> {best_para[1]} for rcnt:{min_err:.3f}%")

    pass

if __name__ == '__main__':
    grid_search()
