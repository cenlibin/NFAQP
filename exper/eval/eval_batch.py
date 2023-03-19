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
from utils import q_error, relative_error, seed_everything, OUTPUT_ROOT, get_logger

SEED = 3407
DATASET_NAME = 'orders'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'small'
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 100
N_SAMPLE_POINT = 16000
MAX_ITERATION = 1
NUM_PREDICATES_RANGE = (1, 1)

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def eval():
    logger = get_logger(OUT_DIR, 'eval.log')
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
    metics = []
    
    query = [table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE) for i in range(N_QUERIES)]
    batch_pred = query_engine.query(query)
    batch_real = [table_wapper.query(q) for q in query]
    

    ms = query_engine.last_qeury_time * 1000 / N_QUERIES
    logger.info(f"ave query lantency:{ms:.3f}")

    # metics = pd.DataFrame(metics, columns=['ms', 'rcnt', 'rave', 'rsum', 'rvar', 'rstd'])
    # metics.to_csv(os.path.join(OUT_DIR, 'eval.csv'))

    # logger.info("mean\n" + str(metics.mean()) + '\n')
    # logger.info(".5\n" + str(metics.quantile(0.5)) + '\n')


#     logger.info(".95", metics.quantile(0.95), '\n')
#     logger.info(".99", metics.quantile(0.99), '\n')
#     logger.info("max", metics.max(), '\n')


if __name__ == '__main__':
    eval()
