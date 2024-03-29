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
from utils import q_error, relative_error, sMAPE,seed_everything, OUTPUT_ROOT, get_logger

SEED = 3407
DATASET_NAME = 'lineitemext'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'tiny'
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 100
N_SAMPLE_POINT = 16000 
MAX_ITERATION = 2
NUM_PREDICATES_RANGE = (1, 5)

Alpha = 0.4


metric = relative_error
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)  
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def eval():
    # logger = get_logger(OUT_DIR, 'eval.log')
    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)

    table_wapper.print_columns_info()
    query_engine = QueryEngine(
        model,
        n_sample_points=N_SAMPLE_POINT,
        max_iteration=MAX_ITERATION,
        integrator=INTEGRATOR,
        deqan_type=DEQUAN_TYPE, 
        dataset_name=DATASET_NAME,
        out_path=OUT_DIR,
        device=DEVICE,
        alpha=Alpha
    )
    print(f"Full range integrator is {query_engine.full_domain_integrate()}")
    metics = []
    for idx in range(N_QUERIES):
        query = table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE)
        sel_real, (cnt_real, ave_real, sum_real, var_real, std_real) = table_wapper.query(query)
        cnt_pred, ave_pred, sum_pred, var_pred, std_pred = query_engine.query(query)

        ms = query_engine.last_qeury_time * 1000

        r_cnt, r_ave, r_sum, r_var, r_std = metric(cnt_pred, cnt_real), metric(ave_pred, ave_real), metric(sum_pred, sum_real), metric(var_pred, var_real), metric(std_pred, std_real)

        print(f'\nquery {idx}:{query} selectivity:{100 * sel_real:.3f}% latency:{ms:.3f} ms')
        print("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_real, ave_real, sum_real, var_real, std_real))
        print("pred:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_pred, ave_pred, sum_pred, var_pred, std_pred))
        print("metric:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f}".format(r_cnt, r_ave, r_sum, r_var, r_std))
        metics.append([ms, r_cnt, r_ave, r_sum, r_var, r_std])
    metics = pd.DataFrame(metics, columns=['ms', 'rcnt', 'rave', 'rsum', 'rvar', 'rstd'])
    metics.to_csv(os.path.join(OUT_DIR, 'eval.csv'))

    print("mean\n" + str(metics.mean()) + '\n')
    print(".5\n" + str(metics.quantile(0.5)) + '\n')


#     print(".95", metics.quantile(0.95), '\n')
#     print(".99", metics.quantile(0.99), '\n')
#     print("max", metics.max(), '\n')


if __name__ == '__main__':
    eval()
