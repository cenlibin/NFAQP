import os
import sys

sys.path.append('/home/clb/AQP')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import torch
import logging
import numpy as np
import pyverdict
import pymysql
from query_engine import QueryEngine
from table_wapper import TableWrapper
from utils import *
from baselines import VerdictEngine, VAEEngine, DeepdbEngine

METRIC = sMAPE  #
SEED = 3407
DATASET_NAME = 'flights'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'small'
REMAKE = True
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

NUM_PREDICATES_RANGE = (1, 4)
OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 100
N_SAMPLE_POINT = 16000 * 10
MAX_ITERATION = 1
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def eval():
    eval_logger = get_logger(OUT_DIR, 'eval.log')
    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
    N = table_wapper.data.shape[0]
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
    print(f"full range integrator is {query_engine.full_domain_integrate()}")

    deepdb_engine = DeepdbEngine(DATASET_NAME, remake=False, table_N=N)

    metics = []
    for idx in range(N_QUERIES):
        query = table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE)
        sel, real = table_wapper.query(query)
        print(f'\nquery {idx}:{query} selectivity:{100 * sel:.3f}%')
        T = TimeTracker()
        flow_pred = query_engine.query(query)
        t1 = T.report_interval_time_ms(f"flow")
        deepdb_pred = deepdb_engine.query(query)
        t4 = T.report_interval_time_ms(f"deepdb")

        fr_cnt, fr_ave, fr_sum, fr_var, fr_std = get_err(flow_pred, real, METRIC)
        fr_mean = mean_err(get_err(flow_pred, real, METRIC))
        dr_cnt, dr_ave, dr_sum = get_err(deepdb_pred, real[:3], METRIC)
        dr_mean = mean_err(get_err(deepdb_pred, real[:3], METRIC))

        cnt_real, ave_real, sum_real, var_real, std_real = real
        cnt_flow, ave_flow, sum_flow, var_flow, std_flow = flow_pred
        cnt_deepdb, ave_deepdb, sum_deepdb = deepdb_pred

        print("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_real, ave_real, sum_real, var_real, std_real))
        print("flow:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_flow, ave_flow, sum_flow, var_flow, std_flow))
        print("deepdb:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} ".format(cnt_deepdb, ave_deepdb, sum_deepdb))

        print("fr_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}% mean:{:.3f}%".
                    format(fr_cnt, fr_ave, fr_sum, fr_var, fr_std, fr_mean))
        print("dr_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% mean:{:.3f}%".
                    format(dr_cnt, dr_ave, dr_sum, dr_mean))
        metics.append([fr_cnt, fr_ave, fr_sum, fr_var, fr_std, fr_mean, t1, dr_cnt, dr_ave, dr_sum, dr_mean, t4])

    metics = pd.DataFrame(metics, columns=['flow_cnt_err', 'flow_avg_err', 'flow_sum_err', 'flow_var_err', 'flow_std_err', 'flow_mean_err', 'flow_latency',
                                   'deepdb_cnt_err', 'deepdb_avg_err', 'deepdb_sum_err', 'deepdb_mean', 'deepdb_latency'])

    print("mean\n" + str(metics.mean()) + '\n')
    print(".5\n" + str(metics.quantile(0.5)) + '\n')



if __name__ == '__main__':
    eval()
