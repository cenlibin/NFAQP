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
from plot import plot_err

METRIC = sMAPE              
SEED = 42332
DATASET_NAME = 'lineitemext'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'tiny'
REMAKE_VERDICTDB = False
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'
N_QUERIES = 200
GAP = 50
INCREASET_N_PREDICATES = False
NUM_PREDICATES_RANGE = [1, 5]
OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'

N_SAMPLE_POINT = 16000
MAX_ITERATION = 1
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

def eval():
    # logger = get_logger(OUT_DIR, 'eval.log')
    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
    N, dim = table_wapper.data.shape
    if INCREASET_N_PREDICATES:
        global N_QUERIES
        N_QUERIES = dim * GAP
        NUM_PREDICATES_RANGE[1] = dim
    print("n_predicates_range ", NUM_PREDICATES_RANGE)
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
    verdict_engine = VerdictEngine(DATASET_NAME, N, remake=REMAKE_VERDICTDB)
    vae_engine = VAEEngine(DATASET_NAME, N, remake=False)
    deepdb_engine = DeepdbEngine(DATASET_NAME, N, remake=False)

    print(f"full range integrator is {query_engine.full_domain_integrate()}")

    eval_csv = []
    for idx in range(N_QUERIES):
        if INCREASET_N_PREDICATES:
            n_p = idx // GAP + 1
            NUM_PREDICATES_RANGE[0] = n_p
            NUM_PREDICATES_RANGE[1] = n_p
        query = table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE)
        sel, real = table_wapper.query(query)
        n_predicates = len(query['where'])
        print(f'\nquery {idx}:{query} selectivity:{100 * sel:.3f}% n_predicates:{n_predicates}')

        T = TimeTracker()
        flow_pred = query_engine.query(query)
        t1 = T.report_interval_time_ms(f"flow")
        verdict_pred = verdict_engine.query(query)
        t2 = T.report_interval_time_ms(f"verdict")
        vae_pred = vae_engine.query(query)
        t3 = T.report_interval_time_ms(f"vae")
        deepdb_pred = deepdb_engine.query(query)
        t4 = T.report_interval_time_ms(f"deepdb")
        

        fr_cnt, fr_ave, fr_sum, fr_var, fr_std = get_err(flow_pred, real, METRIC)
        fr_mean = mean_err(get_err(flow_pred, real, METRIC))
        pr_cnt, pr_ave, pr_sum, pr_var, pr_std = get_err(verdict_pred, real, METRIC)
        pr_mean = mean_err(get_err(verdict_pred, real, METRIC))
        vr_cnt, vr_ave, vr_sum, vr_var, vr_std = get_err(vae_pred, real, METRIC)
        vr_mean = mean_err(get_err(vae_pred, real, METRIC))
        dr_cnt, dr_ave, dr_sum = get_err(deepdb_pred, real[:3], METRIC)
        dr_mean = mean_err(get_err(deepdb_pred, real[:3], METRIC))

        cnt_real, ave_real, sum_real, var_real, std_real = real
        cnt_flow, ave_flow, sum_flow, var_flow, std_flow = flow_pred
        cnt_ver, ave_ver, sum_ver, var_ver, std_ver = verdict_pred
        cnt_vae, ave_vae, sum_vae, var_vae, std_vae = vae_pred
        cnt_deepdb, ave_deepdb, sum_deepdb = deepdb_pred

        
        print("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_real, ave_real, sum_real, var_real, std_real))
        print("flow:{:.3f} ms\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(t1, cnt_flow, ave_flow, sum_flow, var_flow, std_flow))
        print("verdictdb:{:.3f} ms\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(t2, cnt_ver, ave_ver, sum_ver, var_ver, std_ver))
        print("vae:{:.3f} ms\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(t3, cnt_vae, ave_vae, sum_vae, var_vae, std_vae))
        print("deepdb:{:.3f} ms\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} ".
                    format(t4, cnt_deepdb, ave_deepdb, sum_deepdb))
        
        print("flow_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}% mean:{:.3f}%".
                    format(fr_cnt, fr_ave, fr_sum, fr_var, fr_std, fr_mean))
        print("verdictdb_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}% mean:{:.3f}%".
                    format(pr_cnt, pr_ave, pr_sum, pr_var, pr_std, pr_mean))
        print("vae_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f} mean:{:.3f}%%".
                    format(vr_cnt, vr_ave, vr_sum, vr_var, vr_std, vr_mean))
        print("deepdb_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% mean:{:.3f}%%".
                    format(dr_cnt, dr_ave, dr_sum, dr_mean))

        eval_csv.append([sel, n_predicates,
                       fr_cnt, fr_ave, fr_sum, fr_var, fr_std, fr_mean, t1, 
                       pr_cnt, pr_ave, pr_sum, pr_var, pr_std, pr_mean, t2, 
                       vr_cnt, vr_ave, vr_sum, vr_var, vr_std, vr_mean, t3, 
                       dr_cnt, dr_ave, dr_sum, dr_mean, t4,])
        

    eval_csv = pd.DataFrame(eval_csv, columns=['selectivity', 'n_predicates',
                    'flow_cnt_err', 'flow_avg_err', 'flow_sum_err', 'flow_var_err', 'flow_std_err', 'flow_mean_err','flow_latency',
                    'verdict_cnt_err', 'verdict_avg_err', 'verdict_sum_err', 'verdict_var_err', 'verdict_std_err', 'verdict_mean_err', 'verdict_latency',
                    'vae_cnt_err', 'vae_avg_err', 'vae_sum_err', 'vae_var_err', 'vae_std_err', 'vae_mean_err', 'vae_latency',
                    'deepdb_cnt_err', 'deepdb_avg_err', 'deepdb_sum_err', 'deepdb_mean_err', 'deepdb_latency'])
    
    eval_csv.to_csv(os.path.join(OUT_DIR, 'eval.csv'))

    print("mean\n" + str(eval_csv.mean()) + '\n')
    print(".5\n" + str(eval_csv.quantile(0.5)) + '\n')

#     print(".95", metics.quantile(0.95), '\n')
#     print(".99", metics.quantile(0.99), '\n')
#     print("max", metics.max(), '\n')

    plot_err(DATASET_NAME, eval_csv)
if __name__ == '__main__':

    eval()
