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

METRIC = sMAPE               #
SEED = 42332
DATASET_NAME = 'lineitem'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'tiny'
REMAKE_VERDICTDB = False
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'
N_QUERIES = 10
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

def groupby_answer_to_str(res):
    s = ''
    for k in res:
        s += f'{k}:['
        for i in res[k]:
            s += f'{i:.3f}'
            if i != res[k][-1]:
                s += ', '
        s += ']\n'
    return s[:-1]

def eval():
    gap = 200
    # logger = get_logger(OUT_DIR, 'eval.log')
    print(DATASET_NAME)
    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
    N, dim = table_wapper.data.shape
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
        # n_p = idx // gap + 1
        # NUM_PREDICATES_RANGE[0] = n_p
        # NUM_PREDICATES_RANGE[1] = n_p
        query = table_wapper.generate_query(gb=True, num_predicates_ranges=NUM_PREDICATES_RANGE)
        real = table_wapper.query(query)
        print(query)
        print(f'real:\n{groupby_answer_to_str(real)}')
        n_predicates = len(query['where'])
        n_groups = len(real)
        
        T = TimeTracker()
        flow_pred = query_engine.gb_query(query)
        t1 = T.report_interval_time_ms(f"flow")
        
        verdict_pred = verdict_engine.gb_query(query)
        t2 = T.report_interval_time_ms(f"verdict")
        
        vae_pred = vae_engine.gb_query(query)
        t3 = T.report_interval_time_ms(f"vae")
        
        deepdb_pred = deepdb_engine.gb_query(query)
        t4 = T.report_interval_time_ms(f"deepdb")
        
        
        print(f'\nflow:{t1} ms\n{groupby_answer_to_str(flow_pred)}')
        print('err:', groupby_error(flow_pred, real, METRIC))
        print(f'\nverdictdb:{t2} ms\n{groupby_answer_to_str(verdict_pred)}')
        print('err:', groupby_error(verdict_pred, real, METRIC))
        print(f'\nvae:{t3} ms\n{groupby_answer_to_str(vae_pred)}')
        print('err:', groupby_error(vae_pred, real, METRIC))
        print(f'\ndeepdb:{t4} ms\n{groupby_answer_to_str(deepdb_pred)}')
        print('err:', groupby_error(deepdb_pred, real, METRIC)[:-2])

        fr_cnt, fr_ave, fr_sum, fr_var, fr_std = groupby_error(flow_pred, real, METRIC)
        pr_cnt, pr_ave, pr_sum, pr_var, pr_std = groupby_error(verdict_pred, real, METRIC)
        vr_cnt, vr_ave, vr_sum, vr_var, vr_std = groupby_error(vae_pred, real, METRIC)
        dr_cnt, dr_ave, dr_sum, _     ,_       = groupby_error(deepdb_pred, real, METRIC)


        # cnt_real, ave_real, sum_real, var_real, std_real = real
        # cnt_flow, ave_flow, sum_flow, var_flow, std_flow = flow_pred
        # cnt_ver, ave_ver, sum_ver, var_ver, std_ver = verdict_pred
        # cnt_vae, ave_vae, sum_vae, var_vae, std_vae = vae_pred
        # cnt_deepdb, ave_deepdb, sum_deepdb = deepdb_pred

        
        # print("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
        #             format(cnt_real, ave_real, sum_real, var_real, std_real))
        # print("flow:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
        #             format(cnt_flow, ave_flow, sum_flow, var_flow, std_flow))
        # print("verdictdb:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
        #             format(cnt_ver, ave_ver, sum_ver, var_ver, std_ver))
        # print("vae:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
        #             format(cnt_vae, ave_vae, sum_vae, var_vae, std_vae))
        # print("deepdb:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} ".
        #             format(cnt_deepdb, ave_deepdb, sum_deepdb))
        
        print("flow_err:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(fr_cnt, fr_ave, fr_sum, fr_var, fr_std))
        print("verdictdb_err:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(pr_cnt, pr_ave, pr_sum, pr_var, pr_std,))
        print("vae_err:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f}% std:{:.3f} ".
                    format(vr_cnt, vr_ave, vr_sum, vr_var, vr_std,))
        print("deepdb_err:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f}".
                    format(dr_cnt, dr_ave, dr_sum))

        eval_csv.append([n_predicates, n_groups,
                       fr_cnt, fr_ave, fr_sum, fr_var, fr_std, t1, 
                       pr_cnt, pr_ave, pr_sum, pr_var, pr_std, t2, 
                       vr_cnt, vr_ave, vr_sum, vr_var, vr_std, t3, 
                       dr_cnt, dr_ave, dr_sum, t4,])
        

    eval_csv = pd.DataFrame(eval_csv, columns=['n_predicates', 'n_groups',
                    'flow_cnt_err', 'flow_avg_err', 'flow_sum_err', 'flow_var_err', 'flow_std_err', 'flow_latency',
                    'verdict_cnt_err', 'verdict_avg_err', 'verdict_sum_err', 'verdict_var_err', 'verdict_std_err', 'verdict_latency',
                    'vae_cnt_err', 'vae_avg_err', 'vae_sum_err', 'vae_var_err', 'vae_std_err', 'vae_latency',
                    'deepdb_cnt_err', 'deepdb_avg_err', 'deepdb_sum_err', 'deepdb_latency'])
    
    eval_csv.to_csv(os.path.join(OUT_DIR, 'eval-groupby.csv'))
    print("mean\n" + str(eval_csv.mean()) + '\n')
    print(".5\n" + str(eval_csv.quantile(0.5)) + '\n')


    # plot_err(DATASET_NAME, eval_csv)
if __name__ == '__main__':

    eval()
