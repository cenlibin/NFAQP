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
from baselines import VerdictEngine, VAEEngine


METRIC = sMAPE      #
SEED = 3407
DATASET_NAME = 'pm25'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'tiny'
REMAKE = False
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
    logger = get_logger(OUT_DIR, 'eval.log')
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
    logger.info(f"full range integrator is {query_engine.full_domain_integrate()}")

    verdict_engine = VerdictEngine(DATASET_NAME, N, remake=False)
    vae_engine = VAEEngine(DATASET_NAME, N, remake=False)

    metics = []
    for idx in range(N_QUERIES):
        query = table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE)
        sel, real = table_wapper.query(query)
        logger.info(f'\nquery {idx}:{query} selectivity:{100 * sel:.3f}%')
        T = TimeTracker()
        flow_pred = query_engine.query(query)
        t1 = T.report_interval_time_ms(f"flow")
        verdict_pred = verdict_engine.query(query)
        t2 = T.report_interval_time_ms(f"verdict")
        vae_pred = vae_engine.query(query)
        t3 = T.report_interval_time_ms(f"vae")
        

        fr_cnt, fr_ave, fr_sum, fr_var, fr_std = get_err(flow_pred, real, METRIC)
        fr_mean = mean_err(get_err(flow_pred, real, METRIC))
        pr_cnt, pr_ave, pr_sum, pr_var, pr_std = get_err(verdict_pred, real, METRIC)
        pr_mean = mean_err(get_err(verdict_pred, real, METRIC))
        vr_cnt, vr_ave, vr_sum, vr_var, vr_std = get_err(vae_pred, real, METRIC)
        vr_mean = mean_err(get_err(vae_pred, real, METRIC))

        cnt_real, ave_real, sum_real, var_real, std_real = real
        cnt_flow, ave_flow, sum_flow, var_flow, std_flow = flow_pred
        cnt_ver, ave_ver, sum_ver, var_ver, std_ver = verdict_pred
        cnt_vae, ave_vae, sum_vae, var_vae, std_vae = vae_pred


        
        logger.info("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_real, ave_real, sum_real, var_real, std_real))
        logger.info("flow:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_flow, ave_flow, sum_flow, var_flow, std_flow))
        logger.info("verdictdb:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_ver, ave_ver, sum_ver, var_ver, std_ver))
        logger.info("vae:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_vae, ave_vae, sum_vae, var_vae, std_vae))
        
        logger.info("fr_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}% mean:{:.3f}%".
                    format(fr_cnt, fr_ave, fr_sum, fr_var, fr_std, fr_mean))
        logger.info("vr_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}% mean:{:.3f}%".
                    format(pr_cnt, pr_ave, pr_sum, pr_var, pr_std, pr_mean))
        logger.info("vae_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f} mean:{:.3f}%%".
                    format(vr_cnt, vr_ave, vr_sum, vr_var, vr_std, vr_mean))

        metics.append([fr_cnt, fr_ave, fr_sum, fr_var, fr_std, fr_mean, t1, 
                       pr_cnt, pr_ave, pr_sum, pr_var, pr_std, pr_mean, t2, 
                       vr_cnt, vr_ave, vr_sum, vr_var, vr_std, vr_mean, t3, ])
        

    metics = pd.DataFrame(metics, columns=['flow_cnt_err', 'flow_avg_err', 'flow_sum_err', 'flow_var_err', 'flow_std_err', 'flow_mean_err','flow_latency',
                     'verdict_cnt_err', 'verdict_avg_err', 'verdict_sum_err', 'verdict_var_err', 'verdict_std_err', 'verdict_mean_err', 'verdict_latency',
                     'vae_cnt_err', 'vae_avg_err', 'vae_sum_err', 'vae_var_err', 'vae_std_err', 'vae_mean_err', 'vae_latency',])
    
    metics.to_csv(os.path.join(OUT_DIR, 'eval.csv'))

    logger.info("mean\n" + str(metics.mean()) + '\n')
    logger.info(".5\n" + str(metics.quantile(0.5)) + '\n')


#     logger.info(".95", metics.quantile(0.95), '\n')
#     logger.info(".99", metics.quantile(0.99), '\n')
#     logger.info("max", metics.max(), '\n')


if __name__ == '__main__':
    eval()
