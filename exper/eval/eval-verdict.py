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
from utils import q_error, relative_error, seed_everything, OUTPUT_ROOT, get_logger, TimeTracker, sMAPE, HiddenPrints

host = 'localhost'
user = 'root'
password = '7837'
port = 3306
metric = sMAPE
SEED = 3407
DATASET_NAME = 'pm25'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'tiny'
REMAKE = False
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

NUM_PREDICATES_RANGE = (1, 1)
OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 100
N_SAMPLE_POINT = 16000 * 1
MAX_ITERATION = 1
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

SOURCE_DB = 'AQP'
TARGET_DB = 'verdictdb'
dataset = DATASET_NAME
SAMPLE_RATE = 0.01
table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
connect_string = f'jdbc:mysql://{host}:{port}?user={user}&password={password}&useSSL=False&loglevel=error'
mysql_conn = pymysql.connect(
    host=host,
    port=port,
    user=user,
    passwd=password,
    autocommit=True
).cursor()
verdict_conn = pyverdict.VerdictContext(connect_string)

if REMAKE:
    T  = TimeTracker()
    sql = f"DROP TABLE IF EXISTS {TARGET_DB}.{dataset};"
    mysql_conn.execute(sql)
    T.report_interval_time_sec(sql)

    sql = f'CREATE SCRAMBLE {TARGET_DB}.{dataset} FROM {SOURCE_DB}.{dataset} size {SAMPLE_RATE}'
    verdict_conn.sql(sql)
    T.report_interval_time_sec(sql)


def verdictdb_query(query):
    with HiddenPrints():
        target, gb = query['target'], query['gb']
        target = f'`{target}`'
        where = '' if len(query['where']) == 0 else 'WHERE '
        for col, (op, val) in query['where'].items():
            col = f'`{col}`'
            if where != 'WHERE ':
                where += "AND "
            if op == '=':
                val = f'\'{val}\''
            if op == 'between':
                where += f'{col} BETWEEN {val[0]} AND {val[1]} '
            else:
                where += f'{col} {op} {val} '

        pred = []
        for agg in ['COUNT', 'AVG', 'SUM']:
            sql = f'SELECT {agg}({target}) FROM {TARGET_DB}.{DATASET_NAME} {where}'
            verdict_pred = verdict_conn.sql(sql).to_numpy().item()
            verdict_pred = float(verdict_pred if verdict_pred is not None else 0)
            verdict_pred /= (SAMPLE_RATE if agg in ['COUNT', 'SUM'] else 1.0)
            pred.append(verdict_pred)

        p_std = verdict_conn.sql(f'SELECT STDDEV({target}) FROM {TARGET_DB}.{DATASET_NAME} {where}').to_numpy().item()
        p_std = float(p_std if p_std is not None else 0)
        p_var = p_std ** 2
        pred = pred + [p_var, p_std]
        return pred

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
    for idx in range(N_QUERIES):
        query = table_wapper.generate_query(gb=False, num_predicates_ranges=NUM_PREDICATES_RANGE)
        sel_real, cnt_real, ave_real, sum_real, var_real, std_real = table_wapper.query(query)
        T = TimeTracker()
        sel_pred, cnt_pred, ave_pred, sum_pred, var_pred, std_pred = query_engine.query(query)
        t1 = T.report_interval_time_ms(f"flow {query}")
        cnt_ver, ave_ver, sum_ver, var_ver, std_ver = verdictdb_query(query)
        t2 = T.report_interval_time_ms(f"verdict {query}")
        # cnt_ver, ave_ver, sum_ver, var_ver, std_ver = np.array([cnt_ver]), np.array([ave_ver]), np.array([sum_ver]), np.array([var_ver]), np.array([std_ver])

        fr_cnt, fr_ave, fr_sum, fr_var, fr_std = metric(cnt_pred, cnt_real), metric(ave_pred, ave_real), metric(sum_pred, sum_real), metric(var_pred, var_real), metric(std_pred, std_real)
        pr_cnt, pr_ave, pr_sum, pr_var, pr_std = metric(cnt_ver, cnt_real), metric(ave_ver, ave_real), metric(sum_ver, sum_real), metric(var_ver, var_real), metric(std_ver, std_real)

        logger.info(f'\nquery {idx}:{query} selectivity:{100 * sel_real:.3f}%')
        logger.info("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_real, ave_real, sum_real, var_real, std_real))
        logger.info("fpred:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_pred, ave_pred, sum_pred, var_pred, std_pred))
        logger.info("vpred:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_ver, ave_ver, sum_ver, var_ver, std_ver))
        logger.info("fr_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}%".
                    format(fr_cnt, fr_ave, fr_sum, fr_var, fr_std))
        logger.info("vr_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}%".
                    format(pr_cnt, pr_ave, pr_sum, pr_var, pr_std))

        metics.append([fr_cnt, fr_ave, fr_sum, fr_var, fr_std, t1, pr_cnt, pr_ave, pr_sum, pr_var, pr_std, t2])
        

    metics = pd.DataFrame(metics, columns=['flow_cnt_err', 'flow_avg_err', 'flow_sum_err', 'flow_var_err', 'flow_std_err', 'flow_latency',
                     'verdict_cnt_err', 'verdictavg_err', 'verdict_sum_err', 'verdict_var_err', 'verdictstd_err', 'verdictdb_latency'])
    metics.to_csv(os.path.join(OUT_DIR, 'eval.csv'))

    logger.info("mean\n" + str(metics.mean()) + '\n')
    logger.info(".5\n" + str(metics.quantile(0.5)) + '\n')


#     logger.info(".95", metics.quantile(0.95), '\n')
#     logger.info(".99", metics.quantile(0.99), '\n')
#     logger.info("max", metics.max(), '\n')


if __name__ == '__main__':
    eval()
