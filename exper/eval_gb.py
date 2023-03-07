import os
import sys
sys.path.append('/home/clb/AQP')
from utils import TimeTracker
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import torch
import logging
from query_engine import QueryEngine
from table_wapper import TableWrapper
from utils import q_error, relative_error, seed_everything, OUTPUT_ROOT, groupby_relative_error, get_logger
import numpy as np

SEED = 8889
DATASET_NAME = 'order'
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'middle'

MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 100
N_SAMPLE_POINT = 16000
GROUPBY_BATCH_SIZE = 600
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def eval():
    logger = get_logger(OUT_DIR, 'eval-gb.log')
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
        
        query = table_wapper.generate_query(gb=True)
        logger.info(table_wapper.get_qry_sql(query)[0])
        T = TimeTracker()
        index, reals = table_wapper.query(query)
        t0 = T.report_interval_time_ms('real query')
        with torch.no_grad():
            index, preds = query_engine.gb_query(query, batch_size=GROUPBY_BATCH_SIZE)
            t1 = T.report_interval_time_ms("batch query")
            series_index, series_preds = query_engine.gb_serial(query)
            t2 = T.report_interval_time_ms("serial query")
        
        logger.info(f'real group by tooks {t0:.3f} ms, {t0 / len(index):.3f} ms per query')
        logger.info(f'batch group by tooks {t1:.3f} ms, {t1 / len(index):.3f} ms per query')
        logger.info(f'series group by tooks {t2:.3f} ms, {t2 / len(index):.3f} ms per query')
        logger.info(f'aqp speed up {t0 / t1:.3f}x  |   batch speed up {t2 / t1 :.3f}x ')
        rerr = groupby_relative_error(preds, reals)

        for gb_on, s_pred, pred, real in zip(index, series_preds, preds, reals):
            s = f"{gb_on}: "
            for agg, sp, p, r in zip(["sel", "count", "ave", "sum", "std", "var"], s_pred, pred, real):
                s += f'\n|{agg}: {sp:.3f}/{p:.3f}/{r:.3f}({relative_error(sp, r):.3f}%) ({relative_error(p, r):.3f}%)| '
            logger.info(s)

        break 

        pass


if __name__ == '__main__':
    eval()
