import os
import sys

sys.path.append('/home/clb/AQP')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import torch
import logging
from query_engine import QueryEngine
from utils  import DataWrapper
from utils import q_error, relative_error, seed_everything, OUTPUT_ROOT

SEED = 3407
DATASET_NAME = 'lineitem'
DEQUAN_TYPE = 'uniform'
MODEL_SIZE = 'small'
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'

OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
INTEGRATOR = 'Vegas'
N_QUERIES = 100
N_SAMPLE_POINT = 100000
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def eval():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    fh, ch = logging.FileHandler(os.path.join(OUT_DIR, f'eval.log'), 'w', encoding='utf-8'), logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    model = torch.load(OUT_DIR + '/best.pt', map_location=DEVICE)
    data_wapper = DataWrapper(DATASET_NAME, OUT_DIR)
    aqp_engine = QueryEngine(
        model,
        n_sample_points=N_SAMPLE_POINT,
        integrator=INTEGRATOR,
        dataset_name=DATASET_NAME,
        out_path=OUT_DIR,
        device=DEVICE
    )
    logger.info(f"full range integrator is {aqp_engine.full_domain_integrate()}")
    queries = data_wapper.generateNQuery(N_QUERIES)
    metics = []
    for idx, query in enumerate(queries):
        cnt_real, ave_real, sum_real, var_real, std_real = data_wapper.query(query)
        sel_real = cnt_real / data_wapper.n

        cnt_pred, ave_pred, sum_pred, var_pred, std_pred = aqp_engine.query(query)

        ms = aqp_engine.last_qeury_time * 1000

        q_cnt, q_ave, q_sum, q_var, q_std = q_error(cnt_pred, cnt_real), q_error(ave_pred, ave_real), \
            q_error(sum_pred, sum_real), q_error(var_pred, var_real), \
            q_error(std_pred, std_real)

        r_cnt, r_ave, r_sum, r_var, r_std = relative_error(cnt_pred, cnt_real), relative_error(ave_pred, ave_real), \
            relative_error(sum_pred, sum_real), relative_error(var_pred, var_real), \
            relative_error(std_pred, std_real)

        logger.info(f'\nquery {idx}:{query} selectivity:{100 * sel_real:.3f}% latency:{ms:.3f} ms')
        logger.info("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_real, ave_real, sum_real, var_real, std_real))
        logger.info("pred:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_pred, ave_pred, sum_pred, var_pred, std_pred))
        logger.info("q_err:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(q_cnt, q_ave, q_sum, q_var, q_std))
        logger.info("r_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}%".
                    format(r_cnt, r_ave, r_sum, r_var, r_std))
        metics.append([ms, q_cnt, q_ave, q_sum, q_var, q_std,
                       r_cnt, r_ave, r_sum, r_var, r_std])
    metics = pd.DataFrame(metics, columns=['ms', 'qcnt', 'qave', 'qsum', 'qvar',
                                           'qstd', 'rcnt', 'rave', 'rsum', 'rvar', 'rstd'])
    metics.to_csv(os.path.join(OUT_DIR, 'eval.csv'))

    logger.info("mean\n" + str(metics.mean()) + '\n')
    logger.info(".5\n" + str(metics.quantile(0.5)) + '\n')


#     logger.info(".95", metics.quantile(0.95), '\n')
#     logger.info(".99", metics.quantile(0.99), '\n')
#     logger.info("max", metics.max(), '\n')


if __name__ == '__main__':
    eval()
