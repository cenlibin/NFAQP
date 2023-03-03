import os, sys
sys.path.append('/home/clb/AQP')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import torch
from query_engine import QueryEngine
from table_wapper import TableWapper
from utils import MakeFlow, q_error, relative_error, seed_everything
from torchquad import Simpson, set_up_backend
set_up_backend("torch", data_type="float32")
SEED = 1638128
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
DATASET_NAME = 'lineitem-numetric'
MODEL_PATH = "/home/clb/AQP/models/lineitem-numetric-flow/best.pt"


seed_everything(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
def eval():
    
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    aqp_engine = QueryEngine(model, integrator='MonteCarlo', dataset_name=DATASET_NAME, device=DEVICE)
    print(f"full range integrator is {aqp_engine.full_domain_integrate()}")
    data_wapper = TableWrapper(DATASET_NAME)
    queries = data_wapper.generateNQuery(100)
    legal_full_range, actual_full_range = aqp_engine.get_full_range()
    it = Simpson()
    v = it.integrate(
        fn=aqp_engine.pdf,
        dim=aqp_engine.dim,
        N=16000,
        integration_domain=legal_full_range,
        backend='torch'
    )


    metics = []
    for idx, query in enumerate(queries):
        cnt_real, ave_real, sum_real, var_real, std_real = data_wapper.query(query)
        sel_real = cnt_real / data_wapper.n
        cnt_pred, ave_pred, sum_pred, var_pred, std_pred = aqp_engine.query(query)
        q_cnt, q_ave, q_sum, q_var, q_std = q_error(cnt_pred, cnt_real), q_error(ave_pred, ave_real), \
            q_error(sum_pred, sum_real), q_error(var_pred, var_real), \
            q_error(std_pred, std_real)

        r_cnt, r_ave, r_sum, r_var, r_std = relative_error(cnt_pred, cnt_real), relative_error(ave_pred, ave_real), \
            relative_error(sum_pred, sum_real), relative_error(var_pred, var_real), \
            relative_error(std_pred, std_real)

        ms = aqp_engine.last_qeury_time * 1000

        print(f'\nquery {idx}:{query} selectivity:{sel_real:.3f}% lantency:{ms:.3f} ms')
        print("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
              format(cnt_real, ave_real, sum_real, var_real, std_real))
        print("pred:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
              format(cnt_pred, ave_pred, sum_pred, var_pred, std_pred))
        print("q_err:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
              format(q_cnt, q_ave, q_sum, q_var, q_std))
        print("r_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}%".
              format(r_cnt, r_ave, r_sum, r_var, r_std))
        metics.append([ms, q_cnt, q_ave, q_sum, q_var, q_std,
                      r_cnt, r_ave, r_sum, r_var, r_std])
    metics = pd.DataFrame(metics, columns=['ms', 'qcnt', 'qave', 'qsum', 'qvar', 
    'qstd', 'rcnt', 'rave', 'rsum', 'rvar', 'rstd'])

    print("mean", metics.mean(), '\n')
    print(".5", metics.quantile(0.5), '\n')
#     print(".95", metics.quantile(0.95), '\n')
#     print(".99", metics.quantile(0.99), '\n')
#     print("max", metics.max(), '\n')


if __name__ == '__main__':
    eval()
