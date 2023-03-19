import pyverdict
import pymysql
from utils import *
from table_wapper import TableWrapper

SEED = 3489
SOURCE_DB = 'AQP'
TARGET_DB = 'verdictdb'
DATASET_NAME = 'orders'
SAMPLE_RATE = 0.01
DEQUAN_TYPE = 'spline'
MODEL_SIZE = 'small'
N_QUERIES = 100
MODEL_TAG = f'flow-{MODEL_SIZE}'
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'
OUT_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
seed_everything(SEED)
host = 'localhost'
user = 'root'
password = '7837'
port = 3306

table_wapper = TableWrapper(DATASET_NAME, OUT_DIR, DEQUAN_TYPE)
connect_string = f'jdbc:mysql://{host}:{port}?user={user}&password={password}&useSSL=False'
mysql_conn = pymysql.connect(
    host=host,
    port=port,
    user=user,
    passwd=password,
    autocommit=True
).cursor()
verdict_conn = pyverdict.VerdictContext(connect_string)

def verdictdb_query(query):
    target, gb = query['target'], query['gb']
    where = '' if len(query['where']) == 0 else 'WHERE '
    for col, (op, val) in query['where'].items():
        if where != 'WHERE ':
            where += "AND "
        if op == 'between':
            where += f'{col} BETWEEN {val[0]} AND {val[1]} '
        else:
            where += f'{col} {op} {val} '

    pred = []
    for agg in ['COUNT', 'AVG', 'SUM']:
        sql = f'SELECT {agg}({target}) FROM {TARGET_DB}.{DATASET_NAME} {where}'
        pred.append(float(verdict_conn.sql(sql).to_numpy().item()) / (SAMPLE_RATE if agg in ['COUNT', 'SUM'] else 1.0))

    p_std = float(verdict_conn.sql(f'SELECT STDDEV({target}) FROM {TARGET_DB}.{DATASET_NAME} {where}').to_numpy().item())
    p_var = p_std ** 2
    pred = pred + [p_var, p_std]
    return pred



if __name__ == "__main__":

    logger = get_logger(OUT_DIR, 'eval-verdictdb.log')
    # T  = TimeTracker()
    # sql = f"DROP TABLE IF EXISTS {TARGET_DB}.{DATASET_NAME};"
    # mysql_conn.execute(sql)
    # T.report_interval_time_sec(sql)

    # sql = f'CREATE SCRAMBLE {TARGET_DB}.{DATASET_NAME} FROM {SOURCE_DB}.{DATASET_NAME} size {SAMPLE_RATE}'
    # verdict_conn.sql(sql)
    # T.report_interval_time_sec(sql)

    metics = []
    for idx in range(N_QUERIES):
        query = table_wapper.generate_query(gb=False)
        sel_real, cnt_real, ave_real, sum_real, var_real, std_real = table_wapper.query(query)
        T = TimeTracker()
        cnt_pred, ave_pred, sum_pred, var_pred, std_pred = verdictdb_query(query)
        r_cnt, r_ave, r_sum, r_var, r_std = relative_error(cnt_pred, cnt_real), \
                                            relative_error(ave_pred, ave_real), \
                                            relative_error(sum_pred, sum_real), \
                                            relative_error(var_pred, var_real), \
                                            relative_error(std_pred, std_real)
        ms = T.report_interval_time_ms(query)
        logger.info(f'\nquery {idx}:{query} selectivity:{100 * sel_real:.3f}% latency:{ms:.3f} ms')
        logger.info("true:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_real, ave_real, sum_real, var_real, std_real))
        logger.info("pred:\ncnt:{:.3f} ave:{:.3f} sum:{:.3f} var:{:.3f} std:{:.3f} ".
                    format(cnt_pred, ave_pred, sum_pred, var_pred, std_pred))
        logger.info("r_err:\ncnt:{:.3f}% ave:{:.3f}% sum:{:.3f}% var:{:.3f}% std:{:.3f}%".
                    format(r_cnt, r_ave, r_sum, r_var, r_std))
        metics.append([ms, r_cnt, r_ave, r_sum, r_var, r_std])
    metics = pd.DataFrame(metics, columns=['ms', 'rcnt', 'rave', 'rsum', 'rvar', 'rstd'])
    metics.to_csv(os.path.join(OUT_DIR, 'eval-verdictdb.csv'))

    logger.info("mean\n" + str(metics.mean()) + '\n')
    logger.info(".5\n" + str(metics.quantile(0.5)) + '\n')





