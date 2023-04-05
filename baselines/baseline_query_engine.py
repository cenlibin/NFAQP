import pyverdict
import pymysql
import os, sys
import pandas as pd
import numpy as np
sys.path.append('/home/clb/AQP')
from utils import *

# verdict const parameter
SOURCE_DB = 'AQP'
TARGET_DB = 'verdictdb'
SAMPLE_RATE = 0.01

# VAE 
VAE_N_SAMPLE = 10000


class VerdictEngine:
    def __init__(self, dataset_name, table_N, remake=False):

        host = 'localhost'
        user = 'root'
        password = '7837'
        port = 3306
        connect_string = f'jdbc:mysql://{host}:{port}?user={user}&password={password}&useSSL=False&loglevel=error'
        self.mysql_conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            passwd=password,
            autocommit=True
        ).cursor()
        self.verdict_conn = pyverdict.VerdictContext(connect_string)
        self.dataset = dataset_name
        self.N = table_N
        if remake:
            self.generate_sample(dataset_name)

    def query(self, query):
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
                sql = f'SELECT {agg}({target}) FROM {TARGET_DB}.{self.dataset} {where}'
                verdict_pred = self.verdict_conn.sql(sql).to_numpy().item()
                verdict_pred = float(verdict_pred if verdict_pred is not None else 0)
                verdict_pred /= (SAMPLE_RATE if agg in ['COUNT', 'SUM'] else 1.0)
                # if agg == 'COUNT':
                #     pred.append(verdict_pred / self.N)  # sel
                pred.append(verdict_pred)

            p_std = self.verdict_conn.sql(f'SELECT STDDEV({target}) FROM {TARGET_DB}.{self.dataset} {where}').to_numpy().item()
            p_std = float(p_std if p_std is not None else 0)
            p_var = p_std ** 2
            pred = pred + [p_var, p_std]
            return pred


    def generate_sample(self, dataset_name):
        T  = TimeTracker()
        sql = f"DROP TABLE IF EXISTS {TARGET_DB}.{dataset_name};"
        self.mysql_conn.execute(sql)
        T.report_interval_time_sec(sql)

        sql = f'CREATE SCRAMBLE {TARGET_DB}.{dataset_name} FROM {SOURCE_DB}.{dataset_name} size {SAMPLE_RATE}'
        self.verdict_conn.sql(sql)
        T.report_interval_time_sec(sql)


class VAEEngine:
    def __init__(self, dataset_name, table_N, remake=False):
        self.df = pd.read_csv(f"/home/clb/AQP/output/VAE-{dataset_name}/samples_{VAE_N_SAMPLE}.csv")
        self.N = table_N

    def query(self, query):
        OPS = {
            '>': np.greater,
            '<': np.less,
            '>=': np.greater_equal,
            '<=': np.less_equal,
            '=': np.equal,
        }
        predicates, target_col = query['where'], query['target']
        data_np = self.df.to_numpy()
        col_id = {col:id for id, col in enumerate(self.df.columns)}
        target_col_idx = col_id[target_col]
        # target_col_idx = self.get_col_id(target_col)
        mask = np.ones(len(self.df)).astype(np.bool_)
        # Returns a mask of the data for each column.
        for col in self.df.columns:
            # Skips the first predicate in the predicates.
            if col not in predicates:
                continue
            op, val = predicates[col]
            # Returns the index of the column in the data.
            if op in OPS:
                inds = OPS[op](self.df[col], val)
            elif op.lower() in ['between', 'in']:
                lb, ub = val
                inds = OPS['>='](self.df[col], lb) & OPS['<='](self.df[col], ub)
            mask &= inds.array.to_numpy()

        filted_data = data_np[:, target_col_idx][mask]
        count = mask.sum()
        if count == 0:
            return 0, 0, 0, 0, 0
        sel = count / (len(self.df) * 1.0)
        count = sel * self.N
        # sum = filted_data.sum()
        ave = filted_data.mean()
        sum = count * ave
        var = filted_data.var()
        std = filted_data.std()
        return count, ave, sum, var, std

    def generate_sample(self, dataset_name):
        pass