import pyverdict
import pymysql
import os, sys
import pandas as pd
import numpy as np
sys.path.append('/home/clb/AQP')
from utils import *
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), 'deepdb'))
from baselines.deepdb.ensemble_compilation.spn_ensemble import read_ensemble
from baselines.deepdb.evaluation.utils import parse_query
from baselines.deepdb.schemas import *

# VerdictDB
SOURCE_DB = 'AQP'
TARGET_DB = 'verdictdb'
SAMPLE_RATE = 0.01
THRESH_HOLD = 16000
# VAE 


class VerdictEngine:
    def __init__(self, dataset_name, table_N, remake=False):
        if dataset_name not in ['lineitemext']:
            dataset_name += '_10BM'
        # else:
        #     dataset_name += '5g'

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
        # self.verdict_conn = pyverdict.mysql_context(host='localhost', user='root', password=password)
        self.dataset = dataset_name
        self.N = table_N
        self.sample_rate = SAMPLE_RATE

        if remake:
            self.generate_sample(dataset_name)

    def query(self, query):
        with HiddenPrints():
            T = TimeTracker()
            target, gb = query['target'], query['gb']
            # target = f'`{target}`'
            where = '' if len(query['where']) == 0 else 'WHERE '
            for col, (op, val) in query['where'].items():
                # col = f'`{col}`'
                if where != 'WHERE ':
                    where += "AND "
                if op == '=':
                    val = f'\'{val}\''
                if op == 'between':
                    where += f'{col} BETWEEN {val[0]} AND {val[1]} '
                else:
                    where += f'{col} {op} {val} '

            pred = []
            for agg in ['COUNT', 'AVG', 'SUM', 'STDDEV']:
                sql = f'SELECT {agg}({target}) FROM {TARGET_DB}.{self.dataset} {where}'
                verdict_pred = self.verdict_conn.sql(sql).to_numpy().item()
                verdict_pred = float(verdict_pred if verdict_pred is not None else 0)
                verdict_pred /= (self.sample_rate if agg in ['COUNT', 'SUM'] else 1.0)
                # if agg == 'COUNT':
                #     pred.append(verdict_pred / self.N)  # sel
                if agg == 'STDDEV':
                    pred += [verdict_pred ** 2, verdict_pred]
                else:
                    pred.append(verdict_pred)
                if agg == 'SUM':
                    pass
            latency_ms = T.report_interval_time_ms()

            # p_std = self.verdict_conn.sql(f'SELECT STDDEV({target}) FROM {TARGET_DB}.{self.dataset} {where}').to_numpy().item()
            # p_std = float(p_std if p_std is not None else 0)
            # p_var = p_std ** 2
            # pred = pred + [p_var, p_std]
            return latency_ms, pred
    
    def groupby_query(self, query):
        with HiddenPrints():
            T = TimeTracker()
            res = {}
            target, gb = query['target'], query['gb']
            target = f'{target}'
            # gb = f'{gb}'
            where = '' if len(query['where']) == 0 else 'WHERE '
            for col, (op, val) in query['where'].items():
                # col = f'`{col}`'
                if where != 'WHERE ':
                    where += "AND "
                if op == '=':
                    val = f'\'{val}\''
                if op == 'between':
                    where += f'{col} BETWEEN {val[0]} AND {val[1]} '
                else:
                    where += f'{col} {op} {val} '

            for agg_i, agg in enumerate(['COUNT', 'AVG', 'SUM','STDDEV']):
                sql = f'SELECT {gb}, {agg}({target}) FROM {TARGET_DB}.{self.dataset} {where}GROUP BY {gb}'
                # mysql_pred = self.mysql_conn.execute(sql)
                try:
                    verdict_pred = self.verdict_conn.sql(sql)
                except ValueError:
                    verdict_pred = pd.DataFrame([])
                # my = self.mysql_conn.execute(sql)
                verdict_pred = verdict_pred.to_numpy()
                for line in verdict_pred:
                    gb_val, agg_val = line[0], line[1]
                    agg_val = float(agg_val if agg_val is not None else 0)
                    agg_val /= (self.sample_rate if agg in ['COUNT', 'SUM'] else 1.0)
                    if gb_val not in res:
                        res[gb_val] = [0.0] * agg_i
                    if agg == "STDDEV":
                        res[gb_val] += [agg_val ** 2, agg_val]
                    else:
                        res[gb_val].append(agg_val)
                if agg == 'SUM':
                    latency_ms = T.report_interval_time_ms()

            return latency_ms, res


    def generate_sample(self, dataset_name):
        T  = TimeTracker()
        sql = f"DROP TABLE IF EXISTS {TARGET_DB}.{dataset_name};"
        self.mysql_conn.execute(sql)
        T.report_interval_time_sec(sql)

        sql = f'CREATE SCRAMBLE {TARGET_DB}.{dataset_name} FROM {SOURCE_DB}.{dataset_name} size {self.sample_rate}'
        # sql = f'CREATE SCRAMBLE {TARGET_DB}.{dataset_name} FROM {SOURCE_DB}.{dataset_name}'
        self.verdict_conn.sql(sql)
        T.report_interval_time_sec(sql)

    

class VAEEngine:
    def __init__(self, dataset_name, table_N, remake=False):
        N_SAMPLE = 100000
        # N_SAMPLE = int(table_size[dataset_name] * 0.01)
        self.N_SAMPLE = N_SAMPLE
        self.dataset_name = dataset_name
        self.N = table_N
        
        
    def query(self, query, df=None):
        T = TimeTracker()
        OPS = {
            '>': np.greater,
            '<': np.less,
            '>=': np.greater_equal,
            '<=': np.less_equal,
            '=': np.equal,
        }
        if df is None:
            df = pd.read_csv(f"/home/clb/AQP/output/VAE-{self.dataset_name}/samples_{self.N_SAMPLE}.csv")

        predicates, target_col = query['where'], query['target']
        data_np = df.to_numpy()
        col_id = {col:id for id, col in enumerate(df.columns)}
        target_col_idx = col_id[target_col]
        # target_col_idx = self.get_col_id(target_col)
        mask = np.ones(len(df)).astype(np.bool_)
        # Returns a mask of the data for each column.
        for col in df.columns:
            # Skips the first predicate in the predicates.
            if col not in predicates:
                continue
            op, val = predicates[col]
            # Returns the index of the column in the data.
            if op in OPS:
                inds = OPS[op](df[col], val)
            elif op.lower() in ['between', 'in']:
                lb, ub = val
                inds = OPS['>='](df[col], lb) & OPS['<='](df[col], ub)
            mask &= inds.array.to_numpy()

        filted_data = data_np[:, target_col_idx][mask]
        count = mask.sum()
        if count == 0:
            return T.report_interval_time_ms(), (0, 0, 0, 0, 0)
        sel = count / (len(df) * 1.0)
        count = sel * self.N
        # sum = filted_data.sum()
        ave = filted_data.mean()
        sum = count * ave
        latency_ms = T.report_interval_time_ms()
        var = filted_data.var()
        std = filted_data.std()
        return latency_ms, (count, ave, sum, var, std)

    def groupby_query(self, query, df=None):
        T = TimeTracker()
        if df is None:
            df = pd.read_csv(f"/home/clb/AQP/output/VAE-{self.dataset_name}/samples_{self.N_SAMPLE}.csv")

        results = {}
        gb_col = query['gb']
        groups = df.groupby(gb_col)
        mask = np.ones(len(df)).astype(np.bool_)           # if choose the corrsponding rows into filted data
        predicates, target_col = query['where'], query['target']
        OPS = {
            '>': np.greater,
            '<': np.less,
            '>=': np.greater_equal,
            '<=': np.less_equal,
            '=': np.equal,
        }
        # Returns a mask of the data for each column.
        for col in df.columns:
            # Skips the first predicate in the predicates.
            if col not in predicates:
                continue
            op, val = predicates[col]
            # Returns the index of the column in the data.
            if op in OPS:
                inds = OPS[op](df[col], val)
            elif op.lower() in ['between', 'in']:
                lb, ub = val
                inds = OPS['>='](df[col], lb) & OPS['<='](df[col], ub)
            mask &= inds.array.to_numpy()

        groups = df[mask].groupby(gb_col)[target_col]
        scale = self.N_SAMPLE / self.N
        for gb_val, cnt, avg, sum, var, std in zip(groups.count().keys(), groups.count(), groups.mean(), groups.sum(), groups.var(), groups.std()):
            if np.isnan(std):
                std = 0
            if np.isnan(var):
                var = 0
            cnt, sum = cnt / scale, sum / scale
            results[gb_val] = [cnt, avg, sum, var, std]
        latency_ms = T.report_interval_time_ms()
        return latency_ms, results

    def generate_sample(self, dataset_name):
        pass


class DeepdbEngine:

    def __init__(self, dataset_name, table_N, remake=False):
        self.root = f'/home/clb/AQP/output/deepdb-{dataset_name}/'
        data_path = f'/home/clb/AQP/data/{dataset_name}'
        self.dataset = dataset_name
        self.ensemble = read_ensemble([self.root + 'ensemble.pkl'], build_reverse_dict=True)
        schemas = {
            'pm25': gen_pm25_schema,
            'flights': gen_flights_schema,
            'lineitem': gen_lineitem_schema,
            'lineitemext': gen_lineitemext_schema,
            'housing': gen_housing_schema,
        }
        self.schema = schemas[dataset_name](data_path)
        self.schema.tables[0].table_size = table_N
        self.schema.tables[0].sample_rate = 1

        pass

    def query(self, query):
        with HiddenPrints():
            T = TimeTracker()
            result = []
            rdc_spn_selection = False
            pairwise_rdc_path = None
            merge_indicator_exp = True
            max_variants = 1
            exploit_overlapping = True
            debug = False
            show_confidence_intervals = False

            target, gb = query['target'], query['gb']
            # target = f'`{target}`'
            where = '' if len(query['where']) == 0 else 'WHERE '
            for col, (op, val) in query['where'].items():
                # col = f'`{col}`'
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
                sql = f'SELECT {agg}({target}) FROM {self.dataset} {where}'
                qry = parse_query(sql.strip(), self.schema)
                _, p = self.ensemble.evaluate_query(qry, rdc_spn_selection=rdc_spn_selection,
                                                        pairwise_rdc_path=pairwise_rdc_path,
                                                        merge_indicator_exp=merge_indicator_exp,
                                                        max_variants=max_variants,
                                                        exploit_overlapping=exploit_overlapping,
                                                        debug=debug,
                                                        confidence_intervals=show_confidence_intervals)
                pred.append(p)
            return T.report_interval_time_ms(), pred
    
    def groupby_query(self, query):
        with HiddenPrints():
            T = TimeTracker()
            result = {}
            rdc_spn_selection = False
            pairwise_rdc_path = None
            merge_indicator_exp = True
            max_variants = 1
            exploit_overlapping = True
            debug = False
            show_confidence_intervals = False

            target, gb = query['target'], query['gb']
            # target = f'`{target}`'
            where = '' if len(query['where']) == 0 else 'WHERE '
            for col, (op, val) in query['where'].items():
                # col = f'`{col}`'
                if where != 'WHERE ':
                    where += "AND "
                if op == '=':
                    val = f'\'{val}\''
                if op == 'between':
                    where += f'{col} BETWEEN {val[0]} AND {val[1]} '
                else:
                    where += f'{col} {op} {val} '


            for agg in ['COUNT', 'AVG', 'SUM']:

                sql = f'SELECT {agg}({target}) FROM {self.dataset} {where}group by {gb}'

                qry = parse_query(sql.strip(), self.schema)
                try:
                    _, p = self.ensemble.evaluate_query(qry, rdc_spn_selection=rdc_spn_selection,
                                                            pairwise_rdc_path=pairwise_rdc_path,
                                                            merge_indicator_exp=merge_indicator_exp,
                                                            max_variants=max_variants,
                                                            exploit_overlapping=exploit_overlapping,
                                                            debug=debug,
                                                            confidence_intervals=show_confidence_intervals)
                except ValueError:
                    return T.report_interval_time_ms(), {}
                for line in p:
                    gb_val, agg_val = line
                    if gb_val not in result:
                        result[gb_val] = [agg_val]
                    else:
                        result[gb_val].append(agg_val)
            



            return T.report_interval_time_ms(), result
        

class MySQLEngine:
    def __init__(self, dataset_name, table_N, remake=False):
        if dataset_name not in ['lineitemext']:
            dataset_name += '_10BM'
        # else:
        #     dataset_name += '5g'

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
        # self.verdict_conn = pyverdict.VerdictContext(connect_string)
        # self.verdict_conn = pyverdict.mysql_context(host='localhost', user='root', password=password)
        self.dataset = dataset_name
        self.N = table_N
        self.sample_rate = SAMPLE_RATE

        if remake:
            self.generate_sample(dataset_name)

    def query(self, query):
        with HiddenPrints():
            T = TimeTracker()
            target, gb = query['target'], query['gb']
            # target = f'`{target}`'
            where = '' if len(query['where']) == 0 else 'WHERE '
            for col, (op, val) in query['where'].items():
                # col = f'`{col}`'
                if where != 'WHERE ':
                    where += "AND "
                if op == '=':
                    val = f'\'{val}\''
                if op == 'between':
                    where += f'{col} BETWEEN {val[0]} AND {val[1]} '
                else:
                    where += f'{col} {op} {val} '

            pred = []
            for agg in ['COUNT', 'AVG', 'SUM', 'STDDEV']:
                sql = f'SELECT {agg}({target}) FROM {TARGET_DB}.{self.dataset} {where}'
                verdict_pred = self.mysql_conn.execute(sql)
                # verdict_pred = self.verdict_conn.sql(sql).to_numpy().item()
                # verdict_pred = float(verdict_pred if verdict_pred is not None else 0)
                # verdict_pred /= (self.sample_rate if agg in ['COUNT', 'SUM'] else 1.0)
                # if agg == 'COUNT':
                #     pred.append(verdict_pred / self.N)  # sel
                if agg == 'STDDEV':
                    pred += [verdict_pred ** 2, verdict_pred]
                else:
                    pred.append(verdict_pred)
                if agg == 'SUM':
                    pass
            latency_ms = T.report_interval_time_ms()

            # p_std = self.verdict_conn.sql(f'SELECT STDDEV({target}) FROM {TARGET_DB}.{self.dataset} {where}').to_numpy().item()
            # p_std = float(p_std if p_std is not None else 0)
            # p_var = p_std ** 2
            # pred = pred + [p_var, p_std]
            return latency_ms, pred
    
    def groupby_query(self, query):
        with HiddenPrints():
            T = TimeTracker()
            res = {}
            target, gb = query['target'], query['gb']
            target = f'{target}'
            # gb = f'{gb}'
            where = '' if len(query['where']) == 0 else 'WHERE '
            for col, (op, val) in query['where'].items():
                # col = f'`{col}`'
                if where != 'WHERE ':
                    where += "AND "
                if op == '=':
                    val = f'\'{val}\''
                if op == 'between':
                    where += f'{col} BETWEEN {val[0]} AND {val[1]} '
                else:
                    where += f'{col} {op} {val} '

            for agg_i, agg in enumerate(['COUNT', 'AVG', 'SUM','STDDEV']):
                sql = f'SELECT {gb}, {agg}({target}) FROM {TARGET_DB}.{self.dataset} {where}GROUP BY {gb}'
                # mysql_pred = self.mysql_conn.execute(sql)
                try:
                    verdict_pred = self.verdict_conn.sql(sql)
                except ValueError:
                    verdict_pred = pd.DataFrame([])
                # my = self.mysql_conn.execute(sql)
                verdict_pred = verdict_pred.to_numpy()
                for line in verdict_pred:
                    gb_val, agg_val = line[0], line[1]
                    agg_val = float(agg_val if agg_val is not None else 0)
                    agg_val /= (self.sample_rate if agg in ['COUNT', 'SUM'] else 1.0)
                    if gb_val not in res:
                        res[gb_val] = [0.0] * agg_i
                    if agg == "STDDEV":
                        res[gb_val] += [agg_val ** 2, agg_val]
                    else:
                        res[gb_val].append(agg_val)
                if agg == 'SUM':
                    latency_ms = T.report_interval_time_ms()

            return latency_ms, res


    def generate_sample(self, dataset_name):
        T  = TimeTracker()
        sql = f"DROP TABLE IF EXISTS {TARGET_DB}.{dataset_name};"
        self.mysql_conn.execute(sql)
        T.report_interval_time_sec(sql)

        sql = f'CREATE SCRAMBLE {TARGET_DB}.{dataset_name} FROM {SOURCE_DB}.{dataset_name} size {self.sample_rate}'
        # sql = f'CREATE SCRAMBLE {TARGET_DB}.{dataset_name} FROM {SOURCE_DB}.{dataset_name}'
        self.verdict_conn.sql(sql)
        T.report_interval_time_sec(sql)
