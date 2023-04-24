import pickle
from copy import deepcopy
from tqdm import tqdm
import os
from os.path import join, exists
from utils import *

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,

}


class TableWrapper:
    def __init__(self, dataset_name, out_path, dequan_type='spline', seed=45, read_meta=False):
        tracker = TimeTracker()
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.dataset_name = dataset_name
        self.data = load_table(dataset_name)
        self.n, self.dim = self.data.shape
        self.data_np, self.categorical_mapping = discretize_dataset(self.data)
        self.data_dq = dequantilize_dataset(dataset_name, dequan_type)
        self.categorical_mapping = {col: {'id2cate': id_list, 'cate2id': {cate: id for id, cate in enumerate(id_list)}}
                                    for col, id_list in self.categorical_mapping.items()}
        tracker.report_interval_time_ms('Discretize data')
        self.columns = list(self.data.columns)
        self.col2id = {col: i for i, col in enumerate(self.columns)}

        self.numetric_ids, self.categorical_ids = [], []
        self.numetric_cols, self.categorical_cols = [], []
        for col in self.columns:
            id = self.get_col_id(col)
            if col in self.categorical_mapping:
                self.categorical_ids.append(id)
                self.categorical_cols.append(col)
            else:
                self.numetric_ids.append(id)
                self.numetric_cols.append(col)

        self.meta_path = os.path.join(out_path, 'meta.pickle')
        self.query_groupby_sql = open(os.path.join(out_path, 'query-groupby.sql'), 'a')
        self.query_sql = open(os.path.join(out_path, 'query.sql'), 'a')
        if not os.path.exists(self.meta_path) or not read_meta:
            self.create_meta_data()
            tracker.report_interval_time_ms('Create Meta Data')
        else:
            self.read_meta_data()
            tracker.report_interval_time_ms('Reads Meta Data')

    def print_columns_info(self):
        s = f"\nColumns info for {self.dataset_name}:\n"
        for col in self.columns:
            s += f"{col}:{'Category' if col in self.categorical_cols else 'Numeric'}\n"
        s += f'Num Category:{len(self.categorical_cols)} Num Numeric:{len(self.columns) - len(self.categorical_cols)}\n\n'
        print(s)
        return s

    def create_meta_data(self):
        eps = 1e-05
        self.Means = self.data_dq.mean(axis=0)
        self.Stds = self.data_dq.std(axis=0) + eps
        self.Mins = self.data_dq.min(axis=0)
        self.Maxs = self.data_dq.max(axis=0)
        self.minFilter, self.maxFilter = 0, len(self.columns)
        self.is_numetric_col = np.array([1 if col not in self.categorical_mapping else 0 for col in self.columns])
        meta_data = {
            'columns': self.columns,
            'col2id': self.col2id,
            'cate_mapping': self.categorical_mapping,
            'dim': self.dim,
            'n': self.n,
            'Mins': list(self.Mins),
            'Maxs': list(self.Maxs),
            'Means': list(self.Means),
            'Stds': list(self.Stds),
            'minFilter': self.minFilter,
            'maxFilter': self.maxFilter,
            'is_numetric_col': self.is_numetric_col,
            # 'delta': self.delta
        }
        with open(self.meta_path, 'wb') as f:
            pickle.dump(meta_data, f)

    def read_meta_data(self):
        with open(self.meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        self.columns = meta_data['columns']
        self.col2id = meta_data['colMap']
        self.categorical_mapping = meta_data['cateMap']
        self.dim = meta_data['dim']
        self.n = meta_data['n']
        self.Mins = meta_data['Mins']
        self.Maxs = meta_data['Maxs']
        self.Means = meta_data['Means']
        self.Stds = meta_data['Stds']
        self.minFilter = meta_data['minFilter']
        self.maxFilter = meta_data['maxFilter']
        self.is_numetric_col = meta_data['is_numetric_col']

    def get_col_id(self, col):

        return self.col2id[col] if isinstance(col, str) else col

    def get_col_name(self, id):
        return self.columns[int(id)]

    def get_normalized_value(self, col_id, val, norm_type='meanstd'):
        # Returns the normalized value of the column.
        if norm_type == 'meanstd':
            return (val - self.Means[col_id]) / self.Stds[col_id]
        elif norm_type == 'minmax':
            return (val - self.Mins[col_id]) / (self.Maxs[col_id] - self.Mins[col_id])

        else:
            raise TypeError('unsupported normalized type')

    def get_query_range(self, query):
        """get legal range and actual range for an aqp query"""
        legal_range, actual_range = [[0., 1.]] * len(self.columns), [[0., 1.]] * len(self.columns)
        # Returns a list of range values for each column in the query.
        for col_name in self.columns:
            col_idx = self.get_col_id(col_name)
            try:
                op, val = query['where'][col_name]
                # Returns the value of the operation.
                if op in ['>', '>=']:
                    lb, ub = val, self.Maxs[col_idx]
                elif op in ['<', '<=']:
                    lb, ub = self.Mins[col_idx], val
                elif op.lower() in ['in', 'between']:
                    lb, ub = val
                elif op == '=':  # could be str
                    # Get the id of the citation
                    if col_name in self.categorical_mapping:
                        val = self.categorical_mapping[col_name]['cate2id'][val]
                    lb, ub = val, val + 1
                else:
                    raise ValueError("unsupported operation")

            except KeyError:  # no predicate
                lb, ub = self.Mins[col_idx], self.Maxs[col_idx]

            actual_range[col_idx] = [lb, ub]
            # getUnNormalizedValue
            legal_range[col_idx] = [self.get_normalized_value(col_idx, lb), self.get_normalized_value(col_idx, ub)]

        return torch.FloatTensor(legal_range), torch.FloatTensor(actual_range)

    def get_full_query_range(self):

        legal_range, actual_range = [[0., 1.]] * len(self.columns), [[0., 1.]] * len(self.columns)
        # This method is used to calculate the range of the columns in the table.
        for col_name in self.columns:
            col_idx = self.get_col_id(col_name)
            lb, ub = self.Mins[col_idx], self.Maxs[col_idx]
            legal_range[col_idx] = [self.get_normalized_value(col_idx, lb), self.get_normalized_value(col_idx, ub)]
            actual_range[col_idx] = [lb, ub]

        return torch.FloatTensor(legal_range), torch.FloatTensor(actual_range)

    def get_legal_range_N_query(self, queries):
        """ legal ranges for N queries """
        legal_lists = []
        # Add a range of legal lists to the list of legal lists.
        for query in queries:
            legal_lists.append(self.get_query_range(query))
        return legal_lists

    @staticmethod
    def generate_full_query():
        qry = {
            "where": {},
            "col": 1,
            'gb': None
        }
        return qry

    def generate_query(self, gb=False, num_predicates_ranges=None):
        """ generate a AQP query """
        qry = {
            "where": {},
            "target": None,
            'gb': None
        }
        if num_predicates_ranges is not None:
            num_predicates = self.random_state.randint(num_predicates_ranges[0], num_predicates_ranges[1] + 1)
        else:
            num_predicates = self.random_state.randint(1, len(self.categorical_cols) + 1)

        # cols_to_apply_filtes = self.random_state.choice(self.numetric_ids, num_predicates)
        num_point = min(self.random_state.randint(0, 3), num_predicates, len(self.categorical_ids))
        # num_point = min(num_predicates, len(self.categorical_ids))
        num_range = min(num_predicates - num_point, len(self.numetric_ids))

        target_id = self.random_state.choice(self.numetric_ids, 1)
        qry['target'] = self.get_col_name(target_id)
        if gb:
            groupby_id = self.random_state.choice(self.categorical_ids, 1)
            qry['gb'] = self.get_col_name(groupby_id)

        loc = self.random_state.randint(0, self.n)
        tuple0 = self.data.iloc[loc].values
        loc = self.random_state.randint(0, self.n)
        tuple1 = self.data.iloc[loc].values

        range_ids = list(self.random_state.choice(self.numetric_ids, size=num_range, replace=False))
        point_ids = list(self.random_state.choice(self.categorical_ids, size=num_point, replace=False))

        if gb:
            if groupby_id in point_ids:
                point_ids.remove(int(groupby_id))
            num_point -= 1
            num_predicates -= 1

        for id in range_ids:
            col = self.get_col_name(id)
            op = self.random_state.choice(['>=', '<=', 'between'], size=1).item()
            if op == 'between':
                lower, upper = min(tuple0[id], tuple1[id]), max(tuple0[id], tuple1[id])
                if lower == upper:
                    continue
                    # op = self.random_state.choice(['>=', '<='], size=1)
                val = (lower, upper)
            else:
                val = tuple0[id]
                eps = 1e-3
                if op == '>=' and abs(val - self.Maxs[id]) < eps:
                    op = '<='
                elif op == '<=' and abs(val - self.Mins[id]) < eps:
                    op = '>='

            qry['where'][col] = (op, val)

        for id in point_ids:
            col = self.get_col_name(id)
            op = '='
            val = tuple0[id]
            qry['where'][col] = (op, val)
        if not self.is_query_legal(qry):
            print('no legal for ', qry, '\n\n\n')
            qry = self.generate_query(gb, num_predicates_ranges)
        for sql in self.get_qry_sql(qry):
            self.query_sql.write(sql + ';\n' if sql != ' ' else '') if gb is None else self.query_groupby_sql.write(sql + ';\n' if sql != ' ' else '')
        return qry

    def is_query_legal(self, query):
        predicates, target_id = query['where'], self.get_col_id(query['target'])
        legal_range, actual_range = self.get_query_range(predicates)
        from integrators.utils import split_domain
        legal_start, legal_size, legal_volume = split_domain(legal_range)
        if legal_volume == 0:
            return False
        return True

    def generate_groupby_query(self):

        pass

    def get_qry_sql(self, qry):
        from_ = f'FROM {self.dataset_name} '
        if len(qry['where']) == 0:
            where = ''
        else:
            where = 'WHERE '
            for col, (op, val) in qry['where'].items():
                # col = f'`{col}`'
                if where != 'WHERE ':
                    where += 'AND '
                if op == '=':
                    val = f'\'{val}\''
                if op == 'between':
                    lower, upper = val
                    where += f'{col} BETWEEN {lower} AND {upper} '
                else:
                    where += f'{col} {op} {val} '
        target, gb = qry['target'], qry['gb']
        # target = f'`{target}`'
        groupby = f'GROUP BY `{gb}` ' if gb is not None else ''
        sqls = []
        for agg in ['COUNT', 'AVG', 'SUM', 'VARIANCE', 'STDDEV']:
            sql = f'SELECT {agg}({target}) ' + from_ + where + groupby
            sqls.append(sql)
        sqls.append(" ")
        return sqls

    def generate_N_query(self, n):
        # Set the random state of the generator.
        return [self.generate_query() for i in range(n)]

    def query(self, query):
    

        if query['gb'] is not None:
            return self.groupby_query(query)
        else:
            predicates, target_col = query['where'], query['target']
            target_col_idx = self.get_col_id(target_col)
            mask = np.ones(len(self.data)).astype(np.bool_)
            # Returns a mask of the data for each column.
            for col in self.columns:
                # Skips the first predicate in the predicates.
                if col not in predicates:
                    continue
                op, val = predicates[col]
                # Returns the index of the column in the data.
                if op in OPS:
                    inds = OPS[op](self.data[col], val)
                elif op.lower() in ['between', 'in']:
                    lb, ub = val
                    inds = OPS['>='](self.data[col], lb) & OPS['<='](self.data[col], ub)
                mask &= inds.array.to_numpy()

            filted_data = self.data_np[:, target_col_idx][mask]
            count = mask.sum()
            sel = count / (self.n * 1.0)
            sum = filted_data.sum()
            ave = filted_data.mean()
            var = filted_data.var()
            std = filted_data.std()
            return sel, (count, ave, sum, var, std)

    def groupby_query(self, query):
        gb_col = query['gb']
        gb_distinct_vals = self.categorical_mapping[gb_col]['id2cate']
        results = {}
        predicates, target_col = query['where'], query['target']
        target_col_idx = self.get_col_id(target_col)
        mask = np.ones(len(self.data)).astype(np.bool_)
        for col in self.columns:
            # Skips the first predicate in the predicates.
            if col not in predicates:
                continue
            op, val = predicates[col]
            # Returns the index of the column in the data.
            if op in OPS:
                inds = OPS[op](self.data[col], val)
            elif op.lower() in ['between', 'in']:
                lb, ub = val
                inds = OPS['>='](self.data[col], lb) & OPS['<='](self.data[col], ub)
            mask &= inds.array.to_numpy()
        filted_df = self.data.iloc[mask]
        groups = filted_df.groupby(gb_col)[target_col]
        for gb_val, cnt, avg, sum, var, std in zip(groups.count().keys(), groups.count(), groups.mean(), groups.sum(), groups.var(), groups.std()):
            if np.isnan(std):
                std = 0
            if np.isnan(var):
                var = 0
            results[gb_val] = [cnt, avg, sum, var, std]
        # for val, agg in groups.sum():
        #     results[col] = [agg]
        # for val, agg in groups.sum():
        #     results[col] = [agg]


        # target_col_data = self.data_np[:, target_col_idx]
        # for gb_val in tqdm(gb_distinct_vals):
        #     gb_mask = mask & (self.data[gb_col] == gb_val)
        #     filted_data = target_col_data[gb_mask]
        #     count = mask.sum()
        #     if count == 0:
        #         continue
        #     sel = count / (self.n * 1.0)
        #     sum = filted_data.sum()
        #     ave = filted_data.mean()
        #     var = filted_data.var()
        #     std = filted_data.std()
        #     results[gb_val] = [count, ave, sum, var, std]
        #     pass
        return results

    def __del__(self):
        self.query_sql.close()
        self.query_groupby_sql.close()


def make_query(dataset_name, out_dir, dequan_type, n_queries, n_predicates, gb=False):
    query_name = f'queires-{n_queries}-[{n_predicates[0]}, {n_predicates[1]}]{"-gb" if gb else ""}.json'
    if exists(join(out_dir, query_name)) and exists(join(out_dir, 'meta.pickle')):
        return
    wapper = TableWrapper(dataset_name, out_dir, dequan_type, read_meta=False)
    queries = []
    for i in range(n_queries):
        query = wapper.generate_query(gb, num_predicates_ranges=n_predicates)
        query['real'] = wapper.query(query)
        queries.append(query)

    with open(join(out_dir, query_name), 'w', encoding='utf-8') as f:
        f.write(json.dumps(queries, ensure_ascii=False, indent=4))
