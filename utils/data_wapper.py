
import pickle
from .utils import * 
from .time_tracker import TimeTracker
from .preprocess import discretize_dataset, dequantilize_dataset, uniform_dequantize, spline_dequantize


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,

}


class DataWrapper:
    def __init__(self, dataset_name, out_path, seed=45, read_meta=False):
        """
         @brief Initialize the discretization and dequantization of the dataset. This is the function that is called by the constructor
         @param dataset_name name of the dataset to be discretized
         @param out_path path to the output directory
         @param seed seed for random number generator ( default 45 )
         @param read_meta read metadata ( default False ) if True the meta will be
        """
        tracker = TimeTracker()
        self.data = LoadTable(dataset_name)
        self.n, self.dim = self.data.shape
        self.data_np, self.cateMap = discretize_dataset(self.data)
        self.data_dq = dequantilize_dataset(dataset_name)
        self.cateMap = {col: {'id2cate': id_list, 'cate2id': {cate: id for id, cate in enumerate(id_list)}}
                        for col, id_list in self.cateMap.items()}
        tracker.reportIntervalTime('Discretize data')
        self.columns = list(self.data.columns)
        self.colMap = {col: i for i, col in enumerate(self.columns)}
        self.dataset_name = dataset_name
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        self.meta_path = os.path.join(out_path, 'meta.pickle')
        # self.meta_path = PROJECT_DIR + '/meta/{}.pickle'.format(self.dataset_name)
        if not os.path.exists(self.meta_path) or not read_meta:
            eps = 1e-05
            self.Means = self.data_dq.mean(axis=0)
            tracker.reportIntervalTime('cal Means')
            self.Stds = self.data_dq.std(axis=0) + eps
            tracker.reportIntervalTime('cal Stds')

            self.Mins = self.data_dq.min(axis=0)
            tracker.reportIntervalTime('cal Mins')
            self.Maxs = self.data_dq.max(axis=0)
            tracker.reportIntervalTime('cal Maxs')

            self.minFilter, self.maxFilter = 0, len(self.columns)
            self.sensible_to_do_range = np.array(
                [1 if self.data.dtypes[col] != 'object' else 0 for col in self.columns])
            self.saveMetaData()
            tracker.reportIntervalTime('Saves Meta Data')
        else:
            self.readMetaData()
            tracker.reportIntervalTime('Reads Meta Data')

    def saveMetaData(self):
        """
         @brief Save meta data to file for use by : py : meth : ` read `. This is useful when you want to re - use the object
        """
        metaData = {
            'columns': self.columns,
            'colMap': self.colMap,
            'cateMap': self.cateMap,
            'dim': self.dim,
            'n': self.n,
            'Mins': list(self.Mins),
            'Maxs': list(self.Maxs),
            'Means': list(self.Means),
            'Stds': list(self.Stds),
            'minFilter': self.minFilter,
            'maxFilter': self.maxFilter,
            'sensible': self.sensible_to_do_range,
            # 'delta': self.delta
        }
        with open(self.meta_path, 'wb') as f:
            pickle.dump(metaData, f)

    def readMetaData(self):
        """
         @brief Read meta data from file and store in class variables. This is called after the object is created and before any objects are added
        """
        with open(self.meta_path, 'rb') as f:
            metaData = pickle.load(f)
        self.columns = metaData['columns']
        self.colMap = metaData['colMap']
        self.cateMap = metaData['cateMap']
        self.dim = metaData['dim']
        self.n = metaData['n']
        self.Mins = metaData['Mins']
        self.Maxs = metaData['Maxs']
        self.Means = metaData['Means']
        self.Stds = metaData['Stds']
        self.minFilter = metaData['minFilter']
        self.maxFilter = metaData['maxFilter']
        self.sensible_to_do_range = metaData['sensible']

    def getColID(self, col):
        """
         @brief Get the ID of the column. This is useful for determining which columns are in the table and which need to be recalculated when they are added to the result set
         @param col Column name or index.
         @return ID of the column or None if not found ( no error is raised ). Note that col may be a string
        """
        return self.colMap[col] if isinstance(col, str) else col

    def getNormalizedValue(self, col_id, val, norm_type='meanstd'):
        """
         @brief Normalize and return a value. This is a convenience function for normalizing a value to a given mean standard deviation or min / max of the data
         @param col_id column to normalize ( must be in self. Means self. Stds )
         @param val value to normalize ( must be in self. Stds self. Mins self. Maxs )
         @param norm_type type of normalization to use ( meanstd or minmax )
         @return value normalized to the range [ 0 1 ] or raises an exception if normalization is not supported ( default : meanstd
        """
        # Returns the normalized value of the column.
        if norm_type == 'meanstd':
            return (val - self.Means[col_id]) / self.Stds[col_id]
        elif norm_type == 'minmax':
            return (val - self.Mins[col_id]) / (self.Maxs[col_id] - self.Mins[col_id])

        else:
            raise TypeError('unsupported normalized type')

    def getQueryRange(self, query):
        """
         @brief Get the legal and actual range for an aqp query. This is a helper to make it easier to use when you want to know where a query is in the database
         @param query query as returned by self. getQuery
         @return a tuple of two lists : ( lower bound upper bound
        """
        """get legal range and actual range for an aqp query"""
        legal_range, actual_range = [[0., 1.]] * len(self.columns), [[0., 1.]] * len(self.columns)
        # Returns a list of range values for each column in the query.
        for col_name in self.columns:
            col_idx = self.getColID(col_name)
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
                    if col_name in self.cateMap:
                        val = self.cateMap[col_name]['cate2id'][val]
                    lb, ub = val, val + 1
                else:
                    raise ValueError("unsupported operation")

            except KeyError:  # no predicate
                lb, ub = self.Mins[col_idx], self.Maxs[col_idx]

            actual_range[col_idx] = [lb, ub]
            # getUnNormalizedValue
            legal_range[col_idx] = [self.getNormalizedValue(col_idx, lb), self.getNormalizedValue(col_idx, ub)]

        return torch.FloatTensor(legal_range), torch.FloatTensor(actual_range)

    def getFullQueryRange(self):
        """
         @brief Returns legal and actual range for all columns. This is used to determine the range of the query to be executed.
         @return tuple of two lists : 1st list is legal range 2nd list is actual range ( inclusive )
        """
        legal_range, actual_range = [[0., 1.]] * len(self.columns), [[0., 1.]] * len(self.columns)
        # This method is used to calculate the range of the columns in the table.
        for col_name in self.columns:
            col_idx = self.getColID(col_name)
            lb, ub = self.Mins[col_idx], self.Maxs[col_idx]
            legal_range[col_idx] = [self.getNormalizedValue(col_idx, lb), self.getNormalizedValue(col_idx, ub)]
            actual_range[col_idx] = [lb, ub]

        return torch.FloatTensor(legal_range), torch.FloatTensor(actual_range)

    def getLegalRangeNQuery(self, queries):
        """
         @brief Get the Legal Range for N queries. This is a list of N queries that are legal ranges
         @param queries A list of queries to get the Legal Range for
         @return A list of N legal ranges for each query in the queries list. Each legal range is a Range object
        """
        """ legal ranges for N queries """
        legal_lists = []
        # Add a range of legal lists to the list of legal lists.
        for query in queries:
            legal_lists.append(self.getQueryRange(query))
        return legal_lists

    @staticmethod
    def generateFullQuery():
        """
         @brief Generates a full query that can be used to retrieve data. This is a helper function for L { generateQuery } and should be used in conjunction with L { query } when you don't need to query the database.
         @return A dictionary containing the query to be used for the search and the column to be selected by the user
        """
        qry = {
            "where": {},
            "col": 1
        }
        return qry

    def generateAQPQuery(self, rng=None):
        """
         @brief Generate a query to be used in AQP. This is a method of : class : ` ~gensim. models. Sensitivity ` and should be called with a dictionary of parameters that can be passed to
         @param rng random number generator to use
         @return a dictionary of query parameters to be passed to : meth : ` ~gensim. models. Sensitivity
        """
        """ generate a AQP query """
        qry = {
            "where": {},
            "col": None
        }
        rng = self.random_state
        # num_filters = rng.randint(self.minFilter, self.maxFilter)
        num_filters = rng.randint(1, 4)
        target_col_idx = rng.randint(0, self.dim)
        qry['col'] = target_col_idx
        loc = rng.randint(0, self.n)
        tuple0 = self.data.iloc[loc].values
        loc = rng.randint(0, self.n)
        tuple1 = self.data.iloc[loc].values

        cols_idx = rng.choice(len(self.columns), replace=False, size=num_filters)
        cols_name = np.take(self.columns, cols_idx)

        # Add a range of tuples to the query.
        for col_idx, col_name in zip(cols_idx, cols_name):
            # Returns the range of tuples col_idx.
            if not self.sensible_to_do_range[col_idx]:
                # continue
                op = '='
                val = tuple0[col_idx]
            else:
                op = rng.choice(['<=', '>=', 'between'])
                # Returns the predicate to be used for the given operator.
                if op == 'between':
                    lower, upper = min(tuple0[col_idx], tuple1[col_idx]), max(tuple0[col_idx], tuple1[col_idx])
                    # predicate for lower upper case
                    if lower == upper:
                        continue  # unsupport predicate
                    val = (lower, upper)
                else:
                    val = tuple0[col_idx]
                    eps = 1e-3
                    # Compare the operator to the value of the operator.
                    if op == '>=' and abs(val - self.Maxs[col_idx]) < eps:
                        op = '<='
                    elif op == '<=' and abs(val - self.Mins[col_idx]) < eps:
                        op = '>='

            qry['where'][col_name] = (op, val)

        return qry

    def generateNQuery(self, n, rng=None):
        """
         @brief Generate N queries for AQP. This is a wrapper around generateAQPQuery to allow a user to specify a number of queries to be generated
         @param n The number of queries to generate
         @param rng The random number generator to use. If None the one created by self. random_state will be used
         @return A list of queries for AQP ( n ) or an empty list if there are no queries
        """
        """ generate N queries """
        # Set the random state of the generator.
        if rng is None:
            rng = self.random_state
        ret = []
        # Generates a QP query for the next 14 characters.
        for i in range(n):
            ret.append(self.generateAQPQuery(rng))
        return ret

    def query(self, query):
        """
         @brief Query the data to get the statistics. This is a function that takes a query and returns a tuple of the following : count average variance std
         @return A tuple of the following : count averaged variance std ( in case of outliers ) A boolean array of True if the query was
        """
        """ get true count, ave, sum for a query """
        predicates, target_col_idx = query['where'], query['col']
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
        sum = filted_data.sum()
        ave = filted_data.mean()
        var = filted_data.var()
        std = filted_data.std()
        return count, ave, sum, var, std
