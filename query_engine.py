import math
import pickle

import torch
import os
from time import time
from utils import PROJECT_DIR, VEGAS_BIG_N
from integrators import MonteCarloAQP, VegasAQP, VEGAS
from datasets import eps


class QueryEngine:
    def __init__(
            self,
            model,
            dataset_name,
            out_path,
            integrator=None,
            device=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'),
            n_sample_points=100000,
            alpha=0.4,  # alpha & beta are only used in Vegas
            beta=0.2
    ):

        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        self.last_qeury_time = 0
        self.total_query_time = 0
        self._st = 0

        self.map_path = os.path.join(out_path, 'vegas.map')
        self.meta_path = os.path.join(out_path, 'meta.pickle')

        self.read_meta_data()

        # The integrator to use for the simulation.
        if integrator is None or integrator == 'MonteCarlo':
            self.integrator = MonteCarloAQP(
                self.pdf,
                n_sample_points=n_sample_points,
                n_chunks=n_sample_points // 10,
                device=device
        )

        elif integrator == 'Vegas':

            norm_full_domain, _ = self.get_full_range()
            self.integrator = VegasAQP(
                self.pdf,
                target_map=self._get_target_map(),
                full_domain=norm_full_domain,
                n_sample_points=n_sample_points,
                dim=self.dim, 
                device=device,
                alpha=alpha,
                beta=beta,
                max_iteration=4
            )

    def get_normalized_val(self, col_id, val, norm_type='meanstd'):
        """
         @brief Normalize a value to a specified type. This is useful for calculating mean std or min / max of a column
         @param col_id column to normalize ( int )
         @param val value to normalize ( float ) ( int )
         @param norm_type type of normalization ( meanstd min or minmax )
         @return normalized value ( float ) ( int or float ) Normalized value ( float ) ( int or float
        """
        if norm_type == 'meanstd':
            return (val - self.Means[col_id]) / (self.Stds[col_id] + eps)
        elif norm_type == 'minmax':
            return (val - self.Mins[col_id]) / (eps + self.Maxs[col_id] - self.Mins[col_id])

        else:
            norm_val = (val - self.Means[col_id]) / (self.Stds[col_id])
            norm_val = (norm_val - self.Mins[col_id]) / \
                (self.Maxs[col_id] - self.Mins[col_id])
            return norm_val

    def get_query_range(self, predicates):
        """
         @brief Get legal and actual range for an aqp query based on predicate. This is used to determine the range of a QP query that should be used in order to perform the operation.
         @param predicates dictionary of column name and predicate. For example {'column_name': ('>'' <') }
         @return a tuple of two lists. The first list contains the legal range and the second list contains the actual
        """
        """get legal range and actual range for an aqp query"""
        legal_range, actual_range = [[0., 1.]] * len(self.columns), [[0., 1.]] * len(self.columns)
        # Returns a list of range values for each column.
        for col_name in self.columns:
            col_idx = self.get_col_id(col_name)
            try:
                op, val = predicates[col_name]
                # Returns the bin size of the bin size of the column.
                if op in ['>', '>=']:
                    lb, ub = val, self.Maxs[col_idx]
                elif op in ['<', '<=']:
                    lb, ub = self.Mins[col_idx], val
                elif op.lower() in ['in', 'between']:
                    # in this predicate, val should be tupe like (a, b), while is scalar in orther.
                    lb, ub = val
                elif op == '=':  # could be str
                    # Returns the bin size of the categorical column
                    if col_name in self.cateMap:
                        val = self.cateMap[col_name]['cate2id'][val]
                        lb, ub = val, val + 1   # bin size is allways 1 for categorical column
                    else:
                        lb, ub = val, val
                else:
                    raise ValueError("unsupported operation")

            except KeyError:  # full range for empty predicate.
                lb, ub = self.Mins[col_idx], self.Maxs[col_idx]

            actual_range[col_idx] = [lb, ub]
            # getUnNormalizedValue
            legal_range[col_idx] = [self.get_normalized_val(col_idx, lb), self.get_normalized_val(col_idx, ub)]

        return torch.FloatTensor(legal_range).to(self.device), torch.FloatTensor(actual_range).to(self.device)

    def query(self, query):
        """
         @brief Query the data to be integrated. This is a wrapper around : meth : ` integrator. integrate ` which takes a dictionary of predicates and target column indices as input and returns the integration result as
         @return tuple of count average sum var std : math : ` \ sqrt { var } ` where : meth : ` integrator. integrate ` is called with the list of columns that are to be integrated
        """
        self._time_start()
        predicates, target_col_idx = query['where'], query['col']
        legal_range, actual_range = self.get_query_range(predicates)
        sel, ave, var = self.integrator.integrate(
            legal_range,
            actual_range,
            target_col_idx
        )


        count = sel * self.n
        sum = ave * count
        std = math.sqrt(var)
        self._time_stop()
        return count, ave, sum, var, std

    def pdf(self, x):
        """
         @brief Computes the probability density function for the given data. This is a wrapper around log_prob ( x ) and exp ( log_prob ( x ))
         @param x A tensor of shape [ n_samples n_features ]
         @return A tensor of shape [ n_samples n_classes ] containing the probability density function for the given
        """
        with torch.no_grad():
            log_prob = self.model.log_prob(x)
            prob = torch.exp(log_prob)
            return prob

    def get_discretized_predicates(self, cols, ops, vals):
        """
         @brief Get discretized predicates based on column operators and values. This is used to determine whether or not an expression is valid for a column or not
         @param cols A list of column names
         @param ops A list of operator names
         @param vals A list of values to filter on. If value is a list it is converted to a list of code
         @return A tuple of columns ops
        """
        vs = []
        # Add a code to the list of code to the list of code values.
        for c, o, v in zip(cols, ops, vals):
            # Return the code of a CATE code.
            if c in self.cateMap:
                # Convert a list of strings to a CATE code.
                if isinstance(v, list):
                    # Convert a string to a list of codes.
                    if isinstance(v[0], str):
                        v = [self.cateStr2Code[c, v_] for v_ in v]
                elif isinstance(v, str):
                    v = self.cateStr2Code[c][v]
            vs.append(v)
        return cols, ops, vs

    def get_col_id(self, col):
        """
         @brief Get the ID of a column. This is used to identify the column in the data set that is to be displayed to the user.
         @param col The name of the column. It must be a string in the form'col_name'where'col_name'is the case - sensitive
        """
        return self.colMap[col]

    def get_categorical_cols(self, cols):
        """
         @brief Get sensible values for categorical columns. This is a convenience method for getting the sensible values for a list of colums that are to be used as columns in a Categorical object
         @param cols A list of column names
         @return A list of sensible values for the columns in the Categorical object. If there are no sensible values for any of the columns None is
        """
        cs = [self.get_col_id(col) for col in cols]
        return self.sensible_to_do_range[cs]


    def get_full_range(self):
        legal_range, actual_range = [[0., 1.]] * len(self.columns), [[0., 1.]] * len(self.columns)
        # This function computes the range of the range of the columns in the model.
        for col_name in self.columns:
            col_idx = self.get_col_id(col_name)
            lb, ub = self.Mins[col_idx], self.Maxs[col_idx]
            legal_range[col_idx] = [self.get_normalized_val(col_idx, lb), self.get_normalized_val(col_idx, ub)]
            actual_range[col_idx] = [lb, ub]

        return torch.FloatTensor(legal_range).to(self.device), torch.FloatTensor(actual_range).to(self.device)


    def _time_start(self):
        self._st = time()

    def _time_stop(self):
        et = time()
        self.last_qeury_time = (et - self._st)
        self.total_query_time += self.last_qeury_time
        self._st = et

    def _get_target_map(self, remake=False):
        """
        @brief Load or create target map. This is a wrapper to _create_target_map which can be used to re - create the target map after a failure.
        @param remake if True don't try to load the old target map
        @return torch. Tensor or None if it could not be loaded or created ( and remake is False
        """

        st = time()
        # Load and create target map if necessary
        if os.path.exists(self.map_path) and not remake:
            target_map = torch.load(self.map_path, map_location=self.device)
            print('load old target map tooks {:.4f} sec'.format(time() - st))
        else:
            target_map = self._create_target_map()
            print('create new target map tooks {:.4f} sec'.format(time() - st))

        return target_map

    def _create_target_map(self):
        """
         @brief Create a VEGAS target map. It is used to calculate the integration domain and save it to file.
         @return the map created by VEGAS. This map can be used for testing and to compare results
        """
        print('creating new target map')
        vegas = VEGAS()
        bigN = VEGAS_BIG_N
        max_iter = 20
        #bigN = 5000000
        full_domain, _ = self.get_full_range()
        res = vegas.integrate(
            fn=self.pdf,
            dim=len(self.columns),
            N=bigN,
            integration_domain=full_domain,
            use_warmup=True,
            use_grid_improve=True,
            max_iterations=max_iter
        )
        print(f'full integration is {res:.3f}')
        target_map = vegas.map
        torch.save(target_map, self.map_path)
        return target_map

    def full_domain_integrate(self):
        """
         @brief Integrate the full domain. This is a wrapper around : py : meth : ` ~montreal_forced_integrate ` to allow integration of an unsaturated integral over a full domain.
         @return A tuple containing the selection and ave / var for the integration. See : py : meth : ` ~montreal_forced_integrate ` for details
        """
        legal_full_range, actual_full_range = self.get_full_range()
        sel, ave, var = self.integrator.integrate(
            legal_full_range,
            actual_full_range,
            0
        )

        return sel

    def read_meta_data(self):
        """
         @brief Read meta data from file and store in class variables. This is called after the class is instantiated and before any objects are added
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
