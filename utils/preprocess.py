import torch
import os
import sys
import numpy as np
import pandas as pd
import scipy
from .utils import *

def discretize_dataset(data):
    disc_np = data.to_numpy()
    col_ids = {}
    cate_map = {}
    for col_id, col in enumerate(data.columns):
        col_ids[col] = col_id
        cdtype = data.dtypes[col]
        if cdtype == 'object':  # convert categorical type into discretize ID
            cate_map[col] = {}
            val_cnt_index = data[col].value_counts().index  # get each different value as list
            cate_map[col] = list(val_cnt_index.values)
            disc_vals = pd.Categorical(
                data[col], categories=val_cnt_index.values).codes
            disc_np[:, col_id] = disc_vals

    return disc_np.astype(np.float32), cate_map


def uniform_dequantize(col, eps=1e-9):
    '''
        a pre-process step for training data
        given a col of discritize data and dequantize it to more continuious float data
        input: 1-dim np array for a col in table
        output: dequantilized tensor for the input
    '''
    distinct, ID, counts = np.unique(col, return_inverse=True, return_counts=True)
    bin_sizes = np.append(distinct[1:] - distinct[:-1], (distinct[1:] - distinct[:-1]).min())
    noise = np.random.uniform(0.5 * eps, 1 - 0.5 * eps, size=col.shape)
    deq_col = col + bin_sizes[ID] * noise
    return deq_col


def spline_dequantize(col, eps=1e-9):

    """
        implement of spline dequantilization

    """

    # step 1: collect points used to construct CDF

    distinct, ID, counts = np.unique(col, return_inverse=True, return_counts=True)
    cdf_x = np.append(distinct, distinct[-1] + np.diff(distinct).min().item())  # [x0,  x1,      ..., xn, xn + bin]   where bin of end is 1
    cdf_y = np.insert(counts.cumsum() / counts.sum(), 0, 0.0)  # [0, p(a < x1), ..., p(a < xn), 1]

    # step 2: construct CDF
    cdf = scipy.interpolate.PchipInterpolator(x=cdf_x, y=cdf_y, extrapolate=False)

    # step 3: construct inverse CDF
    inverse_cdf = scipy.interpolate.PchipInterpolator(x=cdf_y, y=cdf_x, extrapolate=False)

    # from matplotlib import pyplot as plt
    # plt.cla()
    # x = np.linspace(distinct[0], distinct[-1], 100000)
    # y = cdf(x)
    # plt.plot(x, y)
    # plt.show()
    # plt.cla()
    #
    # x = np.linspace(0, 1, 10000)
    # y = inverse_cdf(x)
    # plt.plot(x, y)
    # plt.show()

    # step 4: get dequantilized value from reverse cdf
    sample = np.random.uniform(0, 1 - eps, size=col.shape)
    prob_start = cdf_y[:-1]
    prob_size = cdf_y[1:] - cdf_y[:-1]
    sample_prob = prob_start[ID] + sample * prob_size[ID]
    deq_col = inverse_cdf(sample_prob)
    return deq_col


def dequantilize_dataset(dataset_name, type='uniform'):
    assert type in ['spline', 'uniform']
    f = spline_dequantize if type == 'spline' else uniform_dequantize
    try:
        deq_df = load_table(f'{dataset_name}-{type}')
    except FileNotFoundError:
        table = load_table(dataset_name)
        data, cate_map = discretize_dataset(table)
        for idx, col_name in enumerate(table.columns):
            print(f'{type} dequantilizing {col_name} {idx + 1}/{len(table.columns)}')
            data[:, idx] = f(data[:, idx])
        deq_df = pd.DataFrame(data=data, columns=table.columns)
        deq_df.to_csv(os.path.join(DATA_PATH, f'{dataset_name}-{type}.csv'), index=False)
    return deq_df

