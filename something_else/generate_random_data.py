import torch
import numpy as np
import pandas as pd


if __name__ == '__main__':
    data = np.random.randn(100000, 6)
    df = pd.DataFrame(data, columns=[f'c{i}' for i in range(6)])
    df.to_csv('/home/clb/AQP/data/random.csv', index=False)

