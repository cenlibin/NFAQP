import numpy as np
import torch
from matplotlib import pyplot as plt
from utils  import * 
from datasets import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from time import sleep

def f():
    N = 20000
    a = torch.randn(N, N).cuda()
    b = torch.rand_like(a).cuda()
    c = torch.randn(N, N).cpu()
    d = torch.rand_like(a).cpu()
    while True:
        a @ b
        c, d = torch.randn(N, N).cpu(), torch.randn(N, N).cpu()
        sleep(1)
    

if __name__ == '__main__':

    # train, val = get_dataset_from_name("lineitem")
    f()

    #data = np.array([0, 2, 4, 7, 2, 7, 0, 9, 7])
    # data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #deq1 = spline_dequantize(data)
    # deq2 = uniform_dequantize(data)

    # dataset = get_dataset_from_name("lineitem")

    # data = LoadTable('BJAQ').to_numpy()
    # for i in range(data.shape[1]):
    #     spline_dequantize(data[:, i])

    pass
    

