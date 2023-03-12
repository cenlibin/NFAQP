import pandas as pd 
import numpy as np
import os, sys
sys.path.append('/home/clb/AQP')
from utils import spline_dequantize
a = np.array([26, 30, 40, 55])
b = np.array([3500, 4000, 5000, 4500])
c = np.array([0, 1, 0, 1])
da = spline_dequantize(a)
db = spline_dequantize(b)
dc = spline_dequantize(c)
pass