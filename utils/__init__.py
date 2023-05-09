import os
# os.system("clear")
from .utils import *
from .preprocess import discretize_dataset, dequantilize_dataset, uniform_dequantize, spline_dequantize
from .mertic import *
from .time_tracker import TimeTracker

table_size = {
    "lineitem":6001215,
    "lineitemext":6001215,
    'pm25':10000000,
    'flights': 10000000,
    'housing': 318356
}