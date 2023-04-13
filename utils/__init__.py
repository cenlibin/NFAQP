from .utils import *
from .preprocess import discretize_dataset, dequantilize_dataset, uniform_dequantize, spline_dequantize
from .mertic import *
from .time_tracker import TimeTracker

table_size = {
    "lineitem":29999795,
    'pm25':462538,
    'flights': 5726566,
    'housing': 318356
}