import datetime
import shutil
from nflows import utils
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
# from utils  import DataPrefetcher, discretize, DataWrapper, uniformDequantize, LoadTable, DATA_PATH
from utils  import *
from utils import *
eps = 1e-5

class TableDataset(Dataset):
    def __init__(self, data):
        self.n, self.dim = data.shape
        self.data = data
    
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n

def get_dataset_from_named(name, dequantilize_type='spline', load_to_device=None):
    T = TimeTracker()
    if name in ['lineitem-numetric', 'lineitem']:
        table = load_table(name)
        T.report_interval_time_ms("Loading table")
        data, cate_map = discretize_dataset(table)
        T.report_interval_time_ms("discretizing")
        data = dequantilize_dataset(name, dequantilize_type).to_numpy().astype(np.float32)
        T.report_interval_time_ms("dequantilize")
        _mean, _std = data.mean(0).reshape([1, -1]), data.std(0).reshape([1, -1])
        T.report_interval_time_ms("cal mean, std")
        data = (data - _mean) / (_std + eps)
        
        
    elif 'power' in name:
        data = load_table(name).to_numpy().astype(np.float32)
        

    elif 'BJAQ' in name:
        table = load_table("BJAQ")
        data = table.to_numpy().astype(np.float32)
        data = dequantilize_dataset(name, dequantilize_type).to_numpy().astype(np.float32)
        _mean, _std = data.mean(0).reshape([1, -1]), data.std(0).reshape([1, -1])
        data = (data - _mean) / (_std + eps)
        
    elif 'random' in name:
        table = load_table('random')
        data, cate_map = discretize_dataset(table)
        data = dequantilize_dataset(name, dequantilize_type).to_numpy().astype(np.float32)
        T.report_interval_time_ms("dequantilize")
        _mean, _std = data.mean(0).reshape([1, -1]), data.std(0).reshape([1, -1])
        T.report_interval_time_ms("cal mean, std")
        data = (data - _mean) / (_std + eps)
        
    elif 'order' in name:
        table = load_table(name)
        T.report_interval_time_ms("Loading table")
        data, cate_map = discretize_dataset(table)
        T.report_interval_time_ms("discretizing")
        data = dequantilize_dataset(name, dequantilize_type).to_numpy().astype(np.float32)
        T.report_interval_time_ms("dequantilize")
        _mean, _std = data.mean(0).reshape([1, -1]), data.std(0).reshape([1, -1])
        T.report_interval_time_ms("cal mean, std")
        data = (data - _mean) / (_std + eps)


    else:
        raise ValueError('No such dataset')
    
    T = TimeTracker()
    np.random.shuffle(data)
    T.report_interval_time_ms("shuffle")
    data = torch.from_numpy(data.astype(np.float32))
    if load_to_device is not None:
        data = data.to(load_to_device)
    split = int(data.shape[0] * 0.8)
    print(f'split dataset {name} in size train:{split}, val:{data.shape[0] - split}')
    return TableDataset(data[:split]), TableDataset(data[split:])




