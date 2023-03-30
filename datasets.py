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



def get_dataset_from_named(name, dequantilize_type='spline', load_to_device=None, re_dequantilize=False):
    T = TimeTracker()
    if name in ['lineitem-numetric', 'lineitem-categorical', 'lineitem']:
        table = load_table(name)
        T.report_interval_time_ms("Loading table")
        data, cate_map = discretize_dataset(table)
        T.report_interval_time_ms("discretizing")
        data = dequantilize_dataset(name, dequantilize_type, remake=re_dequantilize).to_numpy().astype(np.float32)
        T.report_interval_time_ms("dequantilize")
        _mean, _std = data.mean(0).reshape([1, -1]), data.std(0).reshape([1, -1])
        T.report_interval_time_ms("cal mean, std")
        data = (data - _mean) / (_std + eps)
    
    elif 'order' in name:
        table = load_table(name)
        T.report_interval_time_ms("Loading table")
        data, cate_map = discretize_dataset(table)
        T.report_interval_time_ms("discretizing")
        data = dequantilize_dataset(name, dequantilize_type, remake=False).to_numpy().astype(np.float32)
        T.report_interval_time_ms("dequantilize")
        _mean, _std = data.mean(0).reshape([1, -1]), data.std(0).reshape([1, -1])
        T.report_interval_time_ms("cal mean, std")
        data = (data - _mean) / (_std + eps)
    
    else:
        try:
            table = load_table(name)
        except FileNotFoundError:
            raise ValueError('No such dataset')
        T.report_interval_time_ms("Loading table")
        data, cate_map = discretize_dataset(table)
        T.report_interval_time_ms("discretizing")
        data = dequantilize_dataset(name, dequantilize_type, remake=False).to_numpy().astype(np.float32)
        T.report_interval_time_ms("dequantilize")
        _mean, _std = data.mean(0).reshape([1, -1]), data.std(0).reshape([1, -1])
        T.report_interval_time_ms("cal mean, std")
        data = (data - _mean) / (_std + eps)
    
    T = TimeTracker()
    np.random.shuffle(data)
    T.report_interval_time_ms("shuffle")
    data = torch.from_numpy(data.astype(np.float32))
    if load_to_device is not None:
        data = data.to(load_to_device)
    split = int(data.shape[0] * 0.8)
    print(f'split dataset {name} in size train:{split}, val:{data.shape[0] - split}')
    return TableDataset(data[:split]), TableDataset(data[split:])



def get_dataloader_from_named(name, batch_size, dequantilize_type='spline', load_to_device=None, re_dequantilize=False):
    
    train_set, val_set = get_dataset_from_named(name, dequantilize_type=dequantilize_type, load_to_device=load_to_device)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # generator=torch.Generator(device=device)
        # pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size * 4,
        shuffle=False,
        drop_last=False,
        # generator=torch.Generator(device=device)
        # pin_memory=True
    )
    
    return train_loader, val_loader
