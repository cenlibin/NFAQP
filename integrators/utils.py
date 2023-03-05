from time import time


def _add_at_indices(target, indices, source, is_sorted=False):
    target.scatter_add_(dim=0, index=indices, src=source)


def split_domain(domain):
    start = domain[:, 0]
    size = domain[:, 1] - domain[:, 0]
    volumn = size.prod(dim=0)
    return start, size, volumn


class TimeTracker:
    def __init__(self):
        self._st = time()

    def reportIntervalTime(self, msg=''):
        et = time()
        t = et - self._st
        self._st = et
        print('{} cost:{:.4f} ms'.format(msg, t * 1000))
        return t


# def create_new_map():
#     vegas = VEGAS()
#     bigN = 1000000 * 40
#     st = time()
#     res = vegas.integrate(
#         batchProbabilityDensityFunc, dim=dim_features,
#         N=bigN,
#         integration_domain=full_legal_domain,
#         use_warmup=True,
#         use_grid_improve=True,
#         max_iterations=40
#     )
#     print("First intergration tooks ", time() - st)
#     print(res)
#     res = res * data_wrapper.n
#     print('result is ', res)
#     _target_map = vegas.map
#     with open(REUSE_FILE_PATH + "{}.pickle".format(dataset_name), 'wb') as f:
#         pickle.dump(_target_map, f)
#     return _target_map


