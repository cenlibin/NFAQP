from time import time


class TimeTracker:
    def __init__(self):
        self._st = time()

    def report_interval_time_ms(self, msg=''):
        et = time()
        t = et - self._st
        t *= 1000
        self._st = et
        print('{}:{:.4f} ms'.format(msg, t))
        return t

