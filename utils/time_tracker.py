from time import time


class TimeTracker:
    def __init__(self):
        self._st = time()

    def reset(self):
        self._st = time()

    def report_interval_time_ms(self, msg=''):
        et = time()
        t = et - self._st
        t *= 1000
        self._st = et
        print('{}:{:.4f} ms'.format(msg, t))
        return t

    def report_interval_time_sec(self, msg=''):
        et = time()
        t = et - self._st
        self._st = et
        print('{}:{:.4f} sec'.format(msg, t))
        return t

    def report_interval_time(self, msg=''):
        et = time()
        t = et - self._st
        self._st = et
        if t < 1:
            p = 'ms'
            t *= 1000
        else:
            p = 'sec'
        print('{}:{:.4f} {}'.format(msg, t, p))
