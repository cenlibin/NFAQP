from time import time


class TimeTracker:
    def __init__(self):
        self._st = time()

    def reportIntervalTime(self, msg=''):
        et = time()
        t = et - self._st
        self._st = et
        print('{}:{:.4f} ms'.format(msg, t * 1000))
        return t

