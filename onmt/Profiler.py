import os
import sys
import time

DEVNULL = open(os.devnull, "w")
STDOUT = sys.stdout


def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, 'took', end - start, 'time')
        return result
    return f_timer


def tabs(n=1):
    return "".join(["\t"]*n)


class Timer:
    def __init__(self, name, start=True, output=sys.stdout, prefix=""):
        self.name = name
        self.output = output
        self.stime = -1
        self.chkpt_time = -1
        self.chkpt_count = 0
        self.prefix = prefix
        if start:
            self.start()

    def print(self, *args, **kwargs):
        print(str(self.prefix), end=" ", file=self.output)
        print(*args, **kwargs, file=self.output)

    def start(self):
        self.stime = time.time()

    def chkpt(self, chkpt_name=None, append=None):
        """Print time without stopping
        """
        ctime = time.time()
        assert self.stime > 0, "Start timer before"

        name = "%s:%d" % (self.name, self.chkpt_count)
        if chkpt_name is not None:
            name = "%s:%s" % (name, chkpt_name)

        tot_time = ctime - self.stime
        if self.chkpt_count == 0:
            self.print("%s\ttime: %f" % (name, tot_time))
        else:
            chkpt_time = ctime - self.chkpt_time
            self.print("%s\ttime: %f\t[tot: %f]" %
                       (name, chkpt_time, tot_time))

        self.chkpt_time = ctime
        self.chkpt_count += 1
        if append is not None:
            self.print(str(append))

    def stop(self, name="END", append=None):
        """Print and stop
        """
        self.chkpt(name, append)
        self.etime = -1
        self.chkpt_count += 1
        self.chkpt_time = -1

    def reset(self):
        self.stop()
        self.start()
