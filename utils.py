import time

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self, sec):
        if sec < 0:
            sec = "%0.2f" % (sec * 100)
            return sec + " msec"
        elif sec < 60:
            sec = "%0.4f" % sec
            return sec + " sec"
        elif sec < (60 * 60):
            sec = "%0.4f" % (sec / 60)
            return sec + " min"
        else:
            sec = "%0.4f" % (sec / (60 * 60))
            return sec + " hr"
    def elapsed_time(self):
        print("Speed: %s " % self.elapsed(time.time() - self.start_time))


class Settings(object):
    def __init__(self):
        self.ydim = 960
        self.xdim = 540
        self.channels = 3        
        self.model_weights = None

