from MemoryRecognition import *

class StatisticalRecognition(MemoryRecognition):
    def __init__(self):
        super(StatisticalRecognition, self).__init__()

    def tagCurveNoise(self, Min, Max, incr, iters, seed=None):
        working_max = []
        self.system.reset(seed)
        for i in range(iters):
            (ignore, curve) = self.tagCurvature(Min, Max, incr)
            working_max.append(max(curve))
            self.system.reset(seed)
        mean = np.mean(working_max)
        sd = np.std(working_max)
        return mean, sd

    def tagRateNoise(self, Min, Max, incr, iters, seed=None):
        working_max = []
        self.system.reset(seed)
        for i in range(iters):
            (ignore, rate) = self.tagRate(Min, Max, incr)
            working_max.append(max(rate))
            self.system.reset(seed)
        mean = np.mean(working_max)
        sd = np.std(working_max)
        return mean, sd