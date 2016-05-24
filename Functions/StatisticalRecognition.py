import numpy as np
from ParticleSuspension import *
from scipy.optimize import curve_fit

class StatisticalRecognition:
    def __init__(self):
        self.meanRateParams = None
        self.sdRateParams = None
        self.meanCurveParams = None
        self.sdCurveParams = None


    def tagCurveNoise(self, N, Min, Max, incr, iters, areaFrac=0.2, seed=None):
        working_max = []
        x = ParticleSuspension(N, areaFrac, seed)
        for i in range(iters):
            (ignore, curve) = x.tagCurvature(Min, Max, incr)
            working_max.append(max(curve))
            x.reset(seed)
        mean = np.mean(working_max)
        sd = np.std(working_max)
        return mean, sd

    def curveNoiseCollect(self, Nlist, Min, Max, incr, iters, areaFrac=0.2, seed=None):
        means = []
        sds = []
        for N in Nlist:
            (mean, sd) = self.tagCurveNoise(N, Min, Max, incr, iters)
            means.append(mean)
            sds.append(sd)
        return means, sds

    def tagRateNoise(self, N, Min, Max, incr, iters, areaFrac=0.2, seed=None):
        working_max = []
        x = ParticleSuspension(N, areaFrac, seed)
        for i in range(iters):
            (ignore, rate) = x.tagRate(Min, Max, incr)
            working_max.append(max(rate))
            x.reset(seed)
        mean = np.mean(working_max)
        sd = np.std(working_max)
        return mean, sd

    def rateNoiseCollect(self, Nlist, Min, Max, incr, iters, areaFrac=0.2, seed=None):
        means = []
        sds = []
        for N in Nlist:
            (mean, sd) = self.tagRateNoise(N, Min, Max, incr, iters)
            means.append(mean)
            sds.append(sd)
        return means, sds

    def fitFunc(x, a, b, c):
        return a*x**(b) + c

    def noiseFit(self, N, means, sds):
        MeanParams, MeanCovar = curve_fit(self.fitFunc, N, mean, bounds=([-np.inf, -np.inf, -np.inf],[np.inf, 0, np.inf]))
        SdParams, SdCovar = curve_fit(self.fitFunc, N, sd, bounds=([-np.inf, -np.inf, -np.inf],[np.inf, 0, np.inf]))
        return MeanParams, SdParams

    def expectedNoise(sef, N, meanParams, sdParams):
        mean = fitFunc(N, *meanParams)
        sd = fitFunc(N, *sdParams)
        return mean, sd


    