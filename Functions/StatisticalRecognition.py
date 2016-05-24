import numpy as np
from ParticleSuspension import *
from scipy.optimize import curve_fit
import os

class StatisticalRecognition:
    def __init__(self):
        self.meanRateParams = None
        self.sdRateParams = None
        self.meanCurveParams = None
        self.sdCurveParams = None

    # This is the second derivative of the fraction of tagged particles in an untrained system
    # Returns the mean and standard deviation for a system of size N
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

    # This is the second derivative of the fraction of tagged particles in an untrained system
    # Returns a list of mean and standard deviation values that correspond to the 
    # inputed "Nlist" ( a list of number of particles )
    # NOTE: this iters is divided by the number of particles in the system so should be
    # at least as big as the number of particles in your system
    def curveNoiseCollect(self, Nlist, Min, Max, incr, iters, areaFrac=0.2, seed=None):
        means = []
        sds = []
        for N in Nlist:
            (mean, sd) = self.tagCurveNoise(N, Min, Max, incr, int(iters/N))
            means.append(mean)
            sds.append(sd)
        return means, sds

    # This is the first derivative of the fraction of tagged particles in an untrained system
    # Returns the mean and standard deviation for a system of size N
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

    # This is the first derivative of the fraction of tagged particles in an untrained system
    # Returns a list of mean and standard deviation values that correspond to the 
    # inputed "Nlist" ( a list of number of particles )
    # NOTE: this iters is divided by the number of particles in the system so should be
    # at least as big as the number of particles in your system
    def rateNoiseCollect(self, Nlist, Min, Max, incr, iters, areaFrac=0.2, seed=None):
        means = []
        sds = []
        for N in Nlist:
            (mean, sd) = self.tagRateNoise(N, Min, Max, incr, int(iters/N))
            means.append(mean)
            sds.append(sd)
        return means, sds

    def fitFunc(self, x, a, b, c):
        return a*x**(b) + c


    # Fits noise data to fitFunc, returns the parameters to be used in fitFunc
    def noiseFit(self, N, means, sds):
        MeanParams, MeanCovar = curve_fit(self.fitFunc, N, means, bounds=([-np.inf, -np.inf, -np.inf],[np.inf, 0, np.inf]))
        SdParams, SdCovar = curve_fit(self.fitFunc, N, sds, bounds=([-np.inf, -np.inf, -np.inf],[np.inf, 0, np.inf]))
        return MeanParams, SdParams

    # Uses the fit params to produce expected mean noise and
    # standard deviation of noise
    def expectedNoise(self, N, meanParams, sdParams):
        mean = self.fitFunc(N, *meanParams)
        sd = self.fitFunc(N, *sdParams)
        return mean, sd

    # "folder" parameter must be inside of SwellPy/data
    # Can be used for param data or collect data
    def save(self, filename, data, header, folder=None):
        os.chdir("../data")
        if isinstance(folder, str):
            os.chdir(folder)
        data = np.array(data).T
        np.savetxt(filename, data, header=header, fmt = "%10.5lf")

    # "folder" parameter must be inside of SwellPy/data
    # Can be used for param data or collect data
    def load(self, filename, folder=None):
        os.chdir("../data")
        if isinstance(folder, str):
            os.chdir(folder)
        data = np.loadtxt(filename)
        x = data.T
        return x