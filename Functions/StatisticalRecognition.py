import numpy as np
from ParticleSuspension import *
from scipy.optimize import curve_fit
import os


# This is the second derivative of the fraction of tagged particles in an untrained system
# Returns the mean and standard deviation for a system of size N
def tagCurveNoise(N, Min, Max, incr, iters, areaFrac=0.2, seed=None):
    working_max = []
    x = ParticleSuspension(N, areaFrac, seed)
    for i in range(iters):
        (ignore, curve) = x.tagCurve(Min, Max, incr)
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
def curveNoiseCollect(Nlist, Min, Max, incr, iters, areaFrac=0.2, seed=None):
    means = []
    sds = []
    for N in Nlist:
        (mean, sd) = tagCurveNoise(N, Min, Max, incr, int(iters/N))
        means.append(mean)
        sds.append(sd)
    return means, sds

# This is the first derivative of the fraction of tagged particles in an untrained system
# Returns the mean and standard deviation for a system of size N
def tagRateNoise(N, Min, Max, incr, iters, areaFrac=0.2, seed=None):
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
def rateNoiseCollect(Nlist, Min, Max, incr, iters, areaFrac=0.2, seed=None):
    means = []
    sds = []
    for N in Nlist:
        (mean, sd) = tagRateNoise(N, Min, Max, incr, int(iters/N))
        means.append(mean)
        sds.append(sd)
    return means, sds

def fitFunc( x, a, b, c):
    return a*x**(b) + c


# Fits noise data to fitFunc, returns the parameters to be used in fitFunc
def noiseFit(N, means, sds):
    MeanParams, MeanCovar = curve_fit(fitFunc, N, means, bounds=([-np.inf, -np.inf, -np.inf],[np.inf, 0, np.inf]))
    SdParams, SdCovar = curve_fit(fitFunc, N, sds, bounds=([-np.inf, -np.inf, -np.inf],[np.inf, 0, np.inf]))
    return MeanParams, SdParams

# Uses the fit params to produce expected mean noise and
# standard deviation of noise
def expectedNoise(N, meanParams, sdParams):
    mean = fitFunc(N, *meanParams)
    sd = fitFunc(N, *sdParams)
    return mean, sd

# Can be used for param data or collect data
def save(filename, data, header):
    if not os.path.exists("../Data"):
        os.mkdir("../Data")
    data = np.array(data).T
    np.savetxt("../Data/" + filename, data, header=header, fmt = "%10.5lf")

# Can be used for param data or collect data
def load(filename):
    data = np.loadtxt("../Data/" + filename)
    x = data.T
    return x