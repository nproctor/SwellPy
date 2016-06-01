from StatisticalRecognition import *
import numpy as np

def rateData(Nlist, areaFrac, Min, Max, incr, iterPerN, dataFilename = "rateNoiseData.txt"):
	"""
	Generate the mean noise rate values for a range of particle system sizes. 

	Parameters
	----------
		Nlist: 1D int array-like object
			A list or array-like object of ints that represent the number of particles
			for which data will be collected
		areaFrac: float
				Total particle area to box area ratio at swell of 1.0
		Min: float
            The minimum swollen diameter length
        Max: float
            The maximum swollen diameter length, inclusive
        incr: float
            The step size of diameter length when increasing from Min to Max
        iterPerN: int
        	The number of times noise values are taken as if it were a single paticle 
        	system. The number of times values are measured for each system size
        	is avgIters/N
        dataFilename: str, optional
        	The name of the file where the data will be stored. Default is "rateNoiseData.txt"

	"""
	x = StatisticalRecognition()
	(mean, sd) = x.rateNoiseCollect(Nlist, Min, Max, incr, iterPerN)
	header = "Areafrac: %0.4lf, Incr: %0.4lf, IterPerN: %d\nN,mean,sd" %(areaFrac, incr, iterPerN)
	x.save(dataFilename, [Nlist, mean, sd], header)


def rateParams(areaFrac, incr, dataFilename = "rateNoiseData.txt", paramFilename = "rateNoiseParams.txt"):
	"""
	Generate the rate noise fit parameters from the mean noise data. 

	Parameters
	----------
		areaFrac: float
				Total particle area to box area ratio at swell of 1.0
        incr: float
            The step size of diameter length when increasing from Min to Max
        dataFilename: str, optional
        	The name of the file where the data is be stored. Default is "rateNoiseData.txt"
        paramFilename: str, optional
        	The name of the file where the parameters will be stored. Default is "rateNoiseParams.txt"

	"""
	x = StatisticalRecognition()
	(Nlist, means, sds) = x.load(dataFilename)
	[meanParams, sdParams] = x.noiseFit(Nlist, means, sds)
	header = "Params for Rate Noise (AF: %0.4lf, Incr: %0.4lf)\nMeanParams SDParams" %(areaFrac, incr)
	x.save(paramFilename, [meanParams, sdParams], header)

def curveData(Nlist, areaFrac, Min, Max, incr, iterPerN, dataFilename = "curveNoiseData.txt"):
	"""
	Generate the mean noise curve values for a range of particle system sizes. 

	Parameters
	----------
		Nlist: 1D int array-like object
			A list or array-like object of ints that represent the number of particles
			for which data will be collected
		areaFrac: float
				Total particle area to box area ratio at swell of 1.0
		Min: float
            The minimum swollen diameter length
        Max: float
            The maximum swollen diameter length, inclusive
        incr: float
            The step size of diameter length when increasing from Min to Max
        iterPerN: int
        	The number of times noise values are taken as if it were a single paticle 
        	system. The number of times values are measured for each system size
        	is avgIters/N
        dataFilename: str, optional
        	The name of the file where the data will be stored. Default is "curveNoiseData.txt"

	"""
	x = StatisticalRecognition()
	(mean, sd) = x.curveNoiseCollect(Nlist, Min, Max, incr, iterPerN)
	header = "Areafrac: %0.4lf, Incr: %0.4lf, IterPerN: %d\nN,mean,sd" %(areaFrac, incr, iterPerN)
	x.save(dataFilename, [Nlist, mean, sd], header)


def curveParams(areaFrac, incr, dataFilename = "curveNoiseData.txt", paramFilename = "curveNoiseParams.txt"):
	"""
	Generate the curve noise fit parameters from the mean noise data. 

	Parameters
	----------
		areaFrac: float
				Total particle area to box area ratio at swell of 1.0
        incr: float
            The step size of diameter length when increasing from Min to Max
        dataFilename: str, optional
        	The name of the file where the data is be stored. Default is "curveNoiseData.txt"
        paramFilename: str, optional
        	The name of the file where the parameters will be stored. Default is "curveNoiseParams.txt"

	"""
	x = StatisticalRecognition()
	(Nlist, means, sds) = x.load(dataFilename)
	[meanParams, sdParams] = x.noiseFit(Nlist, means, sds)
	header = "Params for Curve Noise (AF: %0.4lf, Incr: %0.4lf)\nMeanParams SDParams" %(areaFrac, incr)
	x.save(paramFilename, [meanParams, sdParams], header)