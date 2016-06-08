import numpy as np
from ParticleSuspension import *
import StatisticalRecognition as sr


def evolution(N, areaFrac, swell, kick, addCycles, iterations):
	""" 
	Saves the plots of actual particle positions, the fraction of particles tagged,
	the rate of tagged particles, and the curvature of tagged particles with respect
	to the swell diameter. The system is swelled and repelled addCycles times before
	a plot is saved. The number of plots saved is equivalent to the number of iterations.

	Parameters 
	----------
			N: int
				The number of particles in the system
			areaFrac: float
				Total particle area to box area ratio at swell of 1.0
			swell: float
				Swollen diameter of the particles
			addCycles: int
				The number of training cycles between plot saves
			iterations: int
				Total number of times plots are generated
	"""
	x = ParticleSuspension(N, areaFrac)
	for i in range(iterations):
		x.trainFor(swell, kick, addCycles)
		x.plot(swell, show=False, save=True, filename="Plots/Plot%d.png" %i)
		x.plotTagFrac(0, 3.0, 0.01, show = False, save=True, filename="FracTag%d.png" %i )
		x.plotTagRate(0, 3.0, 0.01, show = False, save=True, filename="TagRate%d.png" %i )
		x.plotTagCurve(0, 3.0, 0.01, show = False, save=True, filename="TagCurve%d.png" %i )

def swellAtFrac(Nfrac, N, areaFrac, Min, Max, incr, kick):
	""" 
	Find the swell diameter at which a specific fraction of particles are tagged. The system
	is trained on swells [Min, Max] by steps of incr. After each training, the swell at which
	Nfrac fraction of particles are tagged is stored. 
	
	Parameters
	----------
		Nfrac: float
			The fraction of tagged particles 
		N: int
			The number of particles in the system
		areaFrac: float
			Total particle area to box area ratio at swell of 1.0
		Min: float
			The smallest training swell
		Max: float
			The largest training swell (inclusive)
		incr: float
			Increment of training swells
		kick: float
			The max values particles are repelled if tagged 

	Returns
	------- 
		out: list of tuple
			The first element in each tuple is the swell the system was trained
			on. The second element is the swell at which the specified fraction of 
			tagged particles occurs. 
	"""	
	trainOn = np.linspace(Min, Max, (Max-Min)/incr + 1)
	NfracSwell = []
	x = ParticleSuspension(N, areaFrac)
	f = 1.0
	for swell in trainOn:
		x.train(swell, kick)
		while x.tagFracAt(f) != Nfrac:
			f += 0.01
		NfracSwell.append(f)
		f = 0
	return list(zip(trainOn,NfracSwell))


def cyclesToRecognize(N, areaFrac, kick, Min, Max, incr, cycleIncr, tolerance):
	""" 
	Find the number of cycles needed to detect a significant memory over
	a range of swell diameters
	
	Parameters
	----------
		N: int
			The number of particles in the system
		areaFrac: float
			Total particle area to box area ratio at swell of 1.0
		kick: float
			The max values particles are repelled if tagged 
		Min: float
			The smallest training swell
		Max: float
			The largest training swell (inclusive)
		incr: float
			Increment between training swells
		cycleIncr: int
			The number of cycles the system is swelled and repelled
			before checking again for memories
		tolerance: float
			The maximum error allowed when declaring a match between the 
			expected peak location and the actual peak location.

	Returns
	------- 
		out: (M x 2) numpy array
			The first element in each element of the array is the swell
			diameter and the second is the number of cycles it took to form
			a significant memory at that swell. M is the number of swells in 
			the range of swells.  
	"""	
	found = []
	(mean, sd) = loadCurveNoiseMeanSD(N, "curveNoiseParams.txt")
	x = ParticleSuspension(N, areaFrac)
	swells = np.arange(Min, Max + incr, incr)
	cycles = 0
	for swell in swells:
		while x.trainFor(swell, kick, cycle) != 0:
			cycles += cycle
			guess = isSignificant(mean, sd, x)
			if isEqual(swell, guess, tolerance)
				found.append([swell, cycles])
				break
		x.reset()
		cycles = 0
	return np.array(found)


def randTrain(system, swell, kick, maxCycles, seed):
	if isinstance(seed, int):
		np.random.seed(seed)
	cycles = np.random.randint(maxCycles)
	system.trainFor(swell, kick, cycles)
	return cycles

def isEqual(expected, true, tolerance):
	expected.sort()
	true.sort()
	if len(expected) != len(true):
		return False
	for i in range(len(expected)):
		if abs(expected[i] - true[i]) > tolerance:
			return False
	return True

def loadCurveNoiseMeanSD(N, filename)
	(meanParams, sdParams) = sr.load(filename)
	(mean, sd) = sr.expectedNoise(N, meanParams, sdParams)
	return mean, sd


def ranRecogOne(N, areaFrac, kick, Min, Max, incr, maxCycles, tolerance):
	accuracy = []
	(mean, sd) = loadCurveNoiseMeanSD(N, "curveNoiseParams.txt")
	x = ParticleSuspension(N, areaFrac)
	swells = np.arange(Min, Max + incr, incr)
	for swell in swells:
		cycles = randTrain(x, swell, kick, maxCycles, seed=None)
		guess = isSignificant(mean, sd, x)
		if isEqual([swell], guess)
			accuracy.append([swell, cycles, 1])
		else:
			accuracy.append([swell, cycles, 0])
		x.reset()
	return np.array(accuracy)


def ranRecogTwo(N, areaFrac, trainSwell, trainCycles, kick, Min, Max, incr, minRan, maxRan):
	accuracy = []
	(meanParams, sdParams) = sr.load("curveNoiseParams.txt")
	(mean, sd) = sr.expectedNoise(N, meanParams, sdParams)
	x = ParticleSuspension(N, areaFrac)
	swells = np.arange(Min, Max + incr, incr)
	for swell in swells:
		x.trainFor(trainSwell, kick, trainCycles)
		cycle = np.random.randint(minRan, maxRan)
		x.trainFor(swell, kick, cycle)
		guess = isSignificant(mean, sd, x)
		true = np.array([swell, trainSwell])
		true.sort()
		if len(guess) == 2:
			if abs(true[0] - guess[0]) < 0.00001 and abs(true[1] - guess[1]) < 0.00001 :
				accuracy.append([swell, cycle, 1])
			else:
				accuracy.append([swell, cycle, 0])
		else:
			accuracy.append([swell, cycle, 0])
		x.reset()
	return np.array(accuracy)


			

def isSignificant(mean, sd, system):
	sig = []
	(swells, curve) = system.tagCurve(0, 3.0, 0.01)
	for i in range(len(curve)):
		if curve[i] > ( 2*mean):
			sig.append(swells[i])
	return sig

