import numpy as np
from ParticleSuspension import *


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
	Returns: 
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