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