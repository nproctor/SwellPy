import numpy as np
from ParticleSuspension import *


def particleEvolution(N, areaFrac, swell, kick, addCycles, iterations):
	""" Saves the plots of actual particle positions, tagFrac, tagRate, and tagCurvature
	after a set number of cycles for a set number of iterations into Plots folder
	Parameters: N (int) - Number of particles
				areaFrac (double) - total particle area to box area at swell of 1.0
				swell (double) - diameter of the particles
				addCycles (int) - number of training cycles between plot saves
				iterations (int) - number of times plots are generated"""
	x = ParticleSuspension(N, areaFrac)
	for i in range(iterations):
		x.trainFor(swell, kick, addCycles)
		x.plot(swell, show=False, save=True, filename="Plots/Plot%d.png" %i)
		x.plotTagFrac(0, 3.0, 0.01, show = False, save=True, filename="Plots/FracTag%d.png" %i )
		x.plotTagRate(0, 3.0, 0.01, show = False, save=True, filename="Plots/TagRate%d.png" %i )
		x.plotTagCurve(0, 3.0, 0.01, show = False, save=True, filename="Plots/TagCurve%d.png" %i )

