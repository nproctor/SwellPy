import Analysis
import StatisticalRecognition as sr
import os
from matplotlib import pyplot as plt
import matplotlib.colors as col
import random

def collect(fracs, N, AF, Min, Max, incr, kick, iterAll):
	for i in range(iterAll):
		for frac in fracs:
			if os.path.exists("../Data/fracReachSwell%d_%0.2lf.txt" %(i, frac)):
				print("fracReachSwell%d_%0.2lf.txt already exists." %(i, frac))
				continue
			a = Analysis.findFracTagSwell(frac, N, AF, Min, Max, incr, kick)
			sr.save("fracReachSwell%d_%0.2lf.txt" %(i, frac), a, str(N) + " particles, " 
					+ str(AF) + " areaFraction" + "\nFraction=" + str(frac) + "\nTrainOnSwell, FractionReachedSwell")



if __name__ == "__main__":
	#collect([0.99], 1000, 0.2, 0.1, 1.9, 0.01, 0.1, 4)
	plotAll()