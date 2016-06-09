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


def plotAll():
	files = [file for file in os.listdir("../Data") if file.startswith("fracReachSwell")]
	colors = list(col.cnames.values())
	for file in files:
		random.seed(float(file[-8:-5]))
		c = random.choice(colors)
		f = sr.load(file)
		plt.plot(f[0,:], f[1,:], color=c)
		plt.title(file)
		plt.xlim(0, 2.0)
		plt.ylim(0, 4.0)
		plt.xlabel("Training Swell")
		plt.ylabel("Fraction Found Swell")
	plt.show()
	plt.close()



if __name__ == "__main__":
	#collect([0.99], 1000, 0.2, 0.1, 1.9, 0.01, 0.1, 4)
	plotAll()