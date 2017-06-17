import cycles_for_recognition as cfr 
import matplotlib.pyplot as plt

def main():
	# Run for 100 particles
	# and a sample scaling of 1 (this is about 10 samples)
	result = cfr.detect(100, 1)

	# Plot
	plt.plot(result[:,2], result[:,1])
	plt.xlabel("Area Fraction")
	plt.ylabel("Cycles")
	plt.title("Cycles Needed to Implant a Detectable Memory")
	plt.show()


if __name__ == "__main__":
	main()