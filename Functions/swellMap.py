from ParticleSuspension import *


def detect(N, sp, filename):
	f = open(filename, "w")
	f.write("swell, cycles, density\n")
	m = monodisperse(N)
	x = m.percent_to_diameter(0.50)
	xMax = m.percent_to_diameter(1.00)
	incr = 1/(sp*N)
	kick = 1/N
	while (x <= xMax):
		[m, cycles] = train_detectable(m, sp, x)
		density = m.diameter_to_percent(x)
		f.write("%0.5lf, %d, %0.5lf\n" %(x, cycles, density))
		m.reset()
		x += incr
	f.close()


def train_detectable(m, sp, swell):
	cycles = 0
	N = m.N
	kick = 1/N
	incr = 1/(sp*N)
	while (cycles <= N):
		mem = m.detect_memory(sp)
		if (mem.size == 1 and np.isclose(mem, swell, atol = incr).any()):
			return m, cycles
		elif (m.train_for(swell, kick, 1) == 0):
			return m, -1
		cycles += 1
	return m, -1

def detect_only_2(N, sp, filename):
	f = open(filename, "w")
	f.write("swell_1, cycles_1, swell_2, cycles_2\n")
	cycles = 0
	m = monodisperse(N)
	xMin = m.percent_to_diameter(0.60)
	xMax = m.percent_to_diameter(0.90)
	x1 = xMax
	x2 = xMin
	incr = 1/(sp*N)
	kick = 1/N
	while (x1 <= xMax):
		if (x1 == x2):
			x1 += incr
			x2 = xMin
		[m, cycles] = train_detectable(m, sp, x1)
		f.write("%0.04lf, %d," %(x1, cycles))
		#print(x1, cycles, end=" ")
		if (cycles == -1):
			f.write("-1, -1\n")
			x1 += incr
			continue
		[m, cycles] = train_more_detectable(m, sp, np.array([x1]), x2)
		f.write("%0.04lf, %d," %(x2, cycles))
		#print(x2, cycles)
		m.reset()
		x2 += incr;
	f.close()

def train_more_detectable(m, sp, swells_ignore, swell):
	cycles = 0
	N = m.N
	kick = 1/N
	incr = 1/(sp*N)
	size = swells_ignore.size + 1
	while (cycles <= N/2):
		mem = m.detect_memory(sp)
		for ig_swell in swells_ignore:
			if (not np.isclose(mem, ig_swell, atol = incr).any()):
				return m, -1
		if (mem.size == size and np.isclose(mem, swell, atol = incr).any()):
			return m, cycles
		elif (m.train_for(swell, kick, 1) == 0):
			return m, -1
		cycles += 1
	return m, -1






if __name__ == "__main__":
	detect_only(1000, 5, "detect_5_1000_only.txt")
	# m = monodisperse(1000)
	# x = m.percent_to_diameter(0.75)
	# print(train_detectable(m, 2, x)[1])
	# print(train_more_detectable(m, 2, np.array([x]), m.percent_to_diameter(0.72))[1])


