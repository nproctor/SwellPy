from ParticleSuspension import *

"""
Collects data on how many cycles are needed at a specific
swell until the memory is "detectable". 
"""

def detect(N, sp):
	"""
    Collects data on the number of cycles needed until
    a memory is detectable for a range of swells. 

    Parameters
    ----------
        N: int
            The number of particles in the system
        sp: int
            Scales the sample size. A larger "sp" will collect
            more samples. 
    Returns
    -------
    	3x1 array with columns corresponding to swell, cycles, 
    	and area fraction
     """
	m = monodisperse(N)
	x = m.percent_to_diameter(0.01)
	xMax = m.percent_to_diameter(1.00)
	incr = 1/(sp*N)
	kick = 1/N
	result = []
	while (x <= xMax):
		[m, cycles] = train_detectable(m, sp, x)
		density = m.diameter_to_percent(x)
		result.append([x, cycles, density])
		m.reset()
		x += incr
	return np.asarray(result)


def train_detectable(m, sp, swell):
	"""
    Implants a memory one cycle at a time until the memory
    is detectable

    Parameters
    ----------
        m: monodisperse object
        	The monodisperse system of interest
        sp: int
            Scales the sample size. A larger "sp" will collect
            more samples. This MUST be the same size as the "sp"
            parameter in the detect function above if they are being
            used together. 
        swell: float
        	The current swell of interest
    Returns
    -------
    	monodisperse object and 
    	the number of cycles (or -1 if "undetectable")
     """
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