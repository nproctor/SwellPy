from ParticleSuspension import *

class MemoryRecognition:
    def __init__(self):
        self.system = None

    def newSystem(self, N, areaFrac, seed=None):
        x = ParticleSuspension(N, areaFrac, seed)
        self.system = x

    # Query for number of particles at specific swell
    def fracTagAt(self, swell):
        pairs = self.system.tag(swell)
        return len(np.unique(pairs)) / self.system.N

    # Query for number of particles tagged for a range of shears
    def fracTag(self, Min, Max, incr):
        swells = np.arange(Min, Max + incr, incr)
        tagged = np.array(list(map(lambda x: self.fracTagAt(x), swells)))
        return swells, tagged

    
    def tagRate (self, Min, Max, incr):
    	(ignore, tagged) = self.fracTag(Min-incr/2, Max+incr/2, incr)
    	swells = np.arange(Min, Max, incr)
    	rate = ( tagged[1:] - tagged[:-1] ) / incr
    	return swells, rate

    def tagCurvature(self, Min, Max, incr):
    	(ignore, tagRate) = self.tagRate(Min-incr/2, Max+incr/2, incr)
    	swells = np.arange(Min, Max, incr)
    	curve = ( tagRate[1:] - tagRate[:-1] ) / incr
    	return swells, curve
    		