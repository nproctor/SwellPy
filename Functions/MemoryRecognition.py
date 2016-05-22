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
    def fracTagFor(self, Min, Max, incr):
        swells = np.arange(Min, Max, incr)
        tagged = np.array(list(map(lambda x: self.fracTagAt(x), swells)))
        return tagged

