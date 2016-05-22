class MemoryRecognition:
    def __init__(self):
        self.system = None

    def newSystem(self, N, areaFrac, seed=None):
        x = ParticleSystem(N, areaFrac, seed)
        self.system = x

    # Query for number of particles at specific swell
    def numTagAt(self, swell):
        pairs = self.system.tag(swell)
        return len(np.unique(pairs)) / self._N

    # Query for number of particles tagged for a range of shears
    def numTagFor(self, Min, Max, incr):
        swells = np.arange(Min, Max, incr)
        tagged = np.array(list(map(lambda x: self.numTagAt(x), swells)))
        return tagged

