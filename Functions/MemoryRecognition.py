class MemoryRecognition:
    def __init__(self):
        self.system = None

    def newSystem(self, N, areaFrac, seed=None):
        x = ParticleSystem(N, areaFrac, seed)
        self.system = x

    # Query for number of particles tagged at a specific swell
    def numTagAt(self, swell):
        pairs = self.system.tag(swell)
        return len(np.unique(pairs)) / self._N

