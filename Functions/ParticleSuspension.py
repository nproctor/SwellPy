import numpy as np

class ParticleSuspension:
    def __init__(self, N, areaFrac, seed=None):
        self.N = N
        self.areaFrac = areaFrac
        self.boxsize = self.__setBoxsize(N, areaFrac)
        self.centers = self.__setCenters(N, self.boxsize, seed)

    # Sets the boxsize based on the number of paticles and areafraction
    def __setBoxsize (self, N, areaFrac):
        return np.sqrt(N*np.pi/(4*areaFrac))

    # randomly places the particles inside the box
    def __setCenters (self, N, boxsize, seed):
        if ( isinstance(seed, int) ):
            np.random.seed(seed)
        return np.random.uniform(0, boxsize, (N, 2))

    # Manually set the particle placement
    # DOES check for correct format
    # Does NOT check for placed inside of box
    def setCenters(self, centers):
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        try:
            if (centers.shape) != (self.N, 2):
                raise Exception("Centers must be a (%s, 2) array-like object" %self.N)

            if ( (centers < 0).any() or (centers > self.boxsize).any() ):
                raise Exception("Centers out of bounds (0, %0.2f)" %self.boxsize)

            self.centers = centers
        except Exception as e:
            print(e)

    # All else equal, re-randomizes centers
    def reset(self, seed = None):
        if ( isinstance(seed, int)):
            np.random.seed(seed)
        self.boxsize = self.__setBoxsize(self.N, self.areaFrac)
        self.centers = self.__setCenters(self.N, self.boxsize, seed)