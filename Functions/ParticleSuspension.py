import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
import crepel

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
                raise Exception("Error: Centers must be a (%s, 2) array-like object" %self.N)

            if ( (centers < 0).any() or (centers > self.boxsize).any() ):
                print("Warning: centers out of bounds (0, %0.2f)" %self.boxsize)

            self.centers = centers
        except Exception as e:
            print(e)

    # All else equal, re-randomizes centers
    def reset(self, seed = None):
        if ( isinstance(seed, int)):
            np.random.seed(seed)
        self.boxsize = self.__setBoxsize(self.N, self.areaFrac)
        self.centers = self.__setCenters(self.N, self.boxsize, seed)


    # To scale plot of particle position
    def plot(self):
        fig = plt.figure()
        plt.xlim(0, self.boxsize)
        plt.ylim(0, self.boxsize)
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = 1.0))
        plt.show()

    # "Tag" overlapping particles
    def tag(self, swell):
        # Note cKD can retun numpy arrays in query pairs
        # but there is a deallocation bug in the scipy.spatial code
        # converting from a set to an array avoids it
        tree = cKDTree(self.centers, boxsize = self.boxsize)
        pairs = tree.query_pairs(swell)
        # If bug is fixed, remove the following line ...
        pairs = np.array(list(pairs), dtype=np.int64)
        # ... and uncomment the following line 
        #pairs.dtype = np.int64
        return pairs
