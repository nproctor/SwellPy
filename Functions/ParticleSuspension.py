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
        self.centers = None
        self.reset(seed)


    # Sets the boxsize based on the number of paticles and areafraction
    def __setBoxsize (self, N, areaFrac):
        return np.sqrt(N*np.pi/(4*areaFrac))

    # randomly places the particles inside the box
    def reset (self, seed=None):
        if ( isinstance(seed, int) ):
            np.random.seed(seed)
        self.centers = np.random.uniform(0, self.boxsize, (self.N, 2))

    # Manually set the particle placement
    # DOES check for correct format
    # Does NOT check for placed inside of box
    def setCenters(self, centers):
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers, dtype=np.float64)
        try:
            if (centers.shape) != (self.N, 2):
                raise Exception("Error: Centers must be a (%s, 2) array-like object" %self.N)

            if ( (centers < 0).any() or (centers > self.boxsize).any() ):
                print("Warning: centers out of bounds (0, %0.2f)" %self.boxsize)

            self.centers = centers
        except Exception as e:
            print(e)

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
        #pairs = pairs.astype(np.int64)
        return pairs

    def repel(self, pairs, swell, kick):
        if not isinstance(pairs, np.ndarray):
            pairs = np.array(pairs, dtype=np.int64)
        pairs = pairs.astype(np.int64) # This is necessary for the c extension
        centers = self.centers
        boxsize = self.boxsize
        # Fill the tagged pairs with the center coordinates
        fillTagged = np.take(centers, pairs, axis=0)
        # Find the position of the second particle in the tagged pair
        # with respect to the first 
        separation = np.diff(fillTagged, axis=1)
        # Account for tagged across periodic bounary
        np.putmask(separation, separation > swell, separation - boxsize)
        # Normalize
        norm = np.linalg.norm(separation, axis=2).flatten()
        unitSeparation = (separation.T/norm).T
        # Generate kick
        kick = (unitSeparation.T * np.random.uniform(0, kick, unitSeparation.shape[0])).T
        # Since the separation is with respect to the 'first' particle in a pair, 
        # apply positive kick to the 'second' particle and negative kick to the first
        crepel.iterate(centers, pairs[:,1], kick, pairs.shape[0])
        crepel.iterate(centers, pairs[:,0], -kick, pairs.shape[0])
        # Note: this may kick out of bounds -- be sure to wrap!

    def wrap(self):
        centers = self.centers
        boxsize = self.boxsize
        # Wrap if outside of boundaries
        np.putmask(centers, centers>=boxsize, centers-boxsize)
        np.putmask(centers, centers<0, centers+boxsize)


    def train(self, swell, kick):
        i = 0
        pairs = self.tag(swell)
        while ( len(pairs) > 0 ):
            self.repel(pairs, swell, kick)
            self.wrap()
            pairs = self.tag(swell)
            i += 1
        return i


    def trainFor(self, swell, kick, cycles):
        i = 0
        pairs = self.tag(swell)
        while (i < cycles) and ( len(pairs) > 0 ):
            self.repel(pairs, swell, kick)
            self.wrap()
            pairs = self.tag(swell)
            i += 1
        return i

