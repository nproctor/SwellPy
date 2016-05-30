import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
import crepel
import pickle
import os

class ParticleSuspension:
    def __init__(self, N, areaFrac, seed=None):
        self.N = N
        self.areaFrac = areaFrac
        self.boxsize = self.__setBoxsize(N, areaFrac)
        self.centers = None
        self.reset(seed)
        self.recognition = None

    def __setBoxsize (self, N, areaFrac):
        """ Length of the sides of the 2-D box determined by
        the number of particles (N) and area fraction of the particles (areaFrac) """
        return np.sqrt(N*np.pi/(4*areaFrac))

    def reset (self, seed=None):
        """ Randomly positions the particles inside the box. Returns a
        N x 2 numpy array of the x and y components for every particles """
        if ( isinstance(seed, int) ):
            np.random.seed(seed)
        self.centers = np.random.uniform(0, self.boxsize, (self.N, 2))


    def setCenters(self, centers):
        """ Manually set the position of particles in the box. Takes a N x 2 array-like object.
        Raises an exception if centers are not of proper format. Raises a warning if the particles
        are placed outside of the box """
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


    def plot(self, swell = 1.0, show=True, save=False, filename=None):
        """ Shows plot of particles at a specific swell """
        fig = plt.figure()
        plt.title("Particle position")
        plt.xlim(0, self.boxsize)
        plt.ylim(0, self.boxsize)
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = swell/2))
        if save == True:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()


    def tag(self, swell):
        """ Tags overlapping particles at a specific swell. Returns a 
        M x 2 numpy array filled with the indices of tagged particles
        where M is the number of tagged pairs. """
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

    def tagFracAt(self, swell):
        pairs = self.tag(swell)
        return len(np.unique(pairs)) / self.N

    # Query for number of particles tagged for a range of shears
    def tagFrac(self, Min, Max, incr):
        swells = np.arange(Min, Max + incr, incr)
        tagged = np.array(list(map(lambda x: self.tagFracAt(x), swells)))
        return swells, tagged

    
    def tagRate(self, Min, Max, incr):
        (ignore, tagged) = self.tagFrac(Min-incr/2, Max+incr/2, incr)
        swells = np.arange(Min, Max + incr, incr)
        rate = ( tagged[1:] - tagged[:-1] ) / incr
        return swells, rate

    def tagCurvature(self, Min, Max, incr):
        (ignore, tagRate) = self.tagRate(Min-incr/2, Max+incr/2, incr)
        swells = np.arange(Min, Max + incr, incr)
        curve = ( tagRate[1:] - tagRate[:-1] ) / incr
        return swells, curve

    def plotTagFrac(self, Min, Max, incr, show=True, save=False, filename=None):
        (swells, tag) = self.fracTag(Min, Max, incr)
        fig = plt.figure()
        plt.title("Fraction of tagged particles")
        plt.xlabel("Swell")
        plt.plot(swells, tag)
        if save == True:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def plotTagRate(self, Min, Max, incr, show=True, save=False, filename=None):
        (swells, rate) = self.tagRate(Min, Max, incr)
        fig = plt.figure()
        plt.title("Particles tag rate")
        plt.xlabel("Swell")
        plt.xlim(0, Max)
        plt.ylim(0, 15)
        plt.plot(swells, rate)
        if save == True:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def plotTagCurve(self, Min, Max, incr, show=True, save=False, filename=None):
        (swells, curve) = self.tagCurvature(Min, Max, incr)
        fig = plt.figure()
        plt.title("Particle tag curvature")
        plt.xlabel("Swell")
        plt.xlim(0, Max)
        plt.ylim(-600, 600)
        plt.plot(swells, curve)
        if save == True:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()


def save(system, filename):
    f = open(filename, "wb")
    pickle.dump(system, f)
    f.close()

def load(filename):
    f = open(filename, "rb")
    x = pickle.load(f)
    f.close()
    return x
