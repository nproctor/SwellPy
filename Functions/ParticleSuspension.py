import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
import crepel
import pickle
import os

class ParticleSuspension:
    def __init__(self, N, areaFrac, seed=None):
        """
        Create a particle suspension object.

        Parameters
        ----------
            N: int
                The number of particles in the system
            areaFrac: float
                Total particle area to box area ratio at swell of 1.0 (original radius)
            seed: int, optional
            	Seed for initial particle placement randomiztion
        """
        self.N = N
        self.areaFrac = areaFrac
        self.boxsize = self.__setBoxsize(N, areaFrac)
        self.centers = None
        self.reset(seed)
        self.recognition = None
        if not os.path.exists("../Plots"):
            os.mkdir("../Plots")

    def __setBoxsize (self, N, areaFrac):
        """ 
        Length of the sides of the 2-D box determined by the number of particles
        and the area fraction of the particles. Do not directly call this function 
        """
        return np.sqrt(N*np.pi/(4*areaFrac))

    def reset (self, seed=None):
        """ Randomly positions the particles inside the box.
        
        Parameters
        ----------
            seed: int, optional
                The seed to use for randomization 
        """
        if ( isinstance(seed, int) ):
            np.random.seed(seed)
        self.centers = np.random.uniform(0, self.boxsize, (self.N, 2))


    def setCenters(self, centers):
        """ Manually set the position of particles in the box. Raises an exception if 
        centers are not of proper format. Raises a warning if the particles
        are placed outside of the box
        
        Parameters
        ----------
            centers: (N x 2) array-like
                An array or array-like object containing the x, y coordinates 
                of each particle 
        """
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers, dtype=np.float64)
        if (centers.shape) != (self.N, 2):
            raise TypeError("Centers must be a (%s, 2) array-like object" %self.N)
        if ( (centers < 0).any() or (centers > self.boxsize).any() ):
            print("Warning: centers out of bounds (0, %0.2f)" %self.boxsize)
        self.centers = centers


    def plot(self, swell, show=True, save=False, filename="ParticlePlot.png"):
        """
        Show plot of physical particle placement in 2-D box 
        
        Parameters
        ----------
            swell: float
                The diameter length at which the particles are illustrated
            show: bool, default True
                Display the plot after generation
            save: bool, default True
                Save the plot after generation (default False)
            filename: string, default None
                Destination to save the plot if save is True 
        """
        fig = plt.figure()
        plt.title("Particle position")
        plt.xlim(0, self.boxsize)
        plt.ylim(0, self.boxsize)
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = swell/2))
        if save == True:
            plt.savefig("../Plots/" + filename)
        if show == True:
            plt.show()
        plt.close()


    def tag(self, swell):
        """ 
        Get the center indices of the particles that overlap at a 
        specific swell
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles

        Returns
        -------
            pairs: (M x 2) numpy array 
                An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """

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
        """ 
        Repels particles that overlap
        
        Parameters
        ----------
            pairs: (M x 2) array-like object
                An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
            swell: float
                Swollen diameter length of the particles
            kick: float
                The maximum distance particles are repelled 
        """
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
        """
        Applied periodic boundaries to any particles outside of the box. 
        Does not work if particles are outside of the box more than 1x
        the length of the box. 
        """
        centers = self.centers
        boxsize = self.boxsize
        # Wrap if outside of boundaries
        np.putmask(centers, centers>=boxsize, centers-boxsize)
        np.putmask(centers, centers<0, centers+boxsize)


    def train(self, swell, kick):
        """
        Repeatedly tags and repels overlapping particles until swollen particles
        no longer touch
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles
            kick: float
                The maximum distance particles are repelled

        Returns
        -------
            cycles: int
                The number of tagging and repelling cycles until no particles overlapped
        """
        cycles = 0
        pairs = self.tag(swell)
        while ( len(pairs) > 0 ):
            self.repel(pairs, swell, kick)
            self.wrap()
            pairs = self.tag(swell)
            cycles += 1
        return cycles


    def trainFor(self, swell, kick, cycles):
        """ 
        Tag and repel overlapping particles for a specific number of cycles

        Parameters 
        ----------
            swell: float
                Swollen diameter length of the particles
            kick: float
                The maximum distance particles are repelled
            cycles: int
                The max number of cycles particles are tagged and repelled
        
        Returns: 
        -------
            trueCycles: int
                The actual number of cycles the particles are tagged and repelled. 
                Will differ from parameter "cycles" if particles no longer overlapped in a fewer number
                of cycles.
        """          
        trueCycles = 0
        pairs = self.tag(swell)
        while (trueCycles < cycles) and ( len(pairs) > 0 ):
            self.repel(pairs, swell, kick)
            self.wrap()
            pairs = self.tag(swell)
            trueCycles += 1
        return trueCycles

    def tagFracAt(self, swell):
        """
        Returns the fraction of particles that overlap at a specific swell diameter
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles

        Returns
        -------
            out: float
                The fraction of overlapping particles
        """
        pairs = self.tag(swell)
        return len(np.unique(pairs)) / self.N

    def tagFrac(self, Min, Max, incr):
        """
        Return the fraction of particles that are tagged over a range of swell
        diameters.
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles
            Min: float
                The minimum swollen diameter length
            Max: float
                The maximum swollen diameter length, inclusive
            incr: float
                The step size of diameter length when increasing from Min to Max

        Returns
        -------
            swells: float array-like
                The swollen diameter lengths at which the fraction of tagged particles
                is recorded
            tagged: float array-like
                The fraction of particles tagged at each swell diameter in the return 
                object "swells" respectively
        """
        swells = np.arange(Min, Max+incr, incr)
        tagged = np.array(list(map(lambda x: self.tagFracAt(x), swells)))
        return swells, tagged

    
    def tagRate(self, Min, Max, incr):
        """
        Returns the rate at which the fraction of particles overlap over a range of diameters.
        This is the same as measuring the fraction tagged at two swells and dividing by the difference
        of the swells. 
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles
            Min: float
                The minimum swollen diameter length
            Max: float
                The maximum swollen diameter length, inclusive
            incr: float
                The step size of diameter length when increasing from Min to Max

        Returns
        -------
            swells: float array-like
                The swollen diameter lengths at which the rate of tagged particles
                is recorded
            rate: float array-like
                The rate of the fraction of tagged particles at each swell diameter 
                in the return object "swells" respectively
        """
        (ignore, tagged) = self.tagFrac(Min-incr/2, Max+incr/2, incr)
        swells = np.arange(Min, Max + incr, incr)
        rate = ( tagged[1:] - tagged[:-1] ) / incr
        return swells, rate

    def tagCurve(self, Min, Max, incr):
        """
        Returns the curvature at which the fraction of particles overlap over a range of diameters.
        This is the same as measuring the rate at two swells and dividing by the difference
        of the swells. 
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles
            Min: float
                The minimum swollen diameter length
            Max: float
                The maximum swollen diameter length, inclusive
            incr: float
                The step size of diameter length when increasing from Min to Max

        Returns
        -------
            swells: float array-like
                The swollen diameter lengths at which the fraction of tagged particles
                is recorded
            curve: float array-like
                The change in the fraction of tagged particles at each swell diameter 
                in the return object "swells" respectively
        """
        (ignore, tagRate) = self.tagRate(Min-incr/2, Max+incr/2, incr)
        swells = np.arange(Min, Max + incr, incr)
        curve = ( tagRate[1:] - tagRate[:-1] ) / incr
        return swells, curve

    def plotTagFrac(self, Min, Max, incr, show=True, save=False, filename="TagFracPlot.png"):
        (swells, tag) = self.tagFrac(Min, Max, incr)
        fig = plt.figure()
        plt.title("Fraction of tagged particles")
        plt.xlabel("Swell")
        plt.plot(swells, tag)
        if save == True:
            plt.savefig("../Plots/" + filename)
        if show == True:
            plt.show()
        plt.close()

    def plotTagRate(self, Min, Max, incr, show=True, save=False, filename="TagRatePlot.png"):
        (swells, rate) = self.tagRate(Min, Max, incr)
        fig = plt.figure()
        plt.title("Particles tag rate")
        plt.xlabel("Swell")
        plt.xlim(0, Max)
        plt.ylim(0, 15)
        plt.plot(swells, rate)
        if save == True:
            plt.savefig("../Plots/" + filename)
        if show == True:
            plt.show()
        plt.close()

    def plotTagCurve(self, Min, Max, incr, show=True, save=False, filename="TagCurvePlot.png"):
        (swells, curve) = self.tagCurve(Min, Max, incr)
        fig = plt.figure()
        plt.title("Particle tag curvature")
        plt.xlabel("Swell")
        plt.xlim(0, Max)
        plt.ylim(-600, 600)
        plt.plot(swells, curve)
        if save == True:
            plt.savefig("../Plots/" + filename)
        if show == True:
            plt.show()
        plt.close()


def save(system, filename):
    if not os.path.exists("../ParticleCache"):
        os.mkdir("../ParticleCache")
    f = open("../ParticleCache/" + filename, "wb")
    pickle.dump(system, f)
    f.close()

def load(filename):
    f = open("../ParticleCache/" + filename, "rb")
    x = pickle.load(f)
    f.close()
    return x
