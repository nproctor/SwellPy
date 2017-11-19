import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import time
import pickle
import crepel

class ParticleSuspension():
    def __init__(self, N, boxsize=None, seed=None):
        self._name = "ParticleSuspension"
        self.N = N
        if (boxsize):
            self.boxsize = boxsize
        else:
            self.boxsize = np.sqrt(N*np.pi/(4 * 0.2))
        self.centers = None
        self.reset(seed)
    

    def reset(self, seed=None):
        """ Randomly positions the particles inside the box.
        
        Parameters
        ----------
            seed: int, optional
                The seed to use for randomization 
        """
        if (isinstance(seed, int)):
            np.random.seed(seed)
        self.centers = np.random.uniform(0, self.boxsize, (self.N, 2))
    
    def wrap(self):
        """
        Applied periodic boundaries to any particles outside of the box. 
        """
        centers = self.centers
        boxsize = self.boxsize
        # Wrap if outside of boundaries
        np.putmask(centers, centers>=boxsize, centers % boxsize)
        np.putmask(centers, centers<0, centers % boxsize)

    
    def _repel(self, pairs, swell, kick):
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
        np.putmask(separation, separation < -swell, separation + boxsize)
        # Normalize
        norm = np.linalg.norm(separation, axis=2).flatten()
        unitSeparation = (separation.T/norm).T
        # Generate kick
        kick_arr = (unitSeparation.T * np.random.uniform(0, kick, unitSeparation.shape[0])).T
        # Since the separation is with respect to the 'first' particle in a pair, 
        # apply positive kick to the 'second' particle and negative kick to the first
        crepel.iterate(centers, pairs[:,1], kick_arr, pairs.shape[0])
        crepel.iterate(centers, pairs[:,0], -kick_arr, pairs.shape[0])
        # Note: this may kick out of bounds -- be sure to wrap!
    
    def set_centers(self, centers):
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
            raise TypeError("Error: Centers must be a (%s, 2) array-like object" %self.N)
        if ( (centers < 0).any() or (centers > self.boxsize).any() ):
            raise RuntimeError("Warning: centers out of bounds (0, %0.2f)" %self.boxsize)
        self.centers = centers

def save(system, filename = None):
    """
    Pickles the current particle suspension. Filename is generated
    from the day, time, number of particles.

    Returns
    -------
        cycles: int
            The number of tagging and repelling cycles until no particles overlapped
    """
    if (filename):
        f = open(filename, "wb")
    else:
        f = open(system.name + "_%d_%s_%s.p" 
            %(system.N, time.strftime("%d-%m-%Y"), time.strftime("%H.%M.%S")), "wb")
    pickle.dump(system, f)
    f.close()

def load(filename):
    """
    Loads a pickled file from the current directory
    
    Parameters
    ----------
        filename: string
            The name of the pickled particle file

    Returns
    -------
        The particle suspension
    """
    f = open(filename, "rb")
    x = pickle.load(f)
    f.close()
    return x