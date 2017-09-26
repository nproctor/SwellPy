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
            print("Warning: centers out of bounds (0, %0.2f)" %self.boxsize)
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