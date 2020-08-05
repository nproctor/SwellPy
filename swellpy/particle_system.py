import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import time
import pickle
import crepel

""" 
This class defines features that are applicable to both Monodisperse
and Bidisperse systems. It is not intended to be insantiated, but 
can be. It is meant as a building block for other types of particle
systems. Since this class consideres the size of particles to be 
undefined, this system cannot be trained, tagged or repelled. It
can set centers and wrap particles that are outside of the box. 
These operations are all independent of particle
size/shape. 

Note there are some ``private'' methods that are implemented in this
class that are used by child classes, but not by this class. 
Calling these methods from the ParticleSuspension class will not be
successful. 

``Save'' and ``Load'' functionality is also defined in this file
and is NOT a method of the ParticleSystem class.

"""

class ParticleSystem():
    def __init__(self, N, boxsize=None, seed=None):
        self._name = "ParticleSystem"
        self.N = N
        if (boxsize):
            self.boxsize = boxsize
        else:
            self.boxsize = np.sqrt(N*np.pi/(4 * 0.2))
        self.centers = None
        self.reset(seed)
    

    def reset(self, seed=None):
        """ Randomly positions the particles inside the box.
        
        Args:
            seed (int): optional. The seed to use for randomization 
        """
        if (isinstance(seed, int)):
            np.random.seed(seed)
        self.centers = np.random.uniform(0, self.boxsize, (self.N, 2))
    
    def wrap(self):
        """
        Applies periodic boundaries to any particles outside of the box. 
        """
        centers = self.centers
        boxsize = self.boxsize
        # Wrap if outside of boundaries
        np.putmask(centers, centers>=boxsize, centers % boxsize)
        np.putmask(centers, centers<0, centers % boxsize)

    
    def pos_noise(self, noise_type, noise_val):
        """
        adds random noise to the position of each particle, typically used before each swell
        
        Args: 
            noise_type: none for no noise
                        gauss for a gaussian distribution about the particle position,
                        drop for reset fraction of particles each cycle
            noise_val:  standard deviation for gauss
                        fraction of active particles for drop
        """
        if noise_type=='gauss':
            centers = self.centers
            boxsize = self.boxsize
            kicks = np.random.normal(0, noise_val, size=np.shape(centers))
            self.centers = centers+kicks
            pass
        elif noise_type=='drop':
            particles=len(self.centers)
            reset_indicies=[]
            options=np.linspace(0, particles-1, particles)
            options=options.astype(int)
            while len(reset_indicies)<(noise_val*particles):
                i=np.random.randint(0,len(options)-1)
                reset_indicies.append(options[i])
                np.delete(options, i)
            for i in reset_indicies:
                self.centers[i]=self.boxsize*np.random.random_sample((2,))
        else:
            pass
            
        
    
    def _repel(self, pairs, swell, kick):
        """ 
        Repels particles that overlap
        
        Args:
            pairs ((M x 2) np.array): An array object whose elements are pairs of int values that 
                correspond to the center indices of overlapping particles
            swell (float): Swollen diameter length of the particles
            kick (float): The maximum distance particles are repelled 
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
        
        Args:
            centers ((N x 2) array-like): An array or array-like object containing the x, y coordinates 
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
    from the day, time, number of particles, and type of system.

    Args:
        filename (string): optional. The filename used to save the system.
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
    
    Args:
        filename (string): The name of the pickled particle file

    Return:
        (ParticleSystem): The particle system
    """
    f = open(filename, "rb")
    x = pickle.load(f)
    f.close()
    return x
