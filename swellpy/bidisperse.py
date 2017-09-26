import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from .particle_suspension import *


class Bidisperse(ParticleSuspension):
    def __init__(self, N, mod, seed=None):
        """
        Create a particle suspension object with two disctinct
        particle sizes. 

        Parameters
        ----------
            N: int
                The number of particles in the system
            mod: int
                The inverse of the fraction of particles that are large.
                If set to 1, all particles are large. If larger than N, 
                no particles are large.
            seed: int, optional
                Seed for initial particle placement randomiztion
        """
        super(Bidisperse, self).__init__(N)
        self._name = "Bidisperse"
        self.mod = mod


    def tag(self, l_swell, sm_swell):
        """ 
        Get the center indices of the particles that overlap at a 
        specific swell. 
        
        Parameters
        ----------
            l_swell: float
                Swollen diameter length of the larger particles
            sm_swell: float
                Swollen diameter length of the smaller particles

        Returns
        -------
            pairs: (M x 2) numpy array 
                An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """

        # Note cKD can retun numpy arrays in query pairs
        # but there is a deallocation bug in the scipy.spatial code
        # converting from a set to an array avoids it
        mod = self.mod
        tree = cKDTree(self.centers, boxsize = self.boxsize)
        pairs = tree.query_pairs(l_swell)
        pairs = np.array(list(pairs), dtype=np.int64)
        if (len(pairs) > 0):
            # create an array of 0s and 1s where 0s represent large particles
            # only keep pairs from l_pairs that are composed of two large particles
            l_pairs = pairs[np.sum(pairs%mod == 0, axis=1) == 2]
        else:
            l_pairs = np.array([])

        pairs = tree.query_pairs(sm_swell)
        pairs = np.array(list(pairs), dtype=np.int64)
        if (len(pairs) > 0):
            # similarly only keep pairs from sm_pairs that are composed of two small particles
            sm_pairs = pairs[np.sum(pairs%mod == 0, axis=1) == 0]
        else:
            sm_pairs = np.array([])

        pairs = tree.query_pairs((sm_swell+l_swell)/2)
        pairs = np.array(list(pairs), dtype=np.int64)
        if (len(pairs) > 0):
            m_pairs = pairs[np.sum(pairs%mod == 0, axis=1) == 1]
        else:
            m_pairs = np.array([])

        return [l_pairs, m_pairs, sm_pairs]


    def train(self, l_swell, sm_swell, kick):
        """
        Repeatedly tags and repels overlapping particles until swollen particles
        no longer touch
        
        Parameters
        ----------
            l_swell: float
                Swollen diameter length of the larger particles
            sm_swell: float
                Swollen diameter length of the smaller particles
            kick: float
                The maximum distance particles are repelled

        Returns
        -------
            cycles: int
                The number of tagging and repelling cycles until no particles overlapped
        """
        cycles = 0
        [l_pairs, m_pairs, sm_pairs] = self.tag(l_swell, sm_swell)
        while ( (len(l_pairs) + len(m_pairs) + len(sm_pairs)) > 0 ):
            print("%10d tagged\r" %(len(l_pairs)+len(m_pairs)+len(sm_pairs)), end="")
            if (len(l_pairs) > 0):
                self.repel(l_pairs, l_swell, kick)
            if (len(sm_pairs) > 0):
                self.repel(sm_pairs, sm_swell, kick)
            if (len(m_pairs) > 0):
                self.repel(m_pairs, (l_swell + sm_swell)/2, kick)
            self.wrap()
            [l_pairs, m_pairs, sm_pairs] = self.tag(l_swell, sm_swell)
            cycles += 1
        return cycles

    def particle_plot(self, l_swell, sm_swell, extend=False, figsize=(7,7), show=True, filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Parameters
        ----------
            l_swell: float
                Diameter of the larger particles
            sm_swell: float
                Diameter of the smaller particles
            extend: bool, default False
                Show wrap around the periodic boundary. "Original" particles appear darker.
            figsize: tuple of ints, default (7,7)
                Scales the size of the figure
            show: bool, default True
                Display the plot after generation
            filename: string, default None
                Destination to save the plot. If None, the figure is not saved. 
        """
        i = 0
        fig = plt.figure(figsize=figsize)
        if (extend == True):
            plt.xlim(0, 2*self.boxsize)
            plt.ylim(0, 2*self.boxsize)
        else:
            plt.xlim(0, self.boxsize)
            plt.ylim(0, self.boxsize)
        ax = plt.gca()
        ax.axis('off')
        for pair in self.centers:
            if (i % self.mod == 0):
                r = l_swell/2
            else:
                r = sm_swell/2
            ax.add_artist(Circle(xy=(pair), radius = r))
            if (extend == True) :
                ax.add_artist(Circle(xy=(pair + [0, self.boxsize]), radius = r, alpha=0.5))
                ax.add_artist(Circle(xy=(pair + [self.boxsize, 0]), radius = r,alpha=0.5))
                ax.add_artist(Circle(xy=(pair + [self.boxsize, self.boxsize]), radius = r,alpha=0.2))
            i += 1
        fig.tight_layout()
        if save == True:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()