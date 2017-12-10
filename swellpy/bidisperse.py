import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from .particle_system import *


class Bidisperse(ParticleSystem):
    def __init__(self, N1, N2, ratio=0.8, seed=None):
        """
        Create a particle suspension object with two disctinct
        particle sizes. 

        Parameters
        ----------
            N1 (int): The number of one size of particles in the system
            N2 (int): The number of the other size of particles
            ratio (float): ratio between diameters (d of N1 / d of N2)
            seed: int, optional
                Seed for initial particle placement randomiztion
        """
        super(Bidisperse, self).__init__(N1 + N2)
        self.name = "Bidisperse"
        self._N1 = N1
        self._N2 = N2
        self._ratio = ratio

    def equiv_swell(self, area_frac):
        af = np.array(area_frac, ndmin=2)
        d1 = 2 * np.sqrt(af * self.boxsize**2 / (np.pi * (self._N1 * self._ratio**2 + self._N2)))
        return np.concatenate((d1*self._ratio, d1), axis = 0).T

    def tag(self, area_frac):
        swells = self.equiv_swell(area_frac)
        return self._tag(swells[0,0], swells[0,1])


    def _tag(self, swell1, swell2):
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
        tree = cKDTree(self.centers, boxsize = self.boxsize)
        pairs = tree.query_pairs(swell1)
        pairs = np.array(list(pairs), dtype=np.int64)
        if (len(pairs) > 0):
        # create an array of 0s and 1s where 0s represent large particles
        # only keep pairs from pairs1 that are composed of two N1 particles
            pairs1 = pairs[np.sum(pairs < self._N1, axis=1) == 2]
        else:
            pairs1 = np.array([])

        pairs = tree.query_pairs(swell2)
        pairs = np.array(list(pairs), dtype=np.int64)
        if (len(pairs) > 0):
        # similarly only keep pairs from pairs2 that are composed of two N2 particles
            pairs2 = pairs[np.sum(pairs > self._N1, axis=1) == 2]
        else:
            pairs2 = np.array([])

        # keep pairs that contain one N1 and one N2
        pairs = tree.query_pairs((swell1 + swell2)/2)
        pairs = np.array(list(pairs), dtype=np.int64)
        if (len(pairs) > 0):
            pairsBoth = pairs[np.sum(pairs < self._N1, axis=1) == 1]
        else:
            pairsBoth = np.array([])

        return [pairs1, pairs2, pairsBoth]


    def train(self, area_frac, kick, cycles = np.inf):
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
        i = 0
        [[swell1, swell2]] = self.equiv_swell(area_frac)
        [pairs1, pairs2, pairsOther] = self._tag(swell1, swell2)
        while ( (len(pairs1) + len(pairs2) + len(pairsOther)) > 0 and (cycles > i) ):
            print("%10d tagged\r" %(len(pairs1)+len(pairs2)+len(pairsOther)), end="")
            if (len(pairs1) > 0):
                self._repel(pairs1, swell1, kick)
            if (len(pairs2) > 0):
                self._repel(pairs2, swell2, kick)
            if (len(pairsOther) > 0):
                self._repel(pairsOther, (swell1 + swell2)/2, kick)
            self.wrap()
            [pairs1, pairs2, pairsOther] = self._tag(swell1, swell2)
            i += 1
        return i

    def particle_plot(self, area_frac, extend=False, figsize=(6,6), show=True, filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Args:
            l_af (float): area fraction of the larger particles
            sm_af (float): Diameter of the smaller particles
            extend (bool): optional. Show wrap around the periodic boundary. Defaults to false.
            figsize ((int, int)): optional. Size of figure. Defaults to (7,7).
            show (bool): optional. Display the plot after generation. Defaults to true.
            filename (string): optional. Destination to save the plot. If None, the figure is not saved. 
        """
        i = 0
        [[swell1, swell2]] = self.equiv_swell(area_frac)
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
            if (i < self._N1):
                r = swell1/2
            else:
                r = swell2/2
            ax.add_artist(Circle(xy=(pair), radius = r))
            if (extend == True) :
                ax.add_artist(Circle(xy=(pair + [0, self.boxsize]), radius = r, alpha=0.5))
                ax.add_artist(Circle(xy=(pair + [self.boxsize, 0]), radius = r,alpha=0.5))
                ax.add_artist(Circle(xy=(pair + [self.boxsize, self.boxsize]), radius = r,alpha=0.5))
            i += 1
        fig.tight_layout()
        if save == True:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()