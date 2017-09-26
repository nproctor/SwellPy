import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from .particle_suspension import *


class Monodisperse(ParticleSuspension):
    def __init__(self, N, boxsize=None, seed=None):
        """
        Create a particle suspension object.

        Parameters
        ----------
            N: int
                The number of particles in the system
            seed: int, optional
                Seed for initial particle placement randomiztion
        """
        super(Monodisperse, self).__init__(N, boxsize, seed)
        self._name = "Monodisperse"
    
    def equiv_swell(self, area_frac):
        af = np.array(area_frac, ndmin=1)
        return 2 * np.sqrt(af * self.boxsize**2 / (self.N * np.pi))

    def equiv_area_frac(self, swell):
        d = np.array(swell, ndmin=1)
        return (d / 2)**2 * (self.N * np.pi) / self.boxsize**2

    def _tag(self, swell):
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
        pairs = np.array(list(pairs), dtype=np.int64)
        return pairs
    
    def tag(self, area_frac):
        swell = self.equiv_swell(area_frac)
        return self._tag(swell)
    
    def repel(self, pairs, area_frac, kick):
        swell = self.equiv_swell(area_frac)
        self._repel(pairs, swell, kick)


    def train(self, area_frac, kick, cycles=np.inf):
        """
        Repeatedly tags and repels overlapping particles until particles
        no longer touch
        
        Parameters
        ----------
            area_frac: float
                Total area of the swollen particles relative to the boxsize
            kick: float
                The maximum distance particles are repelled relative to the diameter
                of the particles (i.e. a kick of 0.6 is a 60% particle diameter)
            cycles: int
                The upper bound on the number of cycles. Defaults to infinite.

        Returns
        -------
            cycles: int
                The number of tagging and repelling cycles until no particles overlapped
        """
        count = 0
        swell = self.equiv_swell(area_frac)
        pairs = self._tag(swell)
        while (cycles > count and (len(pairs) > 0) ):
            self._repel(pairs, swell, kick)
            self.wrap()
            pairs = self._tag(swell)
            count += 1
        return count


    def particle_plot(self, area_frac, show=True, extend = False, figsize = (7,7), filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Parameters
        ----------
            swell: float
                The diameter length at which the particles are illustrated
            show: bool, default True
                Display the plot after generation
            extend: bool, default False
                Show wrap around the periodic boundary. "Original" particles appear darker.
            figsize: tuple of ints, default (7,7)
                Scales the size of the figure
            filename: string, default None
                Destination to save the plot. If None, the figure is not saved. 
        """
        radius = self.equiv_swell(area_frac)/2
        boxsize = self.boxsize
        fig = plt.figure(figsize = figsize)
        plt.axis('off')
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = radius))
            if (extend):
                ax.add_artist(Circle(xy=(pair) + [0, boxsize], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [boxsize, 0], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [boxsize, boxsize], radius = radius, alpha=0.5))
        if (extend):
            plt.xlim(0, 2*boxsize)
            plt.ylim(0, 2*boxsize)
            plt.plot([0, boxsize*2], [boxsize, boxsize], ls = ':', color = '#333333')
            plt.plot([boxsize, boxsize], [0, boxsize*2], ls = ':', color = '#333333')

        else:
            plt.xlim(0, boxsize)
            plt.ylim(0, boxsize)
        fig.tight_layout()
        if filename != None:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def _tag_count(self, swells):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles

        Returns
        -------
            out: float
                The fraction of overlapping particles
        """
        i = 0
        tagged = np.zeros(swells.size)
        while i < swells.size:
            temp = self._tag(swells[i])
            tagged[i] = np.unique(temp).size/ self.N
            i += 1
        return tagged
    
    def tag_count(self, area_frac):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Parameters
        ----------
            swell: float
                Swollen diameter length of the particles

        Returns
        -------
            out: float
                The fraction of overlapping particles
        """
        swells = self.equiv_swell(area_frac)
        return self._tag_count(swells)

    def _extend_domain(self, domain):
        first = 2 * domain[0] - domain[1]
        if (first < 0):
            first = 0
        last = 2 * domain[-1] - domain[-2]
        domain_extend = np.insert(domain, 0, first)
        domain_extend = np.append(domain_extend, last)
        return domain_extend

    
    def tag_rate(self, area_frac):
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
        af_extended = self._extend_domain(area_frac)
        tagged = self.tag_count(af_extended)
        rate = (tagged[2:] - tagged[:-2])
        return rate

    def tag_curve(self, area_frac):
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
        af_extended = self._extend_domain(area_frac)
        rate = self.tag_rate(af_extended)
        curve = (rate[2:] - rate[:-2])
        return curve

    def tag_plot(self, func, show=True, filename=None):
        area_frac = np.arange(0.1, 1.0, 0.01)
        data = func(area_frac) 
        plt.plot(area_frac, data)
        plt.xlabel("Area Fraction")
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def detect_memory(self, incr):
        """
        Tests the number of tagged particles over a range of swells, and 
        returns a list of swells where memories are detected. 
        
        Parameters
        ----------
            incr: float
                The increment between test swells. Determines accuracy of the
                memory detection. 
        Returns
        -------
            swells: a list of swells where a memory is located
        """
        high = self.percent_to_diameter(1.0)
        low = self.percent_to_diameter(0.05)
        incr = 1/(incr*self.N)
        [swells, curve] = self.tag_curve(low, high, incr)
        zeros = np.zeros(curve.shape)
        pos = np.choose(curve < 0, [curve, zeros])
        neg = np.choose(curve > 0, [curve, zeros])
        indices = peak.indexes(pos, 0.5, incr)
        nindices = peak.indexes(-neg, 0.5, incr)
        matches = []
        for i in indices:
            for j in nindices:
                desc = True
                if (i < j):
                    for k in range(i,j):
                        if (curve[k] < curve[k+1]):
                            desc = False
                    if (desc):
                        matches.append(i)
        return swells[matches]