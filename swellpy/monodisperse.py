import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from .particle_system import ParticleSystem


class Monodisperse(ParticleSystem):
    def __init__(self, N, boxsize=None, seed=None):
        """
        Args:
            N (int): The number of particles in the system
            boxsize (float): optional. Length of the sides of the box
            seed (int): optional. Seed for initial particle placement randomiztion
        """
        super(Monodisperse, self).__init__(N, boxsize, seed)
        self._name = "Monodisperse"
    
    def equiv_swell(self, area_frac):
        """
        Finds the particle diameter that is equivalent to some area fraction.

        Args:
            area_frac (float): the area fraction of interest
        Returns:
            (float): the equivalent diameter
        """
        af = np.array(area_frac, ndmin=1)
        return 2 * np.sqrt(af * self.boxsize**2 / (self.N * np.pi))

    def equiv_area_frac(self, swell):
        """
        Finds the area fraction that is equivalent to some some swell diameter.

        Args:
            swell (float): the particle diameter of interest
        Returns:
            (float) the equivalent area fraction
        """
        d = np.array(swell, ndmin=1)
        return (d / 2)**2 * (self.N * np.pi) / self.boxsize**2

    def _tag(self, swell):
        """ 
        Get the center indices of the particles that overlap at a 
        specific swell
        
        Parameters:
            swell (float): diameter length of the particles

        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
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
        """
        Finds all tagged particles at some area fraction.

        Args:
            area_frac (float): the area fraction of interest
        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """
        swell = self.equiv_swell(area_frac)
        return self._tag(swell)
    
    def repel(self, pairs, area_frac, kick):
        """
        Repels overlapping particles.

        Args:
            pairs (np.array): the pairs of overlapping particles
            area_frac (float): the area fraction of interest
            kick (float): the max kick value the particles are repelled as a percent of the
                inverse diameter
        """
        swell = self.equiv_swell(area_frac)
        self._repel(pairs, swell, kick)

    def train(self, area_frac, kick, cycles=np.inf, noise_type='none', noise_val=0, counter='kicks'):
        """
        Repeatedly tags and repels overlapping particles for some number of cycles
        
        Args:
            area_frac (float or list): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles. Defaults to infinite.
            noise (float): Value for standard deviation of gaussian noise to particle 
                position in each cycle, defaults to 0
            count (kicks or list): whether to count a cycle as one kick or 
                one run over the input list

        Returns:
            (int) the number of tagging and repelling cycles until no particles overlapped
        """
        if not (counter=='kicks' or counter=='list'):
            print('invalid counter parameter, no training performed')
            return
        
        if not type(area_frac) == list:
            area_frac = [area_frac]
        
        count = 0
        while (cycles > count):
            untagged = 0
            for frac in area_frac:
                self.pos_noise(noise_type, noise_val)
                self.wrap()
                swell = self.equiv_swell(frac)
                pairs = self._tag(swell)
                if len(pairs) == 0:
                    untagged += 1
                    continue
                self._repel(pairs, swell, kick)
                self.wrap()
                if counter == 'kicks':
                    count += 1
                    if count >= cycles:
                        break
            if counter == 'list':
                count += 1
            if (untagged == len(area_frac) and noise_val == 0):
                break
        return count


    def particle_plot(self, area_frac, show=True, extend = False, figsize = (7,7), filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Args:
            area_frac (float): The diameter length at which the particles are illustrated
            show (bool): default True. Display the plot after generation
            extend (bool): default False. Show wrap around the periodic boundary.
            figsize ((int,int)): default (7,7). Scales the size of the figure
            filename (string): optional. Destination to save the plot. If None, the figure is not saved. 
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
        
        Args:
            swell (float): swollen diameter length of the particles

        Returns:
            (float): The fraction of overlapping particles
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
        
        Args:
            area_frac (float): area fraction of the particles

        Returns:
            (float): The fraction of overlapping particles
        """
        swells = self.equiv_swell(area_frac)
        return self._tag_count(swells)

    def _extend_domain(self, domain):
        """
        Inserts a value at the beginning of the domain equal to the separation between the first
        two values, and a value at the end of the array determined by the separation of the last
        two values

        Args:
            domain (np.array): array to extend
        Return:
            (np.array) extended domain array
        """
        first = 2 * domain[0] - domain[1]
        if (first < 0):
            first = 0
        last = 2 * domain[-1] - domain[-2]
        domain_extend = np.insert(domain, 0, first)
        domain_extend = np.append(domain_extend, last)
        return domain_extend

    
    def tag_rate(self, area_frac):
        """
        Returns the rate at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the fraction tagged at two area fractions and dividing by the 
        difference of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate tag rate at

        Returns:
            (np.array): The rate of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        tagged = self.tag_count(af_extended)
        rate = (tagged[2:] - tagged[:-2])
        return rate

    def tag_curve(self, area_frac):
        """
        Returns the curvature at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the rate at two area fractions and dividing by the difference
        of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate the tag curvature at

        Returns:
            (np.array): The curvature of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        rate = self.tag_rate(af_extended)
        curve = (rate[2:] - rate[:-2])
        return curve

    def tag_plot(self, area_frac, mode='count', show=True, filename=None):
        """
        Generates a plot of the tag count, rate, or curvature

        Args:
            area_frac (np.array): list of the area fractions to use in the plot
            mode ("count"|"rate"|"curve"): which information you want to plot. Defaults to "count".
            show (bool): default True. Whether or not to show the plot
            filename (string): default None. Filename to save the plot as. If filename=None, the plot is not saved.
        """
        if (mode == 'curve'):
            plt.ylabel('Curve')
            func = self.tag_curve
        elif (mode == 'rate'):
            plt.ylabel('Rate')
            func = self.tag_rate
        else:
            plt.ylabel('Count')
            func = self.tag_count
        data = func(area_frac) 
        plt.plot(area_frac, data)
        plt.xlabel("Area Fraction")
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def detect_memory(self, start, end, incr):
        """
        Tests the number of tagged particles over a range of area fractions, and 
        returns a list of area fractions where memories are detected. 
        
        Args:
            start (float): The first area fraction in the detection
            end (float): The last area fraction in the detection
            incr (float): The increment between test swells. Determines accuracy of the memory detection. 
        Returns:
            (np.array): list of swells where a memory is located
        """
        area_frac = np.arange(start, end, incr)
        curve = self.tag_curve(area_frac)
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
        return area_frac[matches]
