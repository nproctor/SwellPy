import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from .particle_suspension import *


class Monodisperse(ParticleSuspension):
    def __init__(self, N, area_fraction=None, seed=None):
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
        super().__init__(N)
        self.name = "Monodisperse"
        if (area_fraction == None):
            self.boxsize = 1.0
        else:
            self.boxsize = np.sqrt(N*np.pi/(4*area_fraction))
        self.centers = None
        self.reset(seed)

    def percent_to_diameter(self, percent):
        """
        Calculates the swell diameter that corresponds to
        the given area fraction.

        Parameters
        ----------
            percent: float
                The area fraction of interest
        Returns
        -------
            The corresponding particle diameter
        """
        return np.sqrt((percent* self.boxsize**2)/(np.pi*self.N))*2

    def diameter_to_percent(self, diameter):
        """
        Calculates the area fraction that corresponds to a 
        given swell diameter

        Parameters
        ----------
            diameter: float
                The swell diamter of the particles
        Returns
        -------
            The corresponding area fraction
        """
        return (self.N*np.pi*(diameter/2)**2)/(self.boxsize**2)


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
        pairs = np.array(list(pairs), dtype=np.int64)
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
        np.putmask(separation, separation < -swell, separation + boxsize)
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
        Repeatedly tags and repels overlapping particles until particles
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


    def train_for(self, swell, kick, cycles):
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

    def particle_plot(self, swell, show=True, extend = False, figsize = (7,7), filename=None):
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
        fig = plt.figure(figsize = figsize)
        plt.axis('off')
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = swell/2))
            if (extend):
                ax.add_artist(Circle(xy=(pair + [0, self.boxsize]), radius = swell/2, alpha=0.3))
                ax.add_artist(Circle(xy=(pair) + [self.boxsize, 0], radius = swell/2, alpha=0.3))
                ax.add_artist(Circle(xy=(pair) + [self.boxsize, self.boxsize], radius = swell/2, alpha=0.3))
        if (extend):
            plt.xlim(0, 2*self.boxsize)
            plt.ylim(0, 2*self.boxsize)
            plt.plot([0,self.boxsize*2], [self.boxsize, self.boxsize], ls = ':', color = '#333333')
            plt.plot([self.boxsize,self.boxsize], [0, self.boxsize*2], ls = ':', color = '#333333')

        else:
            plt.xlim(0, self.boxsize)
            plt.ylim(0, self.boxsize)
        fig.tight_layout()
        if filename != None:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def tag_count_at(self, swell):
        """
        Returns the number of tagged pairs at a specific swell diameter
        
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

    def tag_count(self, Min, Max, incr):
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
        swell = Min
        swells = []
        while (swell <= Max):
            swells.append(swell)
            swell += incr
        swells = np.array(swells)
        tagged = np.array(list(map(lambda x: self.tag_count_at(x), swells)))
        return swells, tagged

    
    def tag_rate(self, Min, Max, incr):
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
        (swells, tagged) = self.tag_count(Min-incr/2, Max+incr/2, incr)
        rate = ( tagged[1:] - tagged[:-1] ) / incr
        return swells[:-1]+incr/2, rate

    def tag_curve(self, Min, Max, incr):
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
        (swells, tag_rate) = self.tag_rate(Min-incr/2, Max+incr/2, incr)
        curve = ( tag_rate[1:] - tag_rate[:-1] ) / incr
        return swells[:-1]+incr/2, curve

    def tag_plot(self, mode='count', show=True, save=False, filename=None):
        Min = self.percent_to_diameter(0.1)
        Max = self.percent_to_diameter(1)
        incr = 1/(self.N*5)
        if (mode == 'rate'):
            (swells, data) = self.tag_rate(Min, Max, incr)
        elif (mode == 'curve'):
            (swells, data) = self.tag_curve(Min, Max, incr)
        else:
            (swells, data) = self.tag_count(Min, Max, incr)   
        plt.plot(swells, data)
        plt.xlim(Min, Max)
        plt.xlabel("Diameter")
        plt.ylabel("Second Derivative of Tagged Particles")
        if save == True:
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