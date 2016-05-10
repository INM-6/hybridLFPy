#!/usr/bin/env python
"""
Cached networks to use with the population classes, as the only
variables being used is "nodes_ex" and "nodes_in" VERSION THAT WORKS.
"""

import numpy as np
import os
import glob
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
from .gdf import GDF
import matplotlib.pyplot as plt
from mpi4py import MPI


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


############## Functions #######################################################

def remove_axis_junk(ax, which=['right', 'top']):
    """
    Remove axis lines from axes object that exist in list which.

    
    Parameters
    ----------
    ax : `matplotlib.axes.AxesSubplot` object
    which : list of str
        Entries in ['right', 'top', 'bottom', 'left'].


    Returns
    -------
    None
    
    """
    for loc, spine in ax.spines.items():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


################ Classes #######################################################

class CachedNetwork(object):
    """
    Offline processing and storing of network spike events, used by other
    class objects in the package hybridLFPy.


    Parameters
    ----------
    simtime : float
        Simulation duration.
    dt : float,
        Simulation timestep size.
    spike_output_path : str
        Path to gdf-files with spikes.
    label : str
        Prefix of spiking gdf-files.
    ext : str
        File extension of gdf-files.
    GIDs : dict
        dictionary keys are population names and item a list with first
        GID in population and population size            
    autocollect : bool
        If True, class init will process gdf files.
    cmap : str
        Name of colormap, must be in `dir(plt.cm)`.
    
    
    Returns
    -------
    `hybridLFPy.cachednetworks.CachedNetwork` object


    See also
    --------
    CachedFixedSpikesNetwork, CachedNoiseNetwork
    """

    def __init__(self,
                 simtime = 1000.,
                 dt = 0.1,
                 spike_output_path='spike_output_path',
                 label = 'spikes',
                 ext = 'gdf',
                 GIDs={'EX' : [1, 400], 'IN' : [401, 100]},
                 autocollect=True,
                 cmap='Set1',
                 ):
        """
        Offline processing and storing of network spike events, used by other
        class objects in the package `hybridLFPy`.


        Parameters
        ----------
        simtime : float
            Simulation duration.
        dt : float
            Simulation timestep size.
        spike_output_path : str
            Path to gdf-files with spikes.
        label : str
            Prefix of spiking gdf-files.
        ext : str
            File extension of gdf-files.
        GIDs : dict
            dictionary keys are population names and item a list with first
            GID in population and population size            
        autocollect : bool
            If True, class init will process gdf files.
        cmap : str
            Name of colormap, must be in dir(plt.cm).
        
        
        Returns
        -------
        `hybridLFPy.cachednetworks.CachedNetwork` object


        See also
        --------
        CachedFixedSpikesNetwork, CachedNoiseNetwork
            
        """
        # Set some attributes
        self.simtime = simtime
        self.dt = dt
        self.spike_output_path = spike_output_path
        self.label = label
        self.ext = ext
        self.dbname = ':memory:'
        self.GIDs = GIDs
        self.X = GIDs.keys()
        self.X.sort()
        self.autocollect = autocollect

        # Create a dictionary of nodes with proper layernames
        self.nodes = {}
        for X in self.X:
            self.nodes[X] = np.arange(self.GIDs[X][1]) + self.GIDs[X][0]

        #list population sizes
        self.N_X = np.array([self.GIDs[X][1] for X in self.X])

        if self.autocollect:
            #collect the gdf files
            self.collect_gdf()


        # Specify some plot colors used for each population:
        if 'TC' in self.X:
            numcolors = len(self.X)-1
        else:
            numcolors = len(self.X)
            
        self.colors = []
        for i in range(numcolors):
            self.colors += [plt.get_cmap(cmap, numcolors)(i)]

        if 'TC' in self.X:
            self.colors += ['k']
        

    def collect_gdf(self):
        """
        Collect the gdf-files from network sim in folder `spike_output_path`
        into sqlite database, using the GDF-class.
        
        
        Parameters
        ----------
        None
        
        
        Returns
        -------
        None
        
        """
        # Resync
        COMM.Barrier()

        # Raise Exception if there are no gdf files to be read
        if len(glob.glob(os.path.join(self.spike_output_path,
                                      self.label + '*.'+ self.ext))) == 0:
            raise Exception('path to files contain no gdf-files!')

        #create in-memory databases of spikes
        if not hasattr(self, 'dbs'):
            self.dbs = {}
        
        for X in self.X:
            db = GDF(os.path.join(self.dbname),
                     debug=True, new_db=True)
            db.create(re=os.path.join(self.spike_output_path,
                                      '{0}*{1}*{2}'.format(self.label, X,
                                                           self.ext)),
                      index=True)
            self.dbs.update({
                    X : db
                })
      
        COMM.Barrier()


    def get_xy(self, xlim, fraction=1.):
        """
        Get pairs of node units and spike trains on specific time interval.
        
        
        Parameters
        ----------
        xlim : list of floats
            Spike time interval, e.g., [0., 1000.].
        fraction : float in [0, 1.]
            If less than one, sample a fraction of nodes in random order.
        
        
        Returns
        -------
        x : dict
            In `x` key-value entries are population name and neuron spike times.
        y : dict
            Where in `y` key-value entries are population name and neuron gid number.

        """
        x = {}
        y = {}

        for X, nodes in self.nodes.items():
            x[X] = np.array([])
            y[X] = np.array([])

            if fraction != 1:
                nodes = np.random.permutation(nodes)[:int(nodes.size*fraction)]
                nodes.sort()

            spiketimes = self.dbs[X].select_neurons_interval(nodes, T=xlim)
            i = 0
            for times in spiketimes:
                x[X] = np.r_[x[X], times]
                y[X] = np.r_[y[X], np.zeros(times.size) + nodes[i]]
                i += 1
                
        return x, y


    def plot_raster(self, ax, xlim, x, y, pop_names=False,
                    markersize=20., alpha=1., legend=True,
                    marker='o', rasterized=True):
        """
        Plot network raster plot in subplot object.
        
        
        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot` object
            plot axes
        xlim : list
            List of floats. Spike time interval, e.g., [0., 1000.].
        x : dict
            Key-value entries are population name and neuron spike times.
        y : dict
            Key-value entries are population name and neuron gid number.
        pop_names: bool
            If True, show population names on yaxis instead of gid number.
        markersize : float
            raster plot marker size
        alpha : float in [0, 1]
            transparency of marker
        legend : bool
            Switch on axes legends.
        marker : str
            marker symbol for matplotlib.pyplot.plot
        rasterized : bool
            if True, the scatter plot will be treated as a bitmap embedded in
            pdf file output


        Returns
        -------
        None
        
        """
        yoffset = [sum(self.N_X) if X=='TC' else 0 for X in self.X]
        for i, X in enumerate(self.X):
            if y[X].size > 0:
                ax.plot(x[X], y[X]+yoffset[i], marker,
                    markersize=markersize,
                    mfc=self.colors[i],
                    mec='none' if marker in '.ov><v^1234sp*hHDd' else self.colors[i],
                    alpha=alpha,
                    label=X, rasterized=rasterized,
                    clip_on=True)
        
        #don't draw anything for the may-be-quiet TC population
        N_X_sum = 0
        for i, X in enumerate(self.X):
            if y[X].size > 0:
                N_X_sum += self.N_X[i]
        
        ax.axis([xlim[0], xlim[1],
                 self.GIDs[self.X[0]][0], self.GIDs[self.X[0]][0]+N_X_sum])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_ylabel('cell id', labelpad=0)
        ax.set_xlabel('$t$ (ms)', labelpad=0)
        if legend:
            ax.legend()
        if pop_names:
            yticks = []
            yticklabels = []
            for i, X in enumerate(self.X):
                if y[X] != []:
                    yticks.append(y[X].mean()+yoffset[i])
                    yticklabels.append(self.X[i])
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        
        # Add some horizontal lines separating the populations
        for i, X in enumerate(self.X):
            if y[X].size > 0:
                ax.plot([xlim[0], xlim[1]],
                        [y[X].max()+yoffset[i], y[X].max()+yoffset[i]],
                        'k', lw=0.25)


    def plot_f_rate(self, ax, X, i, xlim, x, y, binsize=1, yscale='linear',
                    plottype='fill_between', show_label=False, rasterized=False):
        """
        Plot network firing rate plot in subplot object.
        
        
        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot` object.
        X : str
            Population name.
        i : int
            Population index in class attribute `X`.
        xlim : list of floats
            Spike time interval, e.g., [0., 1000.].
        x : dict
            Key-value entries are population name and neuron spike times.
        y : dict
            Key-value entries are population name and neuron gid number.
        yscale : 'str'
            Linear, log, or symlog y-axes in rate plot.
        plottype : str
            plot type string in `['fill_between', 'bar']`
        show_label : bool
            whether or not to show labels
        

        Returns
        -------
        None
        
        """
        
        bins = np.arange(xlim[0], xlim[1]+binsize, binsize)
        (hist, bins) = np.histogram(x[X], bins=bins)
        
        if plottype == 'fill_between':
            ax.fill_between(bins[:-1], hist * 1000. / self.N_X[i],
                    color=self.colors[i], lw=0.5, label=X, rasterized=rasterized,
                    clip_on=False)
            ax.plot(bins[:-1], hist * 1000. / self.N_X[i],
                    color='k', lw=0.5, label=X, rasterized=rasterized,
                    clip_on=False)
        elif plottype == 'bar':
            ax.bar(bins[:-1], hist * 1000. / self.N_X[i],
                    color=self.colors[i], label=X, rasterized=rasterized ,
                    linewidth=0.25, width=0.9, clip_on=False)
        else:
            mssg = "plottype={} not in ['fill_between', 'bar']".format(plottype)
            raise Exception(mssg)

        remove_axis_junk(ax)

        ax.axis(ax.axis('tight'))

        ax.set_yscale(yscale)

        ax.set_xlim(xlim[0], xlim[1])
        if show_label:
            ax.text(xlim[0] + .05*(xlim[1]-xlim[0]), ax.axis()[3]*1.5, X,
                    va='center', ha='left')


    def raster_plots(self, xlim=[0, 1000], markersize=1, alpha=1., marker='o'):
        """
        Pretty plot of the spiking output of each population as raster and rate.
        
        
        Parameters
        ----------
        xlim : list
            List of floats. Spike time interval, e.g., `[0., 1000.]`.
        markersize : float
            marker size for plot, see `matplotlib.pyplot.plot`
        alpha : float
            transparency for markers, see `matplotlib.pyplot.plot`
        marker : :mod:`A valid marker style <matplotlib.markers>`
        
        
        Returns
        -------
        fig : `matplotlib.figure.Figure` object
        
        """
        x, y = self.get_xy(xlim)

        fig = plt.figure()
        fig.subplots_adjust(left=0.12, hspace=0.15)

        ax0 = fig.add_subplot(211)

        self.plot_raster(ax0, xlim, x, y, markersize=markersize, alpha=alpha,
                         marker=marker)
        remove_axis_junk(ax0)
        ax0.set_title('spike raster')
        ax0.set_xlabel("")

        nrows = len(self.X)
        bottom = np.linspace(0.1, 0.45, nrows+1)[::-1][1:]
        thickn = np.abs(np.diff(bottom))[0]*0.9


        for i, layer in enumerate(self.X):
            ax1 = fig.add_axes([0.12, bottom[i], 0.78, thickn])

            self.plot_f_rate(ax1, layer, i, xlim, x, y, )

            if i == nrows-1:
                ax1.set_xlabel('time (ms)')
            else:
                ax1.set_xticklabels([])

            if i == 4:
                ax1.set_ylabel(r'population rates ($s^{-1}$)')

            if i == 0:
                ax1.set_title(r'population firing rates ($s^{-1}$)')
              
        return fig


class CachedFixedSpikesNetwork(CachedNetwork):
    """
    Subclass of CachedNetwork.
    
    Fake nest output, where each cell in a subpopulation spike
    simultaneously, and each subpopulation is activated at times given in
    kwarg activationtimes.
    
    
    Parameters
    ----------
    activationtimes : list of floats
        Each entry set spike times of all cells in each population
    autocollect : bool
        whether or not to automatically gather gdf file output
    **kwargs : see parent class `hybridLFPy.cachednetworks.CachedNetwork`
    
    
    Returns
    -------
    `hybridLFPy.cachednetworks.CachedFixedSpikesNetwork` object


    See also
    --------
    CachedNetwork, CachedNoiseNetwork, 

    """
    def __init__(self,
                 activationtimes=[200, 300, 400, 500, 600, 700, 800, 900, 1000],
                 autocollect=False,
                 **kwargs):
        """
        Subclass of CachedNetwork
        
        Fake nest output, where each cell in a subpopulation spike
        simultaneously, and each subpopulation is activated at times given in
        kwarg activationtimes.
        
        Parameters
        ----------
        activationtimes : list
            Each entry set spike times of all cells in each population
        autocollect : bool
            whether or not to automatically gather gdf file output
        **kwargs : see parent class `hybridLFPy.cachednetworks.CachedNetwork`
        
        
        Returns
        -------
        `hybridLFPy.cachednetworks.CachedFixedSpikesNetwork` object


        See also
        --------
        CachedNetwork, CachedNoiseNetwork, 
        
        """
        
        CachedNetwork.__init__(self, autocollect=autocollect, **kwargs)

        # Set some attributes
        self.activationtimes = activationtimes
        
        if len(activationtimes) != len(self.N_X):
            raise Exception('len(activationtimes != len(self.N_X))')

        """ Create a dictionary of nodes with proper layernames
         self.nodes = {}.
        """

        if RANK == 0:
            for i, N in enumerate(self.N_X):
                nodes = self.nodes[self.X[i]]
                cell_spt = list(zip(nodes, [self.activationtimes[i]
                                  for x in range(nodes.size)]))
                cell_spt = np.array(cell_spt, dtype=[('a', int), ('b', float)])

                np.savetxt(os.path.join(self.spike_output_path,
                                        self.label + '_{}.gdf'.format(self.X[i])),
                           cell_spt, fmt=['%i', '%.1f'])

        # Resync
        COMM.barrier()

        # Collect the gdf files
        self.collect_gdf()



class CachedNoiseNetwork(CachedNetwork):
    """
    Subclass of CachedNetwork.

    Use Nest to generate N_X poisson-generators each with rate frate,
    and record every vector, and create database with spikes.

    Parameters
    ----------
    frate : list
        Rate of each layer, may be tuple (onset, rate, offset)
    autocollect : bool
        whether or not to automatically gather gdf file output
    **kwargs : see parent class `hybridLFPy.cachednetworks.CachedNetwork`


    Returns
    -------
    `hybridLFPy.cachednetworks.CachedNoiseNetwork` object


    See also
    --------
    CachedNetwork, CachedFixedSpikesNetwork 
    

    """
    def __init__(self,
                 frate=[(200., 15., 210.), 0.992, 3.027, 4.339, 5.962,
                        7.628, 8.669, 1.118, 7.859],
                 autocollect=False,
                 **kwargs):
        """
        Subclass of `CachedNetwork`.
        Use Nest to generate N_X poisson-generators each with rate frate,
        and record every vector, and create database with spikes.
    
        Parameters
        ----------
        frate : list
            Rate of each layer, may be tuple (onset, rate, offset).
        autocollect : bool
            whether or not to automatically gather gdf file output
        **kwargs : see parent class `hybridLFPy.cachednetworks.CachedNetwork`
    
    
        Returns
        -------
        `hybridLFPy.cachednetworks.CachedNoiseNetwork` object
    
    
        See also
        --------
        CachedNetwork, CachedFixedSpikesNetwork 
        
        """
        CachedNetwork.__init__(self, autocollect=autocollect, **kwargs)

        """
        Putting import nest here, avoid making `nest` a mandatory
        `hybridLFPy` dependency.
        """
        import nest


        #set some attributes:
        self.frate = frate
        if len(self.frate) != self.N_X.size:
            raise Exception('self.frate.size != self.N_X.size')

        self.spike_output_path = spike_output_path

        self.total_num_virtual_procs = SIZE

        # Reset nest kernel and set some kernel status variables, destroy old
        # nodes etc in the process
        nest.ResetKernel()

        #if dt is in powers of two, dt must be multiple of ms_per_tic
        if self.dt in 2**np.arange(-32., 0):
            nest.SetKernelStatus({
                "tics_per_ms" : 2**2 / self.dt,
                "resolution": self.dt,
                "print_time": True,
                "overwrite_files" : True,
                "total_num_virtual_procs" : self.total_num_virtual_procs,
                })
        else:
            nest.SetKernelStatus({
                "resolution": self.dt,
                "print_time": True,
                "overwrite_files" : True,
                "total_num_virtual_procs" : self.total_num_virtual_procs,
                })

        nest.SetDefaults("spike_detector", {
            'withtime' : True,
            'withgid' : True,
            'to_file' : True,
            'to_memory' : False,
        })

        # Create some populations of parrot neurons that echo the poisson noise
        self.nodes = {}
        for i, N in enumerate(self.N_X):
            self.nodes[self.X[i]] = nest.Create('parrot_neuron', N)

        if os.path.isfile(os.path.join(self.spike_output_path, self.dbname)):
            mystring = os.path.join(self.spike_output_path, self.dbname)
            print('db %s exist, will not rerun sim or collect gdf!' % mystring)
        else:
            # Create spike detector
            self.spikes = nest.Create("spike_detector", 1,
                            {'label' : os.path.join(self.spike_output_path,
                                                    self.label)})

            """ Create independent poisson spike trains with the some rate,
             but each layer population should really have different rates.
             """
            self.noise = []
            for rate in self.frate:
                if type(rate) == tuple:
                    self.noise.append(nest.Create("poisson_generator", 1,
                                                  { "start" : rate[0],
                                                    "rate" :  rate[1],
                                                    "stop" :  rate[2]}))
                else:
                    self.noise.append(nest.Create("poisson_generator", 1,
                                                  {"rate" : rate}))

            ## Connect parrots and spike detector
            for layer in self.X:
                nest.ConvergentConnect(self.nodes[layer], self.spikes,
                                       model='static_synapse')

            # Connect noise generators and nodes
            for i, layer in enumerate(self.X):
                nest.ConvergentConnect(self.noise[i], self.nodes[layer],
                                       model='static_synapse')

            # Run simulation
            nest.Simulate(self.simtime)

            # Collect the gdf files
            self.collect_gdf()

            # Nodes need to be collected in np.ndarrays:
            for key in list(self.nodes.keys()):
                self.nodes[key] = np.array(self.nodes[key])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
