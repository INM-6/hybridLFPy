#!/usr/bin/env python
'''
Cached networks to use with the population classes, as the only
variables being used is "nodes_ex" and "nodes_in" VERSION THAT WORKS
'''


import numpy as np
import os
import glob
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
from gdf import GDF
import matplotlib.pyplot as plt
from mpi4py import MPI



################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


############## Functions #######################################################

def remove_axis_junk(ax, which=['right', 'top']):
    '''
    Remove axis lines from axes object that exist in list which
    
    Parameters:
    ::

        ax : matplotlib.axes.AxesSubplot object
        which : list
            entries in ['right', 'top', 'bottom', 'left']
    
    '''
    for loc, spine in ax.spines.iteritems():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


################ Classes #######################################################

class CachedNetwork(object):
    '''
    Offline processing and storing of network spike events, used by other class
    objects in the package hybridLFPy

    Parameters:
        ::
            
            simtime : float,
                simulation duration
            dt : float,
                simulation timestep size
            spike_output_path : str,
                path to gdf-files with spikes
            label : str,
                prefix of spiking gdf-files
            ext : str,
                file extension of gdf-files
            dbname : str,
                ':memory:' or filename of new sqlite3 database
            N_X :  np.ndarray,
                the number of neurons in each population
            X : list
                names of each population
            autocollect : bool
                If True, class init will process gdf files
            cmap : str,
                name of colormap, must be in dir(plt.cm)
    '''

    def __init__(self,
                 simtime = 1000.,
                 dt = 0.1,
                 spike_output_path='spike_output_path',
                 label = 'spikes',
                 ext = 'gdf',
                 dbname = ':memory:',
                 N_X=np.array([400, 100]),
                 X=['EX', 'IN'],
                 autocollect=True,
                 cmap='rainbow',
                 ):
        '''
        Offline processing and storing of network spike events, used by other class
        objects in the package hybridLFPy
    
        Parameters:
            ::
                
                simtime : float,
                    simulation duration
                dt : float,
                    simulation timestep size
                spike_output_path : str,
                    path to gdf-files with spikes
                label : str,
                    prefix of spiking gdf-files
                ext : str,
                    file extension of gdf-files
                dbname : str,
                    ':memory:' or filename of new sqlite3 database
                N_X :  list,
                    the number of neurons in each population
                X : list
                    names of each population
                autocollect : bool
                    If True, class init will process gdf files
                cmap : str,
                    name of colormap, must be in dir(plt.cm)
        '''
        #set some attributes
        self.simtime = simtime
        self.dt = dt
        self.spike_output_path = spike_output_path
        self.label = label
        self.ext = ext
        self.dbname = dbname
        self.N_X = np.array(N_X, dtype=int)
        self.X = X
        self.autocollect = autocollect

        #create a dictionary of nodes with proper layernames
        self.nodes = {}
        for i, N in enumerate(self.N_X):
            if i == 0:
                self.nodes[self.X[i]] = np.arange(N) + 1
            else:
                self.nodes[self.X[i]] = np.arange(N) + \
                                            self.N_X.cumsum()[i-1] + 1

        if self.autocollect:
            #collect the gdf files
            self.collect_gdf()


        #specify some colors used for each population:
        if 'TC' in self.X:
            self.colors = ['k']
            numcolors = len(self.X)-1
        else:
            self.colors = []
            numcolors = len(self.X)
            
        for i in range(numcolors):
            self.colors.append(plt.get_cmap(cmap, numcolors)(i))
        

    def collect_gdf(self):
        '''
        collect the gdf-files from network sim in folder spike_output_path
        into sqlite database, using the GDF-class
        '''
        #resync
        COMM.Barrier()

        #raise Exception if there are no gdf files to be read
        if len(glob.glob(os.path.join(self.spike_output_path,
                                      self.label + '*.'+ self.ext))) == 0:
            raise Exception('path to files contain no gdf-files!')

        #if creating in memory db, do across ranks
        if self.dbname == ':memory:':
            self.db = GDF(os.path.join(self.dbname),
                     debug=True, new_db=True)
            self.db.create(re=os.path.join(self.spike_output_path,
                                           self.label + '*.'+ self.ext),
                      index=True)
        else:
            if RANK == 0:
                #put results in db
                db = GDF(os.path.join(self.spike_output_path, self.dbname),
                         debug=True, new_db=True)
                db.create(re=os.path.join(self.spike_output_path,
                                          self.label + '*.'+ self.ext),
                          index=True)
                db.close()

        COMM.Barrier()


    def get_xy(self, xlim, fraction=1.):
        '''
        Get pairs of node units and spike trains on specific time interval
        
        Parameters:
            ::
                
                xlim : list of floats
                    spike time interval, e.g., [0., 1000.]
                fraction : float on [0, 1.]
                    if less than one, sample a fraction of nodes in random order
        
        Return:
            ::
                
                x, y = (dict, dict)
                where in x key-value entries are population name and neuron
                spike times and in y key-value entries are population name and
                neuron gid #

                    
        '''
        if hasattr(self, 'db'):
            db = self.db
        else:
            db = GDF(os.path.join(self.spike_output_path, self.dbname),
                     new_db=False)
        x = {}
        y = {}

        for layer, nodes in self.nodes.iteritems():
            x[layer] = np.array([])
            y[layer] = np.array([])

            if fraction != 1:
                nodes = np.random.permutation(nodes)[:int(nodes.size*fraction)]
                nodes.sort()

            spiketimes = db.select_neurons_interval(nodes, T=xlim)
            i = 0
            for times in spiketimes:
                x[layer] = np.r_[x[layer], times]
                y[layer] = np.r_[y[layer], np.zeros(times.size) + nodes[i]]
                i += 1
        if not hasattr(self, 'db'):
            db.close()
        return x, y


    def plot_raster(self, ax, xlim, x, y, pop_names=False,
                    markersize=1, alpha=1., legend=True, ):
        '''
        plot network raster plot in subplot object
        
        Parameters:
            ::
                
                ax : matplotlib.axes.AxesSubplot object
                xlim : list of floats
                    spike time interval, e.g., [0., 1000.]
                x : dict
                    key-value entries are population name and neuron spike times 
                y : dict
                    key-value entries are population name and neuron gid #
                pop_names: bool
                    if True, show population names on yaxis instead of gid #
                legend : bool
                    switch on axes legends
                **kwargs : see matplotlib.pyplot.plot
        
        '''
        for i, X in enumerate(self.X):
            ax.plot(x[X], y[X], 'o',
                markersize=markersize,
                markerfacecolor=self.colors[i],
                markeredgecolor='none',
                alpha=alpha,
                label=X, rasterized=True,
                clip_on=False)

        ax.axis([xlim[0], xlim[1], 0, self.N_X.sum()])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_ylabel('cell id')
        ax.set_xlabel('time (ms)')
        if legend == True:
            ax.legend()
        if pop_names:
            yticks = []
            yticklabels = []
            for i, X in enumerate(self.X):
                if y[X] != []:
                    yticks.append(y[X].mean())
                    yticklabels.append(self.X[i])
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)


    def plot_f_rate(self, ax, X, i, xlim, x, y, yscale='linear'):
        '''
        plot network firing rate plot in subplot object
        
        Parameters:
            ::
                
                ax : matplotlib.axes.AxesSubplot object
                X : str,
                    population name
                i : int,
                    population index in class attribute X
                xlim : list of floats
                    spike time interval, e.g., [0., 1000.]
                x : dict
                    key-value entries are population name and neuron spike times 
                y : dict
                    key-value entries are population name and neuron gid #
                yscale : 'str'
                    linear, log, or symlog y-axes in rate plot
        '''
        
        bins = np.arange(xlim[0], xlim[1]+1)

        hist = np.histogram(x[X], bins=bins)[0]
        ax.fill_between(bins[:-1], hist * 1000. / self.N_X[i],
                color=self.colors[i], lw=0.5, label=X, rasterized=True,
                clip_on=False)
        ax.plot(bins[:-1], hist * 1000. / self.N_X[i],
                color='k', lw=0.5, label=X, rasterized=False,
                clip_on=False)

        remove_axis_junk(ax)

        ax.axis(ax.axis('tight'))

        ax.set_yscale(yscale)

        ax.set_xlim(xlim[0], xlim[1])
        ax.text(xlim[0] + .05*(xlim[1]-xlim[0]), ax.axis()[3], X,
                va='center', ha='left')


    def raster_plots(self, xlim=[0, 1000], markersize=1, alpha=1.):
        '''
        Pretty plot of the spiking output of each population as raster and rate
        
        Parameters:
            ::
                
                xlim : list of floats
                    spike time interval, e.g., [0., 1000.]
                **kwargs : see matplotlib.pyplot.plot
        
        '''
        x, y = self.get_xy(xlim)

        fig = plt.figure()
        fig.subplots_adjust(left=0.12, hspace=0.15)

        ax0 = fig.add_subplot(211)

        self.plot_raster(ax0, xlim, x, y, markersize=markersize, alpha=alpha)
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
    '''
    subclass of CachedNetwork
    
    Fake nest output, where each cell in a subpopulation spike
    simultaneously, and each subpopulation is activated at times given in
    kwarg activationtimes.
    
    Parameters:
        ::
            
            activationtimes : list of floats
                each entry set spike times of all cells in each population
            **kwargs : see parent class CachedNetwork
    
    '''
    def __init__(self,
                 activationtimes=[200, 300, 400, 500, 600, 700, 800, 900, 1000],
                 autocollect=False,
                 **kwargs):
        '''
        subclass of CachedNetwork
        
        Fake nest output, where each cell in a subpopulation spike
        simultaneously, and each subpopulation is activated at times given in
        kwarg activationtimes.
        
        Parameters:
            ::
                
                activationtimes : list of floats
                    each entry set spike times of all cells in each population
                **kwargs : see class CachedNetwork
        
        '''
        

        CachedNetwork.__init__(self, autocollect=autocollect, **kwargs)

        #set some attributes
        self.activationtimes = activationtimes
        
        if len(activationtimes) != len(self.N_X):
            raise Exception, 'len(activationtimes != len(self.N_X))'

        #create a dictionary of nodes with proper layernames
        self.nodes = {}


        if RANK == 0:
            for i, N in enumerate(self.N_X):
                nodes = self.nodes[self.X[i]]
                val = zip(nodes, [self.activationtimes[i]
                                  for x in range(nodes.size)])
                val = np.array(val, dtype=[('a', int), ('b', float)])
                if i == 0:
                    cell_spt = val
                else:
                    cell_spt = np.r_[cell_spt, val]

            np.savetxt(os.path.join(self.spike_output_path,
                                    'population_spikes.gdf'),
                       cell_spt, fmt=['%i', '%.1f'])

        #resync
        COMM.barrier()

        #collect the gdf files
        self.collect_gdf()



class CachedNoiseNetwork(CachedNetwork):
    '''
    Subclass of CachedNetwork

    Use nest to generate N_X poisson-generators each with rate frate,
    and record every vector, and create database with spikes

    Parameters:
        ::

            frate : list
                rate of each layer, may be tuple (onset, rate, offset)
            **kwargs: See class CachedNetwork

    '''
    def __init__(self,
                 frate=[(200., 15., 210.), 0.992, 3.027, 4.339, 5.962,
                        7.628, 8.669, 1.118, 7.859],
                 autocollect=False,
                 **kwargs):
        '''
        Subclass of CachedNetwork
    
        Use nest to generate N_X poisson-generators each with rate frate,
        and record every vector, and create database with spikes
    
        Parameters:
            ::
    
                frate : list
                    rate of each layer, may be tuple (onset, rate, offset)
                **kwargs: See class CachedNetwork
    
        '''
        CachedNetwork.__init__(self, autocollect=autocollect, **kwargs)

        #putting import nest here, avoid making nest a required
        #dependency
        import nest


        #set some attributes:
        self.frate = frate
        if len(self.frate) != self.N_X.size:
            raise Exception, 'self.frate.size != self.N_X.size'

        self.spike_output_path = spike_output_path

        self.total_num_virtual_procs = SIZE

        #reset nest kernel and set some kernel status variables, destroy old
        #nodes etc in the process
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

        #create some populations of parrot neurons that echo the poisson noise
        self.nodes = {}
        for i, N in enumerate(self.N_X):
            self.nodes[self.X[i]] = nest.Create('parrot_neuron', N)

        if os.path.isfile(os.path.join(self.spike_output_path, self.dbname)):
            mystring = os.path.join(self.spike_output_path, self.dbname)
            print 'db %s exist, will not rerun sim or collect gdf!' % mystring
        else:
            #create spike detector
            self.spikes = nest.Create("spike_detector", 1,
                            {'label' : os.path.join(self.spike_output_path, self.label)})

            #create independent poisson spike trains with the some rate,
            #but each layer population should really have different rates
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

            ##connect parrots and spike detector
            for layer in self.X:
                nest.ConvergentConnect(self.nodes[layer], self.spikes,
                                       model='static_synapse')

            #connect noise generators and nodes
            for i, layer in enumerate(self.X):
                nest.ConvergentConnect(self.noise[i], self.nodes[layer],
                                       model='static_synapse')

            #run simulation
            nest.Simulate(self.simtime)

            #collect the gdf files
            self.collect_gdf()

            #nodes need to be collected in np.ndarrays:
            for key in self.nodes.keys():
                self.nodes[key] = np.array(self.nodes[key])


if __name__ == '__main__':
    pass