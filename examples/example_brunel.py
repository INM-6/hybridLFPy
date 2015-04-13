#!/usr/bin/env python
'''
Hybrid LFP scheme example script, applying the methodology with a model
implementation similar to:

Nicolas Brunel. "Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons". J Comput Neurosci, May 2000, Volume 8,
Issue 3, pp 183-208


Synopsis of the main simulation procedure:
1. Loading of parameterset
    a. network parameters
    b. parameters for hybrid scheme
2. Set up file destinations for different simulation output
3. network simulation
    a. execute network simulation using NEST (www.nest-initiative.org)
    b. merge network output (spikes, currents, voltages)
4. Create a object-representation that uses sqlite3 of all the spiking output 
5. Iterate over post-synaptic populations:
    a. Create Population object with appropriate parameters for
       each specific population
    b. Run all computations for populations
    c. Postprocess simulation output of all cells in population
6. Postprocess all cell- and population-specific output data
7. Create a tarball for all non-redundant simulation output

The full simulation can be evoked by issuing a mpirun call, such as
mpirun -np 4 python example_brunel.py

Not recommended, but running it serially should also work, e.g., calling
python example_brunel.py


Given the size of the network and demands for the multi-compartment LFP-
predictions using the present scheme, running the model on nothing but a large-
scale compute facility is strongly discouraged.

'''
import os
import numpy as np
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from time import time
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
from NeuroTools.parameters import ParameterSet
import h5py
import LFPy
nrn = LFPy.cell.neuron
from mpi4py import MPI

########## matplotlib settings #################################################
plt.close('all')
plt.rcParams.update({'figure.figsize': [10.0, 8.0]})


#set some seed values
SEED = 12345678
SIMULATIONSEED = 12345678
np.random.seed(SEED)


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

#if True, execute full model. If False, do only the plotting. Simulation results
#must exist.
properrun = True


#check if mod file for synapse model specified in alphaisyn.mod is loaded
if not hasattr(nrn.h, 'AlphaISyn'):
    if RANK == 0:
        os.system('nrnivmodl')
    COMM.Barrier()
    nrn.load_mechanisms('.')


################################################################################
## PARAMETERS
################################################################################

#Set up parameters using the NeuroTools.parameters.ParameterSet class.

#Access parameters defined in example script implementing the network using
#pynest, brunel_alpha_nest.py, adapted from the NEST v2.4.1 release. This will
#not execute the network model, but see below.
import brunel_alpha_nest as BN


#set up file destinations differentiating between certain output
PS = ParameterSet(dict(
    #Main folder of simulation output
    savefolder = 'simulation_output_example_brunel',
    
    #make a local copy of main files used in simulations
    sim_scripts_path = os.path.join('simulation_output_example_brunel',
                                    'sim_scripts'),
    
    #destination of single-cell output during simulation
    cells_path = os.path.join('simulation_output_example_brunel', 'cells'),
    
    #destination of cell- and population-specific signals, i.e., compund LFPs,
    #CSDs etc.
    populations_path = os.path.join('simulation_output_example_brunel',
                                    'populations'),
    
    #location of spike output from the network model
    spike_output_path = BN.spike_output_path,
    
    #destination of figure file output generated during model execution
    figures_path = os.path.join('simulation_output_example_brunel', 'figures')
))


#population (and cell type) specific parameters
PS.update(dict(
    #no cell type specificity within each E-I population
    #hence X == x and Y == X
    X = ["EX", "IN"],
    
    #population-specific LFPy.Cell parameters
    cellParams = dict(
        #excitory cells
        EX = dict(
            morphology = 'morphologies/ex.hoc',
            v_init = BN.neuron_params['E_L'],
            rm = BN.neuron_params['tau_m'] * 1E3 / 1.0, #assume cm=1.
            cm = 1.0,
            Ra = 150,
            e_pas = BN.neuron_params['E_L'],    
            nsegs_method = 'lambda_f',
            lambda_f = 100,
            timeres_NEURON = BN.dt,
            timeres_python = BN.dt,
            tstartms = 0,
            tstopms = BN.simtime,
            verbose = False,        
        ),
        #inhibitory cells
        IN = dict(
            morphology = 'morphologies/in.hoc',
            v_init = BN.neuron_params['E_L'],
            rm = BN.neuron_params['tau_m'] * 1E3 / 1.0, #assume cm=1.
            cm = 1.0,
            Ra = 150,
            e_pas = BN.neuron_params['E_L'],    
            nsegs_method = 'lambda_f',
            lambda_f = 100,
            timeres_NEURON = BN.dt,
            timeres_python = BN.dt,
            tstartms = 0,
            tstopms = BN.simtime,
            verbose = False,                    
    )),
    
    #assuming excitatory cells are pyramidal
    rand_rot_axis = dict(
        EX = ['z'],
        IN = ['x', 'y', 'z'],
    ),
    
    
    #kwargs passed to LFPy.Cell.simulate()
    simulationParams = dict(),
    
    #set up parameters corresponding to cylindrical model populations
    populationParams = dict(
        EX = dict(
            number = BN.NE,
            radius = 100,
            z_min = -450,
            z_max = -350,
            min_cell_interdist = 1.,            
            ),
        IN = dict(
            number = BN.NI,
            radius = 100,
            z_min = -450,
            z_max = -350,
            min_cell_interdist = 1.,            
            ),
    ),
    
    #set the boundaries between the "upper" and "lower" layer
    layerBoundaries = [[0., -300],
                      [-300, -500]],
    
    #set the geometry of the virtual recording device
    electrodeParams = dict(
            #contact locations:
            x = [0]*6,
            y = [0]*6,
            z = [x*-100. for x in range(6)],
            #extracellular conductivity:
            sigma = 0.3,
            #contact surface normals, radius, n-point averaging
            N = [[1, 0, 0]]*6,
            r = 5,
            n = 20,
            seedvalue = None,
            #dendrite line sources, soma as sphere source (Linden2014)
            method = 'som_as_point',
            #no somas within the constraints of the "electrode shank":
            r_z = [[-1E199, -600, -550, 1E99],[0, 0, 10, 10]],
    ),
    
    #runtime, cell-specific attributes and output that will be stored
    savelist = [
        'somav',
        'somapos',
        'x',
        'y',
        'z',
        'LFP',
        'CSD',
    ],
    
    #flag for switching on calculation of CSD
    calculateCSD = True,
    
    #time resolution of saved signals
    dt_output = 1. 
))

 
#for each population, define layer- and population-specific connectivity
#parameters
PS.update(dict(
    #number of connections from each presynaptic population onto each
    #layer per postsynaptic population, preserving overall indegree
    k_yXL = dict(
        EX = [[int(0.5*BN.CE), 0],
              [int(0.5*BN.CE), BN.CI]],
        IN = [[0, 0],
              [BN.CE, BN.CI]],
    ),
    
    #set up table of synapse  from each possible 
    J_yX = dict(
        EX = [BN.J_ex*1E-3, BN.J_in*1E-3],
        IN = [BN.J_ex*1E-3, BN.J_in*1E-3],
    ),

    #set up synapse parameters as derived from the network
    synParams = dict(
        EX = dict(
            section = ['apic', 'dend'],
            tau = BN.tauSyn,
            syntype = 'AlphaISyn'
        ),
        IN = dict(
            section = ['dend'],
            tau = BN.tauSyn,
            syntype = 'AlphaISyn'            
        ),
    ),
    
    #set up delays, here using fixed delays of network
    synDelayLoc = dict(
        EX = [BN.delay, BN.delay],
        IN = [BN.delay, BN.delay],
    ),
    #no distribution of delays
    synDelayScale = dict(
        EX = [None, None],
        IN = [None, None],
    ),
    
    
))


#putative mappting between population type and cell type specificity,
#but here all presynaptic senders are also postsynaptic targets
PS.update(dict(
    mapping_Yy = list(zip(PS.X, PS.X))
))

################################################################################
# MAIN simulation procedure                                                    #
################################################################################

#tic toc
tic = time()

######### Perform network simulation ###########################################
if properrun:
    #set up the file destination, removing old results by default
    setup_file_dest(PS, clearDestination=True)

if properrun:
    if RANK == 0:
        os.system("python brunel_alpha_nest.py")
    
#wait for the network simulation to finish, resync MPI threads
COMM.Barrier()


#Create an object representation containing the spiking activity of the network
#simulation output that uses sqlite3. Again, kwargs are derived from the brunel
#network instance.
networkSim = CachedNetwork(
    simtime = BN.simtime,
    dt = BN.dt,
    spike_output_path = BN.spike_output_path,
    label = BN.label,
    ext = 'gdf',
    N_X = [BN.NE, BN.NI],
    X = PS.X,
)


toc = time() - tic
print('NEST simulation and gdf file processing done in  %.3f seconds' % toc)


####### Set up populations #####################################################

if properrun:
    #iterate over each cell type, and create populationulation object
    for i, Y in enumerate(PS.X):
        #create population:
        pop = Population(
                cellParams = PS.cellParams[Y],
                rand_rot_axis = PS.rand_rot_axis[Y],
                simulationParams = PS.simulationParams,
                populationParams = PS.populationParams[Y],
                y = Y,
                layerBoundaries = PS.layerBoundaries,
                electrodeParams = PS.electrodeParams,
                savelist = PS.savelist,
                savefolder = PS.savefolder,
                calculateCSD = PS.calculateCSD,
                dt_output = PS.dt_output, 
                POPULATIONSEED = SIMULATIONSEED + i,
                X = PS.X,
                networkSim = networkSim,
                k_yXL = PS.k_yXL[Y],
                synParams = PS.synParams[Y],
                synDelayLoc = PS.synDelayLoc[Y],
                synDelayScale = PS.synDelayScale[Y],
                J_yX = PS.J_yX[Y],
            )
    
        #run population simulation and collect the data
        pop.run()
        pop.collect_data()
    
        #object no longer needed
        del pop



#close in memory spike output databases
for db in networkSim.dbs.values():
    db.close()


####### Postprocess the simulation output ######################################


#reset seed, but output should be deterministic from now on
np.random.seed(SIMULATIONSEED)

if properrun:
    #do some postprocessing on the collected data, i.e., superposition
    #of population LFPs, CSDs etc
    postproc = PostProcess(y = PS.X,
                           dt_output = PS.dt_output,
                           savefolder = PS.savefolder,
                           mapping_Yy = PS.mapping_Yy,
                           )
    
    #run through the procedure
    postproc.run()
    
    #create tar-archive with output for plotting, ssh-ing etc.
    postproc.create_tar_archive()
    
    
COMM.Barrier()

#tic toc
print('Execution time: %.3f seconds' %  (time() - tic))



################################################################################
# Create set of plots from simulation output
################################################################################


#### Some function declarations ################################################

def remove_axis_junk(ax, which=['right', 'top']):
    '''remove upper and right axis'''
    for loc, spine in ax.spines.items():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def plot_signal_sum(ax, fname='LFPsum.h5', unit='mV',
                    ylabels=True, scalebar=True, vlimround=None,
                    T=[0, 1000], color='k',
                    label=''):
    '''
    on axes plot the signal contributions
    
    args:
    ::
        ax : matplotlib.axes.AxesSubplot object
        T : list, [tstart, tstop], which timeinterval
        ylims : list, set range of yaxis to scale with other plots
        fancy : bool, 
        scaling_factor : float, scaling factor (e.g. to scale 10% data set up)
    '''    
    #open file and get data, samplingrate
    f = h5py.File(fname)
    data = f['data'].value
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    srate = f['srate'].value
    
    #close file object
    f.close()

    # normalize data for plot
    tvec = np.arange(data.shape[1]) * 1000. / srate
    slica = (tvec <= T[1]) & (tvec >= T[0])
    zvec = np.r_[PS.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    vlim = abs(data[:, slica]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim))
    yticklabels=[]
    yticks = []
    
    colors = [color]*data.shape[0]
    
    for i, z in enumerate(PS.electrodeParams['z']):
        if i == 0:
            ax.plot(tvec[slica], data[i, slica] * 100 / vlimround + z,
                    color=colors[i], rasterized=False, label=label,
                    clip_on=False)
        else: 
            ax.plot(tvec[slica], data[i, slica] * 100 / vlimround + z,
                    color=colors[i], rasterized=False, clip_on=False)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)
     
    if scalebar:
        ax.plot([tvec[slica][-1], tvec[slica][-1]],
                [-0, -100], lw=2, color='k', clip_on=False)
        ax.text(tvec[slica][-1]+np.diff(T)*0.02, -50,
                r'%g %s' % (vlimround, unit),
                color='k', rotation='vertical')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
    else:
        ax.yaxis.set_ticklabels([])

    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(r'time (ms)', labelpad=0)


def plot_pop_scatter(ax, somapos, isometricangle, marker, color):
    #scatter plot setting appropriate zorder for each datapoint by binning
    for lower in np.arange(-600, 601, 20):
        upper = lower + 20
        inds = (somapos[:, 1] >= lower) & (somapos[:, 1] < upper)
        if np.any(inds):
            ax.scatter(somapos[inds, 0],
                       somapos[inds, 2] - somapos[inds, 1] * np.sin(isometricangle),
                   s=30, facecolors=color, edgecolors='none', zorder=lower,
                   marker = marker, clip_on=False, rasterized=True)


def plot_population(ax, aspect='equal',
                    isometricangle=np.pi/12,
                    X=['EX', 'IN'],
                    markers=['^', 'o'],
                    colors=['r', 'b'],
                    title='positions'):
    '''
    Plot the geometry of the column model, optionally with somatic locations
    and optionally with reconstructed neurons
    
    kwargs:
    ::
        ax : matplotlib.axes.AxesSubplot
        aspect : str
            matplotlib.axis argument
        isometricangle : float
            pseudo-3d view angle
        plot_somas : bool
            plot soma locations
        plot_morphos : bool
            plot full morphologies
        num_unitsE : int
            number of excitatory morphos plotted per population
        num_unitsI : int
            number of inhibitory morphos plotted per population
        clip_dendrites : bool
            draw dendrites outside of axis
        mainpops : bool
            if True, plot only main pops, e.g. b23 and nb23 as L23I
    
    return:
    ::
        axis : list
            the plt.axis() corresponding to input aspect
    '''

    remove_axis_junk(ax, ['right', 'bottom', 'left', 'top'])

    
    # DRAW OUTLINE OF POPULATIONS 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    #contact points
    ax.plot(PS.electrodeParams['x'],
            PS.electrodeParams['z'],
            '.', marker='o', markersize=5, color='k', zorder=0)

    #outline of electrode       
    x_0 = np.array(PS.electrodeParams['r_z'])[1, 1:-1]
    z_0 = np.array(PS.electrodeParams['r_z'])[0, 1:-1]
    x = np.r_[x_0[-1], x_0[::-1], -x_0[1:], -x_0[-1]]
    z = np.r_[100, z_0[::-1], z_0[1:], 100]
    ax.fill(x, z, color=(0.5, 0.5, 0.5), lw=None, zorder=-0.1)

    #outline of populations:
    #fetch the population radius from some population
    r = PS.populationParams['EX']['radius']

    theta0 = np.linspace(0, np.pi, 20)
    theta1 = np.linspace(np.pi, 2*np.pi, 20)
    
    zpos = np.r_[np.array(PS.layerBoundaries)[:, 0],
                 np.array(PS.layerBoundaries)[-1, 1]]
    
    layers = ['upper', 'lower']
    for i, z in enumerate(np.mean(PS.layerBoundaries, axis=1)):
        ax.text(r, z, ' %s' % layers[i],
                va='center', ha='left', rotation='vertical')

    for i, zval in enumerate(zpos):
        if i == 0:
            ax.plot(r*np.cos(theta0),
                    r*np.sin(theta0)*np.sin(isometricangle)+zval,
                    color='k', zorder=-r, clip_on=False)
            ax.plot(r*np.cos(theta1),
                    r*np.sin(theta1)*np.sin(isometricangle)+zval,
                    color='k', zorder=r, clip_on=False)
        else:
            ax.plot(r*np.cos(theta0),
                    r*np.sin(theta0)*np.sin(isometricangle)+zval,
                    color='gray', zorder=-r, clip_on=False)
            ax.plot(r*np.cos(theta1),
                    r*np.sin(theta1)*np.sin(isometricangle)+zval,
                    color='k', zorder=r, clip_on=False)
    
    ax.plot([-r, -r], [zpos[0], zpos[-1]], 'k', zorder=0, clip_on=False)
    ax.plot([r, r], [zpos[0], zpos[-1]], 'k', zorder=0, clip_on=False)
    
    #plot a horizontal radius scalebar
    ax.plot([0, r], [z_0.min()]*2, 'k', lw=2, zorder=0, clip_on=False)
    ax.text(r / 2., z_0.min()-100, 'r = %i $\mu$m' % int(r), ha='center')
    
    #plot a vertical depth scalebar
    ax.plot([-r]*2, [z_0.min()+50, z_0.min()-50],
        'k', lw=2, zorder=0, clip_on=False)
    ax.text(-r, z_0.min(), r'100 $\mu$m', va='center', ha='right')
    
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    #fake ticks:
    for pos in zpos:
        ax.text(-r, pos, 'z=%i-' % int(pos), ha='right', va='center')
 
    ax.set_title(title)
   
    axis = ax.axis(ax.axis(aspect))


def plot_soma_locations(ax, X, isometricangle, markers, colors):
    for (pop, marker, color) in zip(X, markers, colors):
        #get the somapos
        fname = os.path.join(PS.populations_path,
                             '%s_population_somapos.gdf' % pop)
        
        somapos = np.loadtxt(fname).reshape((-1, 3))
    
        plot_pop_scatter(ax, somapos, isometricangle, marker, color)
        

def plot_morphologies(ax, X, isometricangle, markers, colors):
    for (pop, marker, color) in zip(X, markers, colors):
        #get the somapos
        fname = os.path.join(PS.populations_path,
                             '%s_population_somapos.gdf' % pop)
        
        somapos = np.loadtxt(fname).reshape((-1, 3))
        n = somapos.shape[0]
        
        rotations = [{} for x in range(n)]
        fname = os.path.join(PS.populations_path,
                             '%s_population_rotations.h5' % pop)
        f = h5py.File(fname, 'r')
        
        for key, value in list(f.items()):
            for i, rot in enumerate(value.value):
                rotations[i].update({key : rot})
        
        
        #plot some units
        for j in range(n):
            cell = LFPy.Cell(morphology=PS.cellParams[pop]['morphology'],
                             nsegs_method='lambda_f',
                             lambda_f=100,
                             extracellular=False
                            )
            cell.set_pos(somapos[j, 0], somapos[j, 1], somapos[j, 2])
            cell.set_rotation(**rotations[j])

            #set up a polycollection
            zips = []
            for x, z in cell.get_idx_polygons():
                zips.append(list(zip(x, z-somapos[j, 1] * np.sin(isometricangle))))
            
            polycol = PolyCollection(zips,
                                     edgecolors='k',
                                     facecolors=color,
                                     linewidths=(0.1),
                                     zorder=somapos[j, 1],
                                     clip_on=False,
                                     rasterized=False)

            ax.add_collection(polycol)
            


def plot_two_cells(ax, X, isometricangle, markers, colors):
    
    somapos = np.array([[-75,0,-400],[75, 0, -400]])
    for i, (pop, marker, color) in enumerate(zip(X, markers, colors)):
        
        cell = LFPy.Cell(morphology=PS.cellParams[pop]['morphology'],
                         nsegs_method='lambda_f',
                         lambda_f=100,
                         extracellular=False
                        )
        cell.set_pos(somapos[i, 0], somapos[i, 1], somapos[i, 2])
        
        #set up a polycollection
        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z-somapos[i, 1] * np.sin(isometricangle))))
        
        polycol = PolyCollection(zips,
                                 edgecolors='k',
                                 facecolors=color,
                                 linewidths=(0.1),
                                 zorder=somapos[i, 1],
                                 clip_on=False,
                                 rasterized=False)

        ax.add_collection(polycol)


def normalize(x):
    '''normalize x to have mean 0 and unity standard deviation'''
    x -= x.mean()
    return x / float(x.std())


def plot_correlation(x0, x1, ax, lag=20., title='firing_rate vs LFP'):
    ''' mls
    on axes plot the correlation between x0 and x1
    
    args:
    ::
        x0 : first dataset
        x1 : second dataset - e.g., the multichannel LFP
        ax : matplotlib.axes.AxesSubplot object
        title : text to be used as current axis object title
    '''
    zvec = np.r_[PS.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    
    xcorr_all=np.zeros((np.size(PS.electrodeParams['z']), x0.shape[0]))
    for i, z in enumerate(PS.electrodeParams['z']):
        x2 = x1[i, ]
        xcorr1 = np.correlate(normalize(x0),
                              normalize(x2), 'same') / x0.size
        xcorr_all[i,:]=xcorr1  

    # Find limits for the plot
    vlim = abs(xcorr_all).max()
    vlimround = 2.**np.round(np.log2(vlim))
    
    yticklabels=[]
    yticks = []
    ylimfound=np.zeros((1,2))
    for i, z in enumerate(PS.electrodeParams['z']):
        ind = np.arange(x0.size) - x0.size/2
        ax.plot(ind, xcorr_all[i,::-1] * 100. / vlimround + z, 'k',
                clip_on=True, rasterized=False)
        yticklabels.append('ch. %i' %(i+1))
        yticks.append(z)    

    remove_axis_junk(ax)
    ax.set_title(title)
    ax.set_xlabel(r'lag $\tau$ (ms)')

    ax.set_xlim(-lag, lag)
    ax.set_ylim(z-100, 100)
    
    axis = ax.axis()
    ax.vlines(0, axis[2], axis[3], 'r', 'dotted')

    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create a scaling bar
    ax.plot([lag, lag],
        [0, 100], lw=2, color='k', clip_on=False)
    ax.text(lag, 50, r'CC=%.2f' % vlimround,
            rotation='vertical', va='center')


#the CachedNetwork is needed for plotting
networkSim = CachedNetwork(
    simtime = BN.simtime,
    dt = BN.dt,
    spike_output_path = BN.spike_output_path,
    label = BN.label,
    ext = 'gdf',
    N_X = np.array([BN.NE, BN.NI]),
    X = PS.X,
    cmap='rainbow_r'
)


#turn off interactive plotting
plt.ioff()

if RANK == 0:
    #create network raster plot
    fig = networkSim.raster_plots(xlim=(500, 1000), markersize=2.)
    fig.savefig(os.path.join(PS.figures_path, 'network.pdf'), dpi=300)
    


    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, aspect='equal')
    fig.savefig(os.path.join(PS.figures_path, 'layers.pdf'), dpi=300)
    


    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_soma_locations(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, )
    fig.savefig(os.path.join(PS.figures_path, 'soma_locations.pdf'), dpi=300)
    


    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    aspect='equal')
    plot_morphologies(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, )
    fig.savefig(os.path.join(PS.figures_path, 'populations.pdf'), dpi=300)
    


    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    aspect='equal')
    plot_two_cells(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, )
    fig.savefig(os.path.join(PS.figures_path, 'cell_models.pdf'), dpi=300)
    


    #plot EX morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, X=['EX'], markers=['^'], colors=['r'],
                    aspect='equal')
    plot_morphologies(ax, X=['EX'], markers=['^'], colors=['r'],
                    isometricangle=np.pi/12, )
    fig.savefig(os.path.join(PS.figures_path, 'EX_population.pdf'), dpi=300)



    #plot IN morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, X=['IN'], markers=['o'], colors=['b'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax, X=['IN'], markers=['o'], colors=['b'],
                    isometricangle=np.pi/12, )
    fig.savefig(os.path.join(PS.figures_path, 'IN_population.pdf'), dpi=300)

    

    #plot compound LFP and CSD traces
    fig = plt.figure()
    gs = gridspec.GridSpec(2,8)
    
    ax0 = fig.add_subplot(gs[:,:2])
    ax1 = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 4:])
    ax1.set_title('CSD')
    ax2.set_title('LFP')

    plot_population(ax0, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax0, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, )

    plot_signal_sum(ax1,
                    fname=os.path.join(PS.savefolder, 'CSDsum.h5'),
                    unit='$\mu$Amm$^{-3}$', T=(500, 1000))
    
    plot_signal_sum(ax2,
                    fname=os.path.join(PS.savefolder, 'LFPsum.h5'),
                    unit='mV', T=(500, 1000))
    
    fig.savefig(os.path.join(PS.figures_path, 'compound_signals.pdf'), dpi=300)



    #plot compound LFP and CSD traces
    fig = plt.figure()
    gs = gridspec.GridSpec(2,8)
    
    ax0 = fig.add_subplot(gs[:,:2])
    ax1 = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 4:])
    ax1.set_title('CSD')
    ax2.set_title('LFP')

    plot_population(ax0, X=['EX'], markers=['^'], colors=['r'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax0, X=['EX'], markers=['^'], colors=['r'],
                    isometricangle=np.pi/12, )

    plot_signal_sum(ax1,
                    fname=os.path.join(PS.populations_path, 'EX_population_CSD.h5'),
                    unit='$\mu$Amm$^{-3}$', T=(500, 1000),color='r')
    
    plot_signal_sum(ax2,
                    fname=os.path.join(PS.populations_path, 'EX_population_LFP.h5'),
                    unit='mV', T=(500, 1000), color='r')
    fig.savefig(os.path.join(PS.figures_path, 'population_EX_signals.pdf'), dpi=300)



    #plot compound LFP and CSD traces
    fig = plt.figure()
    gs = gridspec.GridSpec(2,8)
    
    ax0 = fig.add_subplot(gs[:,:2])
    ax1 = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 4:])
    ax1.set_title('CSD')
    ax2.set_title('LFP')

    plot_population(ax0, X=['IN'], markers=['o'], colors=['b'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax0, X=['IN'], markers=['o'], colors=['b'],
                    isometricangle=np.pi/12, )

    plot_signal_sum(ax1,
                    fname=os.path.join(PS.populations_path, 'IN_population_CSD.h5'),
                    unit='$\mu$Amm$^{-3}$', T=(500, 1000),color='b')
    
    plot_signal_sum(ax2,
                    fname=os.path.join(PS.populations_path, 'IN_population_LFP.h5'),
                    unit='mV', T=(500, 1000), color='b')
    fig.savefig(os.path.join(PS.figures_path, 'population_IN_signals.pdf'), dpi=300)



    #correlate global firing rate of network with CSD/LFP across channels
    #compute firing rates
    x, y = networkSim.get_xy((0, BN.simtime))
    bins = np.arange(0, BN.simtime + 1)
        
    xx = np.r_[x['EX'], x['IN']]
        
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    gs = gridspec.GridSpec(4,3)
    
    ax0 = fig.add_subplot(gs[0, :2])
    ax1 = fig.add_subplot(gs[1:, :2])
    ax2 = fig.add_subplot(gs[:, 2])

    ax0.hist(xx, bins=bins[500:], color='r')
    ax0.axis('tight')
    ax0.set_title('spike rate (s$^{-1}$)')

    plot_signal_sum(ax1,
                    fname=os.path.join(PS.savefolder, 'LFPsum.h5'),
                    unit='mV', T=(500, 1000))
    ax1.set_title('LFP')

    fname=os.path.join(PS.savefolder, 'LFPsum.h5')
    f = h5py.File(fname, 'r')
    data = f['data'].value
    
    r, t = np.histogram(xx, bins)
    plot_correlation(r, data[:, 1:], ax2, lag=50, title='rate-LFP xcorr')

    fig.savefig(os.path.join(PS.figures_path, 'compound_signal_correlations.pdf'),
                dpi=300)


    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    aspect='equal')
    plot_morphologies(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12, )

    #keep current axes bounds, so we can put it back
    axis = ax.axis()
    
    #some additional plot annotations
    ax.text(-275, -300, 'EX', clip_on=False, va='center', zorder=200)
    ax.add_patch(plt.Rectangle((-290, -340), fc='r', ec='k', alpha=0.5,
        width=80, height=80, clip_on=False, zorder=200))
    ax.arrow(-210, -300, 50, 50, head_width=10, head_length=10, fc='r',
             lw=5, ec='k', alpha=0.5, zorder=200)
    ax.arrow(-210, -300, 50, -50, head_width=10, head_length=10, fc='r',
             lw=5, ec='k', alpha=0.5, zorder=200)
    
    ax.text(-275, -400, 'IN', clip_on=False, va='center', zorder=200)
    ax.add_patch(plt.Rectangle((-290, -440), fc='b', ec='k', alpha=0.5,
        width=80, height=80, clip_on=False, zorder=200))
    ax.arrow(-210, -400, 50, 0, head_width=10, head_length=10, fc='b',
             lw=5, ec='k', alpha=0.5, zorder=200)    
    
    fig.savefig(os.path.join(PS.figures_path, 'populations_vII.pdf'), dpi=300)

    



    if SIZE == 1:
        plt.show()
    else:
        plt.close('all')

COMM.Barrier()



