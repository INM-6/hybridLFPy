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
from time import time
import nest #not used, but load order determine if network is run in parallel
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
from NeuroTools.parameters import ParameterSet
import h5py
import neuron
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
if not hasattr(neuron.h, 'AlphaISyn'):
    if RANK == 0:
        os.system('nrnivmodl')
    COMM.Barrier()
    neuron.load_mechanisms('.')


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
    
    #set up table of synapse weights from each possible presynaptic population
    J_yX = dict(
        EX = [BN.J_ex*1E-3, BN.J_in*1E-3],
        IN = [BN.J_ex*1E-3, BN.J_in*1E-3],
    ),

    #set up synapse parameters as derived from the network
    synParams = dict(
        EX = dict(
            section = ['apic', 'dend'],
            # tau = [BN.tauSyn, BN.tauSyn],
            syntype = 'AlphaISyn'
        ),
        IN = dict(
            section = ['dend', 'soma'],
            # tau = [BN.tauSyn, BN.tauSyn],
            syntype = 'AlphaISyn'            
        ),
    ),
    
    #set up table of synapse time constants from each presynaptic populations
    tau_yX = dict(
        EX = [BN.tauSyn, BN.tauSyn],
        IN = [BN.tauSyn, BN.tauSyn]
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
    #execute network simulation
    BN.simulate()

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
    GIDs = {'EX' : [1, BN.NE], 'IN' : [BN.NE+1, BN.NI]},
    cmap='rainbow_r',
)


if RANK == 0:
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
                tau_yX = PS.tau_yX[Y],
            )
    
        #run population simulation and collect the data
        pop.run()
        pop.collect_data()
    
        #object no longer needed
        del pop


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
                           savelist = PS.savelist,
                           cells_subfolder = os.path.split(PS.cells_path)[-1],
                           populations_subfolder = os.path.split(PS.populations_path)[-1],
                           figures_subfolder = os.path.split(PS.figures_path)[-1]
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

#import some plotter functions
from example_plotting import *

#turn off interactive plotting
plt.ioff()

if RANK == 0:
    #create network raster plot
    fig = networkSim.raster_plots(xlim=(500, 1000), markersize=2.)
    fig.savefig(os.path.join(PS.figures_path, 'network.pdf'), dpi=300)
    

    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    isometricangle=np.pi/12, aspect='equal')
    fig.savefig(os.path.join(PS.figures_path, 'layers.pdf'), dpi=300)
    

    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_soma_locations(ax, X=['EX', 'IN'],
                        populations_path=PS.populations_path,
                        markers=['^', 'o'], colors=['r', 'b'],
                        isometricangle=np.pi/12, )
    fig.savefig(os.path.join(PS.figures_path, 'soma_locations.pdf'), dpi=300)
    

    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    aspect='equal')
    plot_morphologies(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12,
                    populations_path=PS.populations_path,
                    cellParams=PS.cellParams)
    fig.savefig(os.path.join(PS.figures_path, 'populations.pdf'), dpi=300)
    

    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    aspect='equal')
    plot_individual_morphologies(ax, X=['EX', 'IN'], markers=['^', 'o'],
                                 colors=['r', 'b'],
                                 isometricangle=np.pi/12,
                                 cellParams=PS.cellParams,
                                 populationParams=PS.populationParams)
    fig.savefig(os.path.join(PS.figures_path, 'cell_models.pdf'), dpi=300)
    

    #plot EX morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX'], markers=['^'], colors=['r'],
                    layers = ['upper', 'lower'],
                    aspect='equal')
    plot_morphologies(ax, X=['EX'], markers=['^'], colors=['r'],
                    isometricangle=np.pi/12,
                    populations_path=PS.populations_path,
                    cellParams=PS.cellParams)
    fig.savefig(os.path.join(PS.figures_path, 'EX_population.pdf'), dpi=300)


    #plot IN morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['IN'], markers=['o'], colors=['b'],
                    layers = ['upper', 'lower'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax, X=['IN'], markers=['o'], colors=['b'],
                    isometricangle=np.pi/12,
                    populations_path=PS.populations_path,
                    cellParams=PS.cellParams)
    fig.savefig(os.path.join(PS.figures_path, 'IN_population.pdf'), dpi=300)

    
    #plot compound LFP and CSD traces
    fig = plt.figure()
    gs = gridspec.GridSpec(2,8)
    
    ax0 = fig.add_subplot(gs[:,:2])
    ax1 = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 4:])
    ax1.set_title('CSD')
    ax2.set_title('LFP')

    plot_population(ax0, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax0, X=['EX', 'IN'], markers=['^', 'o'],
                      colors=['r', 'b'],
                    isometricangle=np.pi/12,
                    populations_path=PS.populations_path,
                    cellParams=PS.cellParams)

    plot_signal_sum(ax1, z=PS.electrodeParams['z'],
                    fname=os.path.join(PS.savefolder, 'CSDsum.h5'),
                    unit='$\mu$Amm$^{-3}$', T=(500, 1000))
    
    plot_signal_sum(ax2, z=PS.electrodeParams['z'],
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

    plot_population(ax0, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX'], markers=['^'], colors=['r'],
                    layers = ['upper', 'lower'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax0, X=['EX'], markers=['^'], colors=['r'],
                    isometricangle=np.pi/12,
                    populations_path=PS.populations_path,
                    cellParams=PS.cellParams)

    plot_signal_sum(ax1, z=PS.electrodeParams['z'],
                    fname=os.path.join(PS.populations_path,
                                       'EX_population_CSD.h5'),
                    unit='$\mu$Amm$^{-3}$', T=(500, 1000),color='r')
    
    plot_signal_sum(ax2, z=PS.electrodeParams['z'],
                    fname=os.path.join(PS.populations_path,
                                       'EX_population_LFP.h5'),
                    unit='mV', T=(500, 1000), color='r')
    fig.savefig(os.path.join(PS.figures_path, 'population_EX_signals.pdf'),
                dpi=300)


    #plot compound LFP and CSD traces
    fig = plt.figure()
    gs = gridspec.GridSpec(2,8)
    
    ax0 = fig.add_subplot(gs[:,:2])
    ax1 = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 4:])
    ax1.set_title('CSD')
    ax2.set_title('LFP')

    plot_population(ax0, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['IN'], markers=['o'], colors=['b'],
                    layers = ['upper', 'lower'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax0, X=['IN'], markers=['o'], colors=['b'],
                    isometricangle=np.pi/12,
                    populations_path=PS.populations_path,
                    cellParams=PS.cellParams)

    plot_signal_sum(ax1, z=PS.electrodeParams['z'],
                    fname=os.path.join(PS.populations_path,
                                       'IN_population_CSD.h5'),
                    unit='$\mu$Amm$^{-3}$', T=(500, 1000),color='b')
    
    plot_signal_sum(ax2, z=PS.electrodeParams['z'],
                    fname=os.path.join(PS.populations_path,
                                       'IN_population_LFP.h5'),
                    unit='mV', T=(500, 1000), color='b')
    fig.savefig(os.path.join(PS.figures_path, 'population_IN_signals.pdf'),
                dpi=300)


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

    plot_signal_sum(ax1, z=PS.electrodeParams['z'],
                    fname=os.path.join(PS.savefolder, 'LFPsum.h5'),
                    unit='mV', T=(500, 1000))
    ax1.set_title('LFP')

    fname=os.path.join(PS.savefolder, 'LFPsum.h5')
    f = h5py.File(fname, 'r')
    data = f['data'].value
    
    r, t = np.histogram(xx, bins)
    plot_correlation(z_vec=PS.electrodeParams['z'], x0=r, x1=data[:, 1:],
                     ax=ax2, lag=50, title='rate-LFP xcorr')

    fig.savefig(os.path.join(PS.figures_path,
                             'compound_signal_correlations.pdf'), dpi=300)


    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_population(ax, PS.populationParams, PS.electrodeParams,
                    PS.layerBoundaries,
                    X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    aspect='equal')
    plot_morphologies(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                    isometricangle=np.pi/12,
                    populations_path=PS.populations_path,
                    cellParams=PS.cellParams)

    #keep current axes bounds, so we can put it back
    axis = ax.axis()
    
    #some additional plot annotations
    ax.text(-275, -300, 'EX', clip_on=False, va='center', zorder=500)
    ax.add_patch(plt.Rectangle((-290, -340), fc='r', ec='k', alpha=0.5,
        width=80, height=80, clip_on=False, zorder=500))
    ax.arrow(-210, -300, 50, 50, head_width=20, head_length=20, width=10,
             fc='r', lw=1, ec='w', alpha=1, zorder=500)
    ax.arrow(-210, -300, 50, -50, head_width=20, head_length=20, width=10,
             fc='r', lw=1, ec='w', alpha=1, zorder=500)
    
    ax.text(-275, -400, 'IN', clip_on=False, va='center', zorder=500)
    ax.add_patch(plt.Rectangle((-290, -440), fc='b', ec='k', alpha=0.5,
        width=80, height=80, clip_on=False, zorder=500))
    ax.arrow(-210, -400, 50, 0, head_width=20, head_length=20, width=10, fc='b',
             lw=1, ec='w', alpha=1, zorder=500)    
    
    fig.savefig(os.path.join(PS.figures_path, 'populations_vII.pdf'), dpi=300)


    if SIZE == 1:
        plt.show()
    else:
        plt.close('all')

COMM.Barrier()



