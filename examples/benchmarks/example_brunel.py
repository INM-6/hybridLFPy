#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

'''
# from example_plotting import *
import brunel_alpha_nest as BN
import lfpykit
from mpi4py import MPI
import h5py
from parameters import ParameterSet
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
import nest  # import not used, but we load NEST anyway in order determine if
import neuron  # NEURON compiled with MPI must be imported before NEST/mpi4py
from time import time
from matplotlib import gridspec
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.style
matplotlib.style.use('classic')
# to avoid being aware of MPI
# network is run in parallel


########## matplotlib settings ###########################################
plt.close('all')
plt.rcParams.update({'figure.figsize': [10.0, 8.0]})

#######################################
# Capture command line values
#######################################
# load parameter file
md5 = sys.argv[1]
pset = ParameterSet(os.path.join('parameters', '{}.txt'.format(md5)))

# set some seed values
SEED = pset.GLOBALSEED
SIMULATIONSEED = pset.GLOBALSEED + 1234
np.random.seed(SEED)


################# Initialization of MPI stuff ############################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# if True, execute full model. If False, do only the plotting.
# Simulation results must exist.
properrun = True


# check if mod file for synapse model specified in alphaisyn.mod is loaded
if not hasattr(neuron.h, 'AlphaISyn'):
    if RANK == 0:
        os.system('nrnivmodl')
    COMM.Barrier()
    neuron.load_mechanisms('.')


##########################################################################
# PARAMETERS
##########################################################################

# Set up parameters using the NeuroTools.parameters.ParameterSet class.

# Access parameters defined in example script implementing the network using
# pynest, brunel_alpha_nest.py, adapted from the NEST v2.4.1 release. This will
# not execute the network model, but see below.


# set up file destinations differentiating between certain output
savefolder = md5
# BN.spike_output_path = os.path.join(savefolder, 'spiking_output_path')
PS = ParameterSet(dict(
    # Main folder of simulation output
    savefolder=savefolder,  # 'simulation_output_example_brunel',

    # make a local copy of main files used in simulations
    sim_scripts_path=os.path.join(savefolder, 'sim_scripts'),

    # destination of single-cell output during simulation
    cells_path=os.path.join(savefolder, 'cells'),

    # destination of cell- and population-specific signals, i.e., compund LFPs,
    # CSDs etc.
    populations_path=os.path.join(savefolder, 'populations'),

    # location of spike output from the network model
    spike_output_path=BN.spike_output_path,

    # destination of figure file output generated during model execution
    figures_path=os.path.join(savefolder, 'figures')
))


# population (and cell type) specific parameters
PS.update(dict(
    # no cell type specificity within each E-I population
    # hence X == x and Y == X
    X=["EX", "IN"],

    # population-specific LFPy.Cell parameters
    cellParams=dict(
        # excitory cells
        EX=dict(
            morphology='morphologies/ex.swc',
            v_init=BN.neuron_params['E_L'],
            cm=1.0,
            Ra=150,
            passive=True,
            passive_parameters=dict(
                g_pas=1. / (BN.neuron_params['tau_m'] * 1E3),  # assume cm=1
                e_pas=BN.neuron_params['E_L']),
            nsegs_method='fixed_length',  # 'lambda_f',
            max_nsegs_length=10.,
            # lambda_f=100,
            dt=BN.dt,
            tstart=0,
            tstop=BN.simtime,
            verbose=False,
        ),
        # inhibitory cells
        IN=dict(
            morphology='morphologies/in.swc',
            v_init=BN.neuron_params['E_L'],
            cm=1.0,
            Ra=150,
            passive=True,
            passive_parameters=dict(
                g_pas=1. / (BN.neuron_params['tau_m'] * 1E3),  # assume cm=1
                e_pas=BN.neuron_params['E_L']),
            nsegs_method='fixed_length',  # 'lambda_f',
            max_nsegs_length=10.,
            # lambda_f=100,
            dt=BN.dt,
            tstart=0,
            tstop=BN.simtime,
            verbose=False,
        )),

    # assuming excitatory cells are pyramidal
    rand_rot_axis=dict(
        EX=['z'],
        IN=['x', 'y', 'z'],
    ),


    # kwargs passed to LFPy.Cell.simulate().
    # It can be empty, but if `rec_imem=True`, the signal predictions will be
    # performed using recorded transmembrane currents.
    simulationParams=dict(rec_imem=True),

    # set up parameters corresponding to cylindrical model populations
    populationParams=dict(
        EX=dict(
            number=BN.NE,
            radius=100,
            z_min=-450,
            z_max=-350,
            min_cell_interdist=1.,
            min_r=[[-1E199, -600, -550, 1E99], [0, 0, 10, 10]],
        ),
        IN=dict(
            number=BN.NI,
            radius=100,
            z_min=-450,
            z_max=-350,
            min_cell_interdist=1.,
            min_r=[[-1E199, -600, -550, 1E99], [0, 0, 10, 10]],
        ),
    ),

    # set the boundaries between the "upper" and "lower" layer
    layerBoundaries=[[0., -300],
                     [-300, -500]],

    # set the geometry of the virtual recording device
    electrodeParams=dict(
        # contact locations:
        x=[0] * 6,
        y=[0] * 6,
        z=[x * -100. for x in range(6)],
        # extracellular conductivity:
        sigma=0.3,
        # contact surface normals, radius, n-point averaging
        N=[[1, 0, 0]] * 6,
        r=5,
        n=20,
        seedvalue=None,
        # dendrite line sources, soma as sphere source (Linden2014)
        method='root_as_point',
    ),

    # parameters for LFPykit.LaminarCurrentSourceDensity
    CSDParams=dict(
        z=np.array([[-(i + 1) * 100, -i * 100] for i in range(6)]) + 50.,
        r=np.ones(6) * 100  # same as population radius
    ),

    # runtime, cell-specific attributes and output that will be stored
    savelist=[],

    # time resolution of saved signals
    dt_output=1.
))


# for each population, define layer- and population-specific connectivity
# parameters
PS.update(dict(
    # number of connections from each presynaptic population onto each
    # layer per postsynaptic population, preserving overall indegree
    k_yXL=dict(
        EX=[[int(0.5 * BN.CE), 0],
            [int(0.5 * BN.CE), BN.CI]],
        IN=[[0, 0],
            [BN.CE, BN.CI]],
    ),

    # set up table of synapse weights from each possible presynaptic population
    J_yX=dict(
        EX=[BN.J_ex * 1E-3, BN.J_in * 1E-3],
        IN=[BN.J_ex * 1E-3, BN.J_in * 1E-3],
    ),

    # set up synapse parameters as derived from the network
    synParams=dict(
        EX=dict(
            section=['apic', 'dend', 'soma'],
            # section=['apic', 'dend'],
            # tau = [BN.tauSyn, BN.tauSyn],
            syntype='AlphaISyn'
        ),
        IN=dict(
            section=['apic', 'dend', 'soma'],
            # section=['dend', 'soma'],
            # tau = [BN.tauSyn, BN.tauSyn],
            syntype='AlphaISyn'
        ),
    ),

    # set up table of synapse time constants from each presynaptic populations
    tau_yX=dict(
        EX=[BN.tauSyn, BN.tauSyn],
        IN=[BN.tauSyn, BN.tauSyn]
    ),

    # set up delays, here using fixed delays of network
    synDelayLoc=dict(
        EX=[BN.delay, BN.delay],
        IN=[BN.delay, BN.delay],
    ),
    # no distribution of delays
    synDelayScale=dict(
        EX=[None, None],
        IN=[None, None],
    ),
))


# putative mappting between population type and cell type specificity,
# but here all presynaptic senders are also postsynaptic targets
PS.update(dict(
    mapping_Yy=list(zip(PS.X, PS.X))
))


# In[2]:


##########################################################################
# MAIN simulation procedure                                              #
##########################################################################

# tic toc
tic = time()

######### Perform network simulation #####################################
if properrun:
    # set up the file destination, removing old results by default
    setup_file_dest(PS, clearDestination=True)

if properrun:
    # execute network simulation
    # BN.simulate()
    pass

# wait for the network simulation to finish, resync MPI threads
COMM.Barrier()

# Create an object representation containing the spiking activity of the network
# simulation output that uses sqlite3. Again, kwargs are derived from the brunel
# network instance.
networkSim = CachedNetwork(
    simtime=BN.simtime,
    dt=BN.dt,
    spike_output_path=BN.spike_output_path,
    label=BN.label,
    ext='dat',
    GIDs={'EX': [1, BN.NE], 'IN': [BN.NE + 1, BN.NI]},
    X=['EX', 'IN'],
    cmap='rainbow_r',
    skiprows=3,
)


if RANK == 0:
    toc = time() - tic
    print('NEST simulation and gdf file processing done in  %.3f seconds' % toc)


# Set up LFPykit measurement probes for LFPs and CSDs
if properrun:
    probes = []
    probes.append(lfpykit.RecExtElectrode(cell=None, **PS.electrodeParams))
    probes.append(
        lfpykit.LaminarCurrentSourceDensity(
            cell=None, **PS.CSDParams))


# In[3]:

# measure population simulation times including setup etc.
ticc = time()

# arbor sim.run cumulative time
ticc_run = np.array(0.)

# %%prun -s cumulative -q -l 20 -T prun0
####### Set up populations ###############################################
if properrun:
    # iterate over each cell type, and create populationulation object
    for i, Y in enumerate(PS.X):
        # create population:
        pop = Population(
            cellParams=PS.cellParams[Y],
            rand_rot_axis=PS.rand_rot_axis[Y],
            simulationParams=PS.simulationParams,
            populationParams=PS.populationParams[Y],
            y=Y,
            layerBoundaries=PS.layerBoundaries,
            savelist=PS.savelist,
            savefolder=PS.savefolder,
            probes=probes,
            dt_output=PS.dt_output,
            POPULATIONSEED=SIMULATIONSEED + i,
            X=PS.X,
            networkSim=networkSim,
            k_yXL=PS.k_yXL[Y],
            synParams=PS.synParams[Y],
            synDelayLoc=PS.synDelayLoc[Y],
            synDelayScale=PS.synDelayScale[Y],
            J_yX=PS.J_yX[Y],
            tau_yX=PS.tau_yX[Y],
        )

        # run population simulation and collect the data
        pop.run()
        pop.collect_data()

        # arbor sim.run cumulative time
        ticc_run += pop._cumulative_sim_time

        # object no longer needed
        del pop

# tic toc
tocc = time() - ticc
if RANK == 0:
    with open(os.path.join(savefolder, 'time_pop.txt'), 'w') as f:
        f.write(f'{tocc}')

if RANK == 0:
    tocc_run = np.zeros_like(ticc_run)
else:
    tocc_run = None

COMM.Reduce(ticc_run, tocc_run, op=MPI.SUM, root=0)
if RANK == 0:
    tocc_run /= SIZE
    with open(os.path.join(savefolder, 'time_run.txt'), 'w') as f:
        f.write(f'{float(tocc_run)}')

# In[4]:


# print(open('prun0', 'r').read())


# In[5]:


####### Postprocess the simulation output ################################


# reset seed, but output should be deterministic from now on
np.random.seed(SIMULATIONSEED)

if properrun:
    # do some postprocessing on the collected data, i.e., superposition
    # of population LFPs, CSDs etc
    postproc = PostProcess(y=PS.X,
                           dt_output=PS.dt_output,
                           savefolder=PS.savefolder,
                           mapping_Yy=PS.mapping_Yy,
                           savelist=PS.savelist,
                           probes=probes,
                           cells_subfolder=os.path.split(PS.cells_path)[-1],
                           populations_subfolder=os.path.split(PS.populations_path)[-1],
                           figures_subfolder=os.path.split(PS.figures_path)[-1])

    # run through the procedure
    postproc.run()

    # create tar-archive with output for plotting, ssh-ing etc.
    postproc.create_tar_archive()


COMM.Barrier()

# tic toc
print('Execution time: %.3f seconds' % (time() - tic))
