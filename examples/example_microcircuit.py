#!/usr/bin/env python
'''
Hybrid LFP scheme example script, applying the methodology with the model of:

Potjans, T. and Diesmann, M. "The Cell-Type Specific Cortical Microcircuit:
Relating Structure and Activity in a Full-Scale Spiking Network Model".
Cereb. Cortex (2014) 24 (3): 785-806.
doi: 10.1093/cercor/bhs358

Synopsis of the main simulation procedure:
1. Loading of parameterset
    a. network parameters
    b. parameters for hybrid scheme
2. Set up file destinations for different simulation output
3. network simulation
    a. execute network simulation using NEST (www.nest-simulator.org)
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
    mpirun -np 64 python example_microcircuit.py
where the number 64 is the desired number of MPI threads & CPU cores

Given the size of the network and demands for the multi-compartment LFP-
predictions using the present scheme, running the model on a large scale
compute facility is strongly encouraged.

'''
import os
import numpy as np
from time import time
import nest_simulation
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
import nest_output_processing
import neuron
from mpi4py import MPI


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


#check if mod file for synapse model specified in expisyn.mod is loaded
if not hasattr(neuron.h, 'ExpSynI'):
    if RANK == 0:
        os.system('nrnivmodl')
    COMM.Barrier()
    neuron.load_mechanisms('.')
    

################################################################################
## PARAMETERS
################################################################################

from example_microcircuit_params import multicompartment_params, \
                                point_neuron_network_params

#Full set of parameters including network parameters
params = multicompartment_params()


###############################################################################
# MAIN simulation procedure
###############################################################################

#tic toc
tic = time()

if properrun:
    #set up the file destination
    setup_file_dest(params, clearDestination=True)

######## Perform network simulation ############################################

if properrun:
    ##initiate nest simulation with only the point neuron network parameter class
    networkParams = point_neuron_network_params()
    nest_simulation.sli_run(parameters=networkParams,
                            fname='microcircuit.sli',
                            verbosity='M_WARNING')
    
    #preprocess the gdf files containing spiking output, voltages, weighted and
    #spatial input spikes and currents:
    nest_output_processing.merge_gdf(networkParams,
                                raw_label=networkParams.spike_detector_label,
                                file_type='gdf',
                                fileprefix=params.networkSimParams['label'])


#Create an object representation of the simulation output that uses sqlite3
networkSim = CachedNetwork(**params.networkSimParams)


toc = time() - tic
print 'NEST simulation and gdf file processing done in  %.3f seconds' % toc


####### Set up populations #####################################################

if properrun:
    #iterate over each cell type, run single-cell simulations and create
    #population object
    for i, y in enumerate(params.y):
        #create population:
        pop = Population(
                #parent class parameters
                cellParams = params.yCellParams[y],
                rand_rot_axis = params.rand_rot_axis[y],
                simulationParams = params.simulationParams,
                populationParams = params.populationParams[y],
                y = y,
                layerBoundaries = params.layerBoundaries,
                electrodeParams = params.electrodeParams,
                savelist = params.savelist,
                savefolder = params.savefolder,
                calculateCSD = params.calculateCSD,
                dt_output = params.dt_output, 
                POPULATIONSEED = SIMULATIONSEED + i,
                #daughter class kwargs
                X = params.X,
                networkSim = networkSim,
                k_yXL = params.k_yXL[y],
                synParams = params.synParams[y],
                synDelayLoc = params.synDelayLoc[y],
                synDelayScale = params.synDelayScale[y],
                J_yX = params.J_yX[y],
                recordSingleContribFrac = params.recordSingleContribFrac,
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
    postproc = PostProcess(y = params.y,
                           dt_output = params.dt_output,
                           savefolder = params.savefolder,
                           mapping_Yy = params.mapping_Yy,
                           )
    
    #run through the procedure
    postproc.run()
    
    #create tar-archive with output
    postproc.create_tar_archive()

#tic toc
print 'Execution time: %.3f seconds' %  (time() - tic)


################################################################################
# Create set of plots from simulation output
################################################################################

########## matplotlib settings #################################################
import matplotlib.pyplot as plt
from example_plotting import *

plt.close('all')

#turn off interactive plotting
plt.ioff()

if RANK == 0:
    #create network raster plot
    x, y = networkSim.get_xy((500, 1000), fraction=1)
    fig, ax = plt.subplots(1, figsize=(5,8))
    fig.subplots_adjust(left=0.18)
    networkSim.plot_raster(ax, (500, 1000), x, y, markersize=1, marker='o',
                           alpha=.5,legend=False, pop_names=True)
    remove_axis_junk(ax)
    ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
    ax.set_ylabel('population', labelpad=0.1)
    ax.set_title('network raster')
    fig.savefig(os.path.join(params.figures_path, 'network_raster.pdf'), dpi=300)
    #raise Exception
    
    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.18)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['^' if 'p' in y else '*' for y in params.y],
                    colors=['r' if 'p' in y else 'b' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/12, aspect='equal')
    fig.savefig(os.path.join(params.figures_path, 'layers.pdf'), dpi=300)
    

    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.18)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['^' if 'p' in y else '*' for y in params.y],
                    colors=['r' if 'p' in y else 'b' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_soma_locations(ax, X=params.y,
                        populations_path=params.populations_path,
                        markers=['^' if 'p' in y else '*' for y in params.y],
                        colors=['r' if 'p' in y else 'b' for y in params.y],
                        isometricangle=np.pi/12, )
    fig.savefig(os.path.join(params.figures_path, 'soma_locations.pdf'), dpi=300)
    

    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.18)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['^' if 'p' in y else '*' for y in params.y],
                    colors=['r' if 'p' in y else 'b' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax,
                      X=params.y,
                      markers=['^' if 'p' in y else '*' for y in params.y],
                      colors=['r' if 'p' in y else 'b' for y in params.y],
                      isometricangle=np.pi/12,
                      populations_path=params.populations_path,
                      cellParams=params.yCellParams)
    fig.savefig(os.path.join(params.figures_path, 'populations.pdf'), dpi=300)



    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.18)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['^' if 'p' in y else '*' for y in params.y],
                    colors=['r' if 'p' in y else 'b' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_two_cells(ax,
                   X=params.y,
                   markers=['^' if 'p' in y else '*' for y in params.y],
                   colors=['r' if 'p' in y else 'b' for y in params.y],
                   isometricangle=np.pi/12, cellParams=params.yCellParams,
                   populationParams=params.populationParams)
    fig.savefig(os.path.join(params.figures_path, 'cell_models.pdf'), dpi=300)


    #plot compound LFP and CSD traces
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2,8)
    
    ax0 = fig.add_subplot(gs[:,:2])
    ax1 = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 4:])
    ax1.set_title('CSD')
    ax2.set_title('LFP')

    plot_population(ax0, params.populationParams, params.electrodeParams, params.layerBoundaries,
                    X=params.y,
                    markers=['^' if 'p' in y else '*' for y in params.y],
                    colors=['r' if 'p' in y else 'b' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/12, aspect='equal')
    plot_morphologies(ax0,
                      X=params.y,
                      markers=['^' if 'p' in y else '*' for y in params.y],
                      colors=['r' if 'p' in y else 'b' for y in params.y],
                      isometricangle=np.pi/12,
                      populations_path=params.populations_path,
                      cellParams=params.yCellParams)

    plot_signal_sum(ax1, z=params.electrodeParams['z'],
                    fname=os.path.join(params.savefolder, 'CSDsum.h5'),
                    unit='$\mu$Amm$^{-3}$', T=(500, 1000))
    
    plot_signal_sum(ax2, z=params.electrodeParams['z'],
                    fname=os.path.join(params.savefolder, 'LFPsum.h5'),
                    unit='mV', T=(500, 1000))
    
    fig.savefig(os.path.join(params.figures_path, 'compound_signals.pdf'), dpi=300)

    plt.show()

