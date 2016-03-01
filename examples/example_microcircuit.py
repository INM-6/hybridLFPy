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
    b. merge nest spike output from different MPI ranks
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
import nest
from hybridLFPy import PostProcess, Population, CachedNetwork
from hybridLFPy import setup_file_dest, helpers
from glob import glob
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


################################################################################
# Function declaration(s)
################################################################################

def merge_gdf(model_params, raw_label='spikes_', file_type='gdf',
              fileprefix='spikes'):
    '''
    NEST produces one file per virtual process containing recorder output. 
    This function gathers and combines them into one single file per 
    network population.
    
    Parameters
    ----------
    model_params : object
        network parameters object
    
    Returns
    -------
    None
    
    '''
    def get_raw_gids(model_params):
        '''
        Reads text file containing gids of neuron populations as created within
        the NEST simulation. These gids are not continuous as in the simulation
        devices get created in between.
        
        Parameters
        ----------
        model_params : object
            network parameters object
        
        
        Returns
        -------
        gids : list
            list of neuron ids and value (spike time, voltage etc.)
        
        '''
        gidfile = open(os.path.join(model_params.raw_nest_output_path,
                                    model_params.GID_filename),'r') 
        gids = [] 
        for l in gidfile :
            a = l.split()
            gids.append([int(a[0]),int(a[1])])
        return gids
    
    #some preprocessing
    raw_gids = get_raw_gids(model_params)
    pop_sizes = [raw_gids[i][1]-raw_gids[i][0]+1
                 for i in np.arange(model_params.Npops)]
    raw_first_gids =  [raw_gids[i][0] for i in np.arange(model_params.Npops)]
    converted_first_gids = [int(1 + np.sum(pop_sizes[:i]))
                            for i in np.arange(model_params.Npops)]

    for pop_idx in np.arange(model_params.Npops):
        if pop_idx % SIZE == RANK:
            files = glob(os.path.join(model_params.raw_nest_output_path,
                                      raw_label + str(pop_idx) +
                                      '*.' + file_type))
            gdf = [] # init
            for f in files:
                new_gdf = helpers.read_gdf(f)
                for line in new_gdf:
                    line[0] = line[0] - raw_first_gids[pop_idx] + \
                              converted_first_gids[pop_idx]
                    gdf.append(line)
            
            print 'writing: %s' % os.path.join(model_params.spike_output_path,
                                            fileprefix +
                                            '_%s.gdf' % model_params.X[pop_idx])
            helpers.write_gdf(gdf, os.path.join(model_params.spike_output_path,
                                        fileprefix +
                                        '_%s.gdf' % model_params.X[pop_idx]))
    
    COMM.Barrier()

    return


def dict_of_numpyarray_to_dict_of_list(d):
    '''
    Convert dictionary containing numpy arrays to dictionary containing lists
    
    Parameters
    ----------
    d : dict
        sli parameter name and value as dictionary key and value pairs
    
    Returns
    -------
    d : dict
        modified dictionary
    
    '''
    for key,value in d.iteritems():
        if isinstance(value,dict):  # if value == dict 
            # recurse
            d[key] = dict_of_numpyarray_to_dict_of_list(value)
        elif isinstance(value,np.ndarray): # or isinstance(value,list) :
            d[key] = value.tolist()
    return d


def send_nest_params_to_sli(p):
    '''
    Read parameters and send them to SLI
    
    Parameters
    ----------
    p : dict
        sli parameter name and value as dictionary key and value pairs
    
    Returns
    -------
    None
    '''
    for name in p.keys():
        value = p[name]
        if type(value) == np.ndarray:
            value = value.tolist()
        if type(value) == dict:
            value = dict_of_numpyarray_to_dict_of_list(value)
        if name == 'neuron_model': # special case as neuron_model is a
                                   # NEST model and not a string
            try:
                nest.sli_run('/'+name)
                nest.sli_push(value)
                nest.sli_run('eval')
                nest.sli_run('def')
            except: 
                print 'Could not put variable %s on SLI stack' % (name)
                print type(value)
        else:
            try:
                nest.sli_run('/'+name)
                nest.sli_push(value)
                nest.sli_run('def')
            except: 
                print 'Could not put variable %s on SLI stack' % (name)
                print type(value)
    return


def sli_run(parameters=object(),
            fname='microcircuit.sli',
            verbosity='M_ERROR'):
    '''
    Takes parameter-class and name of main sli-script as input, initiating the
    simulation.
    
    Parameters
    ----------
    parameters : object
        parameter class instance
    fname : str
        path to sli codes to be executed
    verbosity : str,
        nest verbosity flag
    
    Returns
    -------
    None
    
    '''
    # Load parameters from params file, and pass them to nest
    # Python -> SLI
    send_nest_params_to_sli(vars(parameters))
    
    #set SLI verbosity
    nest.sli_run("%s setverbosity" % verbosity)
    
    # Run NEST/SLI simulation
    nest.sli_run('(%s) run' % fname)


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
    #initiate nest simulation with only the point neuron network parameter class
    networkParams = point_neuron_network_params()
    sli_run(parameters=networkParams,
            fname='microcircuit.sli',
            verbosity='M_WARNING')
    
    #preprocess the gdf files containing spiking output, voltages, weighted and
    #spatial input spikes and currents:
    merge_gdf(networkParams,
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
                tau_yX = params.tau_yX[y],
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
    fig.subplots_adjust(left=0.2)
    networkSim.plot_raster(ax, (500, 1000), x, y, markersize=1, marker='o',
                           alpha=.5,legend=False, pop_names=True)
    remove_axis_junk(ax)
    ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
    ax.set_ylabel('population', labelpad=0.1)
    ax.set_title('network raster')
    fig.savefig(os.path.join(params.figures_path, 'network_raster.pdf'),
                dpi=300)
    #raise Exception
    
    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.2)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['*' if 'b' in y else '^' for y in params.y],
                    colors=['b' if 'b' in y else 'r' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/24, aspect='equal')
    ax.set_title('layers')
    fig.savefig(os.path.join(params.figures_path, 'layers.pdf'), dpi=300)
    

    #plot cell locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.2)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['*' if 'b' in y else '^' for y in params.y],
                    colors=['b' if 'b' in y else 'r' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/24, aspect='equal')
    plot_soma_locations(ax, X=params.y,
                        populations_path=params.populations_path,
                        markers=['*' if 'b' in y else '^' for y in params.y],
                        colors=['b' if 'b' in y else 'r' for y in params.y],
                        isometricangle=np.pi/24, )
    ax.set_title('soma positions')
    fig.savefig(os.path.join(params.figures_path, 'soma_locations.pdf'),
                dpi=150)
    
    
    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.2)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['*' if 'b' in y else '^' for y in params.y],
                    colors=['b' if 'b' in y else 'r' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/24, aspect='equal')
    plot_morphologies(ax,
                      X=params.y,
                      markers=['*' if 'b' in y else '^' for y in params.y],
                      colors=['b' if 'b' in y else 'r' for y in params.y],
                      isometricangle=np.pi/24,
                      populations_path=params.populations_path,
                      cellParams=params.yCellParams,
                      fraction=0.02)
    ax.set_title('LFP generators')
    fig.savefig(os.path.join(params.figures_path, 'populations.pdf'), dpi=300)


    #plot morphologies in their respective locations
    fig, ax = plt.subplots(1,1, figsize=(5,8))
    fig.subplots_adjust(left=0.2)
    plot_population(ax, params.populationParams, params.electrodeParams,
                    params.layerBoundaries,
                    X=params.y,
                    markers=['*' if 'b' in y else '^' for y in params.y],
                    colors=['b' if 'b' in y else 'r' for y in params.y],
                    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                    isometricangle=np.pi/24, aspect='equal')
    plot_individual_morphologies(ax,
                   X=params.y,
                   markers=['*' if 'b' in y else '^' for y in params.y],
                   colors=['b' if 'b' in y else 'r' for y in params.y],
                   isometricangle=np.pi/24, cellParams=params.yCellParams,
                   populationParams=params.populationParams)
    ax.set_title('morphologies')
    fig.savefig(os.path.join(params.figures_path, 'cell_models.pdf'), dpi=300)


    #plot compound LFP and CSD traces
    fig = plt.figure(figsize=(13, 8))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, top=0.95,
                        hspace=0.2, wspace=0.2)
    gs = gridspec.GridSpec(2,2)
    
    ax0 = fig.add_subplot(gs[:,0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax0.set_title('network raster')
    ax1.set_title('CSD')
    ax2.set_title('LFP')

    T=(500, 700)
    
    x, y = networkSim.get_xy(T, fraction=1)
    networkSim.plot_raster(ax0, T, x, y, markersize=1, marker='o',
                           alpha=.5,legend=False, pop_names=True)
    remove_axis_junk(ax0)
    ax0.set_xlabel(r'$t$ (ms)', labelpad=0.1)
    ax0.set_ylabel('population', labelpad=0.1)
    

    plot_signal_sum(ax1, z=params.electrodeParams['z'],
                    fname=os.path.join(params.savefolder, 'CSDsum.h5'),
                    unit='$\mu$Amm$^{-3}$', T=T)
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    
    plot_signal_sum(ax2, z=params.electrodeParams['z'],
                    fname=os.path.join(params.savefolder, 'LFPsum.h5'),
                    unit='mV', T=T)
    ax2.set_xlabel('$t$ (ms)')
    
    fig.savefig(os.path.join(params.figures_path, 'compound_signals.pdf'),
                dpi=300)

    plt.show()

