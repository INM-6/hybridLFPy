#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
mpirun -np 64 python cellsim16pops.py

Given the size of the network and demands for the multi-compartment LFP-
predictions using the present scheme, running the model on nothing but a large-
scale compute facility is strongly discouraged.

'''
import os
import numpy as np
from time import time
import neuron # NEURON compiled with MPI must be imported before NEST and mpi4py
              # to avoid NEURON being aware of MPI.
import nest   # Import not used, but done in order to ensure correct execution
import nest_simulation
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
import nest_output_processing
import lfpykit


#set some seed values
SEED = 12345678
SIMULATIONSEED = 12345678
np.random.seed(SEED)



################################################################################
## PARAMETERS
################################################################################

from cellsim16popsParams_modified_regular_input import multicompartment_params, \
                                point_neuron_network_params

#Full set of parameters including network parameters
params = multicompartment_params()

#set up the file destination
setup_file_dest(params, clearDestination=True)


###############################################################################
# MAIN simulation procedure
###############################################################################

#tic toc
tic = time()

######## Perform network simulation ############################################

##initiate nest simulation with only the point neuron network parameter class
networkParams = point_neuron_network_params()
nest_simulation.sli_run(parameters=networkParams,
                       fname='microcircuit.sli',
                       verbosity='M_INFO')

#preprocess the gdf files containing spiking output, voltages, weighted and
#spatial input spikes and currents:
nest_output_processing.merge_gdf(networkParams,
                            raw_label=networkParams.spike_recorder_label,
                            file_type='dat',
                            fileprefix=params.networkSimParams['label'],
                            skiprows=3)
nest_output_processing.merge_gdf(networkParams,
                            raw_label=networkParams.voltmeter_label,
                            file_type='dat',
                            fileprefix='voltages',
                            skiprows=3)
nest_output_processing.merge_gdf(networkParams,
                            raw_label=networkParams.weighted_input_spikes_label,
                            file_type='dat',
                            fileprefix='population_input_spikes',
                            skiprows=3)
##spatial input currents
#nest_output_processing.create_spatial_input_spikes_hdf5(networkParams,
#                                        fileprefix='depth_res_input_spikes-')

# create tar file archive of <raw_nest_output_path> folder as .dat files are
# no longer needed. Also removes .dat files
nest_output_processing.tar_raw_nest_output(params.raw_nest_output_path,
                                           delete_files=True)

#Create an object representation of the simulation output that uses sqlite3
networkSim = CachedNetwork(**params.networkSimParams)


toc = time() - tic
print('NEST simulation and gdf file processing done in  %.3f seconds' % toc)

##### Set up LFPykit measurement probes for LFPs and CSDs
probes = []
probes.append(lfpykit.RecExtElectrode(cell=None, **params.electrodeParams))
probes.append(lfpykit.LaminarCurrentSourceDensity(cell=None, **params.CSDParams))

####### Set up populations #####################################################

#iterate over each cell type, and create populationulation object
for i, y in enumerate(params.y):
    #create population:
    pop = Population(
            #parent class
            cellParams = params.yCellParams[y],
            rand_rot_axis = params.rand_rot_axis[y],
            simulationParams = params.simulationParams,
            populationParams = params.populationParams[y],
            y = y,
            layerBoundaries = params.layerBoundaries,
            probes=probes,
            savelist = params.savelist,
            savefolder = params.savefolder,
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

#do some postprocessing on the collected data, i.e., superposition
#of population LFPs, CSDs etc
postproc = PostProcess(y = params.y,
                       dt_output = params.dt_output,
                       probes=probes,
                       savefolder = params.savefolder,
                       mapping_Yy = params.mapping_Yy,
                       )

#run through the procedure
postproc.run()

#create tar-archive with output for plotting
postproc.create_tar_archive()

#tic toc
print('Execution time: %.3f seconds' %  (time() - tic))
