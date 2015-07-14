'''
Documentation:

This file reformats NEST output in a convinient way.

'''
import os
import numpy as np
from glob import glob
#import h5py_wrapper
#from example_microcircuit_params import point_neuron_network_params
#from analysis_params import params
from hybridLFPy import helpers
from mpi4py import MPI

###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


flattenlist = lambda lst: sum(sum(lst, []),[])


def _get_raw_gids(model_params):
    '''
    Reads text file containing gids of neuron populations as created within the NEST simulation. 
    These gids are not continuous as in the simulation devices get created in between.
    
    Parameters
    ----------
    model_params : object
        network parameters object
    
    
    Returns
    -------
    gids : list
        list of neuron ids and value (spike time, voltage etc.)
    
    '''
    gidfile = open(os.path.join(model_params.raw_nest_output_path, model_params.GID_filename),'r') 
    gids = [] 
    for l in gidfile :
        a = l.split()
        gids.append([int(a[0]),int(a[1])])
    return gids 
    

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
    #some preprocessing
    raw_gids = _get_raw_gids(model_params)
    pop_sizes = [raw_gids[i][1]-raw_gids[i][0]+1 for i in np.arange(model_params.Npops)]
    raw_first_gids =  [raw_gids[i][0] for i in np.arange(model_params.Npops)]
    converted_first_gids = [int(1 + np.sum(pop_sizes[:i])) for i in np.arange(model_params.Npops)]

    for pop_idx in np.arange(model_params.Npops):
        if pop_idx % SIZE == RANK:
            files = glob(os.path.join(model_params.raw_nest_output_path,
                                      raw_label + str(pop_idx) + '*.' + file_type))
            gdf = [] # init
            for f in files:
                new_gdf = helpers.read_gdf(f)
                for line in new_gdf:
                    line[0] = line[0] - raw_first_gids[pop_idx] + converted_first_gids[pop_idx]
                    gdf.append(line)
            
            print 'writing: %s' % os.path.join(model_params.spike_output_path,
                                               fileprefix + '_%s.gdf' % model_params.X[pop_idx])
            helpers.write_gdf(gdf, os.path.join(model_params.spike_output_path,
                                                fileprefix + '_%s.gdf' % model_params.X[pop_idx]))
    
    COMM.Barrier()

    return

