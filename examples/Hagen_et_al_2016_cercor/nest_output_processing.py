#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Documentation:

This file reformats NEST output in a convinient way.

'''
import os
import numpy as np
from glob import glob
from hybridLFPy import helpers
import tarfile
from pathlib import Path
from mpi4py import MPI

###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


flattenlist = lambda lst: sum(sum(lst, []),[])


def get_raw_gids(model_params):
    '''
    Reads text file containing gids of neuron populations as created within the
    NEST simulation. These gids are not continuous as in the simulation devices
    get created in between.
    '''
    gidfile = open(os.path.join(model_params.raw_nest_output_path,
                                model_params.GID_filename), 'r')
    gids = []
    for l in gidfile:
        a = l.split()
        gids.append([int(a[0]),int(a[1])])
    return gids


def merge_gdf(model_params,
              raw_label='spikes_',
              file_type='gdf',
              fileprefix='spikes',
              skiprows=0):
    '''
    NEST produces one file per virtual process per recorder
    (spike detector, voltmeter etc.).
    This function gathers and combines them into one single file per population.
    '''
    #some preprocessing
    raw_gids = get_raw_gids(model_params)
    pop_sizes = [raw_gids[i][1]-raw_gids[i][0]+1
                 for i in np.arange(model_params.Npops)]
    raw_first_gids = [raw_gids[i][0] for i in np.arange(model_params.Npops)]
    converted_first_gids = [int(1 + np.sum(pop_sizes[:i]))
                            for i in np.arange(model_params.Npops)]

    for pop_idx in np.arange(model_params.Npops):
        if pop_idx % SIZE == RANK:
            files = glob(os.path.join(model_params.raw_nest_output_path,
                                      raw_label + '{}*.{}'.format(pop_idx,
                                                                  file_type)))
            gdf = []  # init
            for f in files:
                new_gdf = helpers.read_gdf(f, skiprows)
                for line in new_gdf:
                    line[0] = line[0] - raw_first_gids[pop_idx] + converted_first_gids[pop_idx]
                    gdf.append(line)

            print('writing: {}'.format(os.path.join(model_params.spike_output_path,
                                       fileprefix + '_{}.{}'.format(model_params.X[pop_idx],
                                                                    file_type))))
            helpers.write_gdf(gdf, os.path.join(model_params.spike_output_path,
                                        fileprefix +
                                        '_{}.{}'.format(model_params.X[pop_idx],
                                                        file_type)))

    COMM.Barrier()

    return


def tar_raw_nest_output(raw_nest_output_path,
                        delete_files=True,
                        filepatterns=['voltages*.dat',
                                      'spikes*.dat',
                                      'weighted*.dat'
                                      '*.gdf']):
    '''
    Create tar file of content in `raw_nest_output_path` and optionally
    delete files matching given pattern.

    Parameters
    ----------
    raw_nest_output_path: path
        params.raw_nest_output_path
    delete_files: bool
        if True, delete .dat files
    filepatterns: list of str
        patterns of files being deleted
    '''
    if RANK == 0:
        # create tarfile
        fname = raw_nest_output_path + '.tar'
        with tarfile.open(fname, 'a') as t:
            t.add(raw_nest_output_path)

        # remove files from <raw_nest_output_path>
        if delete_files:
            for pattern in filepatterns:
                for p in Path(raw_nest_output_path).glob(pattern):
                    print('deleting {}'.format(p))
                    while p.isfile():
                        try:
                            p.unlink()
                        except OSError as e:
                            print('Error: {} : {}'.format(p, e.strerror))

    # sync
    COMM.Barrier()

    return
