#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Define ParameterSpace for benchmarking of scaling with different MPI pools
'''
import os
import numpy as np
import operator
import pickle
import hashlib
import subprocess as sp
import json
from parameters import ParameterSpace, ParameterSet, ParameterRange

PSPACES = dict()

# check scaling with MPI pool size
PS_reproducer = ParameterSpace(dict(
    # allow different seeds for different network iterations
    GLOBALSEED=1234,

    # number of Neurons
    POPULATION_SIZE=256,

    # MPI pool size
    NTASKS=ParameterRange([2, 4, 8, 16, 32, 64, 128]),

    NTHREADS=ParameterRange([1, 2]),

    # simulation scripts:
    SIM_SCRIPT='arbor_reproducer.py'
))

PS_reproducer.save('PS_reproducer.txt')
PSPACES.update(dict(PS_reproducer=PS_reproducer))

job_script_hpc = """#!/bin/bash
##################################################################
#SBATCH --account {}
#SBATCH --job-name {}
#SBATCH --time {}
#SBATCH -o logs/{}_stdout.txt
#SBATCH -e logs/{}_error.txt
#SBATCH --ntasks {}
#SBATCH --cpus-per-task={}
##################################################################
# from here on we can run whatever command we want
unset DISPLAY # DISPLAY somehow problematic with Slurm
srun --mpi=pmi2 python -u {} {}
"""

# create output directories, wipe if they exist
for dir in ['jobs', 'parameters', 'logs', 'output']:
    if not os.path.isdir(dir):
        os.mkdir(dir)


env = os.environ

if 'HOSTNAME' in env.keys():
    if env['HOSTNAME'].rfind('jr') >= 0 or env['HOSTNAME'].rfind('jusuf') >= 0:

        # slurm job settings (shared)
        ACCOUNT = 'jinb33' if env['HOSTNAME'].rfind('jr') >= 0 else 'icei-hbp-2020-0004'

        # container for job IDs
        jobIDs = []
        for pset in PS_reproducer.iter_inner():
            # sorted json dictionary
            js = json.dumps(pset, sort_keys=True).encode()
            md5 = hashlib.md5(js).hexdigest()

            # walltime (360 seconds per 1 MPI threads and popscaling 1 and
            # neuron count 512*5)
            wt = 360 * 1. / pset.NTASKS + 240
            wt = '%i:%.2i:%.2i' % (wt // 3600,
                                       (wt - wt // 3600 * 3600) // 60,
                                       (wt - wt // 60 * 60))
            TIME = wt

            # save parameter file
            pset.save(url=os.path.join('parameters', '{}.txt'.format(md5)))

            # create job script
            with open(os.path.join('jobs', '{}.job'.format(md5)), 'w') as f:
                f.writelines(job_script_hpc.format(
                    ACCOUNT,
                    md5,
                    TIME,
                    md5,
                    md5,
                    pset.NTASKS,
                    pset.NTHREADS,
                    pset.SIM_SCRIPT,
                    md5
                ))
            cmd = ' '.join(['sbatch',
                            '{}'.format(os.path.join('jobs',
                                        '{}.job'.format(md5)))])
            print(cmd)
            output = sp.getoutput(cmd)
            jobid = output.split(' ')[-1]
            jobIDs.append((md5, jobid))
    else:
        raise NotImplementedError
else:
    for pset in PS_reproducer.iter_inner():
        # sorted json dictionary
        js = json.dumps(pset, sort_keys=True).encode()
        md5 = hashlib.md5(js).hexdigest()

        # save parameter file
        pset.save(url=os.path.join('parameters', '{}.txt'.format(md5)))

        # run model serially
        cmd = 'mpirun -n {} python {} {}'.format(pset.NTASKS, pset.SIM_SCRIPT, md5)
        print(cmd)
        sp.run(cmd.split(' '))
