#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Define ParameterSpace for benchmarking of scaling with different MPI pools
'''
import os
import numpy as np
import operator
import pickle
import hashlib
# import parameters as ps
import subprocess as sp
import json
from parameters import ParameterSpace, ParameterSet, ParameterRange
import matplotlib.pyplot as plt

PSPACES = dict()

# check scaling with MPI pool size
PS0 = ParameterSpace(dict(
    # allow different seeds for different network iterations
    GLOBALSEED=1234,

    # MPI pool size
    NTASKS=ParameterRange([1, 2, 4, 8, 16, 32, 64, 128]),

    # population size scaling (multiplied with values in
    # populationParams['POP_SIZE']):
    # POPSCALING=ParameterRange([1.]),

    # simulation scripts:
    SIM_SCRIPT=ParameterRange(['example_brunel.py', 'example_brunel_arbor.py'])
))

PS0.save('PS0.txt')
PSPACES.update(dict(PS0=PS0))

jobscript_local = '''#!/bin/bash
# from here on we can run whatever command we want
unset DISPLAY # DISPLAY somehow problematic with Slurm
mpirun -n {} python {} {}
'''

job_script_jusuf = """#!/bin/bash
##################################################################
#SBATCH --account {}
#SBATCH --job-name {}
#SBATCH --time {}
#SBATCH -o logs/{}_stdout.txt
#SBATCH -e logs/{}_error.txt
###SBATCH -N {}
#SBATCH --ntasks {}
#SBATCH --cpus-per-task=2
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

if 'HOSTNAME' in env.keys() and \
   env['HOSTNAME'].rfind('jr') >= 0 or \
   env['HOSTNAME'].rfind('jusuf') >= 0:

    # slurm job settings (shared)
    ACCOUNT = 'jinb33' if env['HOSTNAME'].rfind('jr') >= 0 else 'icei-hbp-2020-0004'
    TIME = '00:10:00'
    LNODES = 1

    # container for job IDs
    jobIDs = []
    for pset in PS0.iter_inner():
        # sorted json dictionary
        js = json.dumps(pset, sort_keys=True).encode()
        md5 = hashlib.md5(js).hexdigest()

        # save parameter file
        pset.save(url=os.path.join('parameters', '{}.txt'.format(md5)))

        # create job script
        with open(os.path.join('jobs', '{}.job'.format(md5)), 'w') as f:
            f.writelines(job_script_jusuf.format(
                ACCOUNT,
                md5,
                TIME,
                md5,
                md5,
                LNODES,
                pset.NTASKS,
                pset.SIM_SCRIPT,
                md5
            ))
        cmd = ' '.join(['sbatch',
                        '{}'.format(os.path.join('jobs',
                                    '{}.job'.format(md5)))])
        print(cmd)
        output = sp.getoutput(cmd)
        # output = sp.getoutput(
        #     'sbatch {}'.format(os.path.join('jobs', '{}.job'.format(md5))))
        jobid = output.split(' ')[-1]
        jobIDs.append((md5, jobid))
else:
    for pset in PS0.iter_inner():
        # sorted json dictionary
        js = json.dumps(pset, sort_keys=True).encode()
        md5 = hashlib.md5(js).hexdigest()

        # save parameter file
        pset.save(url=os.path.join('parameters', '{}.txt'.format(md5)))

        # run model serially
        cmd = 'mpirun -n {} python {} {}'.format(pset.NTASKS, pset.SIM_SCRIPT, md5)
        print(cmd)
        sp.run(cmd.split(' '))
