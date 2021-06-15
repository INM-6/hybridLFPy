#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
create jobscripts for clusters running the Slurm Workload Manager
(http://slurm.schedmd.com) for all network-model and LFP simulations in
Hagen et al., 2016. DOI: 10.1093/cercor/bhw237

All jobs will be submitted to the queue.

Modify according to the set up on your cluster.

'''
import os
import glob

content = '''#!/bin/bash
################################################################################
#SBATCH --job-name {}
#SBATCH --time {}
#SBATCH -o {}
#SBATCH -e {}
#SBATCH --mem-per-cpu={}
#SBATCH --ntasks {}
################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays
{} python {}
'''

# set up
ntasks = 400            # number of MPI threads
stime = '04:00:00'      # expected simulation time 'HH:MM:SS'
# what executable to use w. OpenMPI (mpirun, srun etc.)
mpiexec = 'srun --mpi=pmi2'
jobscriptdir = 'jobs'   # put all job scripts here
logdir = 'logs'         # put all simulation logs here
memPerCPU = '2000MB'      # memory per MPI thread
simscripts = glob.glob("cellsim16pops_*.py")  # list all simulation scripts

if __name__ == '__main__':
    for dir in [jobscriptdir, logdir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    jobscripts = []
    for sim in simscripts:
        job, _ = sim.split('.')

        oe = os.path.join(logdir, job + '.txt')

        fname = os.path.join(jobscriptdir, job + '.job')
        f = open(fname, 'w')
        f.write(
            content.format(
                job,
                stime,
                oe,
                oe,
                memPerCPU,
                ntasks,
                mpiexec,
                sim))
        f.close()

        jobscripts.append(fname)

    for fname in jobscripts:
        os.system('sbatch {}'.format(fname))
