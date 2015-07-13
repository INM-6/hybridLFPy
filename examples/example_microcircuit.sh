#!/bin/sh

#PBS -lnodes=8:ppn=16
#PBS -lwalltime=12:00:00
#PBS -A nn4661k

cd $PBS_O_WORKDIR
mpirun -np 128 python example_microcircuit.py --quiet
