
# Description

This is the README for the code archived for reproducing all data and
figures for the study:

Espen Hagen, David Dahmen, Maria L. Stavrinou, Henrik Lindén, Tom Tetzlaff,
Sacha J van Albada, Sonja Grün, Markus Diesmann, Gaute T. Einevoll.
"Hybrid scheme for modeling local field potentials from point-neuron networks".
Cereb. Cortex (2016)
DOI: 10.1093/cercor/bhw237
First published online: October 20, 2016
Contact: https://github.com/INM-6/hybridLFPy/issues

Here, only the main simulation scripts, data preprocessing and plotting scripts
are provided. Additional software and dependencies has to be obtained elsewhere,
as described below.

## Dependencies (original publication)

In short, the main simulations are run using Python 2.7.x, rely on NEST's
(www.nest-simulator.org) and NEURON's (www.neuron.yale.edu) python extensions,
the scientific python software stack (http://www.scipy.org), LFPy (LFPy.github.io)
and hybridLFPy (INM-6.github.io/hybridLFPy)

The present application of hybridLFPy (v0.1.2) and corresponding simulations
were made possible using:
- Open MPI (v.1.6.2),
- HDF5 (v.1.8.14),
- Python (v.2.7.3) with modules:
    - Cython (v.0.23dev),
    - NeuroTools (v.0.2.0dev),
    - SpikeSort (v.0.13),
    - h5py (v.2.5.0a0),
    - matplotlib (v.1.5.x),
    - mpi4py (v.1.3),
    - numpy (v.1.10.0.dev-c63e1f4),
    - pysqlite (v.2.6.3) and
    - scipy (v.0.17.0.dev0-357a1a0).

Point-neuron network simulations were performed using NEST (v.2.8.0 ff71a29).
This particular revision can be checked out from
https://github.com/espenhgn/nest-simulator/tree/neg_logn_w,
as one minor change from the official NEST v2.8.0 is the support for  lognormal
distribution of weights with a negative sign.
Simulations of multicompartment model neurons uses NEURON
(v.7.4 1186:541994f8f27f) through LFPy (dev. v.aa134c911).
All software was compiled using GCC (v.4.4.6).
Simulations were performed in parallel (256 threads) on the Stallo
high-performance computing facilities (NOTUR, the Norwegian Metacenter for
Computational Science) consisting of 2.6 GHz Intel E5-2670 CPUs running the
Rocks Cluster Distribution (Linux) operating system (v.6.0).

All above software releases are publically available, and is hereby not
included, and will have to be installed separately.

## Dependencies (current)

The current version has been developed around:
- cython                    0.29.23
- h5py                      3.2.1
- hdf5                      1.10.6
- hybridlfpy                0.1.4
- lfpy                      2.2.1
- lfpykit                   0.3
- matplotlib                3.4.2
- meautility                1.4.8
- mpi4py                    3.0.3
- numpy                     1.20.3
- openmpi                   4.1.1
- pynest                    master@7a2dd86f8
- python                    3.9.4
- scipy                     1.6.3

# License

These codes are provided without any warranty whatsoever, and released under
the GPL, see LICENSE file.


# Included files

These are the main files included:

- `cellsim16pops_*.py`:
    Main simulation scripts, corresponding to 11 different configurations
    (modified network states, different types of thalamic activation,
    recording of LFPs from either excitatory and or inhibitory currents,
    prediction of spike-LFP relationships)

    The simulation files has to be executed in parallel on a cluster. A typical
    PBS job script would look like:

        #!/bin/sh
        #PBS -lnodes=16:ppn=16
        #PBS -lwalltime=08:00:00
        cd $PBS_O_WORKDIR
        mpirun -np 256 python cellsim16pops_default.py --quiet

    Submit the job by issuing:

        qsub jobscript.job

    The equivalent Slurm (see http://slurm.schedmd.com) jobscript may look like

        #!/bin/bash
        ################################################################################
        #SBATCH --job-name cellsim16pops_default
        #SBATCH --time 08:00:00
        #SBATCH -o logs/cellsim16pops_default.txt
        #SBATCH -e logs/cellsim16pops_default.txt
        #SBATCH --ntasks 256
        ################################################################################
        unset DISPLAY # slurm appear to create a problem with too many displays
        mpirun python cellsim16pops_default.py

    Submit the job by issuing:

        sbatch jobscript.job


    All simulations has to be done for all figure scripts below to run.
    Simulated output will be saved in under e.g., `simulation_output_default`.
    If the compute cluster has a `/scratch` area, output will be saved in
    under `/scratch/hybrid_model/simulation_output_default`. Please revise
    for the folder hierarchy on your cluster accordingly (inside each
    corresponding `cellsim16popsParams_*.py` file). Tar archive files
    will be created alongside for easy file transfer, named e.g.,
    'simulation_output_default.tar'. Compression is switched off by default
    because of speed and little gain in terms of reduced file sizes.

    Before LFP simulations can be executed, the NEURON NMODL language file
    expsyni.mod specifying the exponential synapse current has to be compiled
    by running the script `nrnivmodl` inside this folder. This script should
    be located along the other NEURON executables, e.g., inside
    `$HOME/NEURON-7.4/nrn/x86_64/bin/`

    Simulation cases:
    - `default`: Network parameters similar to Potjans & Diesmann (2014), Cereb
        Cortex, with spontantaneous activity. (Fig. 6, 8K-O,)
    - `modified_spontan`: Modififed network parameters (increased relative
        inhibition onto population L4E from L4I, lognormal weight distribution,
        see Hagen et al.), spontantaneous activity. (Fig. 8A-E,9,11, 12, 13)
    - `modified_spontan_ex`: Similar to modified_spontan, but inhibitiory currents
        are ignored for LFP predictions. (Fig. 9,)
    - `modified_spontan_inh`: Similar to modified_spontan, but excitatory currents
        are ignored for LFP predictions. (Fig. 9,)
    - `modified_regular_input`: Similar to modified_spontan, but with transient
        thalamic activation every 1 s. (Fig 1, 7, 10)
    - `modified_regular_exc`: Similar to modified_regular_input, inhibition
        ignored in LFP. (Fig. 10)
    - `modified_regular_inh`: Similar to modified_regular_input, excitation
        ignored in LFP. (Fig. 10)
    - `modified_ac_input`: Similar to modified_spontan, but with sinusoidal
        modulation of Poisson spiketrains from the thalamic population.
        (Fig. 8F-J, 10, 11, 12, 13)
    - `modified_ac_exc`: Similar to modified_ac_input, inhibition
        ignored in LFP. (Fig. 10)
    - `modified_ac_inh`: Similar to modified_ac_input, excitation
        ignored in LFP. (Fig. 10)
    - `spikegen`: Similar to modified_spontan, except that network spikes are
        replaced with synchronous spike events for each
        network population, used to compute the average single-spike LFP
        response. (Fig. 13)

- `cellsim16popsParams_*.py`:
    Parameter class definitions for each main simulation file.

- `microcircuit.sli`:
    NEST SLI implementation of the network by Potjans and Diesmann (2014),
    Cerebral Cortex March 2014;24:785�806 doi:10.1093/cercor/bhs358. Used by
    the main Python simulation scripts

- `nest_simulation.py`:
    Some python routines used to parse parameters to NEST's sli namespace
    before invoking the microcircuit.sli script

- `nest_output_processing.py`:
    NEST's gdf file output is typically one file per MPI thread, some
    routines for merging such output is defined here and used after running
    each network simulation.

- `morphologies/*/*.hoc`:
    NEURON-language morphology files used for predictions of local-field
    potentials. By default the 'stretched' geometries will be used. It is
    possible to use simplified 'ballnstick' ones to reduce overall simulation
    times, but this will affect the  numerical result of the simulated output
    as the laminar distribution of in and outgoing currents are not conserved.

- `expsyni.mod`:
    NMODL language specification of the current-based synapse. In order to run
    simulations, run the NEURON-provided bash script nrnivmodl in the root
    of this folder before running simulations.

- `binzegger_connectivity_table.json`:
    JSON format data with the connectivity data from Binzegger et al., 2004 and
    Izhikevich and Edelmann 2008.

- `analysis.py`:
    Applied to the modified_spontan and modified_ac_input simulation cases
    in order to investigate the effect of single-cell LFP spectra on the
    compound LFP signal. Execution in parallel:

        mpirun -np 4 python analysis.py

- `analysis_params.py`:
    global parameters used for some analysis and plotting

- `plotting_helpers.py` and `plot_methods.py`:
    Various common methods and helper functions facilitating generation of
    figure panels

- `figure_*.py`:
    generates the various figures (or parts thereof) as shown in the manuscript
    based on simulated output. Figure files will be saved on the pdf format in
    the folder corresponding to the data it was generated from, e.g., as
    simulation_output_default/figures/figure_08.pdf

- `Fig3/*`:
    Files for generating Fig 3 in the manuscript.

- `run_all_jobs.py`:
    Create jobscripts and submit jobs for clusters running the Slurm Workload
    Manager (http://slurm.schedmd.com)

- `produce_all_figures.py`:
    run all figure-generating scripts at once (for your convenience)

- `LICENSE`:
    General Public License version 3.

- `README.md`:
    This file

- `Dockerfile`:
    Docker container recipe for `x86_64` hosts with all dependencies required
    to run above simulation files. To build and run the container locally,
    get Docker from https://www.docker.com, and issue the following:

        $ docker build -t hybridlfpy -< Dockerfile
        $ docker run -it -p 5000:5000 hybridlfpy:latest


    The `--mount` option can be used to mount a folder on the host to a target folder as:

        $ docker run --mount type=bind,source="$(pwd)",target=/opt -it -p 5000:5000 <image-name>

- `mpich.Dockerfile`:
    Similar to `Dockerfile` but uses a specific `mpich` version suitable for use with Singularity containers on certain HPC facilities  
