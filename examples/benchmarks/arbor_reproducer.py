#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib inline


# In[2]:


from mpi4py import MPI
from time import time
import os
import sys
import numpy as np
import arbor
import scipy.stats
from parameters import ParameterSet

# In[3]:


#######################################
# Capture command line values
#######################################
md5 = sys.argv[1]

# load parameter file
pset = ParameterSet(os.path.join('parameters', '{}.txt'.format(md5)))


################# Initialization of MPI stuff ############################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# set some seed values
SEED = pset.GLOBALSEED
np.random.seed(SEED)


# In[4]:


tstop = 5000
dt = 0.1
cellParams = dict(
            morphology='morphologies/ex.swc',
            v_init=-65.,
            cm=1.0,
            Ra=150,
            g_pas=1. / (10. * 1E3),  # assume cm=1
            e_pas=-65,
            max_cv_length=10,  # µm
        )
population_size = pset.POPULATION_SIZE
synapse_count = 500
nthreads = pset.NTHREADS


# In[5]:


def get_activation_times_from_distribution(n, tstart=0., tstop=1.E6,
                                           distribution=scipy.stats.expon,
                                           rvs_args=dict(loc=0, scale=1),
                                           maxiter=1E6):
    """
    Construct a length n list of ndarrays containing continously increasing
    random numbers on the interval [tstart, tstop], with intervals drawn from
    a chosen continuous random variable distribution subclassed from
    scipy.stats.rv_continous, e.g., scipy.stats.expon or scipy.stats.gamma.

    The most likely initial first entry is
    ``tstart + method.rvs(size=inf, **rvs_args).mean()``

    Parameters
    ----------
    n: int
        number of ndarrays in list
    tstart: float
        minimum allowed value in ndarrays
    tstop: float
        maximum allowed value in ndarrays
    distribution: object
        subclass of scipy.stats.rv_continous. Distributions
        producing negative values should be avoided if continously increasing
        values should be obtained, i.e., the probability density function
        ``(distribution.pdf(**rvs_args))`` should be ``0`` for ``x < 0``,
        which is not explicitly tested for.
    rvs_args: dict
        parameters for method.rvs method. If "size" is in dict, then tstop will
        be ignored, and each ndarray in output list will be
        ``distribution.rvs(**rvs_args).cumsum() + tstart``. If size is not
        given in dict, then values up to tstop will be included
    maxiter: int
        maximum number of iterations


    Returns
    -------
    list of ndarrays
        length n list of arrays containing data

    Raises
    ------
    AssertionError
        if distribution does not have the 'rvs' attribute
    StopIteration
        if number of while-loop iterations reaches maxiter


    Examples
    --------
    Create n sets of activation times with intervals drawn from the exponential
    distribution, with rate expectation lambda 10 s^-1 (thus
    scale=1000 / lambda). Here we assume output in units of ms

    >>> from LFPy.inputgenerators import get_activation_times_from_distribution
    >>> import scipy.stats as st
    >>> import matplotlib.pyplot as plt
    >>> times = get_activation_times_from_distribution(n=10, tstart=0.,
    >>>                                                tstop=1000.,
    >>>                                                distribution=st.expon,
    >>>                                                rvs_args=dict(loc=0.,
    >>>                                                scale=100.))
    """
    assert hasattr(distribution, 'rvs'),         'distribution={} must have the attribute "rvs"'.format(distribution)

    times = []
    if 'size' in rvs_args.keys():
        for i in range(n):
            times += [distribution.rvs(**rvs_args).cumsum() + tstart]
    else:
        for i in range(n):
            values = distribution.rvs(size=1000, **rvs_args).cumsum() + tstart
            iter = 0
            while values[-1] < tstop and iter < maxiter:
                values = np.r_[values, distribution.rvs(
                    size=1000, **rvs_args).cumsum() + values[-1]]
                iter += 1

            if iter == maxiter:
                raise StopIteration('maximum number of iterations reach. Con')

            times += [values[values < tstop]]

    return times


# In[6]:


class BaseRecipe (arbor.recipe):
    def __init__(self, cell):
        super().__init__()

        self.the_cell = cell

        self.iprobe_id = (0, 0)

        self.the_props = arbor.neuron_cable_properties()
        self.the_cat = arbor.default_catalogue()
        self.the_props.register(self.the_cat)

    def num_cells(self):
        return 1

    def num_sources(self, gid):
        return 0

    def num_targets(self, gid):
        return 0

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        return []

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        return [
            arbor.cable_probe_total_current_cell(),
        ]


# In[7]:


class Recipe(BaseRecipe):
    def __init__(self, cell, times=[[1.]], weights=[1.]):
        super().__init__(cell)

        assert(len(times) == len(weights)), 'len(times) != len(weights)'
        self.times = times
        self.weights = weights

    def num_targets(self, gid):
        return len(self.times)

    def event_generators(self, gid):
        events = []
        for i, (w, t) in enumerate(zip(self.weights, self.times)):
            events += [arbor.event_generator(f'{i}', w,
                                             arbor.explicit_schedule(t))]

        return events


# In[8]:


class DummyCell(object):
    def __init__(self, loc_sets):
        self.loc_sets = loc_sets


# In[22]:


class ArborPopulation(object):
    def __init__(self, cellParams,
                 dt=0.1, tstop=1000,
                 population_size=1280,
                 synapse_count=100,
                 synapse_rate=20.,
                 nthreads=2):
        self.cellParams=cellParams
        self.dt = dt
        self.tstop = tstop
        self.population_size = population_size
        self.synapse_count = synapse_count
        self.synapse_rate = synapse_rate
        self.nthreads = nthreads

        self.cellindices = np.arange(self.population_size)
        self.rank_cellindices = self.cellindices[self.cellindices % SIZE == RANK]
        # just set all cell locations and rotations the same
        self.cell_position = {'x': 1., 'y': 2., 'z': 3.}
        self.cell_rotation = {'z': np.pi}

        # cumulative sim time
        self.cumul_sim_time = np.array(0.)

    def cellsim(self, cellindex):
        """

        Parameters
        ----------
        cellindex : int
            cell index between 0 and population size-1.

        Returns
        -------
        None


        See also
        --------
        """
        tic = time()

        # cell decor
        decor = arbor.decor()

        # set initial voltage, temperature, axial resistivity, membrane capacitance
        decor.set_property(
            Vm=self.cellParams['v_init'],  # Initial membrane voltage [mV]
            tempK=300,  # Temperature [Kelvin]
            rL=self.cellParams['Ra'],  # Axial resistivity [Ω cm]
            cm=self.cellParams['cm'] * 1E-2,  # Membrane capacitance [F/m**2]
        )

        # set passive mechanism all over
        pas = arbor.mechanism(
            'pas/e={}'.format(self.cellParams['e_pas'])
            )  # passive mech w. leak reversal potential (mV)
        pas.set('g', self.cellParams['g_pas'])  # leak conductivity (S/cm2)
        decor.paint('(all)', pas)

        # number of CVs per branch
        policy = arbor.cv_policy_max_extent(self.cellParams['max_cv_length'])
        decor.discretization(policy)

        # define morphology (needed for arbor.place_pwlin etc.)
        morphology = arbor.load_swc_arbor(self.cellParams['morphology'])

        # Label dictionary
        defs = {}
        labels = arbor.label_dict(defs)

        # define isometry
        iso = arbor.isometry()  # create isometry
        for key, val in self.cell_rotation.items():
            args = {
                'theta': float(val),
                'axis': tuple((np.array(['x', 'y', 'z']) == key).astype('float'))}
            iso_rot = iso.rotate(**args)
        iso_trans = iso.translate(**self.cell_position)

        # place_pwlin
        p = arbor.place_pwlin(morphology, iso_rot * iso_trans)  # place with isometry

        # need dummycell geometry object
        loc_sets = self.get_loc_sets(p, morphology, labels, decor)

        # get transmembrane currents
        return self.get_I_m(loc_sets, cellindex, decor, morphology, labels)

    def get_loc_sets(self, p, morphology, labels, decor):
        # create cell and set properties
        cable_cell = arbor.cable_cell(morphology, labels, decor)

        # instantiate recipe with cell
        recipe = BaseRecipe(cable_cell)

        # instantiate simulation
        context = arbor.context()
        domains = arbor.partition_load_balance(recipe, context)
        sim = arbor.simulation(recipe, domains, context)

        # set up sampling on probes
        schedule = arbor.regular_schedule(self.dt)
        i_handle = sim.sample(recipe.iprobe_id, schedule, arbor.sampling_policy.exact)

        # need meta data locating each CV
        _, I_m_meta = sim.samples(i_handle)[0]

        # define a list of loc_sets with relative location in between proximal and distal points of each CV
        loc_sets = np.array([
            '(location {} {})'.format(c.branch, np.mean([c.prox, c.dist])) for c in I_m_meta
        ])

        return loc_sets

    def get_I_m(self, loc_sets, cellindex, decor, morphology, labels):
        times = get_activation_times_from_distribution(n=self.synapse_count,
                                                       tstart=0.,
                                                       tstop=self.tstop,
                                                       distribution=scipy.stats.expon,
                                                       rvs_args=dict(loc=0.,
                                                                     scale=1000 / self.synapse_rate)
                                                      )
        weights = np.random.randn(self.synapse_count)
        syn_loc_sets = np.random.choice(loc_sets, size=self.synapse_count)

        # create synapses at each loc_set
        synapse = 'alphaisyn'   # workaround
        synapse_params = {'tau': 5.}
        for i, loc_set in enumerate(syn_loc_sets):
            decor.place(loc_set, arbor.mechanism(synapse, synapse_params), f'{i}')

        # number of CVs per branch
        policy = arbor.cv_policy_max_extent(self.cellParams['max_cv_length'])
        decor.discretization(policy)

        # create cell and set properties
        cable_cell = arbor.cable_cell(morphology, labels, decor)

        # instantiate recipe with cable_cell
        recipe = Recipe(cable_cell, weights=weights, times=times)

        # instantiate simulation
        context = arbor.context(self.nthreads, None)
        domains = arbor.partition_load_balance(recipe, context)
        sim = arbor.simulation(recipe, domains, context)

        # set up sampling on probes
        schedule = arbor.regular_schedule(self.dt)
        i_handle = sim.sample(recipe.iprobe_id, schedule, arbor.sampling_policy.exact)

        # run simulation of transmembrane currents
        tic = time()
        self._bench_sim_run(sim)
        toc = time()
        self.cumul_sim_time += toc - tic

        # extract I_m for each CV
        I_m_samples, _ = sim.samples(i_handle)[0]

        # transmembrane currents in nA
        return I_m_samples[:, 1:]

    def _bench_sim_run(self, sim):
        sim.run(tfinal=self.tstop, dt=self.dt)

    def run(self):
        for cellindex in self.rank_cellindices:
            self.cellsim(cellindex)


# In[23]:


# %%prun -s cumulative -q -l 20 -T prun
pop = ArborPopulation(
    cellParams=cellParams,
    dt=dt,
    tstop=tstop,
    population_size=population_size,
    synapse_count=synapse_count,
    nthreads=nthreads
)

# run population simulation and collect the data
pop.run()


# compute mean time spent calling arbor.simulation.run() per MPI process
if RANK == 0:
    tocc_run = np.array(0.)
else:
    tocc_run = None

COMM.Reduce(pop.cumul_sim_time, tocc_run, op=MPI.SUM, root=0)
if RANK == 0:
    tocc_run /= SIZE
    with open(os.path.join('logs', f'MPI_SIZE_{SIZE}_NTHREADS_{nthreads}.txt'), 'w') as f:
        f.write(f'{float(tocc_run)}')

    print(SIZE, nthreads, tocc_run)

# clean exit?
COMM.Barrier()