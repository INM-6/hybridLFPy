#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib inline


# Hybrid LFP scheme example script, applying the methodology with a model
# implementation similar to:
#
# Nicolas Brunel. "Dynamics of Sparsely Connected Networks of Excitatory and
# Inhibitory Spiking Neurons". J Comput Neurosci, May 2000, Volume 8,
# Issue 3, pp 183-208
#
#
# Synopsis of the main simulation procedure:
#
#     1. Loading of parameterset
#         a. network parameters
#         b. parameters for hybrid scheme
#     2. Set up file destinations for different simulation output
#     3. network simulation
#         a. execute network simulation using NEST (www.nest-initiative.org)
#         b. merge network output (spikes, currents, voltages)
#     4. Create a object-representation that uses sqlite3 of all the spiking output
#     5. Iterate over post-synaptic populations:
#         a. Create Population object with appropriate parameters for each specific population
#         b. Run all computations for populations
#         c. Postprocess simulation output of all cells in population
#     6. Postprocess all cell- and population-specific output data
#     7. Create a tarball for all non-redundant simulation output
#
# The full simulation can be evoked by issuing a mpirun call, such as
# mpirun -np 4 python example_brunel.py
#
# Not recommended, but running it serially should also work, e.g., calling
# python example_brunel.py

# In[2]:


# from example_plotting import *
import brunel_alpha_nest as BN
import lfpykit
from mpi4py import MPI
import h5py
from parameters import ParameterSet
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
import nest  # import not used, but we load NEST anyway in order determine if
import neuron  # NEURON compiled with MPI must be imported before NEST/mpi4py
from time import time
from matplotlib import gridspec
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import scipy.stats as st
import scipy.signal as ss
from LFPy.alias_method import alias_method
import arbor
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.style
matplotlib.style.use('classic')
# to avoid being aware of MPI
# network is run in parallel


# In[3]:


########## matplotlib settings ###########################################
# plt.close('all')
plt.rcParams.update({'figure.figsize': [10.0, 8.0]})

#######################################
# Capture command line values
#######################################
# load parameter file
md5 = sys.argv[1]
pset = ParameterSet(os.path.join('parameters', '{}.txt'.format(md5)))

# set some seed values
SEED = pset.GLOBALSEED
SIMULATIONSEED = pset.GLOBALSEED + 1234
np.random.seed(SEED)


################# Initialization of MPI stuff ############################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# if True, execute full model. If False, do only the plotting.
# Simulation results must exist.
properrun = True

##########################################################################
# PARAMETERS
##########################################################################

# Set up parameters using the NeuroTools.parameters.ParameterSet class.

# Access parameters defined in example script implementing the network using
# pynest, brunel_alpha_nest.py, adapted from the NEST v2.4.1 release. This will
# not execute the network model, but see below.


# set up file destinations differentiating between certain output
savefolder = md5
BN.spike_output_path = os.path.join(savefolder, 'spiking_output_path')
PS = ParameterSet(dict(
    # Main folder of simulation output
    savefolder=savefolder,

    # make a local copy of main files used in simulations
    sim_scripts_path=os.path.join(savefolder,
                                  'sim_scripts'),

    # destination of single-cell output during simulation
    cells_path=os.path.join(savefolder, 'cells'),

    # destination of cell- and population-specific signals, i.e., compund LFPs,
    # CSDs etc.
    populations_path=os.path.join(savefolder,
                                  'populations'),

    # location of spike output from the network model
    spike_output_path=BN.spike_output_path,

    # destination of figure file output generated during model execution
    figures_path=os.path.join(savefolder, 'figures')
))


# population (and cell type) specific parameters
PS.update(dict(
    # no cell type specificity within each E-I population
    # hence X == x and Y == X
    X=["EX", "IN"],

    # population-specific LFPy.Cell parameters
    cellParams=dict(
        # excitory cells
        EX=dict(
            morphology='morphologies/ex.swc',
            v_init=BN.neuron_params['E_L'],
            cm=1.0,
            Ra=150,
            passive=True,
            passive_parameters=dict(
                g_pas=1. / (BN.neuron_params['tau_m'] * 1E3),  # assume cm=1
                e_pas=BN.neuron_params['E_L']),
            # nsegs_method='lambda_f',
            # lambda_f=100,
            max_cv_length=10,  # µm
            dt=BN.dt,
            tstart=0,
            tstop=BN.simtime,
            verbose=False,
        ),
        # inhibitory cells
        IN=dict(
            morphology='morphologies/in.swc',
            v_init=BN.neuron_params['E_L'],
            cm=1.0,
            Ra=150,
            passive=True,
            passive_parameters=dict(
                g_pas=1. / (BN.neuron_params['tau_m'] * 1E3),  # assume cm=1
                e_pas=BN.neuron_params['E_L']),
            # nsegs_method='lambda_f',
            # lambda_f=100,
            max_cv_length=10,  # µm
            dt=BN.dt,
            tstart=0,
            tstop=BN.simtime,
            verbose=False,
        )),

    # assuming excitatory cells are pyramidal
    rand_rot_axis=dict(
        EX=['z'],
        IN=['x', 'y', 'z'],
    ),


    # kwargs passed to LFPy.Cell.simulate().
    # It can be empty, but if `rec_imem=True`, the signal predictions will be
    # performed using recorded transmembrane currents.
    simulationParams=dict(rec_imem=True),

    # set up parameters corresponding to cylindrical model populations
    populationParams=dict(
        EX=dict(
            number=BN.NE,
            radius=100,
            z_min=-450,
            z_max=-350,
            min_cell_interdist=1.,
            min_r=[[-1E199, -600, -550, 1E99], [0, 0, 10, 10]],
        ),
        IN=dict(
            number=BN.NI,
            radius=100,
            z_min=-450,
            z_max=-350,
            min_cell_interdist=1.,
            min_r=[[-1E199, -600, -550, 1E99], [0, 0, 10, 10]],
        ),
    ),

    # set the boundaries between the "upper" and "lower" layer
    layerBoundaries=[[0., -300],
                     [-300, -500]],

    # set the geometry of the virtual recording device
    electrodeParams=dict(
        # contact locations:
        x=[0] * 6,
        y=[0] * 6,
        z=[x * -100. for x in range(6)],
        # extracellular conductivity:
        sigma=0.3,
        # contact surface normals, radius, n-point averaging
        N=[[1, 0, 0]] * 6,
        r=5,
        n=20,
        seedvalue=None,
        # dendrite line sources, soma as sphere source (Linden2014)
        method='root_as_point',
    ),

    # parameters for LFPykit.LaminarCurrentSourceDensity
    CSDParams=dict(
        z=np.array([[-(i + 1) * 100, -i * 100] for i in range(6)]) + 50.,
        r=np.ones(6) * 100  # same as population radius
    ),

    # runtime, cell-specific attributes and output that will be stored
    savelist=[],

    # time resolution of saved signals
    dt_output=1.
))


# for each population, define layer- and population-specific connectivity
# parameters
PS.update(dict(
    # number of connections from each presynaptic population onto each
    # layer per postsynaptic population, preserving overall indegree
    k_yXL=dict(
        EX=[[int(0.5 * BN.CE), 0],
            [int(0.5 * BN.CE), BN.CI]],
        IN=[[0, 0],
            [BN.CE, BN.CI]],
    ),

    # set up table of synapse weights from each possible presynaptic population
    J_yX=dict(
        EX=[BN.J_ex * 1E-3, BN.J_in * 1E-3],
        IN=[BN.J_ex * 1E-3, BN.J_in * 1E-3],
    ),

    # set up synapse parameters as derived from the network
    synParams=dict(
        EX=dict(
            section=['apic', 'dend', 'soma'],
            # section=['apic', 'dend'],
            # tau = [BN.tauSyn, BN.tauSyn],
            syntype='AlphaISyn'
        ),
        IN=dict(
            section=['apic', 'dend', 'soma'],
            # section=['dend', 'soma'],
            # tau = [BN.tauSyn, BN.tauSyn],
            syntype='AlphaISyn'
        ),
    ),

    # set up table of synapse time constants from each presynaptic populations
    tau_yX=dict(
        EX=[BN.tauSyn, BN.tauSyn],
        IN=[BN.tauSyn, BN.tauSyn]
    ),

    # set up delays, here using fixed delays of network
    synDelayLoc=dict(
        EX=[BN.delay, BN.delay],
        IN=[BN.delay, BN.delay],
    ),
    # no distribution of delays
    synDelayScale=dict(
        EX=[None, None],
        IN=[None, None],
    ),
))


# putative mappting between population type and cell type specificity,
# but here all presynaptic senders are also postsynaptic targets
PS.update(dict(
    mapping_Yy=list(zip(PS.X, PS.X))
))


# In[4]:


##########################################################################
# MAIN simulation procedure                                              #
##########################################################################

# tic toc
tic = time()

######### Perform network simulation #####################################
if properrun:
    # set up the file destination, removing old results by default
    setup_file_dest(PS, clearDestination=True)

if properrun:
    # execute network simulation
    BN.simulate()

# wait for the network simulation to finish, resync MPI threads
COMM.Barrier()

# Create an object representation containing the spiking activity of the network
# simulation output that uses sqlite3. Again, kwargs are derived from the brunel
# network instance.
networkSim = CachedNetwork(
    simtime=BN.simtime,
    dt=BN.dt,
    spike_output_path=BN.spike_output_path,
    label=BN.label,
    ext='dat',
    GIDs={'EX': [1, BN.NE], 'IN': [BN.NE + 1, BN.NI]},
    X=['EX', 'IN'],
    cmap='rainbow_r',
    skiprows=3,
)


# In[5]:


if RANK == 0:
    toc = time() - tic
    print('NEST simulation and gdf file processing done in  %.3f seconds' % toc)


# Set up LFPykit measurement probes for LFPs and CSDs
if properrun:
    probes = []
    probes.append(lfpykit.RecExtElectrode(cell=None, **PS.electrodeParams))
    probes.append(
        lfpykit.LaminarCurrentSourceDensity(
            cell=None, **PS.CSDParams))


# In[6]:


# draw locsets accd to y-position weighted by area and some function of y
def get_rand_idx_area_and_distribution_norm(areas,
                                            depth,
                                            n_syn=1,
                                            fun=[st.norm],
                                            funargs=[dict(loc=200, scale=50)],
                                            funweights=[1.]):
    '''Return n_syn CV indices with random probability normalized by membrane area
    multiplied with the value of probability density function constructed as
    weigthed sum of `scipy.stats._continuous_dists` function instances

    Parameters
    ----------
    areas: 1D ndarray
        Area of each CV
    depth: 1D ndarray
        depth of each CV
    n_syn: int
        number of random indices
    fun: iterable of scipy.stats._continuous_distns.*_gen
        iterable of callable methods in scipy.stats module
    funargs: iterable of dicts
        keyword arguments to each element in `fun`
    funweights: list of floats
        scaling

    Returns
    -------
    ndarray
        random indices of size `n_syn` that can be
    '''
    # probabilities for connecting to CV
    p = areas.copy()
    mod = np.zeros(areas.size)
    for f, args, w in zip(fun, funargs, funweights):
        df = f(**args)
        mod += df.pdf(x=depth) * w

    # multiply probs by spatial weighting factor
    p *= mod
    # normalize
    p /= p.sum()
    # find CV inds
    return alias_method(np.arange(areas.size), p, n_syn)


# In[7]:


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


# In[8]:


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


# In[9]:


class ArborPopulation(Population):
    def __init__(self, **kwargs):
        Population.__init__(self, **kwargs)

    def get_synidx(self, cellindex):
        """
        Local function, draw and return synapse locations corresponding
        to a single cell, using a random seed set as
        `POPULATIONSEED` + `cellindex`.


        Parameters
        ----------
        cellindex : int
            Index of cell object.


        Returns
        -------
        synidx : dict
            `LFPy.Cell` compartment indices


        See also
        --------
        Population.get_all_synIdx, Population.fetchSynIdxCell

        """
        # create a cell instance
        cell = self.cellsim(cellindex, return_just_cell=True)

        # compute areas and center of mass of each CV
        CV_areas = []
        for i in range(len(cell._loc_sets)):
            inds = cell._CV_ind == i
            CV_areas = np.r_[CV_areas, cell.area[inds].sum()]

        # center of mass (COM) per segment -- https://mathworld.wolfram.com/ConicalFrustum.html
        # gives geometric centroid as
        # \overline{z}} = h * (R1**2 + 2 * R1*R2 + 3 * R2**2) / (4 * (R1**2 + R1 * R2 + R2**2))
        R = cell.d / 2
        # relative COMs per segment
        h_bar_seg = cell.length * (R[:, 0]**2 + 2 * R[:, 0] * R[:, 1] + 3 * R[:, 1]**2
                                   ) / (4 * (R[:, 0]**2 + R[:, 0] * R[:, 1] + R[:, 1]**2))
        # carthesian coordinates
        r_bar_seg =  np.c_[cell.x[:, 0] + h_bar_seg * (cell.x[:, 1] - cell.x[:, 0]
                                                                ) / cell.length,
                           cell.y[:, 0] + h_bar_seg * (cell.y[:, 1] - cell.y[:, 0]
                                                                ) / cell.length,
                           cell.z[:, 0] + h_bar_seg * (cell.z[:, 1] - cell.z[:, 0]
                                                                ) / cell.length]
        # Volumes / mass
        # V = 1 / 3 * π * h * (R1**2 + R1 * R2 + R2**2)
        V_seg = np.pi * cell.length * (R[:, 0]**2 + R[:, 0] * R[:, 1] + R[:, 1]**2) / 3

        # Center of mass per CV
        r_bar = np.zeros((0, 3))
        for i in range(len(cell._loc_sets)):
            inds = cell._CV_ind == i
            if inds.sum() <= 1:
                r_ = (r_bar_seg[inds, ] * V_seg[inds] / V_seg[inds].sum()).mean(axis=0)
            else:
                # compute mean position weighted by volume/mass (mass monopole location)
                r_ = (r_bar_seg[inds, ].T * V_seg[inds] / V_seg[inds].sum()).sum(axis=-1)
            r_bar = np.vstack((r_bar, r_))


        # local containers
        synidx = {}

        # get synaptic placements and cells from the network,
        # then set spike times,
        for i, X in enumerate(self.X):
            synidx[X] = self.fetchSynIdxCell(# cell=cell,
                                             areas=CV_areas,
                                             depth=r_bar[:, 2],
                                             nidx=self.k_yXL[:, i],
                                             synParams=self.synParams.copy())
        # clean up hoc namespace
        # cell.__del__()

        return synidx

    def fetchSynIdxCell(self,
                        areas,
                        depth,
                        nidx, synParams):
        """
        Find possible synaptic placements for each cell
        As synapses are placed within layers with bounds determined by
        self.layerBoundaries, it will check this matrix accordingly, and
        use the probabilities from `self.connProbLayer to distribute.

        For each layer, the synapses are placed with probability normalized
        by membrane area of each compartment


        Parameters
        ----------
        areas:
        depth:
        nidx : numpy.ndarray
            Numbers of synapses per presynaptic population X.
        synParams : which `LFPy.Synapse` parameters to use.


        Returns
        -------
        syn_idx : list
            List of arrays of synapse placements per connection.


        See also
        --------
        Population.get_all_synIdx, Population.get_synIdx, LFPy.Synapse

        """
        # segment indices in each layer is stored here, list of np.array
        syn_idx = []
        # loop over layer bounds, find synapse locations
        for i, zz in enumerate(self.layerBoundaries):
            if nidx[i] == 0:
                syn_idx.append(np.array([], dtype=int))
            else:
                syn_idx.append(
                    get_rand_idx_area_and_distribution_norm(
                        areas, depth,
                        n_syn=nidx[i],
                        fun=[st.uniform],
                        funargs=[dict(loc=zz.min(), scale=abs(zz.max()-zz.min()))],
                        funweights=[1.],
                    ).astype('int16'))

        return syn_idx

    def insert_all_synapses(self, **kwargs):
        pass

    def insert_synapses(self, **kwargs):
        pass

    def cellsim(self, cellindex, return_just_cell=False):
        """
        Do the actual simulations of LFP, using synaptic spike times from
        network simulation.


        Parameters
        ----------
        cellindex : int
            cell index between 0 and population size-1.
        return_just_cell : bool
            If True, return only the `LFPy.Cell` object
            if False, run full simulation, return None.


        Returns
        -------
        None or `LFPy.Cell` object


        See also
        --------
        hybridLFPy.csd, LFPy.Cell, LFPy.Synapse, LFPy.RecExtElectrode
        """
        tic = time()

        '''
        cell = LFPy.Cell(**self.cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])
        '''

        ##### ARBOR

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
            'pas/e={}'.format(self.cellParams['passive_parameters']['e_pas'])
            )  # passive mech w. leak reversal potential (mV)
        pas.set('g', self.cellParams['passive_parameters']['g_pas'])  # leak conductivity (S/cm2)
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
        for key, val in self.rotations[cellindex].items():
            args = {
                'theta': float(val),
                'axis': tuple((np.array(['x', 'y', 'z']) == key).astype('float'))}
            iso_rot = iso.rotate(**args)
        iso_trans = iso.translate(**self.pop_soma_pos[cellindex])

        # place_pwlin
        p = arbor.place_pwlin(morphology, iso_rot * iso_trans)  # place with isometry

        # need cell geometry object
        cell = self._get_cell(p, morphology, labels, decor)

        if return_just_cell:
            # return only the CellGeometry object (for plotting etc.)
            return cell
        else:
            # proper simulation procedure inserting synapses, record transmembrane currents,
            # make extracellular predictions

            # get transmembrane currents
            I_m = self._get_I_m(cell, cellindex, decor, morphology, labels)

            # compute signal of each probe
            for probe in self.probes:
                probe.cell = cell

                M = self._get_transformation_matrix_for_CV(cell, probe, I_m)

                probe.data = M @ I_m

            # downsample probe.data attribute and unset cell
            for probe in self.probes:
                probe.data = ss.decimate(probe.data,
                                         q=self.decimatefrac
                                         ).astype(np.float32)
                probe.cell = None

            # put all necessary cell output in output dict
            for attrbt in self.savelist:
                attr = getattr(cell, attrbt)
                if isinstance(attr, np.ndarray):
                    self.output[cellindex][attrbt] = attr.astype('float32')
                else:
                    try:
                        self.output[cellindex][attrbt] = attr
                    except BaseException:
                        self.output[cellindex][attrbt] = str(attr)
                self.output[cellindex]['srate'] = 1E3 / self.dt_output

            # collect probe output
            for probe in self.probes:
                self.output[cellindex][probe.__class__.__name__] =                     probe.data.copy()

            print('cell %s population %s in %.2f s' % (cellindex, self.y,
                                                       time() - tic))

    def _get_cell(self, p, morphology, labels, decor):
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

        # Gather geometry of CVs and assign segments to each CV,
        # get segment geometries and mapping to CV indices
        x, y, z, d = [np.array([], dtype=float).reshape((0, 2))] * 4
        CV_ind = np.array([], dtype=int)  # tracks which CV owns segment
        for i, m in enumerate(I_m_meta):
            segs = p.segments([m])
            for j, seg in enumerate(segs):
                x = np.row_stack([x, [seg.prox.x, seg.dist.x]])
                y = np.row_stack([y, [seg.prox.y, seg.dist.y]])
                z = np.row_stack([z, [seg.prox.z, seg.dist.z]])
                d = np.row_stack([d, [seg.prox.radius * 2, seg.dist.radius * 2]])
                CV_ind = np.r_[CV_ind, i]

        # define a list of loc_sets with relative location in between proximal and distal points of each CV
        loc_sets = np.array([
            '(location {} {})'.format(c.branch, np.mean([c.prox, c.dist])) for c in I_m_meta
        ])

        # CellGeometry object
        cell = lfpykit.CellGeometry(
            x=x,
            y=y,
            z=z,
            d=d
        )

        # set some needed attributes:
        cell._CV_ind = CV_ind
        cell._loc_sets = loc_sets

        return cell

    def _get_times_weights_taus_CV_inds(self, cellindex):
        times = np.array([])
        weights = np.array([])
        taus = np.array([])
        CV_inds = np.array([], dtype=int)

        # iterate over presynaptic network populations
        for i, X in enumerate(self.X):
            for j, idx in enumerate(self.synIdx[cellindex][X]):
                weights = np.hstack((weights, np.ones(idx.size) * self.J_yX[i]))
                taus = np.hstack((taus, np.ones(idx.size) * self.tau_yX[i]))
                CV_inds = np.hstack((CV_inds, idx))

                if self.synDelays is not None:
                    synDelays = self.synDelays[cellindex][X][j]
                else:
                    synDelays = None


                try:
                    spikes = self.networkSim.dbs[X].select(self.SpCells[cellindex][X][j])
                except AttributeError:
                    raise AssertionError(
                        'could not open CachedNetwork database objects')

                # convert to object array for slicing
                spikes = np.array(spikes, dtype=object)

                # apply synaptic delays
                if synDelays is not None and idx.size > 0:
                    for k, delay in enumerate(synDelays):
                        if spikes[k].size > 0:
                            spikes[k] += delay

                times = np.hstack((times, spikes))

        # TODO: remove redundant synapses by combining spike trains
        # where weights and taus and CV_inds are equal
        times_ = []
        weights_ = np.array([])
        taus_ = np.array([])
        CV_inds_ = np.array([], dtype=int)

        for tau in np.unique(taus):
            i_0 = taus == tau
            for w in np.unique(weights):
                i_1 = weights == w
                for CV_i in np.unique(CV_inds):
                    i_2 = CV_inds == CV_i
                    i_3 = i_0 & i_1 & i_2
                    if i_3.sum() > 0:
                        st = np.concatenate(times[i_3])
                        st.sort()
                        times_.append(st)
                        weights_ = np.r_[weights_, w]
                        taus_ = np.r_[taus_, tau]
                        CV_inds_ = np.r_[CV_inds_, CV_i]

        times_ = np.array(times_, dtype=object)

        return times_, weights_, taus_, CV_inds_

    def _get_I_m(self, cell, cellindex, decor, morphology, labels):
        # Recipe requires "flat" lists of times and weights for each "connection"
        # plus time constants, CV indices
        times, weights, taus, CV_inds = self._get_times_weights_taus_CV_inds(cellindex)

        # synapse loc_sets (CV midpoints)
        syn_loc_sets = cell._loc_sets[CV_inds]

        # create synapses at each loc_set
        if self.synParams['syntype'] == 'AlphaISyn':
            synapse = 'alphaisyn'   # workaround
            for i, (loc_set, tau) in enumerate(zip(syn_loc_sets, taus)):
                synapse_params = {'tau': tau}
                decor.place(loc_set, arbor.mechanism(synapse, synapse_params), f'{i}')
        else:
            raise NotImplementedError

        # number of CVs per branch
        policy = arbor.cv_policy_max_extent(self.cellParams['max_cv_length'])
        decor.discretization(policy)

        # create cell and set properties
        cable_cell = arbor.cable_cell(morphology, labels, decor)

        # instantiate recipe with cable_cell
        recipe = Recipe(cable_cell, weights=weights, times=times)

        # instantiate simulation
        context = arbor.context()
        domains = arbor.partition_load_balance(recipe, context)
        sim = arbor.simulation(recipe, domains, context)

        # set up sampling on probes
        schedule = arbor.regular_schedule(self.dt)
        i_handle = sim.sample(recipe.iprobe_id, schedule, arbor.sampling_policy.exact)

        # run simulation and collect results.
        # sim.run(tfinal=self.cellParams['tstop'])
        self._bench_sim_run(sim)

        # extract I_m for each CV
        I_m_samples, _ = sim.samples(i_handle)[0]

        # transmembrane currents in nA
        return I_m_samples[:, 1:].T

    def _bench_sim_run(self, sim):
        tic = time()
        sim.run(tfinal=self.cellParams['tstop'] + self.dt, dt=self.dt)
        toc = time()
        self._cumulative_sim_time += toc - tic

    def _get_transformation_matrix_for_CV(self, cell, probe, I_m):
        # mapping per segment (frusta)
        M_tmp = probe.get_transformation_matrix()

        # Define response matrix from M with columns weighted by area of each frusta
        M = np.zeros((probe.z.shape[0], I_m.shape[0]))
        for i in range(I_m.shape[0]):
            inds = cell._CV_ind == i
            M[:, i] = M_tmp[:, inds] @ (cell.area[inds] / cell.area[inds].sum())

        return M


# In[10]:


# %%prun -s cumulative -q -l 20 -T prun0

# measure population simulation times including setup etc.
ticc = time()

# arbor sim.run cumulative time
ticc_run = np.array(0.)

####### Set up populations ###############################################
if properrun:
    # iterate over each cell type, and create populationulation object
    for i, Y in enumerate(PS.X):
        # create population:
        pop = ArborPopulation(
            cellParams=PS.cellParams[Y],
            rand_rot_axis=PS.rand_rot_axis[Y],
            simulationParams=PS.simulationParams,
            populationParams=PS.populationParams[Y],
            y=Y,
            layerBoundaries=PS.layerBoundaries,
            savelist=PS.savelist,
            savefolder=PS.savefolder,
            probes=probes,
            dt_output=PS.dt_output,
            POPULATIONSEED=SIMULATIONSEED + i,
            X=PS.X,
            networkSim=networkSim,
            k_yXL=PS.k_yXL[Y],
            synParams=PS.synParams[Y],
            synDelayLoc=PS.synDelayLoc[Y],
            synDelayScale=PS.synDelayScale[Y],
            J_yX=PS.J_yX[Y],
            tau_yX=PS.tau_yX[Y],
        )

        # run population simulation and collect the data
        pop.run()
        pop.collect_data()

        # arbor sim.run cumulative time
        ticc_run += pop._cumulative_sim_time

        # object no longer needed
        del pop

# tic toc
tocc = time() - ticc
if RANK == 0:
    with open(os.path.join(savefolder, 'time_pop.txt'), 'w') as f:
        f.write(f'{tocc}')

if RANK == 0:
    tocc_run = np.zeros_like(ticc_run)
else:
    tocc_run = None

COMM.Reduce(ticc_run, tocc_run, op=MPI.SUM, root=0)
if RANK == 0:
    tocc_run /= SIZE
    with open(os.path.join(savefolder, 'time_run.txt'), 'w') as f:
        f.write(f'{float(tocc_run)}')


# In[11]:


####### Postprocess the simulation output ################################

# reset seed, but output should be deterministic from now on
np.random.seed(SIMULATIONSEED)

if properrun:
    # do some postprocessing on the collected data, i.e., superposition
    # of population LFPs, CSDs etc
    postproc = PostProcess(y=PS.X,
                           dt_output=PS.dt_output,
                           savefolder=PS.savefolder,
                           mapping_Yy=PS.mapping_Yy,
                           savelist=PS.savelist,
                           probes=probes,
                           cells_subfolder=os.path.split(PS.cells_path)[-1],
                           populations_subfolder=os.path.split(PS.populations_path)[-1],
                           figures_subfolder=os.path.split(PS.figures_path)[-1])

    # run through the procedure
    postproc.run()

    # create tar-archive with output for plotting, ssh-ing etc.
    postproc.create_tar_archive()


COMM.Barrier()

# tic toc
print('Execution time: %.3f seconds' % (time() - tic))
