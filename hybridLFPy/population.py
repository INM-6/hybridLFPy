#!/usr/bin/env python
"""
Class methods defining multicompartment neuron populations in the hybrid scheme
"""
import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI
from .gdf import GDF
from . import csd
from . import helpers
import LFPy
import neuron
from time import time


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


############ class objects #####################################################
class PopulationSuper(object):
    """
    Main population class object, let one set up simulations, execute, and
    compile the results. This class is suitable for subclassing for
    custom cell simulation procedures, inherit things like gathering of
    data written to disk.

    Note that `PopulationSuper.cellsim` do not have any stimuli,
    its main purpose is to gather common methods for inherited Population
    objects.


    Parameters
    ----------
    cellParams : dict
        Parameters for class `LFPy.Cell`
    rand_rot_axis : list
        Axis of which to randomly rotate morphs.
    simulationParams : dict
        Additional args for `LFPy.Cell.simulate()`.
    populationParams : dict
        Constraints for population and cell number.
    y : str
        Population identifier string.
    layerBoundaries : list of lists
        Each element is a list setting upper and lower layer boundary (floats)
    electrodeParams : dict
        parameters for class `LFPy.RecExtElectrode`
    savelist : list
        `LFPy.Cell` arguments to save for each single-cell simulation.
    savefolder : str
        path to where simulation results are stored.
    calculateCSD : bool
        Exctract laminar CSD from transmembrane currents
    dt_output : float
        Time resolution of output, e.g., LFP, CSD etc.
    recordSingleContribFrac : float
        fraction  in [0, 1] of individual neurons in population which output
        will be stored
    POPULATIONSEED : int/float
        Random seed for population, for positions etc.
    verbose : bool
        Verbosity flag.
    output_file : str
        formattable string for population output, e.g., '{}_population_{}'


    Returns
    -------
    hybridLFPy.population.PopulationSuper object


    See also
    --------
    Population, LFPy.Cell, LFPy.RecExtElectrode

    """

    def __init__(self,
                 cellParams={
                    'morphology': 'morphologies/ex.hoc',
                    'Ra': 150,
                    'cm': 1.0,
                    'e_pas': 0.0,
                    'lambda_f': 100,
                    'nsegs_method': 'lambda_f',
                    'rm': 20000.0,
                    'timeres_NEURON': 0.1,
                    'timeres_python': 0.1,
                    'tstartms': 0,
                    'tstopms': 1000.0,
                    'v_init': 0.0,
                    'verbose': False},
                 rand_rot_axis=[],
                 simulationParams={},
                 populationParams={
                    'min_cell_interdist': 1.0,
                    'number': 400,
                    'radius': 100,
                    'z_max': -350,
                    'z_min': -450},
                 y = 'EX',
                 layerBoundaries = [[0.0, -300], [-300, -500]],
                 electrodeParams={
                    'N': [[1, 0, 0], [1, 0, 0], [1, 0, 0],
                          [1, 0, 0], [1, 0, 0], [1, 0, 0]],
                    'method': 'som_as_point',
                    'n': 20,
                    'r': 5,
                    'r_z': [[-1e+199, -600, -550, 1e+99], [0, 0, 10, 10]],
                    'seedvalue': None,
                    'sigma': 0.3,
                    'x': [0, 0, 0, 0, 0, 0],
                    'y': [0, 0, 0, 0, 0, 0],
                    'z': [-0.0, -100.0, -200.0, -300.0, -400.0, -500.0]},
                 savelist=['somapos', 'x', 'y', 'z', 'LFP', 'CSD'],
                 savefolder='simulation_output_example_brunel',
                 calculateCSD=True,
                 dt_output=1.,
                 recordSingleContribFrac=0,
                 POPULATIONSEED=123456,
                 verbose=False,
                 output_file='{}_population_{}'
                 ):

        """
        Main population class object, let one set up simulations, execute, and
        compile the results. This class is suitable for subclassing for
        custom cell simulation procedures, inherit things like gathering of
        data written to disk.

        Note that `PopulationSuper.cellsim` do not have any stimuli,
        its main purpose is to gather common methods for inherited Population
        objects.


        Parameters
        ----------
        cellParams : dict
            Parameters for class `LFPy.Cell`
        rand_rot_axis : list
            Axis of which to randomly rotate morphs.
        simulationParams : dict
            Additional args for `LFPy.Cell.simulate()`.
        populationParams : dict
            Constraints for population and cell number.
        y : str
            Population identifier string.
        layerBoundaries : list of lists
            Each element is a list setting upper and lower layer boundary as
            floats
        electrodeParams : dict
            parameters for class `LFPy.RecExtElectrode`
        savelist : list
            `LFPy.Cell` arguments to save for each single-cell simulation.
        savefolder : str
            path to where simulation results are stored.
        calculateCSD : bool
            Exctract laminar CSD from transmembrane currents
        dt_output : float
            Time resolution of output, e.g., LFP, CSD etc.
        recordSingleContribFrac : float
            fraction  in [0, 1] of individual neurons in population which output
            will be stored
        POPULATIONSEED : int/float
            Random seed for population, for positions etc.
        verbose : bool
            Verbosity flag.
        output_file : str
            formattable string for population output, e.g., '{}_population_{}'


        Returns
        -------
        hybridLFPy.population.PopulationSuper object


        See also
        --------
        Population, LFPy.Cell, LFPy.RecExtElectrode

        """

        self.cellParams = cellParams
        self.dt = self.cellParams['timeres_python']
        self.rand_rot_axis = rand_rot_axis
        self.simulationParams = simulationParams
        self.populationParams = populationParams
        self.POPULATION_SIZE = populationParams['number']
        self.y = y
        self.layerBoundaries = np.array(layerBoundaries)
        self.electrodeParams = electrodeParams
        self.savelist = savelist
        self.savefolder = savefolder
        self.calculateCSD = calculateCSD
        self.dt_output = dt_output
        self.recordSingleContribFrac = recordSingleContribFrac
        self.output_file = output_file
        
        #check that decimate fraction is actually a whole number
        try:
            assert int(self.dt_output / self.dt) == self.dt_output / self.dt
        except AssertionError as ae:
            raise ae, 'dt_output not an integer multiple of dt'
        
        self.decimatefrac = int(self.dt_output / self.dt)
        self.POPULATIONSEED = POPULATIONSEED
        self.verbose = verbose
        
        #put revision info in savefolder
        if self.savefolder is not None:
            os.system('git rev-parse HEAD -> %s/populationRevision.txt' % \
                    self.savefolder)

        #set the random seed for reproducible populations, synapse locations,
        #presynaptic spiketrains
        np.random.seed(self.POPULATIONSEED)

        #using these colors and alphas:
        self.colors = []
        for i in range(self.POPULATION_SIZE):
            i *= 256.
            if self.POPULATION_SIZE > 1:
                i /= self.POPULATION_SIZE - 1.
            else:
                i /= self.POPULATION_SIZE

            try:
                self.colors.append(plt.cm.rainbow(int(i)))
            except:
                self.colors.append(plt.cm.gist_rainbow(int(i)))


        self.alphas = np.ones(self.POPULATION_SIZE)


        self.pop_soma_pos = self.set_pop_soma_pos()
        self.rotations = self.set_rotations()

        self._set_up_savefolder()

        self.CELLINDICES = np.arange(self.POPULATION_SIZE)
        self.RANK_CELLINDICES = self.CELLINDICES[self.CELLINDICES % SIZE
                                                 == RANK]

        #container for single-cell output generated on this RANK
        self.output = dict((i, {}) for i in self.RANK_CELLINDICES)


    def _set_up_savefolder(self):
        """
        Create catalogs for different file output to clean up savefolder.

        Non-public method


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        if self.savefolder == None:
            return

        self.cells_path = os.path.join(self.savefolder, 'cells')
        if RANK == 0:
            if not os.path.isdir(self.cells_path):
                os.mkdir(self.cells_path)

        self.figures_path = os.path.join(self.savefolder, 'figures')
        if RANK == 0:
            if not os.path.isdir(self.figures_path):
                os.mkdir(self.figures_path)

        self.populations_path = os.path.join(self.savefolder, 'populations')
        if RANK == 0:
            if not os.path.isdir(self.populations_path):
                os.mkdir(self.populations_path)

        COMM.Barrier()


    def run(self):
        """
        Distribute individual cell simulations across ranks.

        This method takes no keyword arguments.


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        for cellindex in self.RANK_CELLINDICES:
            self.cellsim(cellindex)

        COMM.Barrier()


    def cellsim(self, cellindex, return_just_cell=False):
        """
        Single-cell `LFPy.Cell` simulation without any stimulus, mostly for
        reference, as no stimulus is added


        Parameters
        ----------
        cellindex : int
            cell index between 0 and POPULATION_SIZE-1.
        return_just_cell : bool
            If True, return only the LFPy.Cell object
            if False, run full simulation, return None.


        Returns
        -------
        None
            if `return_just_cell is False
        cell : `LFPy.Cell` instance
            if `return_just_cell` is True


        See also
        --------
        LFPy.Cell, LFPy.Synapse, LFPy.RecExtElectrode

        """
        tic = time()
        
        electrode = LFPy.RecExtElectrode(**self.electrodeParams)

        cellParams = self.cellParams.copy()
        cell = LFPy.Cell(**cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        if return_just_cell:
            return cell
        else:
            if self.calculateCSD:
                cell.tvec = np.arange(cell.totnsegs)
                cell.imem = np.eye(cell.totnsegs)
                csdcoeff = csd.true_lam_csd(cell,
                                self.populationParams['radius'], electrode.z)
                csdcoeff *= 1E6 #nA mum^-3 -> muA mm^-3 conversion
                del cell.tvec, cell.imem
                cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                              **self.simulationParams)
                cell.CSD = helpers.decimate(cell.dotprodresults[0],
                                            q=self.decimatefrac)
            else:
                cell.simulate(electrode,
                              **self.simulationParams)

            cell.LFP = helpers.decimate(electrode.LFP,
                                        q=self.decimatefrac)


            cell.x = electrode.x
            cell.y = electrode.y
            cell.z = electrode.z
            cell.electrodecoeff = electrode.electrodecoeff


            #put all necessary cell output in output dict
            for attrbt in self.savelist:
                attr = getattr(cell, attrbt)
                if type(attr) == np.ndarray:
                    self.output[cellindex][attrbt] = attr.astype('float32')
                else:
                    try:
                        self.output[cellindex][attrbt] = attr
                    except:
                        self.output[cellindex][attrbt] = str(attr)
                self.output[cellindex]['srate'] = 1E3 / self.dt_output

            print('cell %s population %s in %.2f s' % (cellindex, self.y,
                                                              time()-tic))


    def set_pop_soma_pos(self):
        """
        Set `pop_soma_pos` using draw_rand_pos().

        This method takes no keyword arguments.


        Parameters
        ----------
        None


        Returns
        -------
        numpy.ndarray
            (x,y,z) coordinates of each neuron in the population


        See also
        --------
        PopulationSuper.draw_rand_pos

        """
        tic = time()
        if RANK == 0:
            pop_soma_pos = self.draw_rand_pos(
                min_r = self.electrodeParams['r_z'],
                **self.populationParams)
        else:
            pop_soma_pos = None

        if RANK == 0:
            print('found cell positions in %.2f s' % (time()-tic))

        return COMM.bcast(pop_soma_pos, root=0)


    def set_rotations(self):

        """

        Append random z-axis rotations for each cell in population.

        This method takes no keyword arguments


        Parameters
        ----------
        None


        Returns
        -------
        numpyp.ndarray
            Rotation angle around axis `Population.rand_rot_axis` of each neuron
            in the population


        """
        tic = time()
        if RANK == 0:
            rotations = []
            for i in range(self.POPULATION_SIZE):
                defaultrot = {}
                for axis in self.rand_rot_axis:
                    defaultrot.update({axis : np.random.rand() * 2 * np.pi})
                rotations.append(defaultrot)
        else:
            rotations = None

        if RANK == 0:
            print('found cell rotations in %.2f s' % (time()-tic))

        return COMM.bcast(rotations, root=0)


    def calc_min_cell_interdist(self, x, y, z):
        """
        Calculate cell interdistance from input coordinates.


        Parameters
        ----------
        x, y, z : numpy.ndarray
            xyz-coordinates of each cell-body.


        Returns
        -------
        min_cell_interdist : np.nparray
            For each cell-body center, the distance to nearest neighboring cell

        """
        min_cell_interdist = np.zeros(self.POPULATION_SIZE)

        for i in range(self.POPULATION_SIZE):
            cell_interdist = np.sqrt((x[i] - x)**2
                    + (y[i] - y)**2
                    + (z[i] - z)**2)
            cell_interdist[i] = np.inf
            min_cell_interdist[i] = cell_interdist.min()

        return min_cell_interdist


    def draw_rand_pos(self, radius, z_min, z_max,
                      min_r=np.array([0]), min_cell_interdist=10., **args):
        """
        Draw some random location within radius, z_min, z_max,
        and constrained by min_r and the minimum cell interdistance.
        Returned argument is a list of dicts [{'xpos', 'ypos', 'zpos'},].


        Parameters
        ----------
        radius : float
            Radius of population.
        z_min : float
            Lower z-boundary of population.
        z_max : float
            Upper z-boundary of population.
        min_r : numpy.ndarray
            Minimum distance to center axis as function of z.
        min_cell_interdist : float
            Minimum cell to cell interdistance.
        **args : keyword arguments
            Additional inputs that is being ignored.


        Returns
        -------
        soma_pos : list
            List of dicts of len population size
            where dict have keys xpos, ypos, zpos specifying
            xyz-coordinates of cell at list entry `i`.


        See also
        --------
        PopulationSuper.calc_min_cell_interdist

        """
        x = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        y = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        z = np.random.rand(self.POPULATION_SIZE)*(z_max - z_min) + z_min
        min_r_z = {}
        min_r = np.array(min_r)
        if min_r.size > 0:
            if type(min_r) == type(np.array([])):
                j = 0
                for j in range(min_r.shape[0]):
                    min_r_z[j] = np.interp(z, min_r[0,], min_r[1,])
                    if j > 0:
                        [w] = np.where(min_r_z[j] < min_r_z[j-1])
                        min_r_z[j][w] = min_r_z[j-1][w]
                minrz = min_r_z[j]
        else:
            minrz = np.interp(z, min_r[0], min_r[1])

        R_z = np.sqrt(x**2 + y**2)

        #want to make sure that no somas are in the same place.
        cell_interdist = self.calc_min_cell_interdist(x, y, z)

        [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
            cell_interdist < min_cell_interdist))

        while len(u) > 0:
            for i in range(len(u)):
                x[u[i]] = (np.random.rand()-0.5)*radius*2
                y[u[i]] = (np.random.rand()-0.5)*radius*2
                z[u[i]] = np.random.rand()*(z_max - z_min) + z_min
                if type(min_r) == type(()):
                    for j in range(np.shape(min_r)[0]):
                        min_r_z[j][u[i]] = \
                                np.interp(z[u[i]], min_r[0,], min_r[1,])
                        if j > 0:
                            [w] = np.where(min_r_z[j] < min_r_z[j-1])
                            min_r_z[j][w] = min_r_z[j-1][w]
                        minrz = min_r_z[j]
                else:
                    minrz[u[i]] = np.interp(z[u[i]], min_r[0,], min_r[1,])
            R_z = np.sqrt(x**2 + y**2)

            #want to make sure that no somas are in the same place.
            cell_interdist = self.calc_min_cell_interdist(x, y, z)

            [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
                cell_interdist < min_cell_interdist))

        
        soma_pos = []
        for i in range(self.POPULATION_SIZE):
            soma_pos.append({'xpos' : x[i], 'ypos' : y[i], 'zpos' : z[i]})

        return soma_pos


    def calc_signal_sum(self, measure='LFP'):
        """
        Superimpose each cell's contribution to the compound population signal,
        i.e., the population CSD or LFP


        Parameters
        ----------
        measure : str
            {'LFP', 'CSD'}: Either 'LFP' or 'CSD'.


        Returns
        -------
        numpy.ndarray
            The populations-specific compound signal.

        """
        #compute the total LFP of cells on this RANK
        if self.RANK_CELLINDICES.size > 0:
            for i, cellindex in enumerate(self.RANK_CELLINDICES):
                if i == 0:
                    data = self.output[cellindex][measure]
                else:
                    data += self.output[cellindex][measure]
        else:
            data = np.zeros((len(self.electrodeParams['x']),
                             self.cellParams['tstopms']/self.dt_output + 1),
                dtype=np.float32)

        #container for full LFP on RANK 0
        if RANK == 0:
            DATA = np.zeros_like(data, dtype=np.float32)
        else:
            DATA = None

        #sum to RANK 0 using automatic type discovery with MPI
        COMM.Reduce(data, DATA, op=MPI.SUM, root=0)

        return DATA


    def collectSingleContribs(self, measure='LFP'):
        """
        Collect single cell data and save them to HDF5 file.
        The function will also return signals generated by all cells


        Parameters
        ----------
        measure : str
            {'LFP', 'CSD'}: Either 'LFP' or 'CSD'.


        Returns
        -------
        numpy.ndarray
            output of all neurons in population, axis 0 correspond to neuron ind

        """
        try:
            assert(self.recordSingleContribFrac <= 1 and
                   self.recordSingleContribFrac >= 0)
        except AssertionError as ae:
            raise ae, 'recordSingleContribFrac {} not in [0, 1]'.format(
                                                self.recordSingleContribFrac)

        if not self.recordSingleContribFrac:
            return
        else:
            #reconstruct RANK_CELLINDICES of all RANKs for controlling
            #communication
            if self.recordSingleContribFrac == 1.:
                SAMPLESIZE = self.POPULATION_SIZE
                RANK_CELLINDICES = []
                for i in range(SIZE):
                    RANK_CELLINDICES += [self.CELLINDICES[
                        self.CELLINDICES % SIZE == i]]
            else:
                SAMPLESIZE = int(self.recordSingleContribFrac *
                                 self.POPULATION_SIZE)
                RANK_CELLINDICES = []
                for i in range(SIZE):
                    ids = self.CELLINDICES[self.CELLINDICES % SIZE == i]
                    RANK_CELLINDICES += [ids[ids < SAMPLESIZE]]

            #gather data on this RANK
            if RANK_CELLINDICES[RANK].size > 0:
                for i, cellindex in enumerate(RANK_CELLINDICES[RANK]):
                    if i == 0:
                        data_temp = np.zeros([RANK_CELLINDICES[RANK].size] +
                                            list(self.output[cellindex
                                                             ][measure].shape),
                                            dtype=np.float32)
                    data_temp[i, ] = self.output[cellindex][measure]

            if RANK == 0:
                #container of all output
                data = np.zeros([SAMPLESIZE] +
                                 list(self.output[cellindex][measure].shape),
                                 dtype=np.float32)

                #fill in values from this RANK
                if RANK_CELLINDICES[0].size > 0:
                    for j, k in enumerate(RANK_CELLINDICES[0]):
                        data[k, ] = data_temp[j, ]

                #iterate over all other RANKs
                for i in range(1, len(RANK_CELLINDICES)):
                    if RANK_CELLINDICES[i].size > 0:
                        #receive on RANK 0 from all other RANK
                        data_temp = np.zeros([RANK_CELLINDICES[i].size] +
                                        list(self.output[cellindex
                                                         ][measure].shape),
                                        dtype=np.float32)
                        COMM.Recv([data_temp, MPI.FLOAT], source=i, tag=13)

                        #fill in values
                        for j, k in enumerate(RANK_CELLINDICES[i]):
                            data[k, ] = data_temp[j, ]
            else:
                data = None
                if RANK_CELLINDICES[RANK].size > 0:
                    #send to RANK 0
                    COMM.Send([data_temp, MPI.FLOAT], dest=0, tag=13)

            if RANK == 0:
                #save all single-cell data to file
                fname = os.path.join(self.populations_path,
                                     '%s_%ss.h5' % (self.y, measure))
                f = h5py.File(fname, 'w')
                f.create_dataset('data', data=data, compression=4)
                f['srate'] = self.output[0]['srate']
                f.close()
                assert(os.path.isfile(fname))

                print('file %s_%ss.h5 ok' % (self.y, measure))

            COMM.Barrier()

            return data


    def collect_data(self):
        """
        Collect LFPs, CSDs and soma traces from each simulated population,
        and save to file.


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        #collect some measurements resolved per file and save to file
        for measure in ['LFP', 'CSD']:
            if measure in self.savelist:
                self.collectSingleContribs(measure)


        #calculate lfp from all cell contribs
        lfp = self.calc_signal_sum(measure='LFP')

        #calculate CSD in every lamina
        if self.calculateCSD:
            csd = self.calc_signal_sum(measure='CSD')

        if RANK == 0 and self.POPULATION_SIZE > 0:
            #saving LFPs
            if 'LFP' in self.savelist:
                fname = os.path.join(self.populations_path,
                                     self.output_file.format(self.y,
                                                             'LFP')+'.h5')
                f = h5py.File(fname, 'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=lfp, compression=4)
                f.close()
                del lfp
                assert(os.path.isfile(fname))
                print('save lfp ok')


            #saving CSDs
            if 'CSD' in self.savelist and self.calculateCSD:
                fname = os.path.join(self.populations_path,
                                     self.output_file.format(self.y,
                                                             'CSD')+'.h5')
                f = h5py.File(fname, 'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=csd, compression=4)
                f.close()
                del csd
                assert(os.path.isfile(fname))
                print('save CSD ok')


            #save the somatic placements:
            pop_soma_pos = np.zeros((self.POPULATION_SIZE, 3))
            keys = ['xpos', 'ypos', 'zpos']
            for i in range(self.POPULATION_SIZE):
                for j in range(3):
                    pop_soma_pos[i, j] = self.pop_soma_pos[i][keys[j]]
            fname = os.path.join(self.populations_path,
                                 self.output_file.format(self.y, 'somapos.gdf'))
            np.savetxt(fname, pop_soma_pos)
            assert(os.path.isfile(fname))
            print('save somapos ok')

            #save rotations using hdf5
            fname = os.path.join(self.populations_path,
                                    self.output_file.format(self.y, 'rotations.h5'))
            f = h5py.File(fname, 'w')
            f.create_dataset('x', (len(self.rotations),))
            f.create_dataset('y', (len(self.rotations),))
            f.create_dataset('z', (len(self.rotations),))

            for i, rot in enumerate(self.rotations):
                for key, value in list(rot.items()):
                    f[key][i] = value
            f.close()
            assert(os.path.isfile(fname))
            print('save rotations ok')


        #resync threads
        COMM.Barrier()


class Population(PopulationSuper):
    """
    Class `hybridLFPy.Population`, inherited from class `PopulationSuper`.

    This class rely on spiking times recorded in a network simulation,
    layer-resolved indegrees, synapse parameters, delay parameters, all per
    presynaptic population.


    Parameters
    ----------
    X : list of str
        Each element denote name of presynaptic populations.
    networkSim : `hybridLFPy.cachednetworks.CachedNetwork` object
        Container of network spike events resolved per population
    k_yXL : numpy.ndarray
        Num layers x num presynapse populations array specifying the
        number of incoming connections per layer and per population type.
    synParams : dict of dicts
        Synapse parameters (cf. `LFPy.Synapse` class).
        Each toplevel key denote each presynaptic population,
        bottom-level dicts are parameters passed to `LFPy.Synapse`.
    synDelayLoc : list
        Average synapse delay for each presynapse connection.
    synDelayScale : list
        Synapse delay std for each presynapse connection.
    J_yX : list of floats
        Synapse weights for connections of each presynaptic population, see
        class `LFPy.Synapse`


    Returns
    -------
    `hybridLFPy.population.Population` object


    See also
    --------
    PopulationSuper, CachedNetwork, CachedFixedSpikesNetwork,
    CachedNoiseNetwork, LFPy.Cell, LFPy.RecExtElectrode
    """

    def __init__(self,
                X = ['EX', 'IN'],
                networkSim = 'hybridLFPy.cachednetworks.CachedNetwork',
                k_yXL = [[20,  0], [20, 10]],
                synParams = {
                    'EX': {
                        'section': ['apic', 'dend'],
                        'syntype': 'AlphaISyn',
                        # 'tau': [0.5, 0.5]
                        },
                    'IN': {
                        'section': ['dend'],
                        'syntype': 'AlphaISyn',
                        # 'tau': [0.5, 0.5],
                        },
                    },
                synDelayLoc = [1.5, 1.5],
                synDelayScale = [None, None],
                J_yX = [0.20680155243678455, -1.2408093146207075],
                tau_yX = [0.5, 0.5],
                #calculateCSD = True,
                **kwargs):
        """
        Class `hybridLFPy.Population`, inherited from class `PopulationSuper`.

        This class rely on spiking times recorded in a network simulation,
        layer-resolved indegrees, synapse parameters, delay parameters, all per
        presynaptic population.


        Parameters
        ----------
        X : list of str
            Each element denote name of presynaptic populations.
        networkSim : `hybridLFPy.cachednetworks.CachedNetwork` object
            Container of network spike events resolved per population
        k_yXL : numpy.ndarray
            Num layers x num presynapse populations array specifying the
            number of incoming connections per layer and per population type.
        synParams : dict of dicts
            Synapse parameters (cf. `LFPy.Synapse` class).
            Each toplevel key denote each presynaptic population,
            bottom-level dicts are parameters passed to `LFPy.Synapse`, however,
            time constants `tau' takes one value per presynaptic population. 
        synDelayLoc : list
            Average synapse delay for each presynapse connection.
        synDelayScale : list
            Synapse delay std for each presynapse connection.
        J_yX : list of floats
            Synapse weights for connections from each presynaptic population,
            see class `LFPy.Synapse`
        tau_yX : list of floats
            Synapse time constants for connections from each presynaptic
            population
        #calculateCSD : bool
        #    Flag for computing the ground-source CSD.


        Returns
        -------
        `hybridLFPy.population.Population` object


        See also
        --------
        PopulationSuper, CachedNetwork, CachedFixedSpikesNetwork,
        CachedNoiseNetwork, LFPy.Cell, LFPy.RecExtElectrode
        """
        tic = time()

        PopulationSuper.__init__(self, **kwargs)
        #set some class attributes
        self.X = X
        self.networkSim = networkSim
        self.k_yXL = np.array(k_yXL)


        #local copy of synapse parameters
        self.synParams = synParams
        self.synDelayLoc = synDelayLoc
        self.synDelayScale = synDelayScale
        self.J_yX = J_yX
        self.tau_yX = tau_yX


        #Now loop over all cells in the population and assess
        # - number of synapses in each z-interval (from layerbounds)
        # - placement of synapses in each z-interval

        #get in this order, the
        # - postsynaptic compartment indices
        # - presynaptic cell indices
        # - synapse delays per connection
        self.synIdx = self.get_all_synIdx()
        self.SpCells = self.get_all_SpCells()
        self.synDelays = self.get_all_synDelays()

        if RANK == 0:
            print("population initialized in %.2f seconds" % (time()-tic))


    def get_all_synIdx(self):
        """
        Auxilliary function to set up class attributes containing
        synapse locations given as LFPy.Cell compartment indices

        This function takes no inputs.


        Parameters
        ----------
        None


        Returns
        -------
        synIdx : dict
            `output[cellindex][populationindex][layerindex]` numpy.ndarray of
            compartment indices.


        See also
        --------
        Population.get_synidx, Population.fetchSynIdxCell
        """
        tic = time()

        #containers for synapse idxs existing on this rank
        synIdx = {}


        #ok then, we will draw random numbers across ranks, which have to
        #be unique per cell. Now, we simply record the random state,
        #change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        for cellindex in self.RANK_CELLINDICES:
            #set the random seed on for each cellindex
            np.random.seed(self.POPULATIONSEED + cellindex)

            #find synapse locations for cell in parallel
            synIdx[cellindex] = self.get_synidx(cellindex)

        #reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found synapse locations in %.2f seconds' % (time()-tic))

        #print the number of synapses per layer from which presynapse population
        if self.verbose:
            for cellindex in self.RANK_CELLINDICES:
                for i, synidx in enumerate(synIdx[cellindex]):
                    print('to:\t%s\tcell:\t%i\tfrom:\t%s:' % (self.y,
                                                cellindex, self.X[i]),)
                    idxcount = 0
                    for idx in synidx:
                        idxcount += idx.size
                        print('\t%i' % idx.size,)
                    print('\ttotal %i' % idxcount)

        return synIdx


    def get_all_SpCells(self):
        """
        For each postsynaptic cell existing on this RANK, load or compute
        the presynaptic cell index for each synaptic connection

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        SpCells : dict
            `output[cellindex][populationname][layerindex]`, np.array of
            presynaptic cell indices.


        See also
        --------
        Population.fetchSpCells

        """
        tic = time()

        #container
        SpCells = {}

        #ok then, we will draw random numbers across ranks, which have to
        #be unique per cell. Now, we simply record the random state,
        #change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        for cellindex in self.RANK_CELLINDICES:
            #set the random seed on for each cellindex
            np.random.seed(self.POPULATIONSEED + cellindex + self.POPULATION_SIZE)

            SpCells[cellindex] = {}
            for i, X in enumerate(self.X):
                SpCells[cellindex][X] = self.fetchSpCells(
                    self.networkSim.nodes[X], self.k_yXL[:, i])

        #reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found presynaptic cells in %.2f seconds' % (time()-tic))

        return SpCells


    def get_all_synDelays(self):
        """
        Create and load arrays of connection delays per connection on this rank

        Get random normally distributed synaptic delays,
        returns dict of nested list of same shape as SpCells.

        Delays are rounded to dt.

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        dict
            output[cellindex][populationname][layerindex]`, np.array of
            delays per connection.


        See also
        --------
        numpy.random.normal

        """
        tic = time()

        #ok then, we will draw random numbers across ranks, which have to
        #be unique per cell. Now, we simply record the random state,
        #change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        #container
        delays = {}

        for cellindex in self.RANK_CELLINDICES:
            #set the random seed on for each cellindex
            np.random.seed(self.POPULATIONSEED + cellindex + 2*self.POPULATION_SIZE)

            delays[cellindex] = {}
            for j, X in enumerate(self.X):
                delays[cellindex][X] = []
                for i in self.k_yXL[:, j]:
                    loc = self.synDelayLoc[j]
                    loc /= self.dt
                    scale = self.synDelayScale[j]
                    if scale is not None:
                        scale /= self.dt
                        delay = np.random.normal(loc, scale, i).astype(int)
                        while np.any(delay < 1):
                            inds = delay < 1
                            delay[inds] = np.random.normal(loc, scale,
                                                        inds.sum()).astype(int)
                        delay = delay.astype(float)
                        delay *= self.dt
                    else:
                        delay = np.zeros(i) + self.synDelayLoc[j]
                    delays[cellindex][X].append(delay)

        #reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found delays in %.2f seconds' % (time()-tic))

        return delays


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
        #create a cell instance
        cell = self.cellsim(cellindex, return_just_cell=True)


        #local containers
        synidx = {}

        #get synaptic placements and cells from the network,
        #then set spike times,
        for i, X in enumerate(self.X):
            synidx[X] = self.fetchSynIdxCell(cell=cell,
                                             nidx=self.k_yXL[:, i],
                                             synParams=self.synParams.copy())

        return synidx


    def fetchSynIdxCell(self, cell, nidx, synParams):
        """
        Find possible synaptic placements for each cell
        As synapses are placed within layers with bounds determined by
        self.layerBoundaries, it will check this matrix accordingly, and
        use the probabilities from `self.connProbLayer to distribute.

        For each layer, the synapses are placed with probability normalized
        by membrane area of each compartment


        Parameters
        ----------
        cell : `LFPy.Cell` instance
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

        #segment indices in each layer is stored here, list of np.array
        syn_idx = []
        #loop over layer bounds, find synapse locations
        for i, zz in enumerate(self.layerBoundaries):
            if nidx[i] == 0:
                syn_idx.append(np.array([], dtype=int))
            else:
                syn_idx.append(cell.get_rand_idx_area_norm(
                                section=synParams['section'],
                                nidx=nidx[i],
                                z_min=zz.min(),
                                z_max=zz.max()).astype('int16'))

        return syn_idx

    def cellsim(self, cellindex, return_just_cell = False):
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
        
        cell = LFPy.Cell(**self.cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        if return_just_cell:
            #with several cells, NEURON can only hold one cell at the time
            allsecnames = []
            allsec = []
            for sec in cell.allseclist:
                allsecnames.append(sec.name())
                for seg in sec:
                    allsec.append(sec.name())
            cell.allsecnames = allsecnames
            cell.allsec = allsec
            return cell
        else:
            self.insert_all_synapses(cellindex, cell)

            #electrode object where LFPs are calculated
            electrode = LFPy.RecExtElectrode(**self.electrodeParams)

            if self.calculateCSD:
                cell.tvec = np.arange(cell.totnsegs)
                cell.imem = np.eye(cell.totnsegs)
                csdcoeff = csd.true_lam_csd(cell,
                                self.populationParams['radius'], electrode.z)
                csdcoeff *= 1E6 #nA mum^-3 -> muA mm^-3 conversion
                del cell.tvec, cell.imem
                cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                              **self.simulationParams)
                cell.CSD = helpers.decimate(cell.dotprodresults[0],
                                            q=self.decimatefrac)
            else:
                cell.simulate(electrode,
                              **self.simulationParams)

            cell.LFP = helpers.decimate(electrode.LFP,
                                        q=self.decimatefrac)


            cell.x = electrode.x
            cell.y = electrode.y
            cell.z = electrode.z

            cell.electrodecoeff = electrode.electrodecoeff

            #put all necessary cell output in output dict
            for attrbt in self.savelist:
                attr = getattr(cell, attrbt)
                if type(attr) == np.ndarray:
                    self.output[cellindex][attrbt] = attr.astype('float32')
                else:
                    try:
                        self.output[cellindex][attrbt] = attr
                    except:
                        self.output[cellindex][attrbt] = str(attr)
                self.output[cellindex]['srate'] = 1E3 / self.dt_output

            print('cell %s population %s in %.2f s' % (cellindex, self.y,
                                                              time()-tic))


    def insert_all_synapses(self, cellindex, cell):

        """
        Insert all synaptic events from all presynaptic layers on
        cell object with index `cellindex`.


        Parameters
        ----------
        cellindex : int
            cell index in the population.
        cell : `LFPy.Cell` instance
            Postsynaptic target cell.


        Returns
        -------
        None


        See also
        --------
        Population.insert_synapse

        """
        for i, X in enumerate(self.X): #range(self.k_yXL.shape[1]):
            synParams = self.synParams
            synParams.update({
                'weight' : self.J_yX[i],
                'tau' : self.tau_yX[i],
                })
            for j in range(len(self.synIdx[cellindex][X])):
                if self.synDelays is not None:
                    synDelays = self.synDelays[cellindex][X][j]
                else:
                    synDelays = None
                self.insert_synapses(cell = cell,
                                cellindex = cellindex,
                                synParams = synParams,
                                idx = self.synIdx[cellindex][X][j],
                                X=X,
                                SpCell = self.SpCells[cellindex][X][j],
                                synDelays = synDelays)


    def insert_synapses(self, cell, cellindex, synParams, idx = np.array([]),
                        X='EX', SpCell = np.array([]),
                        synDelays = None):
        """
        Insert synapse with `parameters`=`synparams` on cell=cell, with
        segment indexes given by `idx`. `SpCell` and `SpTimes` picked from
        Brunel network simulation


        Parameters
        ----------
        cell : `LFPy.Cell` instance
            Postsynaptic target cell.
        cellindex : int
            Index of cell in population.
        synParams : dict
            Parameters passed to `LFPy.Synapse`.
        idx : numpy.ndarray
            Postsynaptic compartment indices.
        X : str
            presynaptic population name
        SpCell : numpy.ndarray
            Presynaptic spiking cells.
        synDelays : numpy.ndarray
            Per connection specific delays.


        Returns
        -------
        None


        See also
        --------
        Population.insert_all_synapses

        """
        #Insert synapses in an iterative fashion
        try:
            spikes = self.networkSim.dbs[X].select(SpCell[:idx.size])
        except AttributeError as ae:
            raise ae, 'could not open CachedNetwork database objects'


        #apply synaptic delays
        if synDelays is not None and idx.size > 0:
            for i, delay in enumerate(synDelays):
                if spikes[i].size > 0:
                    spikes[i] += delay

        #create synapse events:
        for i in range(idx.size):
            if len(spikes[i]) == 0:
                pass
                #print 'no spike times, skipping network cell #%i' % SpCell[i]
            else:
                synParams.update({'idx' : idx[i]})
                # Create synapse(s) and setting times using class LFPy.Synapse
                synapse = LFPy.Synapse(cell, **synParams)
                #SpCell is a vector, or do not exist
                synapse.set_spike_times(spikes[i] + cell.tstartms)


    def fetchSpCells(self, nodes, numSyn):
        """
        For N (nodes count) nestSim-cells draw
        POPULATION_SIZE x NTIMES random cell indexes in
        the population in nodes and broadcast these as `SpCell`.

        The returned argument is a list with len = numSyn.size of np.arrays,
        assumes `numSyn` is a list


        Parameters
        ----------
        nodes : numpy.ndarray, dtype=int
            Node # of valid presynaptic neurons.
        numSyn : numpy.ndarray, dtype=int
            # of synapses per connection.


        Returns
        -------
        SpCells : list
            presynaptic network-neuron indices


        See also
        --------
        Population.fetch_all_SpCells
        """
        SpCell = []
        for size in numSyn:
            SpCell.append(np.random.randint(nodes.min(), nodes.max(),
                                            size=size).astype('int32'))
        return SpCell


if __name__ == '__main__':
    import doctest
    doctest.testmod()

