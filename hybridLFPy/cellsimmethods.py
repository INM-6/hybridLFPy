#!/usr/bin/env python
'''Class methods used by cellsim scripts'''
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI
from gdf import GDF
import csd
import helpers
import LFPy
import neuron
from time import time


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
MASTER_MODE = COMM.rank == 0


############ class objects #####################################################
class PopulationSuper(object):
    '''
    Main population class object, let one set up simulations, execute, and
    compile the results. This class is suitable for subclassing for
    custom cell simulation procedures, inherit things like gathering of
    data written to disk.

    Note that PopulationSuper.cellsim do not have any stimuli,
    just here for reference

    Parameters
    ----------
    cellParams : dict
        params for class LFPy.Cell
    rand_rot_axis : list
        axis of which to randomly rotate morphs
    simulationParams : dict
        additional args for LFPy.Cell.simulate()
    populationParams : dict
        constraints for population and cell number
    y : str
        population identifier string:
    electrodeParams : dict
        LFPy.RecExtElectrode-params
    savelist : list
        cell args to save for each cell simulation
    savefolder : str
        where simulation results are stored
    calculateCSD : bool
        exctract laminar CSD
    dt_output : float
        time resolution of output, e.g., LFP, CSD etc
    POPULATIONSEED : int/float
        random seed for population, for pos. etc
    verbose : bool
        verbosity flag

    '''
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
                 savelist=['somav', 'somapos', 'x', 'y', 'z', 'LFP', 'CSD'],
                 savefolder='simulation_output_example_brunel',
                 calculateCSD = True,
                 dt_output = 1.,
                 POPULATIONSEED=123456,
                 verbose=False,
                 ):
        '''
        
        Main population class object, let one set up simulations, execute, and
        compile the results. This class is suitable for subclassing for
        custom cell simulation procedures, inherit things like gathering of
        data written to disk.
    
        Note that PopulationSuper.cellsim do not have any stimuli,
        just here for reference
    
        Parameters
        ----------
        cellParams : dict
            params for class LFPy.Cell
        rand_rot_axis : list
            axis of which to randomly rotate morphs
        simulationParams : dict
            additional args for LFPy.Cell.simulate()
        populationParams : dict
            constraints for population and cell number
        y : str
            population identifier string
        layerBoundaries : list or np.ndarray
            for each layer, specify upper/lower boundaries
        electrodeParams : dict
            LFPy.RecExtElectrode-params
        savelist : list
            cell args to save for each cell simulation
        savefolder : str
            where simulation results are stored
        calculateCSD : bool
            exctract laminar CSD
        dt_output : float
            time resolution of output, e.g., LFP, CSD etc
        POPULATIONSEED : int/float
            random seed for population, for pos. etc
        verbose : bool
            verbosity flag
    
        '''
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
        #check that decimate fraction is actually a whole number
        assert int(self.dt_output / self.dt) == self.dt_output / self.dt
        self.decimatefrac = int(self.dt_output / self.dt)
        self.POPULATIONSEED = POPULATIONSEED
        self.verbose = verbose
        #put revision info in savefolder
        if self.savefolder != None:
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


    def _set_up_savefolder(self):
        '''
        Create catalogs for different file output to clean up savefolder
        '''
        if self.savefolder == None:
            return

        self.cells_path = os.path.join(self.savefolder, 'cells')
        if COMM.rank == 0:
            if not os.path.isdir(self.cells_path):
                os.mkdir(self.cells_path)

        self.figures_path = os.path.join(self.savefolder, 'figures')
        if COMM.rank == 0:
            if not os.path.isdir(self.figures_path):
                os.mkdir(self.figures_path)

        self.populations_path = os.path.join(self.savefolder, 'populations')
        if COMM.rank == 0:
            if not os.path.isdir(self.populations_path):
                os.mkdir(self.populations_path)

        COMM.Barrier()


    def run(self):
        '''
        Distribute individual cell simulations across ranks
        
        '''
        for cellindex in self.RANK_CELLINDICES:
            self.cellsim(cellindex)

        #resync
        COMM.Barrier()


    def cellsim(self, cellindex, return_just_cell=False):
        '''
        LFPy cell simulation without any stimulus, mostly for reference
        
        Parameters
        ----------
        cellindex : int
            cell index between 0 and population size-1
        return_just_cell : bool
            If True, return only the LFPy.Cell object
            if False, run full simulation, return None
        
        '''
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

            if self.simulationParams.has_key('to_file'):
                if self.simulationParams['to_file']:
                    cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                                  file_name=os.path.join(self.cells_path,
                            '%s_lfp_cell%.3i.h5') % (self.y, cellindex),
                                  **self.simulationParams)
                else:
                    cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                                  **self.simulationParams)
                    cell.LFP = electrode.LFP
                    cell.CSD = cell.dotprodresults[0]
            else:
                cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                              **self.simulationParams)
                cell.LFP = helpers.decimate(electrode.LFP,
                                            q=self.decimatefrac)
                cell.CSD = helpers.decimate(cell.dotprodresults[0],
                                            q=self.decimatefrac)

            #downsample somav
            cell.somav = helpers.decimate(cell.somav,
                                          q=self.decimatefrac)

            cell.x = electrode.x
            cell.y = electrode.y
            cell.z = electrode.z
            cell.electrodecoeff = electrode.electrodecoeff

            #access file object
            f = h5py.File(os.path.join(self.cells_path,
                            '%s_lfp_cell%.3i.h5') % (self.y, cellindex),
                          compression='gzip')

            if self.simulationParams.has_key('to_file'):
                if self.simulationParams['to_file']:
                    f.create_dataset('LFP',
                                     data=helpers.decimate(f['electrode000'
                                                             ].value,
                                                   q=self.decimatefrac),
                                     compression=4)
                    f.create_dataset('CSD', helpers.decimate(f['electrode001'
                                                               ].value,
                                                   q=self.decimatefrac),
                                     compression=4)
                    del f['electrode000']
                    del f['electrode001']


            #save stuff from savelist
            f['srate'] = 1E3 / self.dt_output
            for attrbt in self.savelist:
                try:
                    del(f[attrbt])
                except:
                    pass
                attr = getattr(cell, attrbt)
                if type(attr) == np.ndarray:
                    f[attrbt] = attr.astype('float32')
                else:
                    try:
                        f[attrbt] = attr
                    except:
                        f[attrbt] = str(attr)


            #print some stuff
            print 'SIZE %i, RANK %i, Cell %i, Min LFP: %.3f, Max LFP: %.3f' % \
                        (SIZE, RANK, cellindex,
                        f['LFP'].value.min(), f['LFP'].value.max())

            f.close()

            print 'Cell %s saved to file' % cellindex


    def set_pop_soma_pos(self):
        '''
        Set pop_soma_pos using draw_rand_pos()
        
        '''
        if MASTER_MODE:
            pop_soma_pos = self.draw_rand_pos(
                min_r = self.electrodeParams['r_z'],
                **self.populationParams)
        else:
            pop_soma_pos = None
        return COMM.bcast(pop_soma_pos, root=0)


    def set_rotations(self):
        '''
        Append random z-axis rotations for each cell in population
        
        '''
        if MASTER_MODE:
            rotations = []
            for i in range(self.POPULATION_SIZE):
                defaultrot = {}
                for axis in self.rand_rot_axis:
                    defaultrot.update({axis : np.random.rand() * 2 * np.pi})
                rotations.append(defaultrot)
        else:
            rotations = None
        return COMM.bcast(rotations, root=0)


    def calc_min_cell_interdist(self, x, y, z):
        '''
        Calculate cell interdistance from input coordinates
        
        Parameters
        ----------
        x,y,z : np.ndarray
            xyz-coordinates of each cell-body
        
        
        Returns
        ----------           
        np.nparray
            for each cell-body, the distance to nearest neighbor cell
        
        '''
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
        '''
        Draw some random location within radius, z_min, z_max,
        and constrained by min_r and the minimum cell interdistance.
        Returned argument is a list of dicts [{'xpos', 'ypos', 'zpos'}, ]
        
        
        Parameters
        ----------     
        radius : float
            radius of population
        z_min : float
            lower z-boundary of population
        z_mx : float
            upper z-boundary of population
        min_r : np.ndarray
            minimum distance to center axis as function of z
        min_cell_interdist : float,
            minimum cell to cell interdistance
        args : keyword arguments
            simply ignoring additional inputs
        

        Returns
        ----------
        soma_pos: list
            list of dictionaries of length population size
            where dict have keys xpos, ypos, zpos specifying
            xyz-coordinates of cell at list entry i
    
        
        '''
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

        print "assess somatic locations: ",
        while len(u) > 0:
            print len(u),
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

        print 'done!'

        soma_pos = []
        for i in range(self.POPULATION_SIZE):
            soma_pos.append({'xpos' : x[i], 'ypos' : y[i], 'zpos' : z[i]})

        return soma_pos


    def calc_lfp_el_pos(self):
        '''
        Superimpose each cell's contribution to the LFP, and store onto disc
        
        Returns
        ----------
        lfp: np.array
            The populations-specific compound signal
            
        '''
        for i in range(self.POPULATION_SIZE):
            f = h5py.File(os.path.join(self.cells_path,
                                       '%s_lfp_cell%.3i.h5' % (self.y, i)))
            if i == 0:
                lfp = f['LFP'].value
            else:
                lfp += f['LFP'].value
            print '.',
            f.close()

        return lfp


    def calc_csd_el_pos(self):
        '''
        Superimpose each cell's contribution to the CSD, and store onto disc
        
        Returns
        ----------          
        csd: np.array
            The populations-specific compound signal
            
        '''
        for i in range(self.POPULATION_SIZE):
            f = h5py.File(os.path.join(self.cells_path,
                                       '%s_lfp_cell%.3i.h5' % (self.y, i)))
            if i == 0:
                csd = f['CSD'].value
            else:
                csd += f['CSD'].value
            print '.',
            f.close()

        return csd


    def read_lfp_cell_files(self, cellindices=None):
        '''
        Reconstruct base LFPy.Cell-object with simulated data from file storage
        
        Parameters
        ----------               
        cellindicies : np.ndarray
            indices of seletion of cells in the population
            
        Returns
        ----------  
        cells: 
        
        '''
        if cellindices == None:
            cellindices = self.CELLINDICES

        cells = {}
        for cellindex in cellindices:
            cells[cellindex] = self.cellsim(cellindex, return_just_cell=True)

            f = h5py.File(os.path.join(self.cells_path,
                            '%s_lfp_cell%.3i.h5') % (self.y, cellindex))
            print('open file: ' + os.path.join(self.cells_path,
                            '%s_lfp_cell%.3i.h5') % (self.y, cellindex))
            for k in f.iterkeys():
                if k == 'LFP' or k == 'CSD':
                    setattr(cells[cellindex], k, f[k])
                else:
                    setattr(cells[cellindex], k, f[k].value)
            #attach file object
            setattr(cells[cellindex], 'f', f)

        return cells


    def calc_somavs(self):
        '''
        Put all somavs from all cells in a numpy array
        
        Returns
        ----------  
        somavs: np.array
            somatic potentials of all cells in population
        
        '''
        for i in range(self.POPULATION_SIZE):
            f = h5py.File(os.path.join(self.cells_path,
                                       '%s_lfp_cell%.3i.h5' % (self.y, i)))
            if i == 0:
                somavs = f['somav'].value
            else:
                somavs = np.c_[somavs, f['somav'].value]
            print '.',
            f.close()

        return somavs


    def collect_data(self, cellindices=None):
        '''
        collect LFPs, CSDs and somatraces from each simulated population,
        and save to file
        
        '''

        #simplified collection of LFPs from cell
        #objects loaded from file, and the
        #simulation results in terms of the total LFP
        #is saved inside the savefolder.
        if MASTER_MODE and self.POPULATION_SIZE > 0:
            #using cellindices throughout
            if cellindices == None:
                cellindices = np.arange(self.POPULATION_SIZE)


            #calculate lfp from all cell contribs
            lfp = self.calc_lfp_el_pos()
            print 'lfp ok'

            #calculate CSD in every lamina
            if self.calculateCSD:
                csd = self.calc_csd_el_pos()

            #saving
            f = h5py.File(os.path.join(self.populations_path,
                          '%s_population_LFP.h5' % self.y), 'w')
            f['srate'] = 1E3 / self.dt_output
            f.create_dataset('data', data=lfp, compression=4)
            f.close()
            del lfp
            print 'save lfp ok'


            f = h5py.File(os.path.join(self.populations_path,
                          '%s_population_CSD.h5' % self.y), 'w')
            f['srate'] = 1E3 / self.dt_output
            f.create_dataset('data', data=csd, compression=4)
            f.close()
            del csd
            print 'save CSD ok'


            somavs = self.calc_somavs()
            print 'soma potentials ok'

            f = h5py.File(os.path.join(self.populations_path,
                          '%s_population_somatraces.h5' % self.y), 'w')
            f.create_dataset('data', data=somavs, compression=4)
            f['srate'] = 1E3 / self.dt_output
            f.close()
            del somavs
            print 'save somatraces ok'


            #save the somatic placements:
            pop_soma_pos = np.zeros((self.POPULATION_SIZE, 3))
            keys = ['xpos', 'ypos', 'zpos']
            for i in range(self.POPULATION_SIZE):
                for j in range(3):
                    pop_soma_pos[i, j] = self.pop_soma_pos[i][keys[j]]
            np.savetxt(os.path.join(self.populations_path,
                                '%s_population_somapos.gdf' % self.y),
                       pop_soma_pos)
            print 'save somapos ok'


            #save rotations using hdf5
            fname = os.path.join(self.populations_path,
                                    '{}_population_rotations.h5'.format(self.y))
            f = h5py.File(fname, 'w')
            f.create_dataset('x', (len(self.rotations),))
            f.create_dataset('y', (len(self.rotations),))
            f.create_dataset('z', (len(self.rotations),))

            for i, rot in enumerate(self.rotations):
                for key, value in rot.items():
                    f[key][i] = value
            f.close()
            print 'save rotations ok'


        #resync threads
        COMM.Barrier()


class Population(PopulationSuper):
    '''
    Class hybridLFPy.Population, inherited from class PopulationSuper.
    
    This class rely on spiking times of a network simulation, layer-resolved
    input counts, synapse parameters, delay parameters, all per presynaptic
    population.
    
    Parameters
    ----------    
    X : list of strings
        each element denote name of presynaptic populations
    networkSim : hybridLFPy.cachednetworks.Cached*Network object
        container of network spike events resolved per population
    k_yXL : np.array
        num layers x num presynapse populations array specifying the
        number of incoming connections per layer and per population type
    synParams : dict of dicts
        each toplevel key denote each presynaptic population,
        bottom-level dicts are parameters passed to LFPy.Synapse
    synDelayLoc : list,
        Average synapse delay for each presynapse connection
    synDelayScale : list
        Synapse delay std for each presynapse connection
    calculateCSD : bool
        flag for computing the ground-source CSD
            
    '''
    def __init__(self,
                X = ['EX', 'IN'],
                networkSim = 'hybridLFPy.cachednetworks.CachedNetwork',
                k_yXL = [[20,  0], [20, 10]],
                synParams = {
                    'EX': {
                        'section': ['apic', 'dend'],
                        'syntype': 'AlphaISyn',
                        'tau': 0.5},
                    'IN': {
                        'section': ['dend'],
                        'syntype': 'AlphaISyn',
                        'tau': 0.5}},
                synDelayLoc = [1.5, 1.5],
                synDelayScale = [None, None],
                J_yX = [0.20680155243678455, -1.2408093146207075],
                calculateCSD = True,
                **kwargs):
        '''
        Class hybridLFPy.Population, inherited from class PopulationSuper.
        
        This class rely on spiking times of a network simulation, layer-resolved
        input counts, synapse parameters, delay parameters, all per presynaptic
        population.
        
        Parameters
        ----------  
        X : list of strings
            each element denote name of presynaptic populations
        networkSim : hybridLFPy.cachednetworks.Cached*Network object
            container of network spike events resolved per population
        k_yXL : list/np.ndarray
            num layers x num presynapse populations array specifying the
            number of incoming connections per layer and per population type
        synParams : dict of dicts
            each toplevel key denote each presynaptic population,
            bottom-level dicts are parameters passed to LFPy.Synapse
        synDelayLoc : list,
            Average synapse delay for each presynapse connection
        synDelayScale : list
            Synapse delay std for each presynapse connection
        calculateCSD : bool
            flag for computing the ground-source CSD
                
        '''
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


        #CSD calculation flag
        self.calculateCSD = calculateCSD


        #Now loop over all cells in the population and assess
        # - number of synapses in each z-interval (from layerbounds)
        # - placement of synapses in each z-interval
        if MASTER_MODE:
            print 'find synapse locations: ',

        #get in this order, the
        # - postsynaptic compartment indices
        # - presynaptic cell indices
        # - synapse delays per connection
        self.synIdx = self.get_all_synIdx()
        self.SpCells = self.get_all_SpCells()
        self.synDelays = self.get_all_synDelays()


    def get_all_synIdx(self):
        '''
        Auxilliary function to set up class attributes containing
        synapse locations given as LFPy.Cell compartment indices

        This function takes no inputs.
        
        
        Returns
        ----------  
        synIdx: dict
            output[cellindex][populationindex][layerindex] np.ndarray of
            compartment indices
                    
        '''
        tic = time() #timing

        #containers for synapse idxs existing on this rank
        synIdx = {}

        #file names for synapse indices, synIdx*.h5
        fname_all = os.path.join(self.populations_path,
                                      self.y + 'synIdx.h5')

        #if files exist, will just load them without reassessing synapse sites
        if os.path.isfile(fname_all):
            print 'found %s, loading:' % fname_all
            synIdx = helpers.load_dict_of_nested_lists_from_h5(fname_all,
                                                        self.RANK_CELLINDICES)
        #draw synapse locations in parallel
        else:
            #ok then, we will draw random numbers across ranks, which have to
            #be unique per cell. Now, we simply record the random state,
            #change the seed per cell, and put the original state back below.
            randomstate = np.random.get_state()

            for cellindex in self.RANK_CELLINDICES:
                #set the random seed on for each cellindex
                np.random.seed(self.POPULATIONSEED + cellindex)

                #find synapse locations for cell in parallel
                synidx = self.get_synidx(cellindex)
                synIdx[cellindex] = synidx

            #reset the random number generator
            np.random.set_state(randomstate)

            #generated per rank
            fname = os.path.join(self.populations_path,
                                  self.y + 'RANK%.6i' % RANK + 'synIdx.h5')

            #dump to file for loading later
            helpers.dump_dict_of_nested_lists_to_h5(fname, synIdx)

        print 'found synapse locations in %.2f seconds' % (time()-tic)

        #resync
        COMM.Barrier()

        tic = time() #timing

        #create one global file with all synIdx on RANK 0
        if MASTER_MODE:
            if not os.path.isfile(fname_all):
                print 'dumping synIdx*: %s' % fname
                for rank in range(SIZE):
                    fname = os.path.join(self.populations_path,
                                    self.y + 'RANK%.6i' % rank + 'synIdx.h5')

                    #temporary load rank files and dump to one file
                    data = helpers.load_dict_of_nested_lists_from_h5(fname)
                    helpers.dump_dict_of_nested_lists_to_h5(fname_all, data)

                    #deleting temporary files
                    os.remove(fname)

        print 'postprocessed synapse locations in %.2f seconds' % (time()-tic)


        #print the number of synapses per layer from which presynapse population
        if self.verbose:
            for cellindex in self.RANK_CELLINDICES:
                for i, synidx in enumerate(synIdx[cellindex]):
                    print 'to:\t%s\tcell:\t%i\tfrom:\t%s:' % (self.y,
                                                cellindex, self.X[i]),
                    idxcount = 0
                    for idx in synidx:
                        idxcount += idx.size
                        print '\t%i' % idx.size,
                    print '\ttotal %i' % idxcount

        #resync
        COMM.Barrier()

        return synIdx


    def get_all_SpCells(self):
        '''
        For each postsynaptic cell existing on this RANK, load or compute
        the presynaptic cell index for each synaptic connection

        This function takes no kwargs.

        Returns
        ---------- 
        SpCells: dict
            output[cellindex][populationindex][layerindex] np.array of
            presynaptic cell indices
                    

        '''
        #assess the spike time sources from the network classes
        SpCells = {}

        #object containers:
        fname = os.path.join(self.populations_path,
                              self.y + 'SpCells.h5')

        #if files exist, will just load them without reassessing values
        print 'find SpCells: ',
        if not os.path.isfile(fname):
            if MASTER_MODE:
                #assess SpCells for all cells in population (not costly)
                for cellindex in range(self.POPULATION_SIZE):
                    SpCells[cellindex] = []
                    for i, X in enumerate(self.X):
                        SpCells[cellindex].append(self.fetchSpCells(
                            self.networkSim.nodes[X], self.k_yXL[:, i]))

                    print '.',
                print 'done'

                #dump to h5 file so they can be loaded later
                print 'dumping SpCells* to file: %s' % fname

                helpers.dump_dict_of_nested_lists_to_h5(fname, SpCells)

                del SpCells

        #resync
        COMM.Barrier()

        print 'loading: %s' % fname

        #load only data needed on this rank
        SpCells = helpers.load_dict_of_nested_lists_from_h5(fname,
                                                    self.RANK_CELLINDICES)

        #resync
        COMM.Barrier()

        return SpCells



    def get_all_synDelays(self):
        '''
        Create and load arrays of connection delays per connection on this rank
        
        This function takes no kwargs.

        Returns
        -------
        synDelays: dict
            output[cellindex][populationindex][layerindex] np.array of
            delays per connection
        
        '''
        print 'synaptic delays: '
        #assign synaptic delays for every synaptic spike train
        fnameAll = os.path.join(self.populations_path,
                                    self.y + 'synDelays.h5')
        if MASTER_MODE and self.synDelayLoc != None:
            if not os.path.isfile(fnameAll):
                synDelays = self.get_delays()
                print 'dumping synDelays to file: %s' % fnameAll
                helpers.dump_dict_of_nested_lists_to_h5(fnameAll, synDelays)
            else:
                print 'synDelays, loading: %s' % fnameAll

        #resync
        COMM.Barrier()

        #load delays on this rank
        synDelays = helpers.load_dict_of_nested_lists_from_h5(fnameAll,
                                                      self.RANK_CELLINDICES)

        #resync
        COMM.Barrier()

        return synDelays


    def get_synidx(self, cellindex):
        '''
        local function, draw and return synapse locations corresponding
        to a single cell, using a random seed set as
        POPULATIONSEED + cellindex

        
        Parameters
        -------
        cellindex : int,
            index of cell object
        
        Returns
        -------
        synidx: list
        
        '''
        #create a cell instance
        cell = self.cellsim(cellindex, return_just_cell=True)


        #local containers
        synidx = []

        #get synaptic placements and cells from the network,
        #then set spike times,
        for i in range(len(self.X)):
            synidx += [self.fetchSynIdxCell(cell=cell,
                                            nidx=self.k_yXL[:, i],
                                            synParams=self.synParams.copy())]

        return synidx


    def fetchSynIdxCell(self, cell, nidx, synParams):
        '''
        Find possible synaptic placements for each cell
        As synapses are placed within layers with bounds determined by
        self.layerBoundaries, it will check this matrix accordingly, and
        use the probabilities from self.connProbLayer to distribute.

        For each layer, the synapses are placed with probability normalized
        by membrane area of each compartment

        Parameters
        -------    
        cell : LFPy.Cell instance
        nidx : np.ndarray, numbers of synapses per presynaptic population X
        synParams : which synapse parameters to use
        
        Returns
        -------  
        list
            list of arrays of synapse placements per connection
                
        '''
        #segment indices in L1-L6 is stored here, list of np.array
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


    def get_delays(self):
        '''
        get random normally distributed synaptic delays,
        returns nested list of same shape as SpCells.

        Delays are rounded to dt
        
        Returns
        -------  
        delays: dict
            output[cellindex][populationindex][layerindex] np.array of
            delays per connection                    
        
        '''
        delays = {}

        for cellindex in range(self.POPULATION_SIZE):
            delays[cellindex] = []
            for j in range(self.k_yXL.shape[1]):
                delays[cellindex].append([])
                for i in self.k_yXL[:, j]:
                    loc = self.synDelayLoc[j]
                    loc /= self.dt
                    scale = self.synDelayScale[j]
                    if scale != None:
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
                    delays[cellindex][j].append(delay)

        return delays


    def cellsim(self, cellindex, return_just_cell = False):
        '''
        Do the actual simulations of LFP, using synaptic spike times from
        network simulation.
        
        Parameters
        -------  
        cellindex : int
            cell index between 0 and population size-1
        return_just_cell : bool
            If True, return only the LFPy.Cell object
            if False, run full simulation, return None
        
        
        '''
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

            if self.simulationParams.has_key('to_file'):
                if self.simulationParams['to_file']:
                    cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                                  file_name=os.path.join(self.cells_path,
                            '%s_lfp_cell%.3i.h5') % (self.y, cellindex),
                                  **self.simulationParams)
                else:
                    cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                                  **self.simulationParams)
                    cell.LFP = electrode.LFP
                    cell.CSD = cell.dotprodresults[0]
            else:
                cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                              **self.simulationParams)
                cell.LFP = helpers.decimate(electrode.LFP,
                                            q=self.decimatefrac)
                cell.CSD = helpers.decimate(cell.dotprodresults[0],
                                            q=self.decimatefrac)

            #downsample somav
            cell.somav = helpers.decimate(cell.somav, q=self.decimatefrac)

            cell.x = electrode.x
            cell.y = electrode.y
            cell.z = electrode.z

            cell.electrodecoeff = electrode.electrodecoeff

            #access file object
            f = h5py.File(os.path.join(self.cells_path,
                            '%s_lfp_cell%.3i.h5') % (self.y, cellindex),
                          'w', compression='gzip')

            if self.simulationParams.has_key('to_file'):
                if self.simulationParams['to_file']:
                    f.create_dataset('LFP',
                                     helpers.decimate(f['electrode000'].value,
                                                      q=self.decimatefrac),
                                     compression=4)
                    f.create_dataset('CSD',
                                     helpers.decimate(f['electrode001'].value,
                                                      q=self.decimatefrac),
                                     compression=4)
                    del f['electrode000']
                    del f['electrode001']

            #save stuff from savelist
            f['srate'] = 1E3 / self.dt_output
            for attrbt in self.savelist:
                try:
                    del(f[attrbt])
                except:
                    pass
                attr = getattr(cell, attrbt)
                if type(attr) == np.ndarray:
                    f.create_dataset(attrbt, data=attr.astype('float32'),
                                     compression=4)
                else:
                    try:
                        f[attrbt] = attr
                    except:
                        f[attrbt] = str(attr)

            #print some stuff
            print 'SIZE %i, RANK %i, Cell %i, Min LFP: %.3f, Max LFP: %.3f' % \
                        (SIZE, RANK, cellindex,
                        f['LFP'].value.min(), f['LFP'].value.max())

            f.close()

            print 'Cell %s saved to file' % cellindex


    def insert_all_synapses(self, cellindex, cell):
        '''
        Insert all synaptic events from all presynaptic layers on
        cell object with index cellindex
        
        Parameters
        -------  
        cellindex : int
            cell index in the population
        cell : LFPy.Cell instance
            postsynaptic target cell
        
        
        '''
        for X in range(self.k_yXL.shape[1]):
            synParams = self.synParams
            synParams.update({
                'weight' : self.J_yX[X]
                })
            for j in range(len(self.synIdx[cellindex][X])):
                if self.synDelays != None:
                    synDelays = self.synDelays[cellindex][X][j]
                else:
                    synDelays = None
                self.insert_synapses(cell = cell,
                                cellindex = cellindex,
                                synParams = synParams,
                                idx = self.synIdx[cellindex][X][j],
                                SpCell = self.SpCells[cellindex][X][j],
                                SpTimes = os.path.join(self.savefolder,
                                               self.networkSim.dbname),
                                synDelays = synDelays)


    def insert_synapses(self, cell, cellindex, synParams, idx = np.array([]),
                        SpCell = np.array([]), SpTimes=':memory:',
                        synDelays = None):
        '''
        Insert synapse with parameters=synparams on cell=cell, with
        segment indexes given by idx. SpCell and SpTimes picked from Brunel
        network simulation
        
        Parameters
        -------  
        cell : LFPy.Cell instance
            postsynaptic target cell
        cellindex : int
            index of cell in population
        synParams : dict
            parameters passed to LFPy.Synapse
        idx : np.ndarray
            postsynaptic compartment indices
        SpCell : np.ndarray
            presynaptic spiking cells
        SpTimes : str
            ':memory:' or path to on-disk spike time database
        synDelays : np.ndarray
            Per connection specific delays
        
        '''
        #Insert synapses in an iterative fashion
        if hasattr(self.networkSim, 'db'):
            spikes = self.networkSim.db.select(SpCell[:idx.size])
        else:
            db = GDF(SpTimes, new_db=False)
            spikes = db.select(SpCell[:idx.size])
            db.close()

        #apply synaptic delays
        if synDelays != None and idx.size > 0:
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
        '''
        For N (nodes count) nestSim-cells draw
        POPULATION_SIZE x NTIMES random cell indexes in
        the population in nodes and broadcast these as SpCell

        The returned argument is a list with len = numSyn.size of np.arrays,
        assumes numSyn is a list

        Parameters
        -------  
        nodes : np.ndarray, dtype=int,
            node # of valid presynaptic neurons
        numSyn : np.ndarray, dtype=int,
            # of synapses per connection

        '''
        if MASTER_MODE:
            SpCell = []
            for size in numSyn:
                SpCell.append(np.random.randint(nodes.min(), nodes.max(),
                                                size=size).astype('int32'))
            return SpCell
        else:
            return

