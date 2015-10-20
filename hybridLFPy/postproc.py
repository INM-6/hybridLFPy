#!/usr/bin/env python
import numpy as np
import h5py
import os
import glob
import tarfile
from warnings import warn
from mpi4py import MPI

################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class PostProcess(object):
    """
    class `PostProcess`: Methods to deal with the contributions of every
    postsynaptic sub-population.


    Parameters
    ----------
    y : list
        Postsynaptic cell-type or population-names.
    dt_output : float
        Time resolution of output data.
    savefolder : str
        Path to main output folder.
    mapping_Yy : list
        List of tuples, each tuple pairing population with cell type, e.g., [('L4E', 'p4'), ('L4E', 'ss4')].
    cells_subfolder : str
        Folder under `savefolder` containing cell output.
    populations_subfolder : str
        Folder under `savefolder` containing population specific output.
    figures_subfolder : str
        Folder under `savefolder` containing figs.

    """
    def __init__(self,
                 y=['EX', 'IN'],
                 dt_output=1.,
                 mapping_Yy=[('EX', 'EX'),
                               ('IN', 'IN')],
                 savelist=['somav', 'somapos', 'x', 'y', 'z', 'LFP', 'CSD'],
                 savefolder='simulation_output_example_brunel',
                 cells_subfolder='cells',
                 populations_subfolder='populations',
                 figures_subfolder='figures',
                 output_file='{}_population_{}',
                 compound_file='{}sum.h5',
                 ):
        """
        class `PostProcess`: Methods to deal with the contributions of every
        postsynaptic sub-population.

    
        Parameters
        ----------
        y : list
            List of postsynaptic cell-type or population-names.
        dt_output : float
            Time resolution of output data.
        mapping_Yy : list
            List of tuples, each tuple pairing population with cell type,
            e.g., [('L4E', 'p4'), ('L4E', 'ss4')].
        savelist : list
            List of strings, each corresponding to LFPy.Cell attributes
        savefolder : str
            Path to main output folder.
        cells_subfolder : str
            Folder under `savefolder` containing cell output.
        populations_subfolder : str
            Folder under `savefolder` containing population specific output.
        figures_subfolder : str
            Folder under `savefolder` containing figs.
        output_file : str
            formattable file name for population signals, e.g.,
            '{}_population_{}.h5'
        compound_file : str
            formattable file name for population signals, e.g.,
            '{}sum.h5'
        
    
        """
        #set some attributes
        self.y = y
        self.dt_output = dt_output
        self.mapping_Yy = mapping_Yy
        self.savelist = savelist
        self.savefolder = savefolder
        self.cells_path = os.path.join(savefolder, cells_subfolder)
        self.populations_path = os.path.join(savefolder, populations_subfolder)
        self.figures_path = os.path.join(savefolder, figures_subfolder)
        self.output_file = output_file
        self.compound_file = compound_file
        
        #set up subfolders
        if RANK == 0:
            self._set_up_savefolder()
        else:
            pass


    def run(self):
        """ Perform the postprocessing steps, computing compound signals from
        cell-specific output files.
        """
        if RANK == 0:
            if 'LFP' in self.savelist:
                #get the per population LFPs and total LFP from all populations:
                self.LFPdict, self.LFPsum = self.calc_lfp()
                self.LFPdictLayer = self.calc_lfp_layer()
    
                #save global LFP sum, and from L23E, L4I etc.:
                f = h5py.File(os.path.join(self.savefolder,
                                           self.compound_file.format('LFP')
                                           ), 'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=self.LFPsum, compression=4)
                f.close()
    
                for key, value in list(self.LFPdictLayer.items()):
                    f = h5py.File(os.path.join(self.populations_path,
                                               self.output_file.format(key,
                                                                       'LFP.h5')
                                               ), 'w')
                    f['srate'] = 1E3 / self.dt_output
                    f.create_dataset('data', data=value, compression=4)
                    f.close()

            if 'CSD' in self.savelist:
                #get the per population CSDs and total CSD from all populations:
                self.CSDdict, self.CSDsum = self.calc_csd()
                self.CSDdictLayer = self.calc_csd_layer()
    
                #save global CSD sum, and from L23E, L4I etc.:
                f = h5py.File(os.path.join(self.savefolder,
                                           self.compound_file.format('CSD')),
                              'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=self.CSDsum, compression=4)
                f.close()
    
                for key, value in list(self.CSDdictLayer.items()):
                    f = h5py.File(os.path.join(self.populations_path,
                                               self.output_file.format(key,
                                                                       'CSD.h5')
                                               ), 'w')
                    f['srate'] = 1E3 / self.dt_output
                    f.create_dataset('data', data=value, compression=4)
                    f.close()

        else:
            pass

        ##collect matrices with the single cell contributions in parallel
        #self.collectSingleContribs()


    def _set_up_savefolder(self):
        """ Create catalogs for different file output to clean up savefolder.
        """
        if not os.path.isdir(self.cells_path):
            os.mkdir(self.cells_path)

        if not os.path.isdir(self.figures_path):
            os.mkdir(self.figures_path)

        if not os.path.isdir(self.populations_path):
            os.mkdir(self.populations_path)


    def calc_lfp(self):
        """ Sum all the LFP contributions from every cell type.
        """

        LFParray = np.array([])
        LFPdict = {}

        i = 0
        for y in self.y:
            fil = os.path.join(self.populations_path,
                               self.output_file.format(y, 'LFP.h5'))

            f = h5py.File(fil)

            if i == 0:
                LFParray = np.zeros((len(self.y),
                                    f['data'].shape[0], f['data'].shape[1]))

            #fill in
            LFParray[i, ] = f['data'].value

            LFPdict.update({y : f['data'].value})

            f.close()

            i += 1

        return LFPdict,  LFParray.sum(axis=0)


    def calc_csd(self):
        """ Sum all the CSD contributions from every layer.
        """

        CSDarray = np.array([])
        CSDdict = {}

        i = 0
        for y in self.y:
            fil = os.path.join(self.populations_path,
                               self.output_file.format(y, 'CSD.h5'))

            f = h5py.File(fil)

            if i == 0:
                CSDarray = np.zeros((len(self.y),
                                    f['data'].shape[0], f['data'].shape[1]))

            #fill in
            CSDarray[i, ] = f['data'].value

            CSDdict.update({y : f['data'].value})

            f.close()

            i += 1

        return CSDdict,  CSDarray.sum(axis=0)


    def calc_lfp_layer(self):
        """
        Calculate the LFP from concatenated subpopulations residing in a
        certain layer, e.g all L4E pops are summed, according to the `mapping_Yy`
        attribute of the `hybridLFPy.Population` objects.
        """
        LFPdict = {}

        lastY = None
        for Y, y in self.mapping_Yy:
            if lastY != Y:
                try:
                    LFPdict.update({Y : self.LFPdict[y]})
                except KeyError:
                    pass
            else:
                try:
                    LFPdict[Y] += self.LFPdict[y]
                except KeyError:
                    pass
            lastY = Y

        return LFPdict


    def calc_csd_layer(self):
        """
        Calculate the CSD from concatenated subpopulations residing in a
        certain layer, e.g all L4E pops are summed, according to the `mapping_Yy`
        attribute of the `hybridLFPy.Population` objects.
        """
        CSDdict = {}

        lastY = None
        for Y, y in self.mapping_Yy:
            if lastY != Y:
                try:
                    CSDdict.update({Y : self.CSDdict[y]})
                except KeyError:
                    pass
            else:
                try:
                    CSDdict[Y] += self.CSDdict[y]
                except KeyError:
                    pass
            lastY = Y

        return CSDdict


    def create_tar_archive(self):
        """ Create a tar archive of the main simulation outputs.
        """
        #file filter
        EXCLUDE_FILES = glob.glob(os.path.join(self.savefolder, 'cells'))
        EXCLUDE_FILES += glob.glob(os.path.join(self.savefolder,
                                                'populations', 'subsamples'))
        EXCLUDE_FILES += glob.glob(os.path.join(self.savefolder,
                                                'raw_nest_output'))

        def filter_function(tarinfo):
            print(tarinfo.name)
            if len([f for f in EXCLUDE_FILES if os.path.split(tarinfo.name)[-1]
                    in os.path.split(f)[-1]]) > 0 or \
               len([f for f in EXCLUDE_FILES if os.path.split(tarinfo.path)[-1]
                    in os.path.split(f)[-1]]) > 0:
                print('excluding %s' % tarinfo.name)

                return None
            else:
                return tarinfo

        if RANK == 0:
            print('creating archive %s' % (self.savefolder + '.tar'))
            #open file
            f = tarfile.open(self.savefolder + '.tar', 'w')
            #avoid adding files to repo as /scratch/$USER/hybrid_model/...
            arcname = os.path.split(self.savefolder)[-1]

            f.add(name=self.savefolder,
                  arcname=arcname,
                  filter=filter_function)
            f.close()

        #resync
        COMM.Barrier()

