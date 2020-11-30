#!/usr/bin/env python
'''Cythonized helper functions'''

from time import time
import numpy as np
from LFPy.alias_method import alias_method
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t   LTYPE_t

cdef extern from "math.h":
    int floor(DTYPE_t x)
    DTYPE_t sqrt(DTYPE_t x)
    DTYPE_t exp(DTYPE_t x)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list _getSpCell(np.ndarray[LTYPE_t, ndim=1, negative_indices=False] nodes,
               np.ndarray[LTYPE_t, ndim=1, negative_indices=False] numSyn,
               np.ndarray[DTYPE_t, ndim=1, negative_indices=False] Pr,
               ):
    '''
    fetchSpCells() helper function.

    Arguments
    ---------
    nodes : np.ndarray
        integer array of possible presynaptic network node indices
    numSyn : np.ndarray
        integer array of number of synapses
    Pr : np.ndarray
        discrete probability distribution of float values

    Returns
    -------
    SpCell : list
        list presynaptic neuron indices

    '''
    #C-declare local variables
    cdef np.ndarray[LTYPE_t, ndim=1, negative_indices=False] spc
    cdef int n
    cdef list SpCell

    #fill in container with distance-dependent connections
    SpCell = []
    for n in range(numSyn.size):
        #numSyn[n]-size array with units picked according to distance-dependent
        #probability
        spc = alias_method(nodes, Pr, numSyn[n])
        SpCell.append(spc)

    return SpCell


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] _calc_radial_dist_to_cell(DTYPE_t x,
                              DTYPE_t y,
                              np.ndarray[DTYPE_t, ndim=2, negative_indices=False] Xpos,
                              DTYPE_t xextent,
                              DTYPE_t yextent,
                              int edge_wrap):
    '''
    compute radial distance laterally to all neurons in presynaptic
    population X with and without periodic boundary conditions, as seen from
    a single postsynaptic cell in this population


    Arguments
    ---------
    x : float
        cell position along x-axis
    y : float
        cell position along y-axis
    Xpos : np.ndarray
        ndim 2, cell positions in presynaptic population X
    xextent : float
        side length along x-axis
    yextent : float
        side length along y-axis
    edge_wrap : int in [0, 1]
        whether or not we are running with periodic boundaries


    Returns
    -------
    np.ndarray
        lateral distance to all presynaptic cells in population X

    '''
    #c-declare variables
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] xdist, ydist, r
    cdef int i

    r = np.zeros(Xpos.shape[0])

    #account for periodic boundary conditions when computing
    #distance between sources and target
    if edge_wrap == 1:
        xdist = np.abs(Xpos[:, 0] - x)
        ydist = np.abs(Xpos[:, 1] - y)
        for i in range(xdist.size):
            if xdist[i] > xextent - xdist[i]:
                xdist[i] = xextent - xdist[i]
            if ydist[i] > yextent - ydist[i]:
                ydist[i] = yextent - ydist[i]

            r[i] = sqrt(xdist[i]*xdist[i] + ydist[i]*ydist[i])
    else:
        #radial distance without periodic boundaries
        for i in range(r.size):
            r[i] = sqrt((Xpos[i, 0]-x)**2 + (Xpos[i, 1]-y)**2)
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list _fetchSpCells(int cellindex,
                 double x,
                 double y,
                 np.ndarray[DTYPE_t, ndim=2, negative_indices=False] positions,
                 dict topology_connections,
                 np.ndarray[LTYPE_t, ndim=1, negative_indices=False] nodes,
                 np.ndarray[LTYPE_t, ndim=1, negative_indices=False] numSyn):
    """
    For N (nodes count) nestSim-cells draw
    POPULATION_SIZE x NTIMES random cell indexes in
    the population in nodes and broadcast these as `SpCell`.

    The returned argument is a list with len = numSyn.size of np.arrays,
    assumes `numSyn` is a list

    Parameters
    ----------
    cellindex : int
        index of postsynaptic cells
    X : str,
        name of presynaptic population
    nodes : np.ndarray, dtype=int
        Node # of valid presynaptic neurons.
    numSyn : np.ndarray, dtype=int
        # of synapses per connection.


    Returns
    -------
    list
        list of arrays of presynaptic network-neuron indices


    See also
    --------
    Population.fetch_all_SpCells
    """

    #c-declare some variables
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] r, Pr
    cdef DTYPE_t a, tau, p_center, mean, sigma, c, Prsum, mask_radius, ri
    cdef int n, i
    cdef list SpCell

    #connection probability as function of radius
    kernel = topology_connections['kernel']
    mask = topology_connections['mask']
    if list(kernel.keys())[0] == 'exponential':
        #distance to other cells
        r = _calc_radial_dist_to_cell(x,
                                      y,
                                      positions,
                                      topology_connections['extent'][0],
                                      topology_connections['extent'][1],
                                      topology_connections['edge_wrap'])

        #set up connection probability with distance
        a = kernel['exponential']['a']
        tau = kernel['exponential']['tau']
        c = kernel['exponential']['c']
        mask_radius = mask['circular']['radius']

        #compute distance dependent connectivity, apply mask, check for autapses
        Pr = a * np.exp(-r/tau) + c
        Pr[np.where(r > mask_radius)] = 0.
        if not topology_connections['allow_autapses']:
            Pr[np.where(r == 0)] = 0.


        Prsum = Pr.sum()
        if Prsum > 0: #at least a single presynaptic cell must exists
            #normalize connection probabilities (sum of probs is 1.)
            Pr /= Prsum

            ##container for distance-dependent connections
            SpCell = _getSpCell(nodes, numSyn, Pr)
        else: #zero presynaptic neurons exist
            SpCell = [[] for n in numSyn]

    elif list(kernel.keys())[0] == 'gaussian':
        #distance to other cells
        r = _calc_radial_dist_to_cell(x,
                                      y,
                                      positions,
                                      topology_connections['extent'][0],
                                      topology_connections['extent'][1],
                                      topology_connections['edge_wrap'])

        #set up connection probability with distance
        p_center = kernel['gaussian']['p_center']
        sigma = kernel['gaussian']['sigma']
        mean = kernel['gaussian']['mean']
        c = kernel['gaussian']['c']
        mask_radius = mask['circular']['radius']

        #compute distance dependent connectivity, apply mask, check for autapses
        Pr = p_center * np.exp(-((r-mean)*(r-mean))/(2*sigma*sigma)) + c
        Pr[np.where(r > mask_radius)] = 0.
        if not topology_connections['allow_autapses']:
            Pr[np.where(r == 0)] = 0.


        Prsum = Pr.sum()
        if Prsum > 0: #at least a single presynaptic cell must exists
            #normalize connection probabilities (sum of probs is 1.)
            Pr /= Prsum

            ##container for distance-dependent connections
            SpCell = _getSpCell(nodes, numSyn, Pr)
        else: #zero presynaptic neurons exist
            SpCell = [[] for n in numSyn]

    elif list(kernel.keys())[0] == 'random':
        #container for random connections
        SpCell = []
        for n in numSyn:
            SpCell.append(np.random.randint(nodes.min(), nodes.max(),
                                            size=n).astype('int32'))
    else:
        raise NotImplementedError('only random, exponential and gaussian rule supported')

    return SpCell


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict _get_all_SpCells(np.ndarray[LTYPE_t, ndim=1, negative_indices=False] RANK_CELLINDICES,
                     list X,
                     str y,
                     list pop_soma_pos,
                     dict positions,
                     dict topology_connections,
                     dict nodes,
                     np.ndarray[LTYPE_t, ndim=2, negative_indices=False] k_yXL):

    #c-declare variables
    cdef dict SpCells
    cdef int i, cellindex

    #container
    SpCells = {}

    for cellindex in RANK_CELLINDICES:
        ##set the random seed on for each cellindex
        #np.random.seed(self.POPULATIONSEED + cellindex + self.POPULATION_SIZE)

        SpCells[cellindex] = {}
        for i in range(len(X)):
            SpCells[cellindex][X[i]] = _fetchSpCells(
                        cellindex=cellindex,
                        x=pop_soma_pos[cellindex]['x'],
                        y=pop_soma_pos[cellindex]['y'],
                        positions=positions[X[i]],
                        topology_connections=topology_connections[X[i]][y],
                        nodes=nodes[X[i]],
                        numSyn=k_yXL[:, i])
    return SpCells
