#!/usr/env/bin python
'''
# brunel_alpha_nest_topo_exp.py
#
# This file is not part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

# This version uses NEST's Connect functions.

# Modified as an example with the hybrid model scheme for generating
# local field potentials with point-neuron networks
#
# Further modified to set up populations as layers and connections between
# neurons using exponential connectivity within a
# mask and distance-dependent delays.
'''

import os
import nest
import nest.raster_plot

from time import time
import numpy as np
from mpi4py import MPI


def ComputePSPnorm(tauMem, CMem, tauSyn):
    """Compute the maximum of postsynaptic potential
       for a synaptic input current of unit amplitude
       (1 pA)"""

    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)

    # time of maximum
    t_max = 1.0 / b * (-nest.ll_api.sli_func('LambertWm1', -
                                             np.exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude
    return np.exp(1.0) / (tauSyn * CMem * b) * ((np.exp(-t_max / tauMem) -
                                                 np.exp(-t_max / tauSyn)
                                                 ) / b -
                                                t_max * np.exp(-t_max / tauSyn)
                                                )


# Initialization of MPI stuff
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# tic
startbuild = time()

dt = 0.1    # the resolution in ms
simtime = 300.0  # Simulation time in ms
delay = 1.5    # minimum synaptic delay in ms
# NOTE: delays set explicitly in conn_dict_* below with distance dependency

# Parameters for asynchronous irregular firing
g = 6.0
eta = 2.0
epsilon = 0.1  # connection probability

order = 100
NE = 4 * order
NI = 1 * order

# flag for using Poisson or equivalent DC current external drive
Poisson = False

CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron

# Initialize the parameters of the integrate and fire neuron
tauSyn = 0.5
tauMem = 20.0
CMem = 250.0
theta = 20.0
J = 1.0  # 0.1 # postsynaptic amplitude in mV

# normalize synaptic current so that amplitude of a PSP is J
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex = J / J_unit
J_in = -g * J_ex

# threshold rate, equivalent rate of events needed to
# have mean input current equal to threshold
nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * CE

if Poisson:
    # use zero dc input amplitude to the neurons
    dc_amplitude = 0.
else:
    # compute equivalent DC current amplitude instead of Poisson input assuming
    # alpha current synapse
    factor = 0.4  # CHECK
    dc_amplitude = p_rate * J_ex / 1000 / tauSyn * factor

# parameters for "iaf_psc_alpha" cell model
neuron_params = {"C_m": CMem,
                 "tau_m": tauMem,
                 "tau_syn_ex": tauSyn,
                 "tau_syn_in": tauSyn,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta,
                 "I_e": dc_amplitude,
                 }


# layer extent in mum (extent_length x extent_length)
extent_length = 4000.
# mask radius in mum
mask_radius = 2000.
# exponential profile, tau
tau = 300.

area_extent = extent_length**2
area_mask = mask_radius**2 * np.pi


# excitatory connections
conn_kernel_EX = {'exponential': {
    'a': 1., 'c': 0.0, 'tau': 150.}}  # 'a' and 'c' used for LFP predictions
conn_dict_EX = {
    'rule': 'fixed_indegree',
    'indegree': CE,
    'mask': {'circular': {'radius': 2000.0}},
    'allow_autapses': False,
    'allow_multapses': True,
}
conn_delay_EX = {'linear': {'c': delay, 'a': 0.0001}}
syn_dict_EX = {
    'synapse_model': 'excitatory',
    'weight': J_ex,
}

# inhibitory connections
conn_kernel_IN = {'exponential': {
    'a': 1., 'c': 0.0, 'tau': 300.}}
conn_dict_IN = {
    'rule': 'fixed_indegree',
    'indegree': CI,
    'mask': {'circular': {'radius': 2000.0}},
    'allow_autapses': False,
    'allow_multapses': True,
}
conn_delay_IN = {'linear': {'c': delay, 'a': 0.0001}}
syn_dict_IN = {
    'synapse_model': 'inhibitory',
    'weight': J_in,
}


# layer-specific parameters for the excitatory and inhibitory populations
# including layer extent, cell positions, neuron model, periodic boundary flag
layerdict_EX = {
    'extent': [extent_length, extent_length],
    'pos': [[(np.random.rand() - 0.5) * extent_length,
             (np.random.rand() - 0.5) * extent_length] for x in range(NE)
            ],
    'edge_wrap': True,
}
layerdict_IN = {
    'extent': [extent_length, extent_length],
    'pos': [[(np.random.rand() - 0.5) * extent_length,
             (np.random.rand() - 0.5) * extent_length] for x in range(NI)
            ],
    'edge_wrap': True,
}

# integrate and fire with alpha-shaped synapse currents
neuron_model = 'iaf_psc_alpha'

# shared output folder
output_folder = 'simulation_output_example_brunel_topo_exp'

# destination of file output
spike_output_path = os.path.join(output_folder, 'spiking_output_path')

# file prefix for spike detectors
label = 'spikes'

# file prefix for neuron positions
label_positions = 'positions'


def run_model():
    '''
    separate model execution from parameters for safer import from other
    files
    '''
    # create file output destination folders if they do not exist
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        if not os.path.isdir(spike_output_path):
            os.mkdir(spike_output_path)

    nest.ResetKernel()

    nest.SetKernelStatus({"resolution": dt,
                          "print_time": True,
                          "overwrite_files": True, })

    print("Building network")

    nest.SetDefaults(neuron_model, neuron_params)

    # create excitatory and inhibitory layers
    nodes_ex = nest.Create(model=neuron_model,
                           positions=nest.spatial.free(**layerdict_EX))
    nodes_in = nest.Create(model=neuron_model,
                           positions=nest.spatial.free(**layerdict_IN))

    # write GID and positions of neurons in layers
    nest.DumpLayerNodes(nodes_ex, os.path.join(spike_output_path,
                                               label_positions + '-EX.dat'))
    nest.DumpLayerNodes(nodes_in, os.path.join(spike_output_path,
                                               label_positions + '-IN.dat'))

    # distribute membrane potentials between V=0 and threshold V_th
    nest.SetStatus(nodes_ex, "V_m",
                   np.random.rand(NE) * neuron_params["V_th"])
    nest.SetStatus(nodes_in, "V_m",
                   np.random.rand(NI) * neuron_params["V_th"])

    # create Poisson generators
    if Poisson:
        nest.SetDefaults("poisson_generator", {"rate": p_rate})
        noise = nest.Create("poisson_generator")

    # create spiker recorders
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    nest.SetStatus(espikes, [{
        "label": os.path.join(spike_output_path, label + "-EX"),
        "record_to": 'ascii'
    }])

    nest.SetStatus(ispikes, [{
        "label": os.path.join(spike_output_path, label + "-IN"),
        "record_to": 'ascii'
    }])

    print("Connecting devices")

    # copy static_synapse synapse model as excitatory and inhibitory,
    # respectively
    nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex})
    nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in})

    if Poisson:
        nest.Connect(noise, nodes_ex, 'all_to_all', "excitatory")
        nest.Connect(noise, nodes_in, 'all_to_all', "excitatory")

    nest.Connect(nodes_ex, espikes, 'all_to_all', "excitatory")
    nest.Connect(nodes_in, ispikes, 'all_to_all', "excitatory")

    print("Connecting network")

    # We now connect the layers to each other using distance dependent
    # connectivity, in the order of E-E, E-I, I-E and I-I, with fixed indegrees

    print("Excitatory connections")
    if 'exponential' in conn_kernel_EX.keys():
        beta = conn_kernel_EX['exponential']['tau']
        p_dict_EX = {'p': nest.spatial_distributions.exponential(
            nest.spatial.distance, beta=beta)}
    else:
        raise NotImplementedError

    if 'linear' in conn_delay_EX.keys():
        delay_param_EX = \
            conn_delay_EX['linear']['c'] + \
            nest.spatial.distance * conn_delay_EX['linear']['a']
    else:
        raise NotImplementedError

    nest.Connect(nodes_ex, nodes_ex,
                 {**conn_dict_EX, **p_dict_EX},
                 {**syn_dict_EX, **{'delay': delay_param_EX}})
    nest.Connect(nodes_ex, nodes_in,
                 {**conn_dict_EX, **p_dict_EX},
                 {**syn_dict_EX, **{'delay': delay_param_EX}})

    print("Inhibitory connections")
    if 'exponential' in conn_kernel_IN.keys():
        beta = conn_kernel_IN['exponential']['tau']
        p_dict_IN = {'p': nest.spatial_distributions.exponential(
            nest.spatial.distance, beta=beta)}
    else:
        raise NotImplementedError
    if 'linear' in conn_delay_IN.keys():
        delay_param_IN = \
            conn_delay_IN['linear']['c'] + \
            nest.spatial.distance * conn_delay_IN['linear']['a']
    else:
        raise NotImplementedError
    nest.Connect(nodes_in, nodes_ex,
                 {**conn_dict_IN, **p_dict_IN},
                 {**syn_dict_IN, **{'delay': delay_param_IN}})
    nest.Connect(nodes_in, nodes_in,
                 {**conn_dict_IN, **p_dict_IN},
                 {**syn_dict_IN, **{'delay': delay_param_IN}})

    endbuild = time()

    print("Simulating")

    nest.Simulate(simtime)

    # tic-toc
    endsimulate = time()

    # compute and print some statistics of spikes recorded on this RANK
    events_ex = nest.GetStatus(espikes, "n_events")[0]
    rate_ex = events_ex / simtime * 1000.0 / \
        len(nest.GetLocalNodeCollection(nodes_ex).tolist())
    events_in = nest.GetStatus(ispikes, "n_events")[0]
    rate_in = events_in / simtime * 1000.0 / \
        len(nest.GetLocalNodeCollection(nodes_in).tolist())

    num_synapses = nest.GetDefaults("excitatory")["num_connections"] +\
        nest.GetDefaults("inhibitory")["num_connections"]

    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild

    print("Brunel network simulation (Python)")
    print(("Number of neurons : {0}".format(NE + NI)))
    print(("Number of synapses: {0}".format(num_synapses)))
    print(("       Exitatory  : {0}".format(int(CE * NE + NI) + NE + NI)))
    print(("       Inhibitory : {0}".format(int(CI * NE + NI))))
    print(("Excitatory rate   : %.2f Hz" % rate_ex))
    print(("Inhibitory rate   : %.2f Hz" % rate_in))
    print(("Building time     : %.2f s" % build_time))
    print(("Simulation time   : %.2f s" % sim_time))

    # don't wanna look at this right now
    if False:
        nest.raster_plot.from_device(espikes, hist=True)
        nest.raster_plot.from_device(ispikes, hist=True)

    # sorted raster plot:
    if True:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, figsize=(12, 8))

        eevents = np.loadtxt(nest.GetStatus(espikes, 'filenames')[0][0],
                             skiprows=3,
                             dtype=[('senders', np.int), ('times', np.float)])
        ievents = np.loadtxt(nest.GetStatus(ispikes, 'filenames')[0][0],
                             skiprows=3,
                             dtype=[('senders', np.int), ('times', np.float)])
        X = []
        T = []
        for i, j in enumerate(nodes_ex):
            # extract spikes
            t = eevents['times'][eevents['senders'] == j]
            x, y = layerdict_EX['pos'][i]
            if t.size > 0:
                T = np.r_[T, t]
                X = np.r_[X, np.zeros_like(t) + x]
        ax.plot(T, X, 'r.', label='E')

        X = []
        T = []
        for i, j in enumerate(nodes_in):
            # extract spikes
            t = ievents['times'][ievents['senders'] == j]
            x, y = layerdict_IN['pos'][i]
            if t.size > 0:
                T = np.r_[T, t]
                X = np.r_[X, np.zeros_like(t) + x]
        ax.plot(T, X, 'b.', label='I')

        ax.axis()
        ax.legend(loc=1)
        ax.set_title('sorted spike raster')
        ax.set_xlabel('t (ms)')
        ax.set_ylabel(r'x position ($\mu$m)')

        fig.savefig(os.path.join(spike_output_path, 'sorted_raster.pdf'))


if __name__ == '__main__':
    run_model()
