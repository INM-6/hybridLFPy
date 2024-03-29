#!/usr/env/bin python
# -*- coding: utf-8 -*-
#
# brunel_alpha_nest.py
#
# This file is part of NEST.
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
#
# Modified as an example with hybridLFPy (https:/github.com/INM-6/hybridLFPy)
# for calculating local field potentials from spiking point-neuron networks

"""
Random balanced network (alpha synapses) connected with NEST
------------------------------------------------------------

This script simulates an excitatory and an inhibitory population on
the basis of the network used in

Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
183-208 (2000).

In contrast to brunel-alpha-numpy.py, this variant uses NEST's builtin
connection routines to draw the random connections instead of NumPy.

When connecting the network customary synapse models are used, which
allow for querying the number of created synapses. Using spike
detectors the average firing rates of the neurons in the populations
are established. The building as well as the simulation time of the
network are recorded.
"""

"""
Importing all necessary modules for simulation, analysis and plotting.
"""


'''
Definition of functions used in this example. First, define the
Lambert W function implemented in SLI. The second function
computes the maximum of the postsynaptic potential for a synaptic
input current of unit amplitude (1 pA) using the Lambert W
function. Thus function will later be used to calibrate the synaptic
weights.
'''

import nest
import nest.raster_plot
import os
import time
from numpy import exp, random


def LambertWm1(x):
    nest.ll_api.sli_push(x)
    nest.ll_api.sli_run('LambertWm1')
    y = nest.ll_api.sli_pop()
    return y


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)

    # time of maximum
    t_max = 1.0 / b * (-LambertWm1(-exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude
    return exp(1.0) / (tauSyn * CMem * b) * ((exp(-t_max / tauMem) - \
               exp(-t_max / tauSyn)) / b - t_max * exp(-t_max / tauSyn))


'''
Assigning the current time to a variable in order to determine the
build time of the network.
'''

startbuild = time.time()

'''
Assigning the simulation parameters to variables.
'''

dt = 0.1    # the resolution in ms
simtime = 1000.0  # Simulation time in ms
delay = 1.5    # synaptic delay in ms

'''
Definition of the parameters crucial for asynchronous irregular firing
of the neurons.
'''

g = 6.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

'''
Definition of the number of neurons in the network and the number of
neuron recorded from.

Note: We record here all spike events.
'''

order = 100
NE = 4 * order  # number of excitatory neurons
NI = 1 * order  # number of inhibitory neurons
N_neurons = NE + NI   # number of neurons in total

'''
flag for using Poisson or equivalent DC current external drive
'''

Poisson = False

'''
Definition of connectivity parameter
'''

CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE)      # total number of synapses per neuron

'''
Initialization of the parameters of the integrate and fire neuron and
the synapses. The parameter of the neuron are stored in a dictionary.
The synaptic currents are normalized such that the amplitude of the
PSP is J.
'''

tauSyn = 0.5        # synaptic time constant in ms
tauMem = 20.0       # time constant of membrane potential in ms
CMem = 250.0        # capacitance of membrane in in pF
theta = 20.0       # membrane threshold potential in mV
J = 1.0        # postsynaptic amplitude in mV
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
J_in = -g * J_ex    # amplitude of inhibitory postsynaptic current

'''
Definition of threshold rate, which is the external rate needed to fix
the membrane potential around its threshold, the external firing rate
and the rate of the poisson generator which is multiplied by the
in-degree CE and converted to Hz by multiplication by 1000.
'''

nu_th = (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * CE

'''
Adjust parameters accordingly if Poisson background input is used
'''

if Poisson:
    # use zero dc input amplitude to the neurons
    dc_amplitude = 0.
else:
    # compute equivalent DC current amplitude instead of Poisson input assuming
    # alpha current synapse
    factor = 0.4  # CHECK
    dc_amplitude = p_rate * J_ex / 1000 / tauSyn * factor

'''
Parameters for "iaf_psc_alpha" cell model
'''

neuron_params = {"C_m": CMem,
                 "tau_m": tauMem,
                 "tau_syn_ex": tauSyn,
                 "tau_syn_in": tauSyn,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_th": 20.0,
                 "V_m": 0.0,
                 "V_th": theta,
                 "I_e": dc_amplitude, }
'''
Destination of file output
'''

spike_output_path = os.path.join('simulation_output_example_brunel',
                                 'spiking_output_path')

'''
File prefix for spike detectors
'''

label = 'brunel-py'

'''
Main simulation procedure function
'''


def simulate():
    '''instantiate and execute network simulation'''
    # separate model execution from parameters for safe import from other files
    nest.ResetKernel()

    '''
    Configuration of the simulation kernel by the previously defined time
    resolution used in the simulation. Setting "print_time" to True prints
    the already processed simulation time as well as its percentage of the
    total simulation time.
    '''

    nest.SetKernelStatus({"resolution": dt, "print_time": True,
                          "overwrite_files": True})

    print("Building network")

    '''
    Configuration of the model `iaf_psc_alpha` and `poisson_generator`
    using SetDefaults(). This function expects the model to be the
    inserted as a string and the parameter to be specified in a
    dictionary. All instances of theses models created after this point
    will have the properties specified in the dictionary by default.
    '''

    nest.SetDefaults("iaf_psc_alpha", neuron_params)
    nest.SetDefaults("poisson_generator", {"rate": p_rate})

    '''
    Creation of the nodes using `Create`. We store the returned handles in
    variables for later reference. Here the excitatory and inhibitory, as
    well as the poisson generator and two spike detectors. The spike
    detectors will later be used to record excitatory and inhibitory
    spikes.
    '''

    nodes_ex = nest.Create("iaf_psc_alpha", NE)
    nodes_in = nest.Create("iaf_psc_alpha", NI)
    noise = nest.Create("poisson_generator")
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    print("first exc node: {}".format(nodes_ex[0]))
    print("first inh node: {}".format(nodes_in[0]))

    '''
    distribute membrane potentials
    '''
    nest.SetStatus(nodes_ex, "V_m",
                   random.rand(len(nodes_ex)) * neuron_params["V_th"])
    nest.SetStatus(nodes_in, "V_m",
                   random.rand(len(nodes_in)) * neuron_params["V_th"])

    '''
    Configuration of the spike detectors recording excitatory and
    inhibitory spikes using `SetStatus`, which expects a list of node
    handles and a list of parameter dictionaries. Setting the variable
    "to_file" to True ensures that the spikes will be recorded in a .gdf
    file starting with the string assigned to label. Setting "withtime"
    and "withgid" to True ensures that each spike is saved to file by
    stating the gid of the spiking neuron and the spike time in one line.
    '''

    nest.SetStatus(espikes, [{
        "label": os.path.join(spike_output_path, label + "-EX"),
        "record_to": 'ascii',
    }])

    nest.SetStatus(ispikes, [{
        "label": os.path.join(spike_output_path, label + "-IN"),
        "record_to": 'ascii', }])

    print("Connecting devices")

    '''
    Definition of a synapse using `CopyModel`, which expects the model
    name of a pre-defined synapse, the name of the customary synapse and
    an optional parameter dictionary. The parameters defined in the
    dictionary will be the default parameter for the customary
    synapse. Here we define one synapse for the excitatory and one for the
    inhibitory connections giving the previously defined weights and equal
    delays.
    '''

    nest.CopyModel(
        "static_synapse", "excitatory", {
            "weight": J_ex, "delay": delay})
    nest.CopyModel(
        "static_synapse", "inhibitory", {
            "weight": J_in, "delay": delay})

    '''
    Connecting the previously defined poisson generator to the excitatory
    and inhibitory neurons using the excitatory synapse. Since the poisson
    generator is connected to all neurons in the population the default
    rule ('all_to_all') of Connect() is used. The synaptic properties are
    inserted via syn_spec which expects a dictionary when defining
    multiple variables or a string when simply using a pre-defined
    synapse.
    '''

    if Poisson:
        nest.Connect(noise, nodes_ex, 'all_to_all', "excitatory")
        nest.Connect(noise, nodes_in, 'all_to_all', "excitatory")

    '''
    Connecting the first N_neurons nodes of the excitatory and inhibitory
    population to the associated spike detectors using excitatory
    synapses. Here the same shortcut for the specification of the synapse
    as defined above is used.
    '''

    nest.Connect(nodes_ex, espikes, 'all_to_all', "excitatory")
    nest.Connect(nodes_in, ispikes, 'all_to_all', "excitatory")

    print("Connecting network")

    print("Excitatory connections")

    '''
    Connecting the excitatory population to all neurons using the
    pre-defined excitatory synapse. Beforehand, the connection parameter
    are defined in a dictionary. Here we use the connection rule
    'fixed_indegree', which requires the definition of the indegree. Since
    the synapse specification is reduced to assigning the pre-defined
    excitatory synapse it suffices to insert a string.
    '''

    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")

    print("Inhibitory connections")

    '''
    Connecting the inhibitory population to all neurons using the
    pre-defined inhibitory synapse. The connection parameter as well as
    the synapse paramtere are defined analogously to the connection from
    the excitatory population defined above.
    '''

    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

    '''
    Storage of the time point after the buildup of the network in a
    variable.
    '''

    endbuild = time.time()

    '''
    Simulation of the network.
    '''

    print("Simulating")

    nest.Simulate(simtime)

    '''
    Storage of the time point after the simulation of the network in a
    variable.
    '''

    endsimulate = time.time()

    '''
    Reading out the total number of spikes received from the spike
    detector connected to the excitatory population and the inhibitory
    population.
    '''

    events_ex = nest.GetStatus(espikes, "n_events")[0]
    events_in = nest.GetStatus(ispikes, "n_events")[0]

    '''
    Calculation of the average firing rate of the excitatory and the
    inhibitory neurons by dividing the total number of recorded spikes by
    the number of neurons recorded from and the simulation time. The
    multiplication by 1000.0 converts the unit 1/ms to 1/s=Hz.
    '''

    rate_ex = events_ex / simtime * 1000.0 / N_neurons
    rate_in = events_in / simtime * 1000.0 / N_neurons

    '''
    Reading out the number of connections established using the excitatory
    and inhibitory synapse model. The numbers are summed up resulting in
    the total number of synapses.
    '''

    num_synapses = nest.GetDefaults("excitatory")["num_connections"] +\
        nest.GetDefaults("inhibitory")["num_connections"]

    '''
    Establishing the time it took to build and simulate the network by
    taking the difference of the pre-defined time variables.
    '''

    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild

    '''
    Printing the network properties, firing rates and building times.
    '''

    print("Brunel network simulation (Python)")
    print("Number of neurons : {0}".format(N_neurons))
    print("Number of synapses: {0}".format(num_synapses))
    print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
    print("       Inhibitory : {0}".format(int(CI * N_neurons)))
    print("Excitatory rate   : %.2f Hz" % rate_ex)
    print("Inhibitory rate   : %.2f Hz" % rate_in)
    print("Building time     : %.2f s" % build_time)
    print("Simulation time   : %.2f s" % sim_time)

    '''
    Plot a raster of the excitatory neurons and a histogram.
    '''

    if False:
        nest.raster_plot.from_device(espikes, hist=True)
        nest.raster_plot.from_device(ispikes, hist=True)
        nest.raster_plot.show()


if __name__ == '__main__':
    simulate()
