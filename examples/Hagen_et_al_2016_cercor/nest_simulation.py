#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy
import nest


def dict_of_numpyarray_to_dict_of_list(d):
    '''
    Convert dictionary containing numpy arrays to dictionary containing lists
    '''
    for key,value in d.items():
        if isinstance(value,dict):  # if value == dict
            # recurse
            d[key] = dict_of_numpyarray_to_dict_of_list(value)
        elif isinstance(value,numpy.ndarray): # or isinstance(value,list) :
            d[key] = value.tolist()
    return d

def send_nest_params_to_sli(p):
    '''
    Read parameters and send them to SLI
    '''
    for name in list(p.keys()):
        value = p[name]
        if type(value) == numpy.ndarray:
            value = value.tolist()
        if type(value) == dict:
            value = dict_of_numpyarray_to_dict_of_list(value)
        if name == 'neuron_model':  # special case as neuron_model is a
                                    # NEST model and not a string
            try:
                nest.ll_api.sli_run('/'+name)
                nest.ll_api.sli_push(value)
                nest.ll_api.sli_run('eval')
                nest.ll_api.sli_run('def')
            except:
                print('Could not put variable %s on SLI stack' % (name))
                print(type(value))
        else:
            try:
                nest.ll_api.sli_run('/'+name)
                nest.ll_api.sli_push(value)
                nest.ll_api.sli_run('def')
            except:
                print('Could not put variable %s on SLI stack' % (name))
                print(type(value))
    return


def sli_run(parameters=object(),
            fname='microcircuit.sli',
            verbosity='M_ERROR'):
    '''
    Takes parameter-class and name of main sli-script as input, initiating the
    simulation.

    kwargs:
    ::
        parameters : object, parameter class instance
        fname : str, path to sli codes to be executed
        verbosity : 'str', nest verbosity flag
    '''
    # Load parameters from params file, and pass them to nest
    # Python -> SLI
    send_nest_params_to_sli(vars(parameters))

    #set SLI verbosity
    nest.ll_api.sli_run("%s setverbosity" % verbosity)

    # Run NEST/SLI simulation
    nest.ll_api.sli_run('(%s) run' % fname)


if __name__ == '__main__':
    from cellsim16popsParams import point_neuron_network_params
    sli_run(parameters=point_neuron_network_params(),
            fname='microcircuit.sli',
            verbosity='M_ERROR')
