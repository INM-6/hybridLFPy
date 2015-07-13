import numpy as np
import nest


def dict_of_numpyarray_to_dict_of_list(d):
    '''
    Convert dictionary containing numpy arrays to dictionary containing lists
    
    Parameters
    ----------
    d : dict
        sli parameter name and value as dictionary key and value pairs
    
    Returns
    -------
    d : dict
        modified dictionary
    
    '''
    for key,value in d.iteritems():
        if isinstance(value,dict):  # if value == dict 
            # recurse
            d[key] = dict_of_numpyarray_to_dict_of_list(value)
        elif isinstance(value,np.ndarray): # or isinstance(value,list) :
            d[key] = value.tolist()
    return d

def send_nest_params_to_sli(p):
    '''
    Read parameters and send them to SLI
    
    Parameters
    ----------
    p : dict
        sli parameter name and value as dictionary key and value pairs
    
    Returns
    -------
    None
    '''
    for name in p.keys():
        value = p[name]
        if type(value) == np.ndarray:
            value = value.tolist()
        if type(value) == dict:
            value = dict_of_numpyarray_to_dict_of_list(value)
        if name == 'neuron_model': # special case as neuron_model should is a NEST model and not a string
            try:
                nest.sli_run('/'+name)
                nest.sli_push(value)
                nest.sli_run('eval')
                nest.sli_run('def')
            except: 
                print 'Could not put variable %s on SLI stack' % (name)
                print type(value)
        else:
            try:
                nest.sli_run('/'+name)
                nest.sli_push(value)
                nest.sli_run('def')
            except: 
                print 'Could not put variable %s on SLI stack' % (name)
                print type(value)
    return


def sli_run(parameters=object(),
            fname='microcircuit.sli',
            verbosity='M_ERROR'):
    '''
    Takes parameter-class and name of main sli-script as input, initiating the
    simulation.
    
    Parameters
    ----------
    parameters : object
        parameter class instance
    fname : str
        path to sli codes to be executed
    verbosity : str,
        nest verbosity flag
    
    Returns
    -------
    None
    
    '''
    # Load parameters from params file, and pass them to nest
    # Python -> SLI
    send_nest_params_to_sli(vars(parameters))
    
    #set SLI verbosity
    nest.sli_run("%s setverbosity" % verbosity)
    
    # Run NEST/SLI simulation
    nest.sli_run('(%s) run' % fname)


if __name__ == '__main__':
    from cellsim16popsParams import point_neuron_network_params
    sli_run(parameters=point_neuron_network_params(),
            fname='microcircuit.sli',
            verbosity='M_ERROR')