#!/usr/bin/env python
'''
Modified parameters file for the Hybrid LFP scheme, applying the methodology with
the model of:

Potjans, T. and Diesmann, M. "The Cell-Type Specific Cortical Microcircuit:
Relating Structure and Activity in a Full-Scale Spiking Network Model".
Cereb. Cortex (2014) 24 (3): 785-806.
doi: 10.1093/cercor/bhs358


'''
import numpy as np
import os
import json
from mpi4py import MPI #this is needed to initialize other classes correctly


###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


####################################
# HELPER FUNCTIONS                 #
####################################

flattenlist = lambda lst: sum(sum(lst, []),[])


####################################
# SPATIAL CONNECTIVITY EXTRACTION  #
####################################

'''
Include functions that extract information from binzegger.json here
'''



def get_F_y(fname='binzegger_connectivity_table.json', y=['p23']): 
    '''
    Extract frequency of occurrences of those cell types that are modeled.
    The data set contains cell types that are not modeled (TCs etc.)
    The returned percentages are renormalized onto modeled cell-types, i.e. they sum up to 1 
    '''
    # Load data from json dictionary
    f = open(fname,'r')
    data = json.load(f)
    f.close()
    
    occurr = []
    for cell_type in y:
        occurr += [data['data'][cell_type]['occurrence']]
    return list(np.array(occurr)/np.sum(occurr)) 



def get_L_yXL(fname, y, x_in_X, L):
    '''
    compute the layer specificity, defined as:
    ::
    
        L_yXL = k_yXL / k_yX
    '''
    def _get_L_yXL_per_yXL(fname, x_in_X, X_index,
                                  y, layer):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
    
        
        # Get number of synapses
        if layer in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
            #init variables
            k_yXL = 0
            k_yX = 0
            
            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][layer][x] / 100.
                k_yL = data['data'][y]['syn_dict'][layer]['number of synapses per neuron']
                k_yXL += p_yxL * k_yL
                
            for l in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
                for x in x_in_X[X_index]:
                    p_yxL = data['data'][y]['syn_dict'][l][x] / 100.
                    k_yL = data['data'][y]['syn_dict'][l]['number of synapses per neuron']
                    k_yX +=  p_yxL * k_yL
            
            if k_yXL != 0.:
                return k_yXL / k_yX
            else:
                return 0.
        else:
            return 0.


    #init dict
    L_yXL = {}

    #iterate over postsynaptic cell types
    for y_value in y:
        #container
        data = np.zeros((len(L), len(x_in_X)))
        #iterate over lamina
        for i, Li in enumerate(L):
            #iterate over presynapse population inds
            for j in range(len(x_in_X)):
                data[i][j]= _get_L_yXL_per_yXL(fname, x_in_X,
                                                          X_index=j,
                                                          y=y_value,
                                                          layer=Li)
        L_yXL[y_value] = data

    return L_yXL



def get_T_yX(fname, y, y_in_Y, x_in_X, F_y):
    '''
    compute the cell type specificity, defined as:
    ::
    
        T_yX = K_yX / K_YX
            = F_y * k_yX / sum_y(F_y*k_yX) 
    
    '''
    def _get_k_yX_mul_F_y(y, y_index, X_index):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
    
        #init variables
        k_yX = 0.
        
        for l in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][l][x] / 100.
                k_yL = data['data'][y]['syn_dict'][l]['number of synapses per neuron']
                k_yX +=  p_yxL * k_yL
        
        return k_yX * F_y[y_index]


    #container
    T_yX = np.zeros((len(y), len(x_in_X)))
    
    #iterate over postsynaptic cell types
    for i, y_value in enumerate(y):
        #iterate over presynapse population inds
        for j in range(len(x_in_X)):
            k_yX_mul_F_y = 0
            for k, yy in enumerate(sum(y_in_Y, [])):                
                if y_value in yy:
                    for yy_value in yy:
                        ii = np.where(np.array(y) == yy_value)[0][0]
                        k_yX_mul_F_y += _get_k_yX_mul_F_y(yy_value, ii, j)
            
            
            if k_yX_mul_F_y != 0:
                T_yX[i, j] = _get_k_yX_mul_F_y(y_value, i, j) / k_yX_mul_F_y
            
    return T_yX


class general_params(object):
    '''class defining general model parameters'''
    def __init__(self):
        '''class defining general model parameters'''

        ####################################
        # REASON FOR THIS SIMULATION       #
        ####################################

        self.reason = 'Default Potjans model with spontaneous activity'

    ####################################
    #                                  #
    #                                  #
    #     SIMULATION PARAMETERS        #
    #                                  #
    #                                  #
    ####################################  


        ####################################
        # MAIN SIMULATION CONTROL          #
        ####################################

        # simulation step size
        self.dt = 0.1

        # simulation start
        self.tstart = 0

        # simulation stop
        self.tstop = 1200


        ####################################
        # OUTPUT LOCATIONS                 #
        ####################################
        
        # TODO: try except does not work with hambach

        # folder for all simulation output and scripts
        # here, compute clusters have scratch areas for saving
        if os.path.isdir(os.path.join('/', 'scratch', os.environ['USER'])):
            self.savefolder = os.path.join('/', 'scratch', os.environ['USER'],
                                           'hybrid_model',
                                           'simulation_output_example_microcircuit')
        # LOCALLY
        else:
            self.savefolder = 'simulation_output_example_microcircuit'

        # folder for simulation scripts
        self.sim_scripts_path = os.path.join(self.savefolder, 'sim_scripts')

        # folder for each individual cell's output
        self.cells_path = os.path.join(self.savefolder, 'cells')

        # folder for figures
        self.figures_path = os.path.join(self.savefolder, 'figures')

        # folder for population resolved output signals
        self.populations_path = os.path.join(self.savefolder, 'populations')

        # folder for raw nest output files
        self.raw_nest_output_path = os.path.join(self.savefolder,
                                                 'raw_nest_output')
        
        # folder for processed nest output files
        self.spike_output_path = os.path.join(self.savefolder,
                                              'processed_nest_output')

    
    ####################################
    #                                  #
    #                                  #
    #        MODEL PARAMETERS          #
    #                                  #
    #                                  #
    ####################################



        ####################################
        # POPULATIONS                      #
        ####################################

        # Number of populations
        self.Npops = 9

        # number of neurons in each population (unscaled)
        self.full_scale_num_neurons = [[20683,   # layer 23 e
                                        5834 ],  # layer 23 i
                                       [21915,   # layer 4 e
                                        5479 ],  # layer 4 i
                                       [4850,    # layer 5 e
                                        1065 ],  # layer 5 i
                                       [14395,   # layer 6 e
                                        2948 ]]  # layer 6 i

        # Number of thalamic neurons/ point processes
        self.n_thal = 902

        # population names TODO: rename
        self.X = ['TC', 'L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
        self.Y = self.X[1:]

        # TC and cortical population sizes in one list TODO: rename
        self.N_X = np.array([self.n_thal]+flattenlist([self.full_scale_num_neurons]))
  

        ####################################
        # CONNECTIVITY                     #
        ####################################
        
        # intra-cortical connection probabilities between populations
        #                            23e      23i      4e     4i      5e     5i       6e      6i
        self.conn_probs = np.array([[0.1009,0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076,  0.    ],  # 23e
                                    [0.1346,0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042,  0.    ],  # 23i
                                    [0.0077,0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453,  0.    ],  # 4e
                                    [0.0691,0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057,  0.    ],  # 4i
                                    [0.1004,0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204,  0.    ],  # 5e 
                                    [0.0548,0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086,  0.    ],  # 5i
                                    [0.0156,0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396,  0.2252],  # 6e
                                    [0.0364,0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658,  0.1443]]) # 6i

        # connection probabilities for thalamic input
        self.C_th = [[0.0,       # layer 23 e
                      0.0   ],   # layer 23 i    
                     [0.0983,    # layer 4 e
                      0.0619],   # layer 4 i
                     [0.0,       # layer 5 e
                      0.0   ],   # layer 5 i
                     [0.0512,    # layer 6 e
                      0.0196]]   # layer 6 i
             


        # full connection probabilities including TC connections
        self.C_YX = np.c_[flattenlist([self.C_th]), self.conn_probs]
        

        ####################################
        # CONNECTION PROPERTIES            #
        ####################################
                
        # mean EPSP amplitude (mV) for all connections except L4e->L23e
        self.PSP_e = 0.15

        # mean EPSP amplitude (mv) for L4e->L23e connections
        # FIX POLISH NOTATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.PSP_23e_4e = self.PSP_e*2

        # standard deviation of PSC amplitudes relative to mean PSC amplitudes
        # this is sigma/mu in probability distribution
        # Gaussian (lognormal_weights = False): mu is mean, sigma is standard deviation
        # Lognormal (lognormal_weights = False): mean and stdev can be calculated from mu and sigma
        self.PSC_rel_sd = 0.1
                
        # IPSP amplitude relative to EPSP amplitude
        self.g = -4.               

        # set L4i ->L4e stronger in order to get rid of 84 Hz peak
        self.g_4e_4i = self.g*1.15
        
        # Whether to use lognormal weights or not
        self.lognormal_weights = False 

        # mean dendritic delays for excitatory and inhibitory transmission (ms)
        self.delays = [1.5, 0.75] 

        # standard deviation relative to mean delays
        self.delay_rel_sd = 0.5
        
      
        ####################################
        # CELL-TYPE PARAMETERS             #
        ####################################
        
        # Note that these parameters are only relevant for the point-neuron network in case
        # one wants to calculate depth-resolved cell-type specific input currents

        # point to .json connectivity table file
        self.connectivity_table = 'binzegger_connectivity_table.json'

        #list of cell type names used in this script
        #names of every post-syn pop layer
        self.y_in_Y = [
                [['p23'],['b23','nb23']],
                [['p4','ss4(L23)','ss4(L4)'],['b4','nb4']],
                [['p5(L23)','p5(L56)'],['b5','nb5']],
                [['p6(L4)','p6(L56)'],['b6','nb6']]]

        self.y = flattenlist(self.y_in_Y)
        
        #need presynaptic cell type to population mapping
        self.x_in_X = [['TCs', 'TCn']] + sum(self.y_in_Y, [])
        
        
        #map the pre-synaptic populations to the post-syn populations
        self.mapping_Yy = zip(
                  ['L23E', 'L23I', 'L23I',
                   'L4E', 'L4E', 'L4E', 'L4I', 'L4I',
                   'L5E', 'L5E', 'L5I', 'L5I',
                   'L6E', 'L6E', 'L6I', 'L6I'],
                  self.y)

        # Frequency of occurrence of each cell type (F_y); 1-d array 
        self.F_y = get_F_y(fname=self.connectivity_table, y=self.y)

        # Relative frequency of occurrence of each cell type within its population (F_{y,Y})
        self.F_yY = [[get_F_y(fname=self.connectivity_table, y=y) for y in Y] for Y in self.y_in_Y]
        
        # Number of neurons of each cell type (N_y); 1-d array
        self.N_y =  np.array([self.full_scale_num_neurons[layer][pop] * self.F_yY[layer][pop][k] \
                                     for layer, array in enumerate(self.y_in_Y)\
                                     for pop, cell_types in enumerate(array) \
                                     for k, _ in enumerate(cell_types)]).astype(int)
        
        
        #compute the number of synapses as in Potjans&Diesmann 2012
        K_YX = np.zeros(self.C_YX.shape)
        for i in xrange(K_YX.shape[1]):
            K_YX[:, i] = (np.log(1. - self.C_YX[:, i]) /
                                np.log(1. - 1./(self.N_X[1:]*
                                                self.N_X[i])))


        #spatial connection probabilites on each subpopulation
        #Each key must correspond to a subpopulation like 'L23E' used everywhere else,
        #each array maps thalamic and intracortical connections.
        #First column is thalamic connections, and the rest intracortical,
        #ordered like 'L23E', 'L23I' etc., first row is normalised probability of
        #connection withing L1, L2, etc.;
        self.L_yXL = get_L_yXL(fname = self.connectivity_table,
                                         y = self.y,
                                         x_in_X = self.x_in_X,
                                         L = ['1','23','4','5','6'])
        
        #compute the cell type specificity
        self.T_yX = get_T_yX(fname=self.connectivity_table, y=self.y,
                             y_in_Y=self.y_in_Y, x_in_X=self.x_in_X,
                             F_y=self.F_y)
        
        Y, y = zip(*self.mapping_Yy)

        #assess relative distribution of synapses for a given celltype
        self.K_yXL = {}
        #self.T_yX = {}
        for i, (Y, y) in enumerate(self.mapping_Yy):
            #fill in K_yXL (layer specific connectivity)
            self.K_yXL[y] = (self.T_yX[i, ] * K_YX[np.array(self.Y)==Y, ] * self.L_yXL[y]).astype(int)
        
        #number of incoming connections per cell type per layer per cell 
        self.k_yXL = {}
        for y, N_y in zip(self.y, self.N_y):
            self.k_yXL.update({y : (1. * self.K_yXL[y]).astype(int) / N_y})

        #calculate corresponding connectivity to K_yXL
        self.C_yXL = {}
        for y, N_y in zip(self.y, self.N_y):
            self.C_yXL.update({y : 1. - (1.-1./(N_y* self.N_X))**self.K_yXL[y] })


################################################################################################################


class point_neuron_network_params(general_params):
    '''class point-neuron network parameters'''
    def __init__(self):
        '''class point-neuron network parameters'''

        # inherit general params
        general_params.__init__(self)
        

    ####################################
    #                                  #
    #                                  #
    #     SIMULATION PARAMETERS        #
    #                                  #
    #                                  #
    #################################### 

        # use same number of threads as MPI COMM.size()
        self.total_num_virtual_procs = SIZE

        ####################################
        # RNG PROPERTIES                   #
        #################################### 

        # offset for RNGs
        self.seed_offset = 45


        ####################################
        # RECORDING PARAMETERS             #        
        ####################################

        self.to_memory = False

        self.overwrite_existing_files = True 

        # recording can either be done from a fraction of neurons in each population or from a fixed number

        # whether to record spikes from a fixed fraction of neurons in each population. 
        self.record_fraction_neurons_spikes = True 

        if self.record_fraction_neurons_spikes:
            self.frac_rec_spikes = 1.
        else:
            self.n_rec_spikes = 100 

        # whether to record membrane potentials from a fixed fraction of neurons in each population
        self.record_fraction_neurons_voltage = False 

        if self.record_fraction_neurons_voltage:
            self.frac_rec_voltage = 0.1 
        else:
            self.n_rec_voltage = 100 

        # whether to record weighted input spikes from a fixed fraction of neurons in each population
        self.record_fraction_neurons_input_spikes = False

        if self.record_fraction_neurons_input_spikes:
            self.frac_rec_input_spikes = 0.1 
        else:
            self.n_rec_input_spikes = 0 

        # number of recorded neurons for depth resolved input currents
        self.n_rec_depth_resolved_input = 0 
 
        # whether to write any recorded cortical spikes to file
        self.save_cortical_spikes = True 

        # whether to write any recorded membrane potentials to file
        self.save_voltages = False 

        # whether to record thalamic spikes
        self.record_thalamic_spikes = True 

        # whether to write any recorded thalamic spikes to file
        self.save_thalamic_spikes = True 

        # global ID file name
        self.GID_filename = 'population_GIDs.dat'

        # readout global ID file name
        self.readout_GID_filename = 'readout_GIDs.dat' 

        # stem for spike detector file labels
        self.spike_detector_label = 'spikes_'

        # stem for voltmeter file labels
        self.voltmeter_label = 'voltages_'

        # stem for thalamic spike detector file labels
        self.th_spike_detector_label = 'spikes_0'

        # stem for in-degree file labels
        self.in_degree_label = 'in_degrees_'

        # stem for file labels for in-degree from thalamus 
        self.th_in_degree_label = 'in_degrees_th_'

        # stem for weighted input spikes labels
        self.weighted_input_spikes_label = 'weighted_input_spikes_'


    ####################################
    #                                  #
    #                                  #
    #        MODEL PARAMETERS          #
    #                                  #
    #                                  #
    ####################################


        ####################################
        # SCALING                          #
        ####################################  
       
        # scaling parameter for population sizes
        self.area = 1.
        
        # preserve indegrees when downscaling
        self.preserve_K = False         
        

        ####################################
        # SINGLE NEURON PARAMS             #
        ####################################

        # neuron model
        self.neuron_model = '/iaf_psc_exp'      

        # mean of initial membrane potential (mV)
        self.Vm0_mean = -58.0

        # std of initial membrane potential (mV)
        self.Vm0_std = 10.0

        # mean of threshold potential (mV)
        self.V_th_mean = -50.

        # std of threshold potential (mV)
        self.V_th_std = 0.
        
        self.model_params = { 'tau_m': 10.,        # membrane time constant (ms)
                              'tau_syn_ex': 0.5,   # excitatory synaptic time constant (ms)
                              'tau_syn_in': 0.5,   # inhibitory synaptic time constant (ms)
                              't_ref': 2.,         # absolute refractory period (ms)
                              'E_L': -65.,         # resting membrane potential (mV)
                              'V_th': self.V_th_mean, # spike threshold (mV)
                              'C_m': 250.,         # membrane capacitance (pF)
                              'V_reset': -65.      # reset potential (mV)
                              } 
        


        ####################################
        # EXTERNAL INPUTS                  #
        ####################################

        #number of external inputs (Potjans-Diesmann model 2012)
        self.K_bg = [[1600,    # layer 23 e
                      1500],   # layer 23 i
                     [2100,    # layer 4 e
                      1900],   # layer 4 i
                     [2000,    # layer 5 e
                      1900],   # layer 5 i
                     [2900,    # layer 6 e
                      2100]]   # layer 6 i
        
        # rate of Poisson input at each external input synapse (spikess)
        self.bg_rate = 0.       

        # rate of equivalent input used for DC amplitude calculation,
        # set to zero if self.bg_rate > 0.
        self.bg_rate_dc = 8.

        # DC amplitude at each external input synapse (pA)  
        # to each neuron via 'dc_amplitude  = tau_syn_ex/1000*bg_rate*PSC_ext'
        self.dc_amplitude = self.model_params["tau_syn_ex"] * self.bg_rate_dc *\
                            self._compute_J()

        # mean EPSP amplitude (mV) for thalamic and non-thalamic external input spikes
        self.PSP_ext = 0.15 	       

        # mean delay of thalamic input (ms)
        self.delay_th = 1.5  	

        # standard deviation relative to mean delay of thalamic input
        self.delay_th_rel_sd = 0.5   


        ####################################
        # THALAMIC INPUT VERSIONS  	   #
        ####################################
  
        # off-option for start of thalamic input versions
        self.off = 100.*self.tstop

   
        ## poisson_generator (pure Poisson input)
        self.th_poisson_start = self.off  	# onset (ms)
        self.th_poisson_duration = 10.	        # duration (ms)
        self.th_poisson_rate = 120. 	        # rate (spikess)


        ## spike_generator 
        # Note: This can be used with a large Gaussian delay distribution in order to mimic a 
        #       Gaussian pulse packet which is different for each thalamic neuron
        self.th_spike_times = [self.off]	# time of the thalamic pulses (ms)


        ## sinusoidal_poisson_generator (oscillatory Poisson input)
        self.th_sin_start = self.off      	# onset (ms)
        self.th_sin_duration = 5000.  	        # duration (ms)
        self.th_sin_mean_rate = 30. 	        # mean rate (spikess)
        self.th_sin_fluc_rate = 30.  	        # rate modulation amplitude (spikess)
        self.th_sin_freq = 15. 	                # frequency of the rate modulation (Hz)
        self.th_sin_phase = 0.                  # phase of rate modulation (deg)


        ## Gaussian_pulse_packages
        self.th_gauss_times = [self.off]                # package center times
        self.th_gauss_num_spikes_per_packet = 1 	# number of spikes per packet
        self.th_gauss_sd = 5. 				# std of Gaussian pulse packet (ms^2)


        ####################################
        # SPATIAL ORGANIZATION             #
        ####################################
        
        # needed for spatially resolved input currents
        
        # number of layers TODO: find a better solution for that
        self.num_input_layers = 5


    def _compute_J(self):
        '''
        Compute the current amplitude corresponding to the exponential
        synapse model PSP amplitude
        
        Derivation using sympy:
        ::
            from sympy import *
            #define symbols
            t, tm, Cm, ts, Is, Vmax = symbols('t tm Cm ts Is Vmax')
            
            #assume zero delay, t >= 0
            #using eq. 8.10 in Sterrat et al
            V = tm*ts*Is*(exp(-t/tm) - exp(-t/ts)) / (tm-ts) / Cm
            print 'V = %s' % V
            
            #find time of V == Vmax
            dVdt = diff(V, t)
            print 'dVdt = %s' % dVdt
            
            [t] = solve(dVdt, t)
            print 't(t@dVdT==Vmax) = %s' % t
            
            #solve for Is at time of maxima
            V = tm*ts*Is*(exp(-t/tm) - exp(-t/ts)) / (tm-ts) / Cm
            print 'V(%s) = %s' % (t, V)
            
            [Is] = solve(V-Vmax, Is)
            print 'Is = %s' % Is
        
        resulting in:
        ::
            Cm*Vmax*(-tm + ts)/(tm*ts*(exp(tm*log(ts/tm)/(tm - ts))
                                     - exp(ts*log(ts/tm)/(tm - ts))))
        
        '''
        #LIF params
        tm = self.model_params['tau_m']
        Cm = self.model_params['C_m']
        
        #synapse
        ts = self.model_params['tau_syn_ex']
        Vmax = self.PSP_e
        
        #max current amplitude
        J = Cm*Vmax*(-tm + ts)/(tm*ts*(np.exp(tm*np.log(ts/tm)/(tm - ts))
                                     - np.exp(ts*np.log(ts/tm)/(tm - ts))))
        
        #unit conversion pF*mV -> nA
        J *= 1E-3
        
        return J
 

class multicompartment_params(point_neuron_network_params):
    '''
    Inherited class defining additional attributes needed by e.g., the
    classes population.Population and population.DummyNetwork

    This class do not take any kwargs    

    '''
    def __init__(self):
        '''
        Inherited class defining additional attributes needed by e.g., the
        classes population.Population and population.DummyNetwork
        
        This class do not take any kwargs    
        
        '''
        
        # initialize parent classes
        point_neuron_network_params.__init__(self)
 

    ####################################
    #                                  #
    #                                  #
    #     SIMULATION PARAMETERS        #
    #                                  #
    #                                  #
    ####################################
       

        #######################################
        # PARAMETERS FOR LOADING NEST RESULTS #
        #######################################


        # parameters for class population.DummyNetwork class
        self.networkSimParams = {
            'simtime' :     self.tstop - self.tstart,
            'dt' :          self.dt,
            'spike_output_path' : self.spike_output_path,
            'label' :       'population_spikes',
            'ext' :         'gdf',
            'GIDs' : self.get_GIDs(),
        }


        # Switch for current source density computations
        self.calculateCSD = True


    ####################################
    #                                  #
    #                                  #
    #        MODEL PARAMETERS          #
    #                                  #
    #                                  #
    ####################################

       
        ####################################
        # SCALING (VOLUME not density)     #
        ####################################  
           
        self.SCALING = 1.
        
  
        ####################################
        # MORPHOLOGIES                     #
        ####################################

        # list of morphology files with default location, testing = True
        # will point to simplified morphologies
        testing = True
        if testing:
            self.PATH_m_y = os.path.join('morphologies', 'ballnsticks')
            self.m_y = [Y + '_' + y + '.hoc' for Y, y in self.mapping_Yy]
        else:
            self.PATH_m_y = os.path.join('morphologies', 'stretched')
            self.m_y = [
                'L23E_oi24rpy1.hoc',
                'L23I_oi38lbc1.hoc',
                'L23I_oi38lbc1.hoc',
            
                'L4E_53rpy1.hoc',
                'L4E_j7_L4stellate.hoc',
                'L4E_j7_L4stellate.hoc',
                'L4I_oi26rbc1.hoc',
                'L4I_oi26rbc1.hoc',
            
                'L5E_oi15rpy4.hoc',
                'L5E_j4a.hoc',
                'L5I_oi15rbc1.hoc',
                'L5I_oi15rbc1.hoc',
            
                'L6E_51-2a.CNG.hoc',
                'L6E_oi15rpy4.hoc',
                'L6I_oi15rbc1.hoc',
                'L6I_oi15rbc1.hoc',
                ]


        ####################################
        # CONNECTION WEIGHTS               #
        ####################################
        
        # compute the synapse weight from fundamentals of exp synapse LIF neuron
        self.J = self._compute_J()
        
        # set up matrix containing the synapse weights between any population X
        # and population Y, including exceptions for certain connections
        J_YX = np.zeros(self.C_YX.shape)
        J_YX += self.J
        J_YX[:, 2::2] *= self.g
        if hasattr(self, 'PSP_23e_4e'):
            J_YX[0, 3] *= self.PSP_23e_4e / self.PSP_e
        if hasattr(self, 'g_4e_4i'):
            J_YX[2, 4] *= self.g_4e_4i / self.g
        

        # extrapolate weights between populations X and
        # cell type y in population Y
        self.J_yX = {}
        for Y, y in self.mapping_Yy:
            [i] = np.where(np.array(self.Y) == Y)[0]
            self.J_yX.update({y : J_YX[i, ]})
        
    
        ####################################
        # GEOMETRY OF CORTICAL COLUMN      #
        ####################################
        
        # set the boundaries of each layer, L1->L6,
        # and mean depth of soma layers
        self.layerBoundaries = np.array([[     0.0,   -81.6],
                                          [  -81.6,  -587.1],
                                          [ -587.1,  -922.2],
                                          [ -922.2, -1170.0],
                                          [-1170.0, -1491.7]])
        
        # assess depth of each 16 subpopulation
        self.depths = self._calcDepths()
        
        # make a nice structure with data for each subpopulation
        self.y_zip_list = zip(self.y, self.m_y,
                            self.depths, self.N_y)



        ##############################################################
        # POPULATION PARAMS (cells, population, synapses, electrode) #
        ##############################################################


        # Global LFPy.Cell-parameters, by default shared between populations
        # Some passive parameters will not be fully consistent with LIF params
        self.cellParams = {
            'v_init' : self.model_params['E_L'],
            'passive' : True,
            'rm' : self.model_params['tau_m'] * 1E3 / 1.0, #assyme cm=1
            'cm' : 1.0,
            'Ra' : 150,
            'e_pas' : self.model_params['E_L'],    
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
            'timeres_NEURON' : self.dt,
            'timeres_python' : self.dt,
            'tstartms' : self.tstart,
            'tstopms' : self.tstop,
            'verbose' : False,
        }
        

        # layer specific LFPy.Cell-parameters as nested dictionary
        self.yCellParams = self._yCellParams()
        
        
        # set the axis of which each cell type y is randomly rotated,
        # SS types and INs are rotated around both x- and z-axis
        # in the population class, while P-types are
        # only rotated around the z-axis
        self.rand_rot_axis = {}
        for y, _, _, _ in self.y_zip_list:
            #identify pyramidal cell populations:
            if 'p' in y:
                self.rand_rot_axis.update({y : ['z']})
            else:
                self.rand_rot_axis.update({y : ['x', 'z']})
        
        
        # additional simulation kwargs, see LFPy.Cell.simulate() docstring
        self.simulationParams = {}
        
                
        # a dict setting the number of cells N_y and geometry
        # of cell type population y
        self.populationParams = {}
        for y, _, depth, N_y in self.y_zip_list:
            self.populationParams.update({
                y : {
                    'number' : int(N_y*self.SCALING),
                    'radius' : np.sqrt(1000**2 / np.pi),
                    'z_min' : depth - 25,
                    'z_max' : depth + 25,
                    'min_cell_interdist' : 1.,            
                }
            })

        # Set up cell type specific synapse parameters in terms of synapse model
        # and synapse locations
        self.synParams = {}
        for y in self.y:
            if 'p' in y:
                #pyramidal types have apical dendrites
                section = ['apic', 'dend']
            else:
                #other cell types do not
                section = ['dend']

            self.synParams.update({
                y : {
                    'syntype' : 'ExpSynI',  #current based exponential synapse
                    'section' : section,
                    # 'tau' : self.model_params["tau_syn_ex"],
                },
            })

        
        # set up dictionary of synapse time constants specific to each
        # postsynaptic cell type and presynaptic population
        self.tau_yX = {}
        for y in self.y:
            self.tau_yX.update({
                y : [self.model_params["tau_syn_in"] if 'I' in X else
                     self.model_params["tau_syn_ex"] for X in self.X]
            })

        #synaptic delay parameters, loc and scale is mean and std for every
        #network population, negative values will be removed
        self.synDelayLoc, self.synDelayScale = self._synDelayParams()

         
        # Define electrode geometry corresponding to a laminar electrode,
        # where contact points have a radius r, surface normal vectors N,
        # and LFP calculated as the average LFP in n random points on
        # each contact. Recording electrode emulate NeuroNexus array,
        # contact 0 is superficial
        self.electrodeParams = {
            #contact locations:
            'x' : np.zeros(16),
            'y' : np.zeros(16),
            'z' : -np.mgrid[0:16] * 100,
            #extracellular conductivity:
            'sigma' : 0.3,
            #contact surface normals, radius, n-point averaging
            'N' : np.array([[1, 0, 0]]*16),
            'r' : 7.5,
            'n' : 50,
            'seedvalue' : None,
            #dendrite line sources, soma sphere source (Linden2014)
            'method' : 'som_as_point',
            #no somas within the constraints of the "electrode shank":
            'r_z': np.array([[-1E199, -1600, -1550, 1E99],[0, 0, 10, 10]]),
        }
        
        
        #these variables will be saved to file for each cell and electrdoe object
        self.savelist = [
            'somav',
            'timeres_NEURON',
            'timeres_python',
            'somapos',
            'x',
            'y',
            'z',
            'LFP',
            'CSD',
            'morphology',
            'default_rotation',
            'electrodecoeff',
        ]
        
        
        #########################################
        # MISC                                  #
        #########################################
        
        #time resolution of downsampled data in ms
        self.dt_output = 1.
        
        #set fraction of neurons from population which LFP output is stored
        self.recordSingleContribFrac = 0.


    def get_GIDs(self):
        GIDs = {}
        ind = 1
        for i, (X, N_X) in enumerate(zip(self.X, self.N_X)):
            GIDs[X] = [ind, N_X]
            ind += N_X
        return GIDs


    def _synDelayParams(self):
        '''
        set up the detailed synaptic delay parameters,
        loc is mean delay,
        scale is std with low bound cutoff,
        assumes numpy.random.normal is used later
        '''
        delays = {}
        #mean delays
        loc = np.zeros((len(self.y), len(self.X)))
        loc[:, 0] = self.delays[0]
        loc[:, 1::2] = self.delays[0]
        loc[:, 2::2] = self.delays[1]
        #standard deviations
        scale = loc * self.delay_rel_sd
        
        #prepare output
        delay_loc = {}
        for i, y in enumerate(self.y):
            delay_loc.update({y : loc[i]})
        
        delay_scale = {}
        for i, y in enumerate(self.y):
            delay_scale.update({y : scale[i]})
                
        return delay_loc, delay_scale


    def _calcDepths(self):
        '''
        return the cortical depth of each subpopulation
        '''
        depths = self.layerBoundaries.mean(axis=1)[1:]

        depth_y = []
        for y in self.y:
            if y in ['p23', 'b23', 'nb23']:
                depth_y = np.r_[depth_y, depths[0]]
            elif y in ['p4', 'ss4(L23)', 'ss4(L4)', 'b4', 'nb4']:
                depth_y = np.r_[depth_y, depths[1]]
            elif y in ['p5(L23)', 'p5(L56)', 'b5', 'nb5']:
                depth_y = np.r_[depth_y, depths[2]]
            elif y in ['p6(L4)', 'p6(L56)', 'b6', 'nb6']:
                depth_y = np.r_[depth_y, depths[3]]
            else:
                raise Exception, 'this aint right'
                
        return depth_y


    def _yCellParams(self):
        '''
        Return dict with parameters for each population.
        The main operation is filling in cell type specific morphology
        '''
        #cell type specific parameters going into LFPy.Cell        
        yCellParams = {}
        for layer, morpho, _, _ in self.y_zip_list:
            yCellParams.update({layer : self.cellParams.copy()})
            yCellParams[layer].update({
                'morphology' : os.path.join(self.PATH_m_y, morpho),
            })
        return yCellParams
 
        
if __name__ == '__main__':
    params = multicompartment_params()
     
    print dir(params)
