#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis module for example files
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from mpi4py import MPI
import glob

import hybridLFPy.helpers as hlp

###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

from cellsim16popsParams_modified_spontan import multicompartment_params as params_modified_spontan
from cellsim16popsParams_modified_ac_input import multicompartment_params as params_modified_ac_input

import analysis_params
ana_params = analysis_params.params()


######################################
### ANALYSIS FUNCTIONS             ###
######################################

def create_analysis_dir(params):
    if RANK == 0:
        destination = os.path.join(params.savefolder, ana_params.analysis_folder)
        if not os.path.isdir(destination):
            os.mkdir(destination)
    #sync
    COMM.Barrier()


def create_downsampled_data(params):
    '''
    Creates one CSD or LFP file with downsampled data per cell type
    '''
    maxsamples = 1

    for data_type in ['LFP','CSD']:

        if RANK == 0:
            if not os.path.isdir(os.path.join(params.savefolder, 'populations', 'subsamples')):
                os.mkdir((os.path.join(params.savefolder,'populations','subsamples')))
        COMM.Barrier()

        try:
            assert(ana_params.scaling <= params.recordSingleContribFrac)
        except AssertionError:
            raise AssertionError('scaling parameter must be less than simulation recordSingleContribFrac')
        
        samples = int(1. / ana_params.scaling)
        if samples > maxsamples:
            samples = maxsamples
        
        COUNTER = 0
        for j, layer in enumerate(params.y_in_Y): # loop over layers
            for k, pop in enumerate(layer): # loop over populations
                for i, y in enumerate(pop): # loop over cell types
                    if COUNTER % SIZE == RANK:
                        # Load data
                        fname = os.path.join(params.savefolder, 'populations', '%s_%ss.h5' \
                                             % (y, data_type))
                        f = h5py.File(fname)
                        print(('Load %s' % str(f.filename)))
                        raw_data = f['data'][()]
                        srate = f['srate'][()]
                        f.close()

                        ## shuffle data
                        #np.random.shuffle(raw_data)

                        # sample size
                        N = int(params.N_y[np.array(params.y) == y]*ana_params.scaling)
                        try:
                            assert(N <= raw_data.shape[0])
                        except AssertionError:
                            raise AssetionError('shape mismatch with sample size')


                        for sample in range(samples): # loop over samples

                            # slice data
                            data = raw_data[sample*N:(sample+1)*N]

                            # create cell resolved file
                            fname = os.path.join(params.savefolder,'populations','subsamples',
                                                 '%s_%ss_%i_%i.h5' \
                                                 % (y, data_type, ana_params.scaling*100, sample))
                            f = h5py.File(fname, 'w')
                            print(('Write %s' % str(f.filename)))
                            f['data'] = data
                            f['srate'] = srate
                            f.close()

                            # create cell type resolved file
                            fname = os.path.join(params.savefolder,'populations','subsamples',
                                                 '%s_population_%s_%i_%i.h5' \
                                                 % (y,data_type,ana_params.scaling*100,sample))
                            f = h5py.File(fname, 'w')
                            print(('Write %s' % str(f.filename)))
                            f['data'] = data.sum(axis=0)
                            f['srate'] = srate
                            f.close()

                    COUNTER += 1

        COMM.Barrier()
        f = h5py.File(os.path.join(params.savefolder,'populations', '%s_%ss.h5' % (y,data_type)), 'r')
        datashape = f['data'].shape
        f.close()

        COUNTER = 0
        for sample in range(samples): # loop over samples
            if COUNTER % SIZE == RANK:
                # initialize full sum signal
                data_full = np.zeros(datashape[1:])
                for j,layer in enumerate(params.y_in_Y): # loop over layers
                    for k,pop in enumerate(layer): # loop over populations

                        # initialize population resolved sum signal
                        data_Y = np.zeros(datashape[1:])

                        for i,y in enumerate(pop): # loop over cell types

                            # Load data
                            fname = os.path.join(params.savefolder, 'populations',
                                                 'subsamples', '%s_population_%s_%i_%i.h5' \
                                                 % (y, data_type, ana_params.scaling*100,
                                                    sample))
                            f = h5py.File(fname, 'r')    
                            # Update population sum:
                            data_Y += f['data'][()]
                            srate = f['srate'][()]
                            f.close()

                        # write population sum
                        fname = os.path.join(params.savefolder,'populations','subsamples',
                                             '%s_population_%s_%i_%i.h5' \
                                             % (params.Y[2*j+k], data_type,
                                                ana_params.scaling*100, sample))
                        f = h5py.File(fname,'w')
                        print(('Write %s' % str(f.filename)))
                        f['data'] = data_Y
                        f['srate'] = srate
                        f.close() 

                        # update full sum
                        data_full += data_Y

                # write sum
                fname = os.path.join(params.savefolder,'populations','subsamples',
                                     '%ssum_%i_%i.h5' % (data_type,ana_params.scaling*100,sample))
                f = h5py.File(fname,'w')
                print(('Write %s' % str(f.filename)))
                f['data'] = data_full
                f['srate'] = srate
                f.close()

            COUNTER += 1

        COMM.Barrier()
        


def calc_signal_power(params):

    '''
    calculates power spectrum of sum signal for all channels

    '''

    for i, data_type in enumerate(['CSD','LFP','CSD_10_0', 'LFP_10_0']):
        if i % SIZE == RANK:

            # Load data
            if data_type in ['CSD','LFP']:
                fname=os.path.join(params.savefolder, data_type+'sum.h5')
            else:
                fname=os.path.join(params.populations_path, 'subsamples',
                                   str.split(data_type,'_')[0] + 'sum_' +
                                   str.split(data_type,'_')[1] + '_' +
                                   str.split(data_type,'_')[2] + '.h5')
            #open file
            f = h5py.File(fname)
            data = f['data'][()]
            srate = f['srate'][()] 
            tvec = np.arange(data.shape[1]) * 1000. / srate
        
            # slice
            slica = (tvec >= ana_params.transient)
            data = data[:,slica]
    
            # subtract mean
            dataT = data.T - data.mean(axis=1)
            data = dataT.T
            f.close()
    
            #extract PSD
            PSD=[]
            for i in np.arange(len(params.electrodeParams['z'])):
                if ana_params.mlab:
                    Pxx, freqs=plt.mlab.psd(data[i], NFFT=ana_params.NFFT,
                                        Fs=srate, noverlap=ana_params.noverlap,
                                        window=ana_params.window)
                else:
                    [freqs, Pxx] = hlp.powerspec([data[i]], tbin= 1.,
                                             Df=ana_params.Df, pointProcess=False)
                    mask = np.where(freqs >= 0.)
                    freqs = freqs[mask]
                    Pxx = Pxx.flatten()
                    Pxx = Pxx[mask]
                    Pxx = Pxx/tvec[tvec >= ana_params.transient].size**2
                PSD +=[Pxx.flatten()]
                
            PSD=np.array(PSD)
    
            # Save data
            f = h5py.File(os.path.join(params.savefolder, ana_params.analysis_folder,
                                       data_type + ana_params.fname_psd),'w')
            f['freqs']=freqs
            f['psd']=PSD
            f['transient']=ana_params.transient
            f['mlab']=ana_params.mlab
            f['NFFT']=ana_params.NFFT
            f['noverlap']=ana_params.noverlap
            f['window']=str(ana_params.window)
            f['Df']=str(ana_params.Df)
            f.close()

    return 


def calc_uncorrelated_signal_power(params):
    
    '''This function calculates the depth-resolved power spectrum of signals
    without taking into account any cross-correlation.'''


    for i, data_type in enumerate(['LFP','CSD']):
        if i % SIZE == RANK:
    
            # Determine size of PSD matrix
    
            f = h5py.File(os.path.join(params.savefolder, data_type + 'sum.h5'),'r')
            data = f['data'][()]
            srate = f['srate'][()]
            if ana_params.mlab:
                Psum, freqs = plt.mlab.psd(data[0], NFFT=ana_params.NFFT, Fs=srate,
                                    noverlap=ana_params.noverlap, window=ana_params.window)
            else:
                [freqs, Psum] = hlp.powerspec([data[0]], tbin= 1./srate*1000.,
                                       Df=ana_params.Df, pointProcess=False)
            f.close()
            P = np.zeros((data.shape[0],Psum.shape[0]))
               
            for y in params.y:
    
                print(('processing ', y))
                # Load data
                f = h5py.File(os.path.join(params.populations_path, '%s_%ss' %
                                           (y,data_type) + '.h5'),'r')
                data_y = f['data'][()][:,:, ana_params.transient:]
                # subtract mean
                for j in range(len(data_y)):
                    data_yT = data_y[j].T - data_y[j].mean(axis=1)
                    data_y[j] = data_yT.T
                srate = f['srate'][()]
                tvec = np.arange(data_y.shape[2]) * 1000. / srate
                f.close()
    
            
                for j in range(len(data_y)): # loop over cells
                    if ana_params.mlab:
                        for ch in range(len(params.electrodeParams['z'])): # loop over channels
                            P_j_ch, freqs = plt.mlab.psd(data_y[j,ch],
                                                         NFFT=ana_params.NFFT, Fs=srate,
                                                         noverlap=ana_params.noverlap,
                                                         window=ana_params.window)
                                       
                            P[ch] += P_j_ch
                    else:
                        [freqs, P_j] = hlp.powerspec(data_y[j], tbin= 1./srate*1000.,
                                                     Df=ana_params.Df, pointProcess=False)
                        mask = np.where(freqs >= 0.)
                        freqs = freqs[mask]
                        P_j = P_j[:,mask][:,0,:]
                        P_j = P_j/tvec[tvec >= ana_params.transient].size**2
                        
                        P += P_j
    
            #rescale PSD as they may be computed from a fraction of single cell LFPs
            P /= params.recordSingleContribFrac
            
            # Save data
            f = h5py.File(os.path.join(params.savefolder, ana_params.analysis_folder,
                                       data_type +  ana_params.fname_psd_uncorr),'w')
            f['freqs']=freqs
            f['psd']=P
            f['transient']=ana_params.transient
            f['mlab']=ana_params.mlab
            f['NFFT']=ana_params.NFFT
            f['noverlap']=ana_params.noverlap
            f['window']=str(ana_params.window)
            f['Df']=str(ana_params.Df)        
            f.close()

    return
 

def calc_variances(params):
    '''
    This function calculates the variance of the sum signal and all population-resolved signals
    '''

    depth = params.electrodeParams['z']

    ############################
    ### CSD                  ###
    ############################
 
    for i, data_type in enumerate(['CSD','LFP']):
        if i % SIZE == RANK:
    
            f_out = h5py.File(os.path.join(params.savefolder, ana_params.analysis_folder,
                                           data_type + ana_params.fname_variances), 'w')
            f_out['depths']=depth
      
            for celltype in params.y:
                f_in = h5py.File(os.path.join(params.populations_path,
                                              '%s_population_%s' % (celltype,data_type) + '.h5' ))
                var = f_in['data'][()][:, ana_params.transient:].var(axis=1)
                f_in.close()
                f_out[celltype]= var
            
            f_in = h5py.File(os.path.join(params.savefolder, data_type + 'sum.h5' ))
            var= f_in['data'][()][:, ana_params.transient:].var(axis=1)
            f_in.close()
            f_out['sum']= var
        
            f_out.close()

    return


if __name__ == '__main__':
    params = params_modified_spontan()
    create_analysis_dir(params)
    create_downsampled_data(params)
    calc_signal_power(params)
    calc_uncorrelated_signal_power(params)
    calc_variances(params)

    params = params_modified_ac_input()
    create_analysis_dir(params)
    create_downsampled_data(params)
    calc_signal_power(params)
    calc_uncorrelated_signal_power(params)
    calc_variances(params)

