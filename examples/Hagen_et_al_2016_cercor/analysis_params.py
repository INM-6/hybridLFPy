#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared parameters for analysis
"""

import matplotlib.pyplot as plt

# global flag for black and white figures
bw = False
colorE = 'r'
colorI = 'b'
colorP = 'r'


class params():
    def __init__(self,
                 textsize=6,
                 titlesize=7,):
        
        # label for merged spike files
        self.pop_spike_file_label = ['population_spikes-th.gdf',
                                     'population_spikes-0-0.gdf',
                                     'population_spikes-0-1.gdf',
                                     'population_spikes-1-0.gdf',
                                     'population_spikes-1-1.gdf',
                                     'population_spikes-2-0.gdf',
                                     'population_spikes-2-1.gdf',
                                     'population_spikes-3-0.gdf',
                                     'population_spikes-3-1.gdf'] 

        # start-up transient that is cut-off
        self.cutoff = 0 #200.

        # bin size
        self.tbin = 1.

        # frequency filtering
        self.Df = 1.


        self.analysis_folder = 'data_analysis'
        self.fname_psd = 'psd.h5'
        self.fname_psd_uncorr = 'psd_uncorr.h5'
        self.fname_meanInpCurrents = 'meanInpCurrents.h5'
        self.fname_meanVoltages = 'meanVoltages.h5'
        self.fname_variances = 'variances.h5'

        self.scaling = 0.1


        self.transient = 200
        self.Df = None
        self.mlab = True
        self.NFFT = 256
        self.noverlap = 128
        self.window = plt.mlab.window_hanning


        self.PLOSwidth1Col = 3.27  # in inches
        self.PLOSwidth2Col= 6.83 
        self.inchpercm = 2.54
        self.frontierswidth=8.5 
        self.PLOSwidth= 6.83 
        self.textsize = textsize
        self.titlesize = titlesize

        ##global colormap for populations
        #self.bw = True

        #default plot parameters goes here, may be overridden by below functions        
        plt.rcdefaults()
        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm,
                                self.frontierswidth/self.inchpercm],
            'figure.dpi' : 160,
            'xtick.labelsize' : self.textsize,
            'ytick.labelsize' : self.textsize,
            'font.size' : self.textsize,
            'axes.labelsize' : self.textsize,
            'axes.titlesize' : self.titlesize,
            'axes.linewidth': 0.75,
            'lines.linewidth': 0.5,
            'legend.fontsize' : self.textsize / 1.25,
            'savefig.dpi' : 300
        })
        
        # Set default plotting parameters
        self.set_default_fig_style()



    def set_default_fig_style(self):
        '''default figure size'''
        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm, self.frontierswidth/self.inchpercm],
        })


    def set_large_fig_style(self):
        '''twice width figure size'''
        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm*2, self.frontierswidth/self.inchpercm],
        })   

    def set_broad_fig_style(self):
        '''4 times width, 1.5 times height'''
        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm*4, self.frontierswidth/self.inchpercm*1.5],
        })   


    def set_enormous_fig_style(self):
        '''2 times width, 2 times height'''

        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm*2, self.frontierswidth/self.inchpercm*2],
        })   


    def set_PLOS_1column_fig_style(self, ratio=1):
        '''figure size corresponding to Plos 1 column'''
        plt.rcParams.update({
            'figure.figsize' : [self.PLOSwidth1Col,self.PLOSwidth1Col*ratio],
        })


    def set_PLOS_2column_fig_style(self, ratio=1):
        '''figure size corresponding to Plos 2 columns'''
        plt.rcParams.update({
            'figure.figsize' : [self.PLOSwidth2Col, self.PLOSwidth2Col*ratio],
        })



