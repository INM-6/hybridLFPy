#!/usr/bin/env python
# -*- coding: utf-8 -*-
from builtins import open, zip
import os
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
import plotting_helpers as phlp
from plot_methods import getMeanInpCurrents, getMeanVoltages, plot_population, plot_signal_sum
from cellsim16popsParams_modified_spontan import multicompartment_params 
import analysis_params
from hybridLFPy import CachedNetwork, helpers
import pickle as pickle


def plot_multi_scale_output_a(fig):    
    #get the mean somatic currents and voltages,
    #write pickles if they do not exist:
    if not os.path.isfile(os.path.join(params.savefolder, 'data_analysis',
                                       'meanInpCurrents.pickle')):
        meanInpCurrents = getMeanInpCurrents(params, params.n_rec_input_spikes,
                                        os.path.join(params.spike_output_path,
                                                     'population_input_spikes'))
        f = open(os.path.join(params.savefolder, 'data_analysis',
                              'meanInpCurrents.pickle'), 'wb')
        pickle.dump(meanInpCurrents, f)
        f.close()
    else:
        f = open(os.path.join(params.savefolder, 'data_analysis',
                              'meanInpCurrents.pickle'), 'rb')
        meanInpCurrents = pickle.load(f)
        f.close()

    if not os.path.isfile(os.path.join(params.savefolder, 'data_analysis',
                                       'meanVoltages.pickle')):
        meanVoltages = getMeanVoltages(params, params.n_rec_voltage,
                                       os.path.join(params.spike_output_path,
                                                       'voltages'))
        f = open(os.path.join(params.savefolder, 'data_analysis',
                              'meanVoltages.pickle'), 'wb')
        pickle.dump(meanVoltages, f)
        f.close()
    else:
        f = open(os.path.join(params.savefolder, 'data_analysis',
                              'meanVoltages.pickle'), 'rb')
        meanVoltages = pickle.load(f)
        f.close()
    

    #load spike as database
    networkSim = CachedNetwork(**params.networkSimParams)
    if analysis_params.bw:
        networkSim.colors = phlp.get_colors(len(networkSim.X))

    show_ax_labels = True
    show_insets = False

    transient=200
    T=[800, 1000]
    T_inset=[900, 920]

    sep = 0.025/2 #0.017
    
    left = 0.075
    bottom = 0.55
    top = 0.975
    right = 0.95
    axwidth = 0.16
    numcols = 4
    insetwidth = axwidth/2
    insetheight = 0.5
    
    lefts = np.linspace(left, right-axwidth, numcols)
    
    
    #fig = plt.figure()
    ############################################################################ 
    # A part, plot spike rasters
    ############################################################################
    ax1 = fig.add_axes([lefts[0], bottom, axwidth, top-bottom])
    #fig.text(0.005,0.95,'a',fontsize=8, fontweight='demibold')
    if show_ax_labels:
        phlp.annotate_subplot(ax1, ncols=4, nrows=1.02, letter='A', )
    ax1.set_title('network activity')
    plt.locator_params(nbins=4)
    
    x, y = networkSim.get_xy(T, fraction=1)
    networkSim.plot_raster(ax1, T, x, y, markersize=0.2, marker='_', alpha=1.,
                           legend=False, pop_names=True, rasterized=False)
    phlp.remove_axis_junk(ax1)
    ax1.set_xlabel(r'$t$ (ms)', labelpad=0.1)
    ax1.set_ylabel('population', labelpad=0.1)
    
    # Inset
    if show_insets:
        ax2 = fig.add_axes([lefts[0]+axwidth-insetwidth, top-insetheight, insetwidth, insetheight])
        plt.locator_params(nbins=4)
        x, y = networkSim.get_xy(T_inset, fraction=0.4)
        networkSim.plot_raster(ax2, T_inset, x, y, markersize=0.25, alpha=1.,
                               legend=False)
        phlp.remove_axis_junk(ax2)
        ax2.set_xticks(T_inset)
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_ylabel('')
        ax2.set_xlabel('')

    
    ############################################################################
    # B part, plot firing rates
    ############################################################################
    
    nrows = len(networkSim.X)-1
    high = top
    low = bottom
    thickn = (high-low) / nrows - sep
    bottoms = np.linspace(low, high-thickn, nrows)[::-1]
    
    x, y = networkSim.get_xy(T, fraction=1)  
    
    #dummy ax to put label in correct location
    ax_ = fig.add_axes([lefts[1], bottom, axwidth, top-bottom])
    ax_.axis('off')
    if show_ax_labels:
        phlp.annotate_subplot(ax_, ncols=4, nrows=1, letter='B')        
    
    for i, X in enumerate(networkSim.X[:-1]):
        ax3 = fig.add_axes([lefts[1], bottoms[i], axwidth, thickn])
        plt.locator_params(nbins=4)
        phlp.remove_axis_junk(ax3)
        networkSim.plot_f_rate(ax3, X, i, T, x, y, yscale='linear',
                               plottype='fill_between', show_label=False,
                               rasterized=False)
        ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
        if i != nrows -1:    
            ax3.set_xticklabels([])
    
        if i == 3:
            ax3.set_ylabel(r'(s$^{-1}$)', labelpad=0.1)
    
        if i == 0:
            ax3.set_title(r'firing rates ')

        ax3.text(0, 1, X,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax3.transAxes)
        
    for loc, spine in ax3.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')
    ax3.set_xlabel(r'$t$ (ms)', labelpad=0.1)

      
    ############################################################################
    # C part, plot somatic synapse input currents population resolved 
    ############################################################################
        
    #set up subplots
    nrows = len(list(meanInpCurrents.keys()))
    high = top
    low = bottom
    thickn = (high-low) / nrows - sep
    bottoms = np.linspace(low, high-thickn, nrows)[::-1]

    ax_ = fig.add_axes([lefts[2], bottom, axwidth, top-bottom])
    ax_.axis('off')
    if show_ax_labels:
        phlp.annotate_subplot(ax_, ncols=4, nrows=1, letter='C')        
    
    for i, Y in enumerate(params.Y):
        value = meanInpCurrents[Y]
        
        tvec = value['tvec']
        inds = (tvec <= T[1]) & (tvec >= T[0])
        ax3 = fig.add_axes([lefts[2], bottoms[i], axwidth, thickn])
        plt.locator_params(nbins=4)

        if i == 0:
            ax3.plot(tvec[inds][::10],
                     helpers.decimate(value['E'][inds], 10),
                     'k' if analysis_params.bw else analysis_params.colorE, #lw=0.75, #'r',
                     rasterized=False,label='exc.')
            ax3.plot(tvec[inds][::10],
                     helpers.decimate(value['I'][inds], 10),
                     'gray' if analysis_params.bw else analysis_params.colorI, #lw=0.75, #'b',
                     rasterized=False,label='inh.')
            ax3.plot(tvec[inds][::10],
                     helpers.decimate(value['E'][inds] + value['I'][inds], 10),
                     'k', lw=1, rasterized=False, label='sum')
        else:
            ax3.plot(tvec[inds][::10], helpers.decimate(value['E'][inds], 10),
                     'k' if analysis_params.bw else analysis_params.colorE, #lw=0.75, #'r',
                     rasterized=False)
            ax3.plot(tvec[inds][::10], helpers.decimate(value['I'][inds], 10),
                     'gray' if analysis_params.bw else analysis_params.colorI, #lw=0.75, #'b',
                     rasterized=False)
            ax3.plot(tvec[inds][::10],
                     helpers.decimate(value['E'][inds] + value['I'][inds], 10),
                     'k', lw=1, rasterized=False)
        phlp.remove_axis_junk(ax3)

        
        ax3.axis(ax3.axis('tight'))
        ax3.set_yticks([ax3.axis()[2], 0, ax3.axis()[3]])
        ax3.set_yticklabels([np.round((value['I'][inds]).min(), decimals=1),
                             0,
                             np.round((value['E'][inds]).max(), decimals=1)])

        
        ax3.text(0, 1, Y,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax3.transAxes)        
    
        if i == nrows-1:
            ax3.set_xlabel('$t$ (ms)', labelpad=0.1)
        else:
            ax3.set_xticklabels([])
        
        if i == 3:
            ax3.set_ylabel(r'(nA)', labelpad=0.1)
    
        if i == 0:
            ax3.set_title('input currents')
            ax3.legend(loc=1,prop={'size':4})
        phlp.remove_axis_junk(ax3)
        ax3.set_xlim(T)
        


    ############################################################################
    # D part, plot membrane voltage population resolved 
    ############################################################################
        
    nrows = len(list(meanVoltages.keys()))    
    high = top
    low = bottom
    thickn = (high-low) / nrows - sep
    bottoms = np.linspace(low, high-thickn, nrows)[::-1]
    
    colors = phlp.get_colors(len(params.Y)) 

    ax_ = fig.add_axes([lefts[3], bottom, axwidth, top-bottom])
    ax_.axis('off')
    if show_ax_labels:
        phlp.annotate_subplot(ax_, ncols=4, nrows=1, letter='D')        
    
    for i, Y in enumerate(params.Y):
        value = meanVoltages[Y]
        
        tvec = value['tvec']
        inds = (tvec <= T[1]) & (tvec >= T[0])
        
        ax4 = fig.add_axes([lefts[3], bottoms[i], axwidth, thickn])
        ax4.plot(tvec[inds][::10], helpers.decimate(value['data'][inds], 10), color=colors[i],
                 zorder=0, rasterized=False)
                
        
        phlp.remove_axis_junk(ax4)
        
        plt.locator_params(nbins=4)
        
        ax4.axis(ax4.axis('tight'))
        ax4.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        ax4.text(0, 1, Y,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax4.transAxes)        
    
        if i == nrows-1:
            ax4.set_xlabel('$t$ (ms)', labelpad=0.1)
        else:
            ax4.set_xticklabels([])
        
        if i == 3:
            ax4.set_ylabel(r'(mV)', labelpad=0.1)
    
        if i == 0:
            ax4.set_title('voltages')
        
        ax4.set_xlim(T)
    


def plot_multi_scale_output_b(fig, X='L5E'):
    '''docstring me'''

    show_ax_labels = True
    show_insets = False
    show_images = False

    T=[800, 1000]
    T_inset=[900, 920]

    
    left = 0.075
    bottom = 0.05
    top = 0.475
    right = 0.95
    axwidth = 0.16
    numcols = 4
    insetwidth = axwidth/2
    insetheight = 0.5
    
    lefts = np.linspace(left, right-axwidth, numcols)
    lefts += axwidth/2



    #lower row of panels
    #fig = plt.figure()
    #fig.subplots_adjust(left=0.12, right=0.9, bottom=0.36, top=0.9, wspace=0.2, hspace=0.3)

    ############################################################################    
    # E part, soma locations
    ############################################################################

    ax4 = fig.add_axes([lefts[0], bottom, axwidth, top-bottom], frameon=False)
    plt.locator_params(nbins=4)
    ax4.xaxis.set_ticks([])
    ax4.yaxis.set_ticks([])
    if show_ax_labels:
        phlp.annotate_subplot(ax4, ncols=4, nrows=1, letter='E')
    plot_population(ax4, params, isometricangle=np.pi/24, rasterized=False)
    
    
    ############################################################################    
    # F part, CSD
    ############################################################################

    ax5 = fig.add_axes([lefts[1], bottom, axwidth, top-bottom])
    plt.locator_params(nbins=4)
    phlp.remove_axis_junk(ax5)
    if show_ax_labels:
        phlp.annotate_subplot(ax5, ncols=4, nrows=1, letter='F')
    plot_signal_sum(ax5, params, fname=os.path.join(params.savefolder, 'LaminarCurrentSourceDensity_sum.h5'),
                        unit='$\mu$A mm$^{-3}$',
                        T=T,
                        ylim=[ax4.axis()[2], ax4.axis()[3]],
                        rasterized=False)
    ax5.set_title('CSD', va='center')
    
    # Inset
    if show_insets:
        ax6 = fig.add_axes([lefts[1]+axwidth-insetwidth, top-insetheight, insetwidth, insetheight])
        plt.locator_params(nbins=4)
        phlp.remove_axis_junk(ax6)
        plot_signal_sum_colorplot(ax6, params, os.path.join(params.savefolder, 'LaminarCurrentSourceDensity_sum.h5'),
                                  unit=r'$\mu$Amm$^{-3}$', T=T_inset,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='bwr_r')
        ax6.set_xticks(T_inset)
        ax6.set_yticklabels([])

    #show traces superimposed on color image
    if show_images:
        plot_signal_sum_colorplot(ax5, params, os.path.join(params.savefolder, 'LaminarCurrentSourceDensity_sum.h5'),
                                  unit=r'$\mu$Amm$^{-3}$', T=T,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='jet_r')
        

    
    ############################################################################
    # G part, LFP 
    ############################################################################

    ax7 = fig.add_axes([lefts[2], bottom, axwidth, top-bottom])
    plt.locator_params(nbins=4)
    if show_ax_labels:
        phlp.annotate_subplot(ax7, ncols=4, nrows=1, letter='G')
    phlp.remove_axis_junk(ax7)
    plot_signal_sum(ax7, params, fname=os.path.join(params.savefolder, 'RecExtElectrode_sum.h5'),
                    unit='mV', T=T, ylim=[ax4.axis()[2], ax4.axis()[3]],
                    rasterized=False)
    ax7.set_title('LFP',va='center')
    
    # Inset
    if show_insets:
        ax8 = fig.add_axes([lefts[2]+axwidth-insetwidth, top-insetheight, insetwidth, insetheight])
        plt.locator_params(nbins=4)
        phlp.remove_axis_junk(ax8)
        plot_signal_sum_colorplot(ax8, params, os.path.join(params.savefolder, 'RecExtElectrode_sum.h5'),
                                  unit='mV', T=T_inset,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='bwr_r')   
        ax8.set_xticks(T_inset)
        ax8.set_yticklabels([])

    #show traces superimposed on color image
    if show_images:
        plot_signal_sum_colorplot(ax7, params, os.path.join(params.savefolder, 'RecExtElectrode_sum.h5'),
                                  unit='mV', T=T,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='bwr_r')





if __name__ == '__main__':
    params = multicompartment_params()
    ana_params = analysis_params.params()
    ana_params.set_PLOS_2column_fig_style(ratio=1)

    plt.close('all')

    savefolders = [
        'simulation_output_modified_spontan'
    ]

    for i, savefolder in enumerate(savefolders):
        # path to simulation files
        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path

        fig = plt.figure()
        plot_multi_scale_output_a(fig)        
        plot_multi_scale_output_b(fig)
        fig.savefig('figure_06.pdf', dpi=300,
                    bbox_inches='tight', pad_inches=0, compression=9)
        fig.savefig('figure_06.eps',
                    bbox_inches='tight', pad_inches=0.01)
        
    plt.show()

