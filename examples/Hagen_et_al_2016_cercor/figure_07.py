#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotting_helpers as phlp
from plot_methods import plot_signal_sum, plot_signal_sum_colorplot
from cellsim16popsParams_modified_regular_input import multicompartment_params 
import analysis_params
from hybridLFPy import CachedNetwork
    

if __name__ == '__main__':

    params = multicompartment_params()
    ana_params = analysis_params.params()
    ana_params.set_PLOS_2column_fig_style(ratio=0.5)

    params.figures_path = os.path.join(params.savefolder, 'figures')
    params.spike_output_path = os.path.join(params.savefolder,
                                                       'processed_nest_output')
    params.networkSimParams['spike_output_path'] = params.spike_output_path

    #load spike as database
    networkSim = CachedNetwork(**params.networkSimParams)
    if analysis_params.bw:
        networkSim.colors = phlp.get_colors(len(networkSim.X))




    transient=200
    T=[890, 920]
    
    show_ax_labels = True
    show_images = True
    # show_images = False if analysis_params.bw else True
    
    gs = gridspec.GridSpec(9,4)
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.075, top=0.925, hspace=0.35, wspace=0.35)
    ############################################################################ 
    # A part, plot spike rasters
    ############################################################################
    ax1 = fig.add_subplot(gs[:, 0])
    if show_ax_labels:
        phlp.annotate_subplot(ax1, ncols=1, nrows=1, letter='A', linear_offset=0.065)
    ax1.set_title('spiking activity')
    plt.locator_params(nbins=4)
    
    x, y = networkSim.get_xy(T, fraction=1)
    networkSim.plot_raster(ax1, T, x, y,
                           markersize=0.2,
                           marker='_',
                           alpha=1.,
                           legend=False, pop_names=True,
                           rasterized=False)
    a = ax1.axis()
    ax1.vlines(x['TC'][0], a[2], a[3], 'k', lw=0.25)
    phlp.remove_axis_junk(ax1)
    ax1.set_xlabel(r'$t$ (ms)', )
    ax1.set_ylabel('population', labelpad=0.1)
    


    ############################################################################
    # B part, plot firing rates
    ############################################################################
    # create invisible axes to position labels correctly
    ax_ = fig.add_subplot(gs[:, 1])
    if show_ax_labels:    
        phlp.annotate_subplot(ax_, ncols=1, nrows=1, letter='B', linear_offset=0.065)
    ax_.set_title('firing rates ')   
    ax_.axis('off')

    x, y = networkSim.get_xy(T, fraction=1)  
    colors = ['k'] + phlp.get_colors(len(params.Y))      
    
    for i, X in enumerate(networkSim.X):
        ax3 = fig.add_subplot(gs[i, 1])
        plt.locator_params(nbins=4)
        phlp.remove_axis_junk(ax3)
        networkSim.plot_f_rate(ax3, X, i, T, x, y, yscale='linear',
                               plottype='bar')
        ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
        if i != len(networkSim.X) -1:    
            ax3.set_xticklabels([])
    
        if i == 4:
            ax3.set_ylabel(r'(s$^{-1}$)', labelpad=0.1)

        ax3.text(0.02, 1, X,
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax3.transAxes)
    
  
    for loc, spine in ax3.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')
    ax3.set_xlabel(r'$t$ (ms)', labelpad=0.1)


    ############################################################################    
    # C part, CSD
    ############################################################################
    ax5 = fig.add_subplot(gs[:, 2])
    plt.locator_params(nbins=4)
    phlp.remove_axis_junk(ax5)
    if show_ax_labels:
        phlp.annotate_subplot(ax5, ncols=1, nrows=1, letter='C', linear_offset=0.065)
    plot_signal_sum(ax5, params,
                    fname=os.path.join(params.savefolder, 'CSDsum.h5'),
                    unit='$\mu$A mm$^{-3}$',
                    T=T,
                    ylim=[-1550, 50],
                    rasterized=False)
    ax5.set_title('CSD')
    a = ax5.axis()
    ax5.vlines(x['TC'][0], a[2], a[3], 'k', lw=0.25)
    

    #show traces superimposed on color image
    if show_images:
        im = plot_signal_sum_colorplot(ax5, params, os.path.join(params.savefolder, 'CSDsum.h5'),
                                  unit=r'$\mu$Amm$^{-3}$', T=T,
                                  ylim=[-1550, 50],
                                  fancy=False, colorbar=False,
                                  cmap=plt.get_cmap('gray', 21) if analysis_params.bw else plt.get_cmap('bwr_r', 21),
                                  rasterized=False)

        cb = phlp.colorbar(fig, ax5, im,
                           width=0.05, height=0.5,
                           hoffset=-0.05, voffset=0.5)
        cb.set_label('($\mu$Amm$^{-3}$)', labelpad=0.1)



    ############################################################################
    # D part, LFP 
    ############################################################################
    ax7 = fig.add_subplot(gs[:, 3])
    plt.locator_params(nbins=4)
    if show_ax_labels:
        phlp.annotate_subplot(ax7, ncols=1, nrows=1, letter='D', linear_offset=0.065)
    phlp.remove_axis_junk(ax7)
    plot_signal_sum(ax7, params,
                    fname=os.path.join(params.savefolder, 'LFPsum.h5'),
                    unit='mV', T=T, ylim=[-1550, 50],
                    rasterized=False)
    ax7.set_title('LFP ')
    a = ax7.axis()
    ax7.vlines(x['TC'][0], a[2], a[3], 'k', lw=0.25)
    

    #show traces superimposed on color image
    if show_images:
        im = plot_signal_sum_colorplot(ax7, params, os.path.join(params.savefolder, 'LFPsum.h5'),
                                  unit='mV', T=T,
                                  ylim=[-1550, 50],
                                  fancy=False, colorbar=False,
                                  cmap=plt.get_cmap('gray', 21) if analysis_params.bw else plt.get_cmap('RdBu', 21),
                                  rasterized=False)
                                  
        cb = phlp.colorbar(fig, ax7, im,
                           width=0.05, height=0.5,
                           hoffset=-0.05, voffset=0.5)
        cb.set_label('(mV)', labelpad=0.1)

    ax7.set_yticklabels([])
    

    fig.savefig('figure_07.pdf',
                dpi=450, compression=9,
                )
    fig.savefig('figure_07.eps')
    plt.show()
  
