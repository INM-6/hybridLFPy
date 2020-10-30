#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
import os

import plotting_helpers as phlp
import analysis_params

######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

from cellsim16popsParams_modified_spontan import multicompartment_params 


######################################
### IMPORT PANELS                  ###
######################################

from plot_methods import plotPowers, plot_population, plot_signal_sum_colorplot, plot_signal_sum
import analysis_params
from figure_10 import fig_exc_inh_contrib


######################################
### FIGURE                         ###
######################################



    
'''Plot signal (total power) decomposition as function of depth and show single population LFP'''
def fig_lfp_decomposition(fig, axes, params, transient=200, X=['L23E', 'L6E'], show_xlabels=True):
    # ana_params.set_PLOS_2column_fig_style(ratio=0.5)
    # fig, axes = plt.subplots(1,5)
    # fig.subplots_adjust(left=0.06, right=0.96, wspace=0.4, hspace=0.2)
    
    if analysis_params.bw:
        # linestyles = ['-', '-', '--', '--', '-.', '-.', ':', ':']
        linestyles = ['-', '-', '-', '-', '-', '-', '-', '-']
        markerstyles = ['s', 's', 'v', 'v', 'o', 'o', '^', '^']
    else:
        if plt.matplotlib.__version__ == '1.5.x':
            linestyles = ['-', ':']*(len(params.Y) // 2)
            print(('CSD variance semi log plots may fail with matplotlib.__version__ {}'.format(plt.matplotlib.__version__)))
        else:
            linestyles = ['-', (0, (1,1))]*(len(params.Y) // 2) #cercor version
        # markerstyles = ['s', 's', 'v', 'v', 'o', 'o', '^', '^']
        markerstyles = [None]*len(params.Y)
        linewidths = [1.25 for i in range(len(linestyles))]
    
    plt.delaxes(axes[0])
    
    #population plot
    axes[0] = fig.add_subplot(261)
    axes[0].xaxis.set_ticks([])
    axes[0].yaxis.set_ticks([])
    axes[0].set_frame_on(False)
    plot_population(axes[0], params, aspect='tight', isometricangle=np.pi/32,
                           plot_somas = False, plot_morphos = True,
                           num_unitsE = 1, num_unitsI=1,
                           clip_dendrites=False, main_pops=True,
                           rasterized=False)
    phlp.annotate_subplot(axes[0], ncols=5, nrows=1, letter='A')
    axes[0].set_aspect('auto')
    axes[0].set_ylim(-1550, 50)
    axis = axes[0].axis()
    
    
    
    phlp.remove_axis_junk(axes[1])
    plot_signal_sum(axes[1], params,
                    fname=os.path.join(params.populations_path, X[0] + '_population_RecExtElectrode.h5'),
                    unit='mV', T=[800,1000], ylim=[axis[2], axis[3]],
                    rasterized=False)
    
    # CSD background colorplot
    im = plot_signal_sum_colorplot(axes[1], params, os.path.join(params.populations_path,  X[0] + '_population_LaminarCurrentSourceDensity.h5'),
                              unit=r'$\mu$Amm$^{-3}$', T=[800,1000],
                              colorbar=False,
                              ylim=[axis[2], axis[3]], fancy=False,
                              cmap=plt.get_cmap('gray', 21) if analysis_params.bw else plt.get_cmap('bwr_r', 21),
                              rasterized=False)

    cb = phlp.colorbar(fig, axes[1], im,
                       width=0.05, height=0.5,
                       hoffset=-0.05, voffset=0.5)
    cb.set_label('($\mu$Amm$^{-3}$)', labelpad=0.)

    axes[1].set_ylim(-1550, 50)
    axes[1].set_title('LFP and CSD ({})'.format(X[0]), va='baseline')
    phlp.annotate_subplot(axes[1], ncols=3, nrows=1, letter='B')
     
    #quickfix on first axes
    axes[0].set_ylim(-1550, 50)
    if show_xlabels:
        axes[1].set_xlabel(r'$t$ (ms)',labelpad=0.)
    else:
        axes[1].set_xlabel('')
    


    phlp.remove_axis_junk(axes[2])
    plot_signal_sum(axes[2], params,
                    fname=os.path.join(params.populations_path, X[1] + '_population_RecExtElectrode.h5'), ylabels=False,
                    unit='mV', T=[800,1000], ylim=[axis[2], axis[3]],
                    rasterized=False)
    
    # CSD background colorplot
    im = plot_signal_sum_colorplot(axes[2], params, os.path.join(params.populations_path, X[1] + '_population_LaminarCurrentSourceDensity.h5'),
                              unit=r'$\mu$Amm$^{-3}$', T=[800,1000], ylabels=False,
                              colorbar=False,
                              ylim=[axis[2], axis[3]], fancy=False, 
                              cmap=plt.get_cmap('gray', 21) if analysis_params.bw else plt.get_cmap('bwr_r', 21),
                              rasterized=False)

    cb = phlp.colorbar(fig, axes[2], im,
                       width=0.05, height=0.5,
                       hoffset=-0.05, voffset=0.5)
    cb.set_label('($\mu$Amm$^{-3}$)', labelpad=0.)

    axes[2].set_ylim(-1550, 50)
    axes[2].set_title('LFP and CSD ({})'.format(X[1]), va='baseline')
    phlp.annotate_subplot(axes[2], ncols=1, nrows=1, letter='C')
    if show_xlabels:
        axes[2].set_xlabel(r'$t$ (ms)',labelpad=0.)
    else:
        axes[2].set_xlabel('')
    

    plotPowers(axes[3], params, params.Y, 'LaminarCurrentSourceDensity', linestyles=linestyles, transient=transient, markerstyles=markerstyles, linewidths=linewidths)
    axes[3].axis(axes[3].axis('tight'))
    axes[3].set_ylim(-1550, 50)
    axes[3].set_yticks(-np.arange(16)*100)
    if show_xlabels:
        axes[3].set_xlabel(r'$\sigma^2$ ($(\mu$Amm$^{-3})^2$)', va='center')
    axes[3].set_title('CSD variance', va='baseline')
    axes[3].set_xlim(left=1E-7)
    phlp.remove_axis_junk(axes[3])
    phlp.annotate_subplot(axes[3], ncols=1, nrows=1, letter='D')
    
    
    plotPowers(axes[4], params, params.Y, 'RecExtElectrode', linestyles=linestyles, transient=transient, markerstyles=markerstyles, linewidths=linewidths)
    axes[4].axis(axes[4].axis('tight'))
    axes[4].set_ylim(-1550, 50)
    axes[4].set_yticks(-np.arange(16)*100)
    if show_xlabels:
        axes[4].set_xlabel(r'$\sigma^2$ (mV$^2$)', va='center')
    axes[4].set_title('LFP variance', va='baseline')
    axes[4].legend(bbox_to_anchor=(1.37, 1.0), frameon=False)
    axes[4].set_xlim(left=1E-7)
    phlp.remove_axis_junk(axes[4])
    phlp.annotate_subplot(axes[4], ncols=1, nrows=1, letter='E')
    
    
    
    
    return fig

if __name__ == '__main__':
    plt.close('all')
    
    params = multicompartment_params()
    ana_params = analysis_params.params()


    ana_params.set_PLOS_2column_fig_style(ratio=1)
    fig, axes = plt.subplots(2,5)
    fig.subplots_adjust(left=0.06, right=0.96, wspace=0.4, hspace=0.2, bottom=0.05, top=0.95)

    # params.figures_path = os.path.join(params.savefolder, 'figures')
    # params.populations_path = os.path.join(params.savefolder, 'populations')
    # params.spike_output_path = os.path.join(params.savefolder,
    #                                                    'processed_nest_output')
    # params.networkSimParams['spike_output_path'] = params.spike_output_path

    fig_lfp_decomposition(fig, axes[0], params, transient=200, show_xlabels=False)
    fig_exc_inh_contrib(fig, axes[1], params,
                        savefolders=['simulation_output_modified_spontan_exc',
                                     'simulation_output_modified_spontan_inh',
                                     'simulation_output_modified_spontan'],
                        T=[800, 1000], transient=200, panel_labels='FGHIJ')
    fig.savefig('figure_09.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig('figure_09.eps',
                bbox_inches='tight', pad_inches=0.01)


    plt.show()
    
