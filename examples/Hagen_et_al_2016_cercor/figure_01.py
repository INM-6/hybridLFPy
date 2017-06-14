import numpy as np
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

import plotting_helpers as phlp
from hybridLFPy import CachedNetwork
import analysis_params

######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

from cellsim16popsParams_modified_regular_input import multicompartment_params 


######################################
### IMPORT PANELS                  ###
######################################

from plot_methods import network_sketch, plotConnectivity, plot_population, plot_signal_sum



######################################
### FIGURE                         ###
######################################


def fig_intro(params, ana_params, T=[800, 1000], fraction=0.05, rasterized=False):
    '''set up plot for introduction'''
    ana_params.set_PLOS_2column_fig_style(ratio=0.5)
    
    #load spike as database
    networkSim = CachedNetwork(**params.networkSimParams)
    if analysis_params.bw:
        networkSim.colors = phlp.get_colors(len(networkSim.X))

    #set up figure and subplots
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 4)
    
    
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.5, hspace=0.)


    #network diagram
    ax0_1 = fig.add_subplot(gs[:, 0], frameon=False)
    ax0_1.set_title('point-neuron network', va='bottom')

    network_sketch(ax0_1, yscaling=1.3)
    ax0_1.xaxis.set_ticks([])
    ax0_1.yaxis.set_ticks([])
    phlp.annotate_subplot(ax0_1, ncols=4, nrows=1, letter='A', linear_offset=0.065)
   
    
    #network raster
    ax1 = fig.add_subplot(gs[:, 1], frameon=True)
    phlp.remove_axis_junk(ax1)
    phlp.annotate_subplot(ax1, ncols=4, nrows=1, letter='B', linear_offset=0.065)
       
    x, y = networkSim.get_xy(T, fraction=fraction)
    # networkSim.plot_raster(ax1, T, x, y, markersize=0.1, alpha=1.,legend=False, pop_names=True)
    networkSim.plot_raster(ax1, T, x, y, markersize=0.2, marker='_', alpha=1.,legend=False, pop_names=True, rasterized=rasterized)
    ax1.set_ylabel('')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.set_title('spiking activity', va='bottom')
    a = ax1.axis()
    ax1.vlines(x['TC'][0], a[2], a[3], 'k', lw=0.25)


    #population
    ax2 = fig.add_subplot(gs[:, 2], frameon=False)
    ax2.xaxis.set_ticks([])
    ax2.yaxis.set_ticks([])
    plot_population(ax2, params, isometricangle=np.pi/24, plot_somas=False,
                    plot_morphos=True, num_unitsE=1, num_unitsI=1,
                    clip_dendrites=True, main_pops=True, title='',
                    rasterized=rasterized)
    ax2.set_title('multicompartment\nneurons', va='bottom', fontweight='normal')
    phlp.annotate_subplot(ax2, ncols=4, nrows=1, letter='C', linear_offset=0.065)
    

    #LFP traces in all channels
    ax3 = fig.add_subplot(gs[:, 3], frameon=True)
    phlp.remove_axis_junk(ax3)
    plot_signal_sum(ax3, params, fname=os.path.join(params.savefolder, 'LFPsum.h5'),
                unit='mV', vlimround=0.8,
                T=T, ylim=[ax2.axis()[2], ax2.axis()[3]],
                rasterized=False)
    ax3.set_title('LFP', va='bottom')
    ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
    phlp.annotate_subplot(ax3, ncols=4, nrows=1, letter='D', linear_offset=0.065)
    a = ax3.axis()
    ax3.vlines(x['TC'][0], a[2], a[3], 'k', lw=0.25)
    
    
    #draw some arrows:
    ax = plt.gca()
    ax.annotate("", xy=(0.27, 0.5), xytext=(.24, 0.5),
                xycoords="figure fraction",
            arrowprops=dict(facecolor='black', arrowstyle='simple'),
            )
    ax.annotate("", xy=(0.52, 0.5), xytext=(.49, 0.5),
                xycoords="figure fraction",
            arrowprops=dict(facecolor='black', arrowstyle='simple'),
            )
    ax.annotate("", xy=(0.78, 0.5), xytext=(.75, 0.5),
                xycoords="figure fraction",
            arrowprops=dict(facecolor='black', arrowstyle='simple'),
            )

    
    return fig


if __name__ == '__main__':
    plt.close('all')

    params = multicompartment_params()
    ana_params = analysis_params.params()

    
    savefolders = [
        'simulation_output_modified_regular_input'
    ]

    for i, savefolder in enumerate(savefolders):
        # path to simulation files
        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path
        
        fig = fig_intro(params, ana_params, T=[875, 950], fraction=1.)
        
        fig.savefig('figure_01.pdf',
                    dpi=450,
                    bbox_inches='tight', pad_inches=0)
        fig.savefig('figure_01.eps',
                    bbox_inches='tight', pad_inches=0.01)
    plt.show()
