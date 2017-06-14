#!/usr/bin/env python
'''
Example plot for LFPy: Single-synapse contribution to the LFP
'''
import os
import sys
sys.path += [os.path.split(os.environ['PWD'])[0]]

import LFPy
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from analysis_params import params as PlotParams
import analysis_params
import plotting_helpers as phlp

plotparams = PlotParams()

plt.close('all')

################################################################################
# Main script, set parameters and create cell, synapse and electrode objects
################################################################################


# Define cell parameters
cell_parameters = {          # various cell parameters,
    #'morphology' : 'patdemo/cells/j4a.hoc', # Mainen&Sejnowski, 1996
    'rm' : 10000.,      # membrane resistance
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150,        # axial resistance
    'v_init' : -65.,    # initial crossmembrane potential
    'e_pas' : -65.,     # reversal potential passive mechs
    'passive' : True,   # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100.,
    'dt' : 2.**-4,      # [ms] dt's should be in powers of 2 for both, need
                        # binary representation
    'tstart' : 0.,    # start time of simulation, recorders start at t=0
    'tstop' : 10.,   # stop simulation at 200 ms. These can be overridden
                        # by setting these arguments i cell.simulation()
}


# Create a grid of measurement locations, in (mum)
X, Z = np.mgrid[-800:801:25, -400:1201:25]
Y = np.zeros(X.shape)

# Define electrode parameters
grid_electrode_parameters = {
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()
}


# Define electrode parameters
point_electrode_parameters = {
    'sigma' : 0.3,  # extracellular conductivity
    'x' : np.array([-150., -250.]),
    'y' : np.array([   0.,    0.]),
    'z' : np.array([-200.,  600.]),
}


def run_sim(morphology='patdemo/cells/j4a.hoc',
            cell_rotation=dict(x=4.99, y=-4.33, z=3.14),
            closest_idx=dict(x=-200., y=0., z=800.)):
    '''set up simple cell simulation with LFPs in the plane'''

    # Create cell
    cell = LFPy.Cell(morphology=morphology, **cell_parameters)
    # Align cell
    cell.set_rotation(**cell_rotation)
    
    # Define synapse parameters
    synapse_parameters = {
        'idx' : cell.get_closest_idx(**closest_idx),
        'e' : 0.,                   # reversal potential
        'syntype' : 'ExpSynI',       # synapse type
        'tau' : 0.5,                 # synaptic time constant
        'weight' : 0.0878,            # synaptic weight
        'record_current' : True,    # record synapse current
    }
    
    # Create synapse and set time of synaptic input
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([1.]))

    
    # Create electrode object
    
    # Run simulation, electrode object argument in cell.simulate
    print "running simulation..."
    cell.simulate(rec_imem=True,rec_isyn=True)
    
    grid_electrode = LFPy.RecExtElectrode(cell,**grid_electrode_parameters)
    point_electrode = LFPy.RecExtElectrode(cell,**point_electrode_parameters)
    
    grid_electrode.calc_lfp()
    point_electrode.calc_lfp()
    
    print "done"
    
    return cell, synapse, grid_electrode, point_electrode


def plot_sim(ax, cell, synapse, grid_electrode, point_electrode, letter='a'):
    '''create a plot'''
    
    fig = plt.figure(figsize = (3.27*2/3, 3.27*2/3))
    
    ax = fig.add_axes([.1,.05,.9,.9], aspect='equal', frameon=False)
    
    phlp.annotate_subplot(ax, ncols=1, nrows=1, letter=letter, fontsize=16)
    
    cax = fig.add_axes([0.8, 0.2, 0.02, 0.2], frameon=False)
    
    
    LFP = np.max(np.abs(grid_electrode.LFP),1).reshape(X.shape)
    im = ax.contour(X, Z, np.log10(LFP), 
               50,
               cmap='RdBu',
               linewidths=1.5,
               zorder=-2)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('$|\phi(\mathbf{r}, t)|_\mathrm{max}$ (nV)')
    cbar.outline.set_visible(False)
    #get some log-linear tickmarks and ticklabels
    ticks = np.arange(np.ceil(np.log10(LFP.min())), np.ceil(np.log10(LFP.max())))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(10.**ticks * 1E6) #mv -> nV
    
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(zip(x, z))
    polycol = PolyCollection(zips,
                             edgecolors='k',
                             linewidths=0.5,
                             facecolors='k')
    ax.add_collection(polycol)
    
    ax.plot([100, 200], [-400, -400], 'k', lw=1, clip_on=False)
    ax.text(150, -470, r'100$\mu$m', va='center', ha='center')
    
    ax.axis('off')
    
    
    ax.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=5,
            markeredgecolor='k',
            markerfacecolor='r')
    
    color_vec = ['blue','green']
    for i in xrange(2):
        ax.plot(point_electrode_parameters['x'][i],
                        point_electrode_parameters['z'][i],'o',ms=6,
                        markeredgecolor='none',
                        markerfacecolor=color_vec[i])
    
    
    plt.axes([.11, .075, .25, .2])
    plt.plot(cell.tvec,point_electrode.LFP[0]*1e6,color=color_vec[0], clip_on=False)
    plt.plot(cell.tvec,point_electrode.LFP[1]*1e6,color=color_vec[1], clip_on=False)
    plt.axis('tight')
    ax = plt.gca()
    ax.set_ylabel(r'$\phi(\mathbf{r}, t)$ (nV)') #rotation='horizontal')
    ax.set_xlabel('$t$ (ms)', va='center')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.axes([.11, 0.285, .25, .2])
    plt.plot(cell.tvec,synapse.i*1E3, color='red', clip_on=False)
    plt.axis('tight')
    ax = plt.gca()
    ax.set_ylabel(r'$I_{i, j}(t)$ (pA)', ha='center', va='center') #, rotation='horizontal')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticklabels([])

    return fig


def plot_sim_tstep(fig, ax, cell, synapse, grid_electrode, point_electrode, tstep=0,
                   letter='a',title='', cbar=True, show_legend=False):
    '''create a plot'''
    ax.set_title(title)
    
    if letter != None:
        phlp.annotate_subplot(ax, ncols=3, nrows=1, letter=letter, linear_offset=0.05, fontsize=16)

    
    
    LFP = grid_electrode.LFP[:, tstep].reshape(X.shape).copy()
    LFP *= 1E6 #mv -> nV
    vlim = 50
    levels = np.linspace(-vlim*2, vlim*2, 401)
    cbarticks = np.mgrid[-50:51:20]
    #cbarticks = [-10**np.floor(np.log10(vlim)),
    #             0,
    #             10**np.floor(np.log10(vlim)),]
    #force dashed for negative values
    linestyles = []
    for level in levels:
        if analysis_params.bw:
            if level > 0:
                linestyles.append('-')
            elif level == 0:
                linestyles.append((0, (5, 5)))
            else:
                linestyles.append((0, (1.0, 1.0)))
        else:
            # linestyles.append('-')
            if level > 0:
                linestyles.append('-')
            elif level == 0:
                linestyles.append((0, (5, 5)))
            else:
                linestyles.append('-')
    if np.any(LFP != np.zeros(LFP.shape)):
        im = ax.contour(X, Z, LFP,
                   levels=levels,
                   cmap='gray' if analysis_params.bw else 'RdBu',
                   vmin=-vlim,
                   vmax=vlim,
                   linewidths=3,
                   linestyles=linestyles,
                   zorder=-2,
                   rasterized=False)
        

        bbox = np.array(ax.get_position()).flatten()
        if cbar:
            cax = fig.add_axes((bbox[2]-0.01, 0.2, 0.01, 0.4), frameon=False)
            cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%i'), values=[-vlim, vlim])
            cbar.set_ticks(cbarticks)
            cbar.set_label('$\phi(\mathbf{r}, t)$ (nV)', labelpad=0)
            cbar.outline.set_visible(False)
        
        if show_legend:
            proxy = [plt.Line2D((0,1),(0,1), color='gray' if analysis_params.bw else plt.get_cmap('RdBu', 3)(2), ls='-', lw=3),
                     plt.Line2D((0,1),(0,1), color='gray' if analysis_params.bw else plt.get_cmap('RdBu', 3)(1), ls=(0, (5, 5)), lw=3),
                     plt.Line2D((0,1),(0,1), color='gray' if analysis_params.bw else plt.get_cmap('RdBu', 3)(0), ls=(0, (1, 1)), lw=3), ]
            
            ax.legend(proxy, [r'$\phi(\mathbf{r}, t) > 0$ nV',
                               r'$\phi(\mathbf{r}, t) = 0$ nV',
                               r'$\phi(\mathbf{r}, t) < 0$ nV'],
                      loc=1,
                      bbox_to_anchor=(1.2, 1),
                      fontsize=10,
                      frameon=False)
        
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(zip(x, z))
    polycol = PolyCollection(zips,
                             edgecolors='k',
                             linewidths=0.5,
                             facecolors='k')
    ax.add_collection(polycol)
    
    ax.plot([100, 200], [-400, -400], 'k', lw=2, clip_on=False)
    ax.text(150, -470, r'100$\mu$m', va='center', ha='center')
    
    ax.axis('off')
    
    
    ax.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=6,
            markeredgecolor='k',
            markerfacecolor='w' if analysis_params.bw else 'r')
    
    color_vec = ['k' if analysis_params.bw else 'b', 'gray' if analysis_params.bw else 'g']
    for i in xrange(2):
        ax.plot(point_electrode_parameters['x'][i],
                        point_electrode_parameters['z'][i],'o',ms=6,
                        markeredgecolor='k',
                        markerfacecolor=color_vec[i])
    
    
    bbox = np.array(ax.get_position()).flatten()
    ax1 = fig.add_axes((bbox[0], bbox[1], 0.05, 0.2))
    ax1.plot(cell.tvec,point_electrode.LFP[0]*1e6,color=color_vec[0], clip_on=False)
    ax1.plot(cell.tvec,point_electrode.LFP[1]*1e6,color=color_vec[1], clip_on=False)
    axis = ax1.axis(ax1.axis('tight'))
    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.vlines(cell.tvec[tstep], axis[2], axis[3], lw=0.2)
    ax1.set_ylabel(r'$\phi(\mathbf{r}, t)$ (nV)', labelpad=0) #rotation='horizontal')
    ax1.set_xlabel('$t$ (ms)', labelpad=0)
    for loc, spine in ax1.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2 = fig.add_axes((bbox[0], bbox[1]+.6, 0.05, 0.2))
    ax2.plot(cell.tvec,synapse.i*1E3, color='k' if analysis_params.bw else 'r',
             clip_on=False)
    axis = ax2.axis(ax2.axis('tight'))
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.vlines(cell.tvec[tstep], axis[2], axis[3])
    ax2.set_ylabel(r'$I_{i, j}(t)$ (pA)', labelpad=0) #, rotation='horizontal')
    for loc, spine in ax2.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_xticklabels([])



if __name__ == '__main__':
    
    tstep = 2 * 16 # 2 ms mark

    plotparams.set_PLOS_2column_fig_style(ratio=0.33)


    fig, axes = plt.subplots(1,3)
    fig.subplots_adjust(left=0.05, bottom=0.1, top=0.925, right=0.95, wspace=0.4)

    cell, synapse, grid_electrode, point_electrode = run_sim(morphology='../morphologies/stretched/L4E_53rpy1.hoc',
                                                             cell_rotation=dict(z=np.pi),
                                                             closest_idx=dict(x=-100, y=0., z=650.))
    plot_sim_tstep(fig, axes[0], cell, synapse, grid_electrode, point_electrode, tstep=tstep, letter='A', title='p4, apical input', cbar=False, show_legend=False)
    

    cell, synapse, grid_electrode, point_electrode = run_sim(morphology='../morphologies/stretched/L4E_53rpy1.hoc',
                                                             cell_rotation=dict(z=np.pi),
                                                             closest_idx=dict(x=0, y=0., z=-200.))
    plot_sim_tstep(fig, axes[1], cell, synapse, grid_electrode, point_electrode, tstep=tstep, letter='B', title='p4, basal input', cbar=False, show_legend=False)

    
    cell, synapse, grid_electrode, point_electrode = run_sim(morphology='../morphologies/stretched/L4E_j7_L4stellate.hoc',
                                                             cell_rotation=dict(z=np.pi/2), #dict(z=np.pi),
                                                             closest_idx=dict(x=0., y=0., z=-200))
    plot_sim_tstep(fig, axes[2], cell, synapse, grid_electrode, point_electrode, tstep=tstep, letter='C', title='ss4(L4)', cbar=True, show_legend=False)

    fig.savefig('Fig03.pdf', dpi=450/2, bbox_inches='tight', pad_inches=0)
    fig.savefig('Fig03.eps', bbox_inches='tight', pad_inches=0.01)

    plt.show()
