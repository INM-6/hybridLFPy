#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
file containing plotter functions for example scripts
'''
import os
import numpy as np
import h5py
import LFPy
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection

def remove_axis_junk(ax, which=['right', 'top']):
    '''remove upper and right axis'''
    for loc, spine in ax.spines.items():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def plot_signal_sum(ax, z, fname='LFPsum.h5', unit='mV',
                    ylabels=True, scalebar=True, vlimround=None,
                    T=[0, 1000], color='k',
                    label=''):
    '''
    on axes plot the signal contributions
    
    args:
    ::
        ax : matplotlib.axes.AxesSubplot object
        z : np.ndarray
        T : list, [tstart, tstop], which timeinterval
        ylims : list, set range of yaxis to scale with other plots
        fancy : bool, 
        scaling_factor : float, scaling factor (e.g. to scale 10% data set up)
    '''    
    #open file and get data, samplingrate
    f = h5py.File(fname)
    data = f['data'].value
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    srate = f['srate'].value
    
    #close file object
    f.close()

    # normalize data for plot
    tvec = np.arange(data.shape[1]) * 1000. / srate
    slica = (tvec <= T[1]) & (tvec >= T[0])
    zvec = np.r_[z]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    vlim = abs(data[:, slica]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim))
    yticklabels=[]
    yticks = []
    
    colors = [color]*data.shape[0]
    
    for i, z in enumerate(z):
        if i == 0:
            ax.plot(tvec[slica], data[i, slica] * 100 / vlimround + z,
                    color=colors[i], rasterized=False, label=label,
                    clip_on=False)
        else: 
            ax.plot(tvec[slica], data[i, slica] * 100 / vlimround + z,
                    color=colors[i], rasterized=False, clip_on=False)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)
     
    if scalebar:
        ax.plot([tvec[slica][-1], tvec[slica][-1]],
                [-0, -100], lw=2, color='k', clip_on=False)
        ax.text(tvec[slica][-1]+np.diff(T)*0.02, -50,
                r'%g %s' % (vlimround, unit),
                color='k', rotation='vertical')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
    else:
        ax.yaxis.set_ticklabels([])

    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(r'time (ms)', labelpad=0)


def plot_pop_scatter(ax, somapos, isometricangle, marker, color):
    #scatter plot setting appropriate zorder for each datapoint by binning
    for lower in np.arange(-600, 601, 20):
        upper = lower + 20
        inds = (somapos[:, 1] >= lower) & (somapos[:, 1] < upper)
        if np.any(inds):
            ax.scatter(somapos[inds, 0],
                       somapos[inds, 2] - somapos[inds, 1] *
                       np.sin(isometricangle),
                   s=30, facecolors=color, edgecolors='gray', linewidth=0.1,
                   zorder=lower,
                   marker = marker, clip_on=False, rasterized=True)


def plot_population(ax,
                    populationParams,
                    electrodeParams,
                    layerBoundaries,
                    aspect='equal',
                    isometricangle=np.pi/12,
                    X=['EX', 'IN'],
                    markers=['^', 'o'],
                    colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    title='positions'):
    '''
    Plot the geometry of the column model, optionally with somatic locations
    and optionally with reconstructed neurons
    
    kwargs:
    ::
        ax : matplotlib.axes.AxesSubplot
        aspect : str
            matplotlib.axis argument
        isometricangle : float
            pseudo-3d view angle
        plot_somas : bool
            plot soma locations
        plot_morphos : bool
            plot full morphologies
        num_unitsE : int
            number of excitatory morphos plotted per population
        num_unitsI : int
            number of inhibitory morphos plotted per population
        clip_dendrites : bool
            draw dendrites outside of axis
        mainpops : bool
            if True, plot only main pops, e.g. b23 and nb23 as L23I
    
    return:
    ::
        axis : list
            the plt.axis() corresponding to input aspect
    '''

    remove_axis_junk(ax, ['right', 'bottom', 'left', 'top'])

    
    # DRAW OUTLINE OF POPULATIONS 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    #contact points
    ax.plot(electrodeParams['x'],
            electrodeParams['z'],
            '.', marker='o', markersize=5, color='k', zorder=0)

    #outline of electrode       
    x_0 = np.array(electrodeParams['r_z'])[1, 1:-1]
    z_0 = np.array(electrodeParams['r_z'])[0, 1:-1]
    x = np.r_[x_0[-1], x_0[::-1], -x_0[1:], -x_0[-1]]
    z = np.r_[100, z_0[::-1], z_0[1:], 100]
    ax.fill(x, z, color=(0.5, 0.5, 0.5), lw=None, zorder=-0.1)

    #outline of populations:
    #fetch the population radius from some population
    r = populationParams[X[0]]['radius']

    theta0 = np.linspace(0, np.pi, 20)
    theta1 = np.linspace(np.pi, 2*np.pi, 20)
    
    zpos = np.r_[np.array(layerBoundaries)[:, 0],
                 np.array(layerBoundaries)[-1, 1]]
    
    for i, z in enumerate(np.mean(layerBoundaries, axis=1)):
        ax.text(r, z, ' %s' % layers[i],
                va='center', ha='left', rotation='vertical')

    for i, zval in enumerate(zpos):
        if i == 0:
            ax.plot(r*np.cos(theta0),
                    r*np.sin(theta0)*np.sin(isometricangle)+zval,
                    color='k', zorder=-r, clip_on=False)
            ax.plot(r*np.cos(theta1),
                    r*np.sin(theta1)*np.sin(isometricangle)+zval,
                    color='k', zorder=r, clip_on=False)
        else:
            ax.plot(r*np.cos(theta0),
                    r*np.sin(theta0)*np.sin(isometricangle)+zval,
                    color='gray', zorder=-r, clip_on=False)
            ax.plot(r*np.cos(theta1),
                    r*np.sin(theta1)*np.sin(isometricangle)+zval,
                    color='k', zorder=r, clip_on=False)
    
    ax.plot([-r, -r], [zpos[0], zpos[-1]], 'k', zorder=0, clip_on=False)
    ax.plot([r, r], [zpos[0], zpos[-1]], 'k', zorder=0, clip_on=False)
    
    #plot a horizontal radius scalebar
    ax.plot([0, r], [z_0.min()]*2, 'k', lw=2, zorder=0, clip_on=False)
    ax.text(r / 2., z_0.min()-100, 'r = %i $\mu$m' % int(r), ha='center')
    
    #plot a vertical depth scalebar
    ax.plot([-r]*2, [z_0.min()+50, z_0.min()-50],
        'k', lw=2, zorder=0, clip_on=False)
    ax.text(-r, z_0.min(), r'100 $\mu$m', va='center', ha='right')
    
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    #fake ticks:
    for pos in zpos:
        ax.text(-r, pos, 'z=%i-' % int(pos), ha='right', va='center')
 
    ax.set_title(title)
   
    axis = ax.axis(ax.axis(aspect))


def plot_soma_locations(ax, X, populations_path, isometricangle, markers, colors):
    for (pop, marker, color) in zip(X, markers, colors):
        #get the somapos
        fname = os.path.join(populations_path,
                             '%s_population_somapos.gdf' % pop)
        
        somapos = np.loadtxt(fname).reshape((-1, 3))
    
        plot_pop_scatter(ax, somapos, isometricangle, marker, color)
        

def plot_morphologies(ax, X, isometricangle, markers, colors,
                      populations_path, cellParams, fraction=1):
    for (pop, marker, color) in zip(X, markers, colors):
        #get the somapos
        fname = os.path.join(populations_path,
                             '%s_population_somapos.gdf' % pop)
        
        somapos = np.loadtxt(fname).reshape((-1, 3))
        n = somapos.shape[0]
        
        rotations = [{} for x in range(n)]
        fname = os.path.join(populations_path,
                             '%s_population_rotations.h5' % pop)
        f = h5py.File(fname, 'r')
        
        for key, value in list(f.items()):
            for i, rot in enumerate(value.value):
                rotations[i].update({key : rot})
        
        
        #plot some units
        for j in range(int(n*fraction)):
            cell = LFPy.Cell(morphology=cellParams[pop]['morphology'],
                             nsegs_method='lambda_f',
                             lambda_f=100,
                             extracellular=False
                            )
            cell.set_pos(somapos[j, 0], somapos[j, 1], somapos[j, 2])
            cell.set_rotation(**rotations[j])

            #set up a polycollection
            zips = []
            for x, z in cell.get_idx_polygons():
                zips.append(list(zip(x, z-somapos[j, 1] * np.sin(isometricangle)
                                     )))
            
            polycol = PolyCollection(zips,
                                     edgecolors='gray',
                                     facecolors=color,
                                     linewidths=(0.1),
                                     zorder=somapos[j, 1],
                                     clip_on=False,
                                     rasterized=False)

            ax.add_collection(polycol)
            


def plot_individual_morphologies(ax, X, isometricangle, markers, colors,
                                 cellParams, populationParams):
    
    depth = []
    offsets = []
    depth0 = np.inf
    offset = -75
    for i, y in enumerate(X):
        d = populationParams[y]
        depth += [(d['z_min']+d['z_max'])/2]
        if depth0 == depth[-1]:
            offset += 150
        else:
            offset = -75 - i*20
        depth0 = depth[-1]
        offsets += [offset]
    somapos = np.c_[offsets, np.zeros(len(offsets)), depth]
    
    for i, (pop, marker, color) in enumerate(zip(X, markers, colors)):
        
        cell = LFPy.Cell(morphology=cellParams[pop]['morphology'],
                         nsegs_method='lambda_f',
                         lambda_f=100,
                         extracellular=False
                        )
        cell.set_pos(somapos[i, 0], somapos[i, 1], somapos[i, 2])
        
        #set up a polycollection
        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z-somapos[i, 1] * np.sin(isometricangle))))
        
        polycol = PolyCollection(zips,
                                 edgecolors='gray',
                                 facecolors=color,
                                 linewidths=(0.1),
                                 zorder=somapos[i, 1],
                                 clip_on=False,
                                 rasterized=False)

        ax.add_collection(polycol)


def normalize(x):
    '''normalize x to have mean 0 and unity standard deviation'''
    x = x.astype(float)
    x -= x.mean()
    return x / float(x.std())


def plot_correlation(z_vec, x0, x1, ax, lag=20., title='firing_rate vs LFP'):
    ''' mls
    on axes plot the correlation between x0 and x1
    
    args:
    ::
        x0 : first dataset
        x1 : second dataset - e.g., the multichannel LFP
        ax : matplotlib.axes.AxesSubplot object
        title : text to be used as current axis object title
    '''
    zvec = np.r_[z_vec]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    
    xcorr_all=np.zeros((np.size(z_vec), x0.shape[0]))
    for i, z in enumerate(z_vec):
        x2 = x1[i, ]
        xcorr1 = np.correlate(normalize(x0),
                              normalize(x2), 'same') / x0.size
        xcorr_all[i,:]=xcorr1  

    # Find limits for the plot
    vlim = abs(xcorr_all).max()
    vlimround = 2.**np.round(np.log2(vlim))
    
    yticklabels=[]
    yticks = []
    ylimfound=np.zeros((1,2))
    for i, z in enumerate(z_vec):
        ind = np.arange(x0.size) - x0.size/2
        ax.plot(ind, xcorr_all[i,::-1] * 100. / vlimround + z, 'k',
                clip_on=True, rasterized=False)
        yticklabels.append('ch. %i' %(i+1))
        yticks.append(z)    

    remove_axis_junk(ax)
    ax.set_title(title)
    ax.set_xlabel(r'lag $\tau$ (ms)')

    ax.set_xlim(-lag, lag)
    ax.set_ylim(z-100, 100)
    
    axis = ax.axis()
    ax.vlines(0, axis[2], axis[3], 'r', 'dotted')

    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create a scaling bar
    ax.plot([lag, lag],
        [0, 100], lw=2, color='k', clip_on=False)
    ax.text(lag, 50, r'CC=%.2f' % vlimround,
            rotation='vertical', va='center')
