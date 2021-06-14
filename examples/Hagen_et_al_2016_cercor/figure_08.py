#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

from hybridLFPy import CachedNetwork
from hybridLFPy.gdf import GDF
import hybridLFPy.helpers as hlp
import plotting_helpers as phlp
import analysis_params

######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

from cellsim16popsParams_default import multicompartment_params

#
ana_params = analysis_params.params()


######################################
### IMPORT PANELS                  ###
######################################

from plot_methods import plot_signal_sum_colorplot, plot_signal_sum, plot_signal_power_colorplot, calc_signal_power

######################################
### FIGURE                         ###
######################################


def fig_network_input_structure(fig, params, bottom=0.1, top=0.9, transient=200, T=[800, 1000], Df= 0., mlab= True, NFFT=256, srate=1000,
             window=plt.mlab.window_hanning, noverlap=256*3/4, letters='abcde', flim=(4, 400),
             show_titles=True, show_xlabels=True, show_CSD=False):
    '''
    This figure is the top part for plotting a comparison between the PD-model
    and the modified-PD model

    '''
    #load spike as database
    networkSim = CachedNetwork(**params.networkSimParams)
    if analysis_params.bw:
        networkSim.colors = phlp.get_colors(len(networkSim.X))


    # ana_params.set_PLOS_2column_fig_style(ratio=ratio)
    # fig = plt.figure()
    # fig.subplots_adjust(left=0.06, right=0.94, bottom=0.09, top=0.92, wspace=0.5, hspace=0.2)

    #use gridspec to get nicely aligned subplots througout panel
    gs1 = gridspec.GridSpec(5, 5, bottom=bottom, top=top)


    ############################################################################
    # A part, full dot display
    ############################################################################

    ax0 = fig.add_subplot(gs1[:, 0])
    phlp.remove_axis_junk(ax0)
    phlp.annotate_subplot(ax0, ncols=5, nrows=1, letter=letters[0],
                     linear_offset=0.065)

    x, y = networkSim.get_xy(T, fraction=1)
    networkSim.plot_raster(ax0, T, x, y,
                           markersize=0.2, marker='_',
                           alpha=1.,
                           legend=False, pop_names=True,
                           rasterized=False)
    ax0.set_ylabel('population', labelpad=0.)
    ax0.set_xticks([800,900,1000])

    if show_titles:
        ax0.set_title('spiking activity',va='center')
    if show_xlabels:
        ax0.set_xlabel(r'$t$ (ms)', labelpad=0.)
    else:
        ax0.set_xlabel('')

    ############################################################################
    # B part, firing rate spectra
    ############################################################################


    # Get the firing rate from Potjan Diesmann et al network activity
    #collect the spikes x is the times, y is the id of the cell.
    T_all=[transient, networkSim.simtime]
    bins = np.arange(transient, networkSim.simtime+1)

    x, y = networkSim.get_xy(T_all, fraction=1)

    # create invisible axes to position labels correctly
    ax_ = fig.add_subplot(gs1[:, 1])
    phlp.annotate_subplot(ax_, ncols=5, nrows=1, letter=letters[1],
                                 linear_offset=0.065)
    if show_titles:
        ax_.set_title('firing rate PSD', va='center')

    ax_.axis('off')

    colors = phlp.get_colors(len(params.Y)) + ['k']

    COUNTER = 0
    label_set = False

    tits = ['L23E/I', 'L4E/I', 'L5E/I', 'L6E/I', 'TC']

    if x['TC'].size > 0:
        TC = True
    else:
        TC = False

    BAxes = []
    for i, X in enumerate(params.Y + ['TC']):

        if i % 2 == 0:
            ax1 = fig.add_subplot(gs1[COUNTER, 1])
            phlp.remove_axis_junk(ax1)

            if x[X].size > 0:
                ax1.text(0.05, 0.85, tits[COUNTER],
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax1.transAxes)
            BAxes.append(ax1)


        #firing rate histogram
        hist = np.histogram(x[X], bins=bins)[0].astype(float)
        hist -= hist.mean()

        if mlab:
            Pxx, freqs=plt.mlab.psd(hist, NFFT=NFFT,
                                    Fs=srate, noverlap=noverlap, window=window)
        else:
            [freqs, Pxx] = hlp.powerspec([hist], tbin= 1.,
                                        Df=Df, pointProcess=False)
            mask = np.where(freqs >= 0.)
            freqs = freqs[mask]
            Pxx = Pxx.flatten()
            Pxx = Pxx[mask]
            Pxx = Pxx/(T_all[1]-T_all[0])**2

        if x[X].size > 0:
            ax1.loglog(freqs[1:], Pxx[1:],
                       label=X, color=colors[i],
                       clip_on=True)
            ax1.axis(ax1.axis('tight'))
            ax1.set_ylim([5E-4,5E2])
            ax1.set_yticks([1E-3,1E-1,1E1])
            if label_set == False:
                ax1.set_ylabel(r'(s$^{-2}$/Hz)', labelpad=0.)
                label_set = True
            if i > 1:
                ax1.set_yticklabels([])
            if i >= 6 and not TC and show_xlabels or X == 'TC' and TC and show_xlabels:
                ax1.set_xlabel('$f$ (Hz)', labelpad=0.)
            if TC and i < 8 or not TC and i < 6:
                ax1.set_xticklabels([])

        else:
            ax1.axis('off')

        ax1.set_xlim(flim)


        if i % 2 == 0:
            COUNTER += 1

        ax1.yaxis.set_minor_locator(plt.NullLocator())


    ############################################################################
    # c part, LFP traces and CSD color plots
    ############################################################################

    ax2 = fig.add_subplot(gs1[:, 2])

    phlp.annotate_subplot(ax2, ncols=5, nrows=1, letter=letters[2],
                     linear_offset=0.065)


    phlp.remove_axis_junk(ax2)
    plot_signal_sum(ax2, params,
                    fname=os.path.join(params.savefolder, 'RecExtElectrode_sum.h5'),
                    unit='mV', T=T, ylim=[-1600, 40],
                    rasterized=False)

    # CSD background colorplot
    if show_CSD:
        im = plot_signal_sum_colorplot(ax2, params, os.path.join(params.savefolder, 'LaminarCurrentSourceDensity_sum.h5'),
                                  unit=r'($\mu$Amm$^{-3}$)', T=[800, 1000],
                                  colorbar=False,
                                  ylim=[-1600, 40], fancy=False, cmap=plt.cm.get_cmap('bwr_r', 21),
                                  rasterized=False)
        cb = phlp.colorbar(fig, ax2, im,
                           width=0.05, height=0.4,
                           hoffset=-0.05, voffset=0.3)
        cb.set_label('($\mu$Amm$^{-3}$)', labelpad=0.1)

    ax2.set_xticks([800,900,1000])
    ax2.axis(ax2.axis('tight'))

    if show_titles:
        if show_CSD:
            ax2.set_title('LFP & CSD', va='center')
        else:
            ax2.set_title('LFP', va='center')
    if show_xlabels:
        ax2.set_xlabel(r'$t$ (ms)', labelpad=0.)
    else:
        ax2.set_xlabel('')


    ############################################################################
    # d part, LFP power trace for each layer
    ############################################################################

    freqs, PSD = calc_signal_power(params, fname=os.path.join(params.savefolder,
                                                           'RecExtElectrode_sum.h5'),
                                        transient=transient, Df=Df, mlab=mlab,
                                        NFFT=NFFT, noverlap=noverlap,
                                        window=window)

    channels = [0, 3, 7, 11, 13]

    # create invisible axes to position labels correctly
    ax_ = fig.add_subplot(gs1[:, 3])
    phlp.annotate_subplot(ax_, ncols=5, nrows=1, letter=letters[3],
                                 linear_offset=0.065)

    if show_titles:
        ax_.set_title('LFP PSD',va='center')

    ax_.axis('off')

    for i, ch in enumerate(channels):

        ax = fig.add_subplot(gs1[i, 3])
        phlp.remove_axis_junk(ax)

        if i == 0:
            ax.set_ylabel('(mV$^2$/Hz)', labelpad=0)

        ax.loglog(freqs[1:],PSD[ch][1:], color='k')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if i < 4:
            ax.set_xticklabels([])
        ax.text(0.75, 0.85,'ch. %i' %(channels[i]+1),
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=6,
                transform=ax.transAxes)
        ax.tick_params(axis='y', which='minor', bottom='off')
        ax.axis(ax.axis('tight'))
        ax.yaxis.set_minor_locator(plt.NullLocator())

        ax.set_xlim(flim)
        ax.set_ylim(1E-7,2E-4)
        if i != 0 :
            ax.set_yticklabels([])


    if show_xlabels:
        ax.set_xlabel('$f$ (Hz)', labelpad=0.)

    ############################################################################
    # e part signal power
    ############################################################################

    ax4 = fig.add_subplot(gs1[:, 4])

    phlp.annotate_subplot(ax4, ncols=5, nrows=1, letter=letters[4],
                     linear_offset=0.065)

    fname=os.path.join(params.savefolder, 'RecExtElectrode_sum.h5')
    im = plot_signal_power_colorplot(ax4, params, fname=fname, transient=transient, Df=Df,
                                mlab=mlab, NFFT=NFFT, window=window,
                                cmap=plt.cm.get_cmap('gray_r', 12),
                                vmin=1E-7, vmax=1E-4)
    phlp.remove_axis_junk(ax4)

    ax4.set_xlim(flim)

    cb = phlp.colorbar(fig, ax4, im,
                       width=0.05, height=0.5,
                       hoffset=-0.05, voffset=0.5)
    cb.set_label('(mV$^2$/Hz)', labelpad=0.1)


    if show_titles:
        ax4.set_title('LFP PSD', va='center')
    if show_xlabels:
        ax4.set_xlabel(r'$f$ (Hz)', labelpad=0.)
    else:
        ax4.set_xlabel('')

    return fig


if __name__ == '__main__':

    params = multicompartment_params()

    savefolders = [
        'simulation_output_default',
        'simulation_output_modified_ac_input',
        'simulation_output_modified_spontan',
    ]
    letterslist = [
        'KLMNO',
        'FGHIJ',
        'ABCDE',
    ]

    show_titles = [False, False, True]
    show_xlabels = [True, False, False]

    axh = 0.2825
    bottoms = np.linspace(0.035, 0.965-axh, 3)
    tops = bottoms + axh

    ana_params.set_PLOS_2column_fig_style(ratio=1.25)
    fig = plt.figure()
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.09, top=0.92, wspace=0.5, hspace=0.2)

    for savefolder, letters, ttl, xlbl, bottom, top in zip(savefolders, letterslist, show_titles, show_xlabels, bottoms, tops):
        # path to simulation files
        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                           'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path


        fig_network_input_structure(fig, params, bottom=bottom, top=top,
                                    transient=200, T=[800, 1000], Df=0., mlab= True, NFFT=256, srate=1000,
                                    window=plt.mlab.window_hanning, noverlap=128, letters=letters,
                                    show_titles=ttl, show_xlabels=xlbl)

    fig.savefig('figure_08.pdf',
                dpi=450,
                bbox_inches='tight', pad_inches=0)
    fig.savefig('figure_08.eps', bbox_inches='tight', pad_inches=0.01)


    plt.show()
