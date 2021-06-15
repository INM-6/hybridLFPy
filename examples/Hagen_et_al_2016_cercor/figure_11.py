#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cellsim16popsParams_modified_spontan import multicompartment_params
import analysis_params
import plotting_helpers as phlp
import h5py
import os
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')


######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

#params = multicompartment_params()

ana_params = analysis_params.params()


######################################
### FIGURE                         ###
######################################


'''
Run analysis.py first to generate data files.
'''


def fig_lfp_corr(params, savefolders, transient=200,
                 channels=[0, 3, 7, 11, 13], Df=None,
                 mlab=True, NFFT=256, noverlap=128,
                 window=plt.mlab.window_hanning,
                 letterslist=['AB', 'CD'], data_type='LFP'):
    '''This figure compares power spectra for correlated and uncorrelated signals

    '''
    ana_params.set_PLOS_2column_fig_style(ratio=0.5)
    fig = plt.figure()
    fig.subplots_adjust(
        left=0.07,
        right=0.95,
        bottom=0.1,
        wspace=0.3,
        hspace=0.1)

    gs = gridspec.GridSpec(5, 4)

    for i, (savefolder, letters) in enumerate(zip(savefolders, letterslist)):
        # path to simulation files
        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path

        # Including correlations
        f = h5py.File(
            os.path.join(
                params.savefolder,
                ana_params.analysis_folder,
                data_type +
                ana_params.fname_psd),
            'r')
        freqs = f['freqs'][()]
        LFP_PSD_corr = f['psd'][()]
        f.close()

        # Excluding correlations
        f = h5py.File(
            os.path.join(
                params.savefolder,
                ana_params.analysis_folder,
                data_type +
                ana_params.fname_psd_uncorr),
            'r')
        freqs = f['freqs'][()]
        LFP_PSD_uncorr = f['psd'][()]
        f.close()

        ##################################
        ###  Single channel LFP PSDs   ###
        ##################################

        ax = fig.add_subplot(gs[0, (i % 2) * 2])
        phlp.remove_axis_junk(ax)
        ax.loglog(freqs, LFP_PSD_corr[channels[0]], color='k', label='$P$')
        ax.loglog(freqs,
                  LFP_PSD_uncorr[channels[0]],
                  color='gray' if analysis_params.bw else analysis_params.colorP,
                  lw=1,
                  label='$\tilde{P}$')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.text(
            0.80,
            0.82,
            'ch. %i' % (channels[0] + 1),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=6,
            transform=ax.transAxes)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel('(mV$^2$/Hz)', labelpad=0.)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', which='minor', bottom='off')
        ax.set_xlim(4E0, 4E2)
        ax.set_ylim(1E-8, 1.5E-4)
        ax.set_yticks([1E-8, 1E-6, 1E-4])
        ax.set_title('power spectra')
        phlp.annotate_subplot(ax, ncols=4, nrows=5, letter=letters[0],
                              linear_offset=0.065)

        ax = fig.add_subplot(gs[1, (i % 2) * 2])
        phlp.remove_axis_junk(ax)
        ax.loglog(freqs, LFP_PSD_corr[channels[1]], color='k', label='corr')
        ax.loglog(freqs,
                  LFP_PSD_uncorr[channels[1]],
                  color='gray' if analysis_params.bw else analysis_params.colorP,
                  lw=1,
                  label='uncorr')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.text(
            0.80,
            0.82,
            'ch. %i' % (channels[1] + 1),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=6,
            transform=ax.transAxes)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', which='minor', bottom='off')
        ax.set_xlim(4E0, 4E2)
        ax.set_ylim(1E-8, 1.5E-4)
        ax.set_yticks([1E-8, 1E-6, 1E-4])
        ax.set_yticklabels([])

        ax = fig.add_subplot(gs[2, (i % 2) * 2])
        phlp.remove_axis_junk(ax)
        ax.loglog(freqs, LFP_PSD_corr[channels[2]], color='k', label='corr')
        ax.loglog(freqs,
                  LFP_PSD_uncorr[channels[2]],
                  color='gray' if analysis_params.bw else analysis_params.colorP,
                  lw=1,
                  label='uncorr')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.text(
            0.80,
            0.82,
            'ch. %i' % (channels[2] + 1),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=6,
            transform=ax.transAxes)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', which='minor', bottom='off')
        ax.set_xlim(4E0, 4E2)
        ax.set_ylim(1E-8, 1.5E-4)
        ax.set_yticks([1E-8, 1E-6, 1E-4])
        ax.set_yticklabels([])

        ax = fig.add_subplot(gs[3, (i % 2) * 2])
        phlp.remove_axis_junk(ax)
        ax.loglog(freqs, LFP_PSD_corr[channels[3]], color='k', label='corr')
        ax.loglog(freqs,
                  LFP_PSD_uncorr[channels[3]],
                  color='gray' if analysis_params.bw else analysis_params.colorP,
                  lw=1,
                  label='uncorr')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.text(
            0.80,
            0.82,
            'ch. %i' % (channels[3] + 1),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=6,
            transform=ax.transAxes)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', which='minor', bottom='off')
        ax.set_xlim(4E0, 4E2)
        ax.set_ylim(1E-8, 1.5E-4)
        ax.set_yticks([1E-8, 1E-6, 1E-4])
        ax.set_yticklabels([])

        ax = fig.add_subplot(gs[4, (i % 2) * 2])
        phlp.remove_axis_junk(ax)
        ax.loglog(freqs, LFP_PSD_corr[channels[4]], color='k', label='corr')
        ax.loglog(freqs,
                  LFP_PSD_uncorr[channels[4]],
                  color='gray' if analysis_params.bw else analysis_params.colorP,
                  lw=1,
                  label='uncorr')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel(r'$f$ (Hz)', labelpad=0.2)
        ax.text(
            0.80,
            0.82,
            'ch. %i' % (channels[4] + 1),
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=6,
            transform=ax.transAxes)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='y', which='minor', bottom='off')
        ax.set_xlim(4E0, 4E2)
        ax.set_ylim(1E-8, 1.5E-4)
        ax.set_yticks([1E-8, 1E-6, 1E-4])
        ax.set_yticklabels([])

        ##################################
        ###  LFP PSD ratios            ###
        ##################################

        ax = fig.add_subplot(gs[:, (i % 2) * 2 + 1])
        phlp.annotate_subplot(ax, ncols=4, nrows=1, letter=letters[1],
                              linear_offset=0.065)
        phlp.remove_axis_junk(ax)
        ax.set_title('power ratio')
        PSD_ratio = LFP_PSD_corr / LFP_PSD_uncorr

        zvec = np.r_[params.electrodeParams['z']]
        inds = freqs >= 1  # frequencies greater than 4 Hz
        im = ax.pcolormesh(freqs[inds],
                           zvec,
                           PSD_ratio[:,
                                     inds],
                           rasterized=False,
                           cmap=plt.get_cmap('gray_r',
                                             12) if analysis_params.bw else plt.get_cmap('Reds',
                                                                                         12),
                           norm=LogNorm(vmin=10**-0.25,
                                        vmax=10**2.75),
                           shading='auto')
        ax.set_xscale('log')

        ax.set_yticks(zvec)
        yticklabels = ['ch. %i' % i for i in np.arange(len(zvec)) + 1]
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(r'$f$ (Hz)', labelpad=0.2)
        plt.axis('tight')
        ax.set_xlim([4E0, 4E2])

        cb = phlp.colorbar(fig, ax, im,
                           width=0.05, height=0.5,
                           hoffset=-0.05, voffset=0.0)
        cb.set_label('(-)', labelpad=0.1)

    return fig


if __name__ == '__main__':

    params = multicompartment_params()

    savefolders = [
        'simulation_output_modified_spontan',
        'simulation_output_modified_ac_input'
    ]
    letterslist = ['AB', 'CD']

    fig = fig_lfp_corr(
        params,
        savefolders,
        transient=200,
        channels=[
            0,
            3,
            7,
            11,
            13],
        Df=None,
        mlab=True,
        NFFT=256,
        noverlap=128,
        window=plt.mlab.window_hanning,
        letterslist=letterslist)

    fig.savefig('figure_11.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig('figure_11.eps', bbox_inches='tight', pad_inches=0.01)

    plt.show()
