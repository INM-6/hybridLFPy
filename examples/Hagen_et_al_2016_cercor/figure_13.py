#!/usr/bin/env python
# -*- coding: utf-8 -*-
from plot_methods import plotting_correlation, network_sketch, plot_population, plot_signal_sum, calc_signal_power, plot_population
from cellsim16popsParams_modified_spontan import multicompartment_params
import analysis_params
import plotting_helpers as phlp
from hybridLFPy import CachedNetwork, CachedFixedSpikesNetwork, GDF, helpers
import h5py
import os
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib import gridspec
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')


######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

params = multicompartment_params()

ana_params = analysis_params.params()


######################################
### IMPORT PANELS                  ###
######################################


######################################
### FIGURE                         ###
######################################


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  xxxxxx/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  (1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def fig_kernel_lfp(
    savefolders, params, transient=200, T=[
        800., 1000.], X='L5E', lags=[
            20., 20.], channels=[
                0, 3, 7, 11, 13]):
    '''
    This function calculates the  STA of LFP, extracts kernels and recontructs the LFP from kernels.

    Arguments
    ::
      transient : the time in milliseconds, after which the analysis should begin
                so as to avoid any starting transients
      X : id of presynaptic trigger population

    '''

    # Electrode geometry
    zvec = np.r_[params.electrodeParams['z']]

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    ana_params.set_PLOS_2column_fig_style(ratio=1)
    # Start the figure
    fig = plt.figure()
    fig.subplots_adjust(
        left=0.06,
        right=0.95,
        bottom=0.05,
        top=0.95,
        hspace=0.23,
        wspace=0.55)

    # create grid_spec
    gs = gridspec.GridSpec(2 * len(channels) + 1, 7)

    ###########################################################################
    # spikegen "network" activity
    ##########################################################################

    # path to simulation files
    params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                     'simulation_output_spikegen')
    params.figures_path = os.path.join(params.savefolder, 'figures')
    params.spike_output_path = os.path.join(params.savefolder,
                                            'processed_nest_output')
    params.networkSimParams['spike_output_path'] = params.spike_output_path

    # Get the spikegen LFP:
    f = h5py.File(
        os.path.join(
            params.savefolder,
            'RecExtElectrode_sum.h5'),
        'r')
    srate = f['srate'][()]
    tvec = np.arange(f['data'].shape[1]) * 1000. / srate

    # slice
    inds = (tvec < params.tstop) & (tvec >= transient)

    data_sg_raw = f['data'][()].astype(float)
    data_sg = data_sg_raw[:, inds]
    f.close()

    # kernel width
    kwidth = 20.

    # create some dummy spike times
    activationtimes = np.array([x * 100. for x in range(3, 11)] + [200.])
    networkSimSpikegen = CachedNetwork(**params.networkSimParams)

    x, y = networkSimSpikegen.get_xy([transient, params.tstop])

    ###########################################################################
    # Part A: spatiotemporal kernels, all presynaptic populations
    ##########################################################################

    titles = ['TC',
              'L23E/I',
              'LFP kernels \n L4E/I',
              'L5E/I',
              'L6E/I',
              ]

    COUNTER = 0
    for i, X__ in enumerate(([['TC']]) +
                            list(zip(params.X[1::2], params.X[2::2]))):
        ax = fig.add_subplot(gs[:len(channels), i])
        if i == 0:
            phlp.annotate_subplot(
                ax,
                ncols=7,
                nrows=4,
                letter=alphabet[0],
                linear_offset=0.02)

        for j, X_ in enumerate(X__):
            # create spikegen histogram for population Y
            cinds = np.arange(activationtimes[np.arange(-1, 8)][COUNTER] - kwidth,
                              activationtimes[np.arange(-1, 8)][COUNTER] + kwidth + 2)
            x0_sg = np.histogram(x[X_], bins=cinds)[0].astype(float)

            if X_ == ('TC'):
                color = 'k' if analysis_params.bw else analysis_params.colorE
                # lw = plt.rcParams['lines.linewidth']
                # zorder=1
            else:
                color = (
                    'k' if analysis_params.bw else analysis_params.colorE,
                    'gray' if analysis_params.bw else analysis_params.colorI)[j]
            lw = 0.75 if color in [
                'gray', 'r', 'b'] else plt.rcParams['lines.linewidth']
            zorder = 0 if 'I' in X_ else 1

            # plot kernel as correlation of spikegen LFP signal with delta
            # spike train
            xcorr, vlimround = plotting_correlation(params,
                                                    x0_sg / x0_sg.sum()**2,
                                                    data_sg_raw[:, cinds[:-1].astype(int)] * 1E3,
                                                    ax, normalize=False,
                                                    lag=kwidth,
                                                    color=color,
                                                    scalebar=False,
                                                    lw=lw, zorder=zorder)
            if i > 0:
                ax.set_yticklabels([])

            # Create scale bar
            ax.plot([kwidth, kwidth], [-1500 + j * 3 * 100, -1400 +
                    j * 3 * 100], lw=2, color=color, clip_on=False)
            ax.text(kwidth * 1.08, -1450 + j * 3 * 100, '%.1f $\\mu$V' %
                    vlimround, rotation='vertical', va='center')

            ax.set_xlim((-5, kwidth))
            ax.set_xticks([-20, 0, 20])
            ax.set_xticklabels([-20, 0, 20])

            COUNTER += 1

        ax.set_title(titles[i])

    ################################################
    # Iterate over savefolders
    ################################################

    for i, (savefolder, lag) in enumerate(zip(savefolders, lags)):

        # path to simulation files
        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path

        # load spike as database inside function to avoid buggy behaviour
        networkSim = CachedNetwork(**params.networkSimParams)

        # Get the Compound LFP: LFPsum : data[nchannels, timepoints ]
        f = h5py.File(
            os.path.join(
                params.savefolder,
                'RecExtElectrode_sum.h5'),
            'r')
        data_raw = f['data'][()]
        srate = f['srate'][()]
        tvec = np.arange(data_raw.shape[1]) * 1000. / srate
        # slice
        inds = (tvec < params.tstop) & (tvec >= transient)
        data = data_raw[:, inds]
        # subtract mean
        dataT = data.T - data.mean(axis=1)
        data = dataT.T
        f.close()

        # Get the spikegen LFP:
        f = h5py.File(os.path.join(os.path.split(params.savefolder)[0],
                                   'simulation_output_spikegen',
                                   'RecExtElectrode_sum.h5'), 'r')
        data_sg_raw = f['data'][()]

        f.close()

        #######################################################################
        # Part B: STA LFP
        #######################################################################

        titles = ['stLFP(%s)\n(spont.)' % X, 'stLFP(%s)\n(AC. mod.)' % X]
        ax = fig.add_subplot(gs[:len(channels), 5 + i])
        if i == 0:
            phlp.annotate_subplot(ax, ncols=15, nrows=4,
                                  letter=alphabet[i + 1], linear_offset=0.02)

        # collect the spikes x is the times, y is the id of the cell.
        x, y = networkSim.get_xy([0, params.tstop])

        # Get the spikes for the population of interest given as 'Y'
        bins = np.arange(0, params.tstop + 2) + 0.5
        x0_raw = np.histogram(x[X], bins=bins)[0]
        x0 = x0_raw[inds].astype(float)

        # correlation between firing rate and LFP deviation
        # from mean normalized by the number of spikes
        xcorr, vlimround = plotting_correlation(params,
                                                x0 / x0.sum(),
                                                data * 1E3,
                                                ax, normalize=False,
                                                #unit='%.3f mV',
                                                lag=lag,
                                                scalebar=False,
                                                color='k',
                                                title=titles[i],
                                                )

        # Create scale bar
        ax.plot([lag, lag],
                [-1500, -1400], lw=2, color='k',
                clip_on=False)
        ax.text(lag * 1.08, -1450, '%.1f $\\mu$V' % vlimround,
                rotation='vertical', va='center')

        [Xind] = np.where(np.array(networkSim.X) == X)[0]

        # create spikegen histogram for population Y
        x0_sg = np.zeros(x0.shape, dtype=float)
        x0_sg[activationtimes[Xind].astype(int)] += params.N_X[Xind]

        ax.set_yticklabels([])
        ax.set_xticks([-lag, 0, lag])
        ax.set_xticklabels([-lag, 0, lag])

        #######################################################################
        # Part C, F: LFP and reconstructed LFP
        #######################################################################

        # create grid_spec
        gsb = gridspec.GridSpec(2 * len(channels) + 1, 8)

        ax = fig.add_subplot(gsb[1 + len(channels):, (i * 4):(i * 4 + 2)])
        phlp.annotate_subplot(ax,
                              ncols=8 / 2.,
                              nrows=4,
                              letter=alphabet[i * 3 + 2],
                              linear_offset=0.02)

        # extract kernels, force negative lags to be zero
        kernels = np.zeros((len(params.N_X), 16, int(kwidth * 2)))
        for j in range(len(params.X)):
            kernels[j, :, int(kwidth):] = data_sg_raw[:, (j + 2)
                                                      * 100:int(kwidth) + (j + 2) * 100] / params.N_X[j]

        LFP_reconst_raw = np.zeros(data_raw.shape)

        for j, pop in enumerate(params.X):
            x0_raw = np.histogram(x[pop], bins=bins)[0].astype(float)
            for ch in range(kernels.shape[1]):
                LFP_reconst_raw[ch] += np.convolve(x0_raw, kernels[j, ch],
                                                   'same')

        # slice
        LFP_reconst = LFP_reconst_raw[:, inds]
        # subtract mean
        LFP_reconstT = LFP_reconst.T - LFP_reconst.mean(axis=1)
        LFP_reconst = LFP_reconstT.T
        vlimround = plot_signal_sum(ax, params,
                                    fname=os.path.join(params.savefolder,
                                                       'RecExtElectrode_sum.h5'),
                                    unit='mV', scalebar=True,
                                    T=T, ylim=[-1550, 50],
                                    color='k', label='$real$',
                                    rasterized=False,
                                    zorder=1)

        plot_signal_sum(ax, params, fname=LFP_reconst_raw,
                        unit='mV', scaling_factor=1., scalebar=False,
                        vlimround=vlimround,
                        T=T, ylim=[-1550, 50],
                        color='gray' if analysis_params.bw else analysis_params.colorP,
                        label='$reconstr$',
                        rasterized=False,
                        lw=1, zorder=0)
        ax.set_title('LFP & population \n rate predictor')
        if i > 0:
            ax.set_yticklabels([])

        #######################################################################
        # Part D,G: Correlation coefficient
        #######################################################################

        ax = fig.add_subplot(gsb[1 + len(channels):, i * 4 + 2:i * 4 + 3])
        phlp.remove_axis_junk(ax)
        phlp.annotate_subplot(ax,
                              ncols=8. / 1,
                              nrows=4,
                              letter=alphabet[i * 3 + 3],
                              linear_offset=0.02)

        cc = np.zeros(len(zvec))
        for ch in np.arange(len(zvec)):
            cc[ch] = np.corrcoef(data[ch], LFP_reconst[ch])[1, 0]

        ax.barh(
            zvec,
            cc,
            height=80,
            align='center',
            color='0.5',
            linewidth=0.5)

        # superimpose the chance level, obtained by mixing one input vector n times
        # while keeping the other fixed. We show boxes drawn left to right where
        # these denote mean +/- two standard deviations.
        N = 1000
        method = 'randphase'  # or 'permute'
        chance = np.zeros((cc.size, N))
        for ch in np.arange(len(zvec)):
            x1 = LFP_reconst[ch]
            x1 -= x1.mean()
            if method == 'randphase':
                x0 = data[ch]
                x0 -= x0.mean()
                X00 = np.fft.fft(x0)
            for n in range(N):
                if method == 'permute':
                    x0 = np.random.permutation(datas[ch])
                elif method == 'randphase':
                    X0 = np.copy(X00)
                    # random phase information such that spectra is preserved
                    theta = np.random.uniform(
                        0, 2 * np.pi, size=X0.size // 2 - 1)
                    # half-sided real and imaginary component
                    real = abs(X0[1:X0.size // 2]) * np.cos(theta)
                    imag = abs(X0[1:X0.size // 2]) * np.sin(theta)

                    # account for the antisymmetric phase values
                    X0.imag[1:imag.size + 1] = imag
                    X0.imag[imag.size + 2:] = -imag[::-1]
                    X0.real[1:real.size + 1] = real
                    X0.real[real.size + 2:] = real[::-1]
                    x0 = np.fft.ifft(X0).real

                chance[ch, n] = np.corrcoef(x0, x1)[1, 0]

        # p-values, compute the fraction of chance correlations > cc at each
        # channel
        p = []
        for h, x in enumerate(cc):
            p += [(chance[h, ] >= x).sum() / float(N)]

        print(('p-values:', p))

        # compute the 99% percentile of the chance data
        right = np.percentile(chance, 99, axis=-1)

        ax.plot(right, zvec, ':', color='k', lw=1.)
        ax.set_ylim([-1550, 50])
        ax.set_yticklabels([])
        ax.set_yticks(zvec)
        ax.set_xlim([0, 1.])
        ax.set_xticks([0.0, 0.5, 1])
        ax.yaxis.tick_left()
        ax.set_xlabel('$cc$ (-)', labelpad=0.1)
        ax.set_title('corr. \n coef.')

        print('correlation coefficients:')
        print(cc)

        #######################################################################
        # Part E,H: Power spectra
        #######################################################################

        # compute PSDs ratio between ground truth and estimate
        freqs, PSD_data = calc_signal_power(params, fname=data,
                                            transient=transient, Df=None,
                                            mlab=True,
                                            NFFT=256, noverlap=128,
                                            window=plt.mlab.window_hanning)
        freqs, PSD_LFP_reconst = calc_signal_power(params, fname=LFP_reconst,
                                                   transient=transient, Df=None,
                                                   mlab=True,
                                                   NFFT=256, noverlap=128,
                                                   window=plt.mlab.window_hanning)

        zv = np.r_[params.electrodeParams['z']]
        zv = np.r_[zv, zv[-1] + np.diff(zv)[-1]]
        inds = freqs >= 1  # frequencies greater than 1 Hz

        for j, ch in enumerate(channels):

            ax = fig.add_subplot(
                gsb[1 + len(channels) + j, (i * 4 + 3):(i * 4 + 4)])
            if j == 0:
                phlp.annotate_subplot(ax,
                                      ncols=8. / 1,
                                      nrows=4.5 * len(channels),
                                      letter=alphabet[i * 3 + 4],
                                      linear_offset=0.02)
                ax.set_title('PSD')

            phlp.remove_axis_junk(ax)
            ax.loglog(freqs[inds], PSD_data[ch, inds], 'k', label='LFP',
                      clip_on=True, zorder=1)
            ax.loglog(freqs[inds], PSD_LFP_reconst[ch, inds],
                      'gray' if analysis_params.bw else analysis_params.colorP,
                      label='predictor', clip_on=True, lw=1, zorder=0)
            ax.set_xlim([4E0, 4E2])
            ax.set_ylim([1E-8, 1E-4])
            ax.tick_params(axis='y', which='major', pad=0)
            ax.set_yticks([1E-8, 1E-6, 1E-4])
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.text(0.8, 0.9, 'ch. %i' % (ch + 1),
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=6,
                    transform=ax.transAxes)

            if j == 0:
                ax.set_ylabel('(mV$^2$/Hz)', labelpad=0.)
            if j > 0:
                ax.set_yticklabels([])
            if j == len(channels) - 1:
                ax.set_xlabel(r'$f$ (Hz)', labelpad=0.)
            else:
                ax.set_xticklabels([])

    return fig, PSD_LFP_reconst, PSD_data

#
# def fig_kernel_lfp_CINPLA(savefolders, params, transient=200, X='L5E', lags=[20, 20]):
#
#     '''
#     This function calculates the  STA of LFP, extracts kernels and recontructs the LFP from kernels.
#
#     kwargs:
#     ::
#       transient : the time in milliseconds, after which the analysis should begin
#                 so as to avoid any starting transients
#       X : id of presynaptic trigger population
#
#     '''
#
#     # Electrode geometry
#     zvec = np.r_[params.electrodeParams['z']]
#
#     alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#
#
#
#     ana_params.set_PLOS_2column_fig_style(ratio=0.5)
#     # Start the figure
#     fig = plt.figure()
#     fig.subplots_adjust(left=0.06, right=0.95, bottom=0.075, top=0.925, hspace=0.23, wspace=0.55)
#
#     # create grid_spec
#     gs = gridspec.GridSpec(1, 7)
#
#
#     ###########################################################################
#     # Part A: spikegen "network" activity
#     ############################################################################
#
#     # path to simulation files
#     params.savefolder = 'simulation_output_spikegen'
#     params.figures_path = os.path.join(params.savefolder, 'figures')
#     params.spike_output_path = os.path.join(params.savefolder,
#                                                        'processed_nest_output')
#     params.networkSimParams['spike_output_path'] = params.spike_output_path
#
#
#     # Get the spikegen LFP:
#     f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
#     srate = f['srate'][()]
#     tvec = np.arange(f['data'].shape[1]) * 1000. / srate
#
#     # slice
#     inds = (tvec < params.tstop) & (tvec >= transient)
#
#     data_sg_raw = f['data'][()].astype(float)
#     f.close()
#     #
#     # kernel width
#     kwidth = 20
#
#     # extract kernels
#     kernels = np.zeros((len(params.N_X), 16, 100))
#     for j in range(len(params.X)):
#         kernels[j] = data_sg_raw[:, 100+kwidth+j*100:100+kwidth+(j+1)*100] / params.N_X[j]
#
#
#     # create some dummy spike times
#     activationtimes = np.array([x*100 for x in range(3,11)] + [200])
#     networkSimSpikegen = CachedNetwork(**params.networkSimParams)
#
#     x, y = networkSimSpikegen.get_xy([transient, params.tstop])
#
#     ###########################################################################
#     # Part A: spatiotemporal kernels, all presynaptic populations
#     ############################################################################
#
#     titles = ['TC',
#               'L23E/I',
#               'LFP kernels \n L4E/I',
#               'L5E/I',
#               'L6E/I',
#               ]
#
#     COUNTER = 0
#     for i, X__ in enumerate(([['TC']]) + list(zip(params.X[1::2], params.X[2::2]))):
#         ax = fig.add_subplot(gs[0, i])
#         if i == 0:
#             phlp.annotate_subplot(ax, ncols=7, nrows=4, letter=alphabet[0], linear_offset=0.02)
#
#         for j, X_ in enumerate(X__):
#             # create spikegen histogram for population Y
#             cinds = np.arange(activationtimes[np.arange(-1, 8)][COUNTER]-kwidth,
#                               activationtimes[np.arange(-1, 8)][COUNTER]+kwidth+2)
#             x0_sg = np.histogram(x[X_], bins=tvec[cinds])[0].astype(float)
#
#             if X_ == ('TC'):
#                 color='r'
#             else:
#                 color=('r', 'b')[j]
#
#
#             # plot kernel as correlation of spikegen LFP signal with delta spike train
#             xcorr, vlimround = plotting_correlation(x0_sg/x0_sg.sum()**2,
#                                  data_sg_raw[:, cinds[:-1].astype(int)]*1E3,
#                                  ax, normalize=False,
#                                  lag=kwidth,
#                                  color=color,
#                                  scalebar=False)
#             if i > 0:
#                 ax.set_yticklabels([])
#
#             ## Create scale bar
#             ax.plot([kwidth, kwidth],
#                 [-1500 + j*3*100, -1400 + j*3*100], lw=2, color=color,
#                 clip_on=False)
#             ax.text(kwidth*1.08, -1450 + j*3*100, '%.1f $\mu$V' % vlimround,
#                         rotation='vertical', va='center')
#
#             ax.set_xlim((-5, kwidth))
#             ax.set_xticks([-20, 0, 20])
#             ax.set_xticklabels([-20, 0, 20])
#
#             COUNTER += 1
#
#         ax.set_title(titles[i])
#
#
#     for i, (savefolder, lag) in enumerate(zip(savefolders, lags)):
#
#         # path to simulation files
#         params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
#                                          savefolder)
#         params.figures_path = os.path.join(params.savefolder, 'figures')
#         params.spike_output_path = os.path.join(params.savefolder,
#                                                 'processed_nest_output')
#         params.networkSimParams['spike_output_path'] = params.spike_output_path
#
#         #load spike as database inside function to avoid buggy behaviour
#         networkSim = CachedNetwork(**params.networkSimParams)
#
#
#
#         # Get the Compound LFP: LFPsum : data[nchannels, timepoints ]
#         f = h5py.File(os.path.join(params.savefolder, 'LFPsum.h5'))
#         data_raw = f['data'][()]
#         srate = f['srate'][()]
#         tvec = np.arange(data_raw.shape[1]) * 1000. / srate
#         # slice
#         inds = (tvec < params.tstop) & (tvec >= transient)
#         data = data_raw[:,inds]
#         # subtract mean
#         dataT = data.T - data.mean(axis=1)
#         data = dataT.T
#         f.close()
#
#         # Get the spikegen LFP:
#         f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
#         data_sg_raw = f['data'][()]
#         # slice
#         data_sg = data_sg_raw[:,inds[data_sg_raw.shape[1]]]
#         f.close()
#
#
#
#
#         ########################################################################
#         # Part B: STA LFP
#         ########################################################################
#
#         ax = fig.add_subplot(gs[0, 5 + i])
#         phlp.annotate_subplot(ax, ncols=15, nrows=4, letter=alphabet[i+1],
#                               linear_offset=0.02)
#
#         # collect the spikes x is the times, y is the id of the cell.
#         x, y = networkSim.get_xy([0,params.tstop])
#
#         # Get the spikes for the population of interest given as 'Y'
#         bins = np.arange(0, params.tstop+2)
#         x0_raw = np.histogram(x[X], bins=bins)[0]
#         x0 = x0_raw[inds].astype(float)
#
#         # correlation between firing rate and LFP deviation
#         # from mean normalized by the number of spikes
#         xcorr, vlimround = plotting_correlation(x0/x0.sum(),
#                              data*1E3,
#                              ax, normalize=False,
#                              #unit='%.3f mV',
#                              lag=lag,
#                              scalebar=False,
#                              color='k',
#                              title='stLFP\n(trigger %s)' %X,
#                              )
#
#         # Create scale bar
#         ax.plot([lag, lag],
#             [-1500, -1400], lw=2, color='k',
#             clip_on=False)
#         ax.text(lag*1.04, -1450, '%.1f $\mu$V' % vlimround,
#                     rotation='vertical', va='center')
#
#
#         [Xind] = np.where(np.array(networkSim.X) == X)[0]
#
#         # create spikegen histogram for population Y
#         x0_sg = np.zeros(x0.shape, dtype=float)
#         x0_sg[activationtimes[Xind]] += params.N_X[Xind]
#
#         ax.set_yticklabels([])
#         ax.set_xticks([-lag, 0, lag])
#         ax.set_xticklabels([-lag, 0, lag])
#
#
#     return fig
#
#
# def fig_kernel_lfp_EITN_I(savefolders, params, transient=200, T=[800., 1000.], X='L5E',
#                    lags=[20, 20], channels=[0,3,7,11,13]):
#
#     '''
#     This function calculates the  STA of LFP, extracts kernels and recontructs the LFP from kernels.
#
#     Arguments
#     ::
#       transient : the time in milliseconds, after which the analysis should begin
#                 so as to avoid any starting transients
#       X : id of presynaptic trigger population
#
#     '''
#
#     # Electrode geometry
#     zvec = np.r_[params.electrodeParams['z']]
#
#     alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#
#
#
#     ana_params.set_PLOS_2column_fig_style(ratio=0.5)
#     # Start the figure
#     fig = plt.figure()
#     fig.subplots_adjust(left=0.06, right=0.95, bottom=0.08, top=0.90, hspace=0.23, wspace=0.55)
#
#     # create grid_spec
#     gs = gridspec.GridSpec(1, 7)
#
#
#     ###########################################################################
#     # spikegen "network" activity
#     ############################################################################
#
#
#     # path to simulation files
#     params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
#                                          'simulation_output_spikegen')
#     params.figures_path = os.path.join(params.savefolder, 'figures')
#     params.spike_output_path = os.path.join(params.savefolder,
#                                                        'processed_nest_output')
#     params.networkSimParams['spike_output_path'] = params.spike_output_path
#
#
#     # Get the spikegen LFP:
#     f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
#     srate = f['srate'][()]
#     tvec = np.arange(f['data'].shape[1]) * 1000. / srate
#
#     # slice
#     inds = (tvec < params.tstop) & (tvec >= transient)
#
#     data_sg_raw = f['data'][()].astype(float)
#     data_sg = data_sg_raw[:, inds]
#     f.close()
#
#     # kernel width
#     kwidth = 20
#
#     # create some dummy spike times
#     activationtimes = np.array([x*100 for x in range(3,11)] + [200])
#     networkSimSpikegen = CachedNetwork(**params.networkSimParams)
#
#     x, y = networkSimSpikegen.get_xy([transient, params.tstop])
#
#
#     ###########################################################################
#     # Part A: spatiotemporal kernels, all presynaptic populations
#     ############################################################################
#
#     titles = ['TC',
#               'L23E/I',
#               'LFP kernels \n L4E/I',
#               'L5E/I',
#               'L6E/I',
#               ]
#
#     COUNTER = 0
#     for i, X__ in enumerate(([['TC']]) + list(zip(params.X[1::2], params.X[2::2]))):
#         ax = fig.add_subplot(gs[0, i])
#         if i == 0:
#             phlp.annotate_subplot(ax, ncols=7, nrows=4, letter=alphabet[0], linear_offset=0.02)
#
#         for j, X_ in enumerate(X__):
#             # create spikegen histogram for population Y
#             cinds = np.arange(activationtimes[np.arange(-1, 8)][COUNTER]-kwidth,
#                               activationtimes[np.arange(-1, 8)][COUNTER]+kwidth+2)
#             x0_sg = np.histogram(x[X_], bins=cinds)[0].astype(float)
#
#             if X_ == ('TC'):
#                 color='r'
#             else:
#                 color=('r', 'b')[j]
#
#
#             # plot kernel as correlation of spikegen LFP signal with delta spike train
#             xcorr, vlimround = plotting_correlation(params,
#                                                     x0_sg/x0_sg.sum()**2,
#                                                     data_sg_raw[:, cinds[:-1].astype(int)]*1E3,
#                                                     ax, normalize=False,
#                                                     lag=kwidth,
#                                                     color=color,
#                                                     scalebar=False)
#             if i > 0:
#                 ax.set_yticklabels([])
#
#             ## Create scale bar
#             ax.plot([kwidth, kwidth],
#                 [-1500 + j*3*100, -1400 + j*3*100], lw=2, color=color,
#                 clip_on=False)
#             ax.text(kwidth*1.08, -1450 + j*3*100, '%.1f $\mu$V' % vlimround,
#                         rotation='vertical', va='center')
#
#             ax.set_xlim((-5, kwidth))
#             ax.set_xticks([-20, 0, 20])
#             ax.set_xticklabels([-20, 0, 20])
#
#             COUNTER += 1
#
#         ax.set_title(titles[i])
#
#
#     ################################################
#     # Iterate over savefolders
#     ################################################
#
#     for i, (savefolder, lag) in enumerate(zip(savefolders, lags)):
#
#         # path to simulation files
#         params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
#                                          savefolder)
#         params.figures_path = os.path.join(params.savefolder, 'figures')
#         params.spike_output_path = os.path.join(params.savefolder,
#                                                 'processed_nest_output')
#         params.networkSimParams['spike_output_path'] = params.spike_output_path
#
#         #load spike as database inside function to avoid buggy behaviour
#         networkSim = CachedNetwork(**params.networkSimParams)
#
#
#
#         # Get the Compound LFP: LFPsum : data[nchannels, timepoints ]
#         f = h5py.File(os.path.join(params.savefolder, 'LFPsum.h5'))
#         data_raw = f['data'][()]
#         srate = f['srate'][()]
#         tvec = np.arange(data_raw.shape[1]) * 1000. / srate
#         # slice
#         inds = (tvec < params.tstop) & (tvec >= transient)
#         data = data_raw[:, inds]
#         # subtract mean
#         dataT = data.T - data.mean(axis=1)
#         data = dataT.T
#         f.close()
#
#         # Get the spikegen LFP:
#         f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
#         data_sg_raw = f['data'][()]
#
#         f.close()
#
#
#
#
#         ########################################################################
#         # Part B: STA LFP
#         ########################################################################
#
#         titles = ['stLFP(%s)\n(spont.)' % X, 'stLFP(%s)\n(AC. mod.)' % X]
#         ax = fig.add_subplot(gs[0, 5 + i])
#         if i == 0:
#             phlp.annotate_subplot(ax, ncols=15, nrows=4, letter=alphabet[i+1],
#                                   linear_offset=0.02)
#
#         #collect the spikes x is the times, y is the id of the cell.
#         x, y = networkSim.get_xy([0,params.tstop])
#
#         # Get the spikes for the population of interest given as 'Y'
#         bins = np.arange(0, params.tstop+2) + 0.5
#         x0_raw = np.histogram(x[X], bins=bins)[0]
#         x0 = x0_raw[inds].astype(float)
#
#         # correlation between firing rate and LFP deviation
#         # from mean normalized by the number of spikes
#         xcorr, vlimround = plotting_correlation(params,
#                                                 x0/x0.sum(),
#                                                 data*1E3,
#                                                 ax, normalize=False,
#                                                 #unit='%.3f mV',
#                                                 lag=lag,
#                                                 scalebar=False,
#                                                 color='k',
#                                                 title=titles[i],
#                                                 )
#
#         # Create scale bar
#         ax.plot([lag, lag],
#             [-1500, -1400], lw=2, color='k',
#             clip_on=False)
#         ax.text(lag*1.08, -1450, '%.1f $\mu$V' % vlimround,
#                     rotation='vertical', va='center')
#
#
#         [Xind] = np.where(np.array(networkSim.X) == X)[0]
#
#         # create spikegen histogram for population Y
#         x0_sg = np.zeros(x0.shape, dtype=float)
#         x0_sg[activationtimes[Xind]] += params.N_X[Xind]
#
#
#         ax.set_yticklabels([])
#         ax.set_xticks([-lag, 0, lag])
#         ax.set_xticklabels([-lag, 0, lag])
#
#     return fig
#
#
# def fig_kernel_lfp_EITN_II(savefolders, params, transient=200, T=[800., 1000.], X='L5E',
#                    lags=[20, 20], channels=[0,3,7,11,13]):
#
#     '''
#     This function calculates the  STA of LFP, extracts kernels and recontructs the LFP from kernels.
#
#     Arguments
#     ::
#       transient : the time in milliseconds, after which the analysis should begin
#                 so as to avoid any starting transients
#       X : id of presynaptic trigger population
#
#     '''
#
#     # Electrode geometry
#     zvec = np.r_[params.electrodeParams['z']]
#
#     alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#
#
#
#     ana_params.set_PLOS_2column_fig_style(ratio=0.5)
#     # Start the figure
#     fig = plt.figure()
#     fig.subplots_adjust(left=0.06, right=0.95, bottom=0.08, top=0.9, hspace=0.23, wspace=0.55)
#
#     # create grid_spec
#     gs = gridspec.GridSpec(len(channels), 7)
#
#
#     ###########################################################################
#     # spikegen "network" activity
#     ############################################################################
#
#
#     # path to simulation files
#     savefolder = 'simulation_output_spikegen'
#     params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
#                                          savefolder)
#     params.figures_path = os.path.join(params.savefolder, 'figures')
#     params.spike_output_path = os.path.join(params.savefolder,
#                                                        'processed_nest_output')
#     params.networkSimParams['spike_output_path'] = params.spike_output_path
#
#
#     # Get the spikegen LFP:
#     f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
#     srate = f['srate'][()]
#     tvec = np.arange(f['data'].shape[1]) * 1000. / srate
#
#     # slice
#     inds = (tvec < params.tstop) & (tvec >= transient)
#
#     data_sg_raw = f['data'][()].astype(float)
#     data_sg = data_sg_raw[:, inds]
#     f.close()
#
#     # kernel width
#     kwidth = 20
#
#     # create some dummy spike times
#     activationtimes = np.array([x*100 for x in range(3,11)] + [200])
#     networkSimSpikegen = CachedNetwork(**params.networkSimParams)
#
#     x, y = networkSimSpikegen.get_xy([transient, params.tstop])
#
#
#     ############################################################################
#     ## Part A: spatiotemporal kernels, all presynaptic populations
#     #############################################################################
#     #
#     #titles = ['TC',
#     #          'L23E/I',
#     #          'LFP kernels \n L4E/I',
#     #          'L5E/I',
#     #          'L6E/I',
#     #          ]
#     #
#     #COUNTER = 0
#     #for i, X__ in enumerate(([['TC']]) + zip(params.X[1::2], params.X[2::2])):
#     #    ax = fig.add_subplot(gs[:len(channels), i])
#     #    if i == 0:
#     #        phlp.annotate_subplot(ax, ncols=7, nrows=4, letter=alphabet[0], linear_offset=0.02)
#     #
#     #    for j, X_ in enumerate(X__):
#     #        # create spikegen histogram for population Y
#     #        cinds = np.arange(activationtimes[np.arange(-1, 8)][COUNTER]-kwidth,
#     #                          activationtimes[np.arange(-1, 8)][COUNTER]+kwidth+2)
#     #        x0_sg = np.histogram(x[X_], bins=cinds)[0].astype(float)
#     #
#     #        if X_ == ('TC'):
#     #            color='r'
#     #        else:
#     #            color=('r', 'b')[j]
#     #
#     #
#     #        # plot kernel as correlation of spikegen LFP signal with delta spike train
#     #        xcorr, vlimround = plotting_correlation(params,
#     #                                                x0_sg/x0_sg.sum()**2,
#     #                                                data_sg_raw[:, cinds[:-1]]*1E3,
#     #                                                ax, normalize=False,
#     #                                                lag=kwidth,
#     #                                                color=color,
#     #                                                scalebar=False)
#     #        if i > 0:
#     #            ax.set_yticklabels([])
#     #
#     #        ## Create scale bar
#     #        ax.plot([kwidth, kwidth],
#     #            [-1500 + j*3*100, -1400 + j*3*100], lw=2, color=color,
#     #            clip_on=False)
#     #        ax.text(kwidth*1.08, -1450 + j*3*100, '%.1f $\mu$V' % vlimround,
#     #                    rotation='vertical', va='center')
#     #
#     #        ax.set_xlim((-5, kwidth))
#     #        ax.set_xticks([-20, 0, 20])
#     #        ax.set_xticklabels([-20, 0, 20])
#     #
#     #        COUNTER += 1
#     #
#     #    ax.set_title(titles[i])
#
#
#     ################################################
#     # Iterate over savefolders
#     ################################################
#
#     for i, (savefolder, lag) in enumerate(zip(savefolders, lags)):
#
#         # path to simulation files
#         params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
#                                          savefolder)
#         params.figures_path = os.path.join(params.savefolder, 'figures')
#         params.spike_output_path = os.path.join(params.savefolder,
#                                                 'processed_nest_output')
#         params.networkSimParams['spike_output_path'] = params.spike_output_path
#
#         #load spike as database inside function to avoid buggy behaviour
#         networkSim = CachedNetwork(**params.networkSimParams)
#
#
#
#         # Get the Compound LFP: LFPsum : data[nchannels, timepoints ]
#         f = h5py.File(os.path.join(params.savefolder, 'LFPsum.h5'))
#         data_raw = f['data'][()]
#         srate = f['srate'][()]
#         tvec = np.arange(data_raw.shape[1]) * 1000. / srate
#         # slice
#         inds = (tvec < params.tstop) & (tvec >= transient)
#         data = data_raw[:, inds]
#         # subtract mean
#         dataT = data.T - data.mean(axis=1)
#         data = dataT.T
#         f.close()
#
#         # Get the spikegen LFP:
#         f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
#         data_sg_raw = f['data'][()]
#
#         f.close()
#         #
#         #
#         #
#         #
#         #########################################################################
#         ## Part B: STA LFP
#         #########################################################################
#         #
#         #titles = ['staLFP(%s)\n(spont.)' % X, 'staLFP(%s)\n(AC. mod.)' % X]
#         #ax = fig.add_subplot(gs[:len(channels), 5 + i])
#         #if i == 0:
#         #    phlp.annotate_subplot(ax, ncols=15, nrows=4, letter=alphabet[i+1],
#         #                          linear_offset=0.02)
#         #
#         #collect the spikes x is the times, y is the id of the cell.
#         x, y = networkSim.get_xy([0,params.tstop])
#         #
#         ## Get the spikes for the population of interest given as 'Y'
#         bins = np.arange(0, params.tstop+2) + 0.5
#         x0_raw = np.histogram(x[X], bins=bins)[0]
#         x0 = x0_raw[inds].astype(float)
#         #
#         ## correlation between firing rate and LFP deviation
#         ## from mean normalized by the number of spikes
#         #xcorr, vlimround = plotting_correlation(params,
#         #                                        x0/x0.sum(),
#         #                                        data*1E3,
#         #                                        ax, normalize=False,
#         #                                        #unit='%.3f mV',
#         #                                        lag=lag,
#         #                                        scalebar=False,
#         #                                        color='k',
#         #                                        title=titles[i],
#         #                                        )
#         #
#         ## Create scale bar
#         #ax.plot([lag, lag],
#         #    [-1500, -1400], lw=2, color='k',
#         #    clip_on=False)
#         #ax.text(lag*1.08, -1450, '%.1f $\mu$V' % vlimround,
#         #            rotation='vertical', va='center')
#         #
#         #
#         #[Xind] = np.where(np.array(networkSim.X) == X)[0]
#         #
#         ## create spikegen histogram for population Y
#         #x0_sg = np.zeros(x0.shape, dtype=float)
#         #x0_sg[activationtimes[Xind]] += params.N_X[Xind]
#         #
#         #
#         #ax.set_yticklabels([])
#         #ax.set_xticks([-lag, 0, lag])
#         #ax.set_xticklabels([-lag, 0, lag])
#
#
#         ###########################################################################
#         # Part C, F: LFP and reconstructed LFP
#         ############################################################################
#
#         # create grid_spec
#         gsb = gridspec.GridSpec(len(channels), 8)
#
#
#         ax = fig.add_subplot(gsb[:, (i*4):(i*4+2)])
#         phlp.annotate_subplot(ax, ncols=8/2., nrows=4, letter=alphabet[i*3+2],
#                               linear_offset=0.02)
#
#         # extract kernels, force negative lags to be zero
#         kernels = np.zeros((len(params.N_X), 16, kwidth*2))
#         for j in range(len(params.X)):
#             kernels[j, :, kwidth:] = data_sg_raw[:, (j+2)*100:kwidth+(j+2)*100]/params.N_X[j]
#
#         LFP_reconst_raw = np.zeros(data_raw.shape)
#
#         for j, pop in enumerate(params.X):
#             x0_raw = np.histogram(x[pop], bins=bins)[0].astype(float)
#             for ch in range(kernels.shape[1]):
#                 LFP_reconst_raw[ch] += np.convolve(x0_raw, kernels[j, ch],
#                                                    'same')
#
#         # slice
#         LFP_reconst = LFP_reconst_raw[:, inds]
#         # subtract mean
#         LFP_reconstT = LFP_reconst.T - LFP_reconst.mean(axis=1)
#         LFP_reconst = LFP_reconstT.T
#         vlimround = plot_signal_sum(ax, params,
#                                     fname=os.path.join(params.savefolder,
#                                                            'LFPsum.h5'),
#                     unit='mV', scalebar=True,
#                         T=T, ylim=[-1550, 50],
#                         color='k', label='$real$',
#                         rasterized=False)
#
#         plot_signal_sum(ax, params, fname=LFP_reconst_raw,
#                     unit='mV', scaling_factor= 1., scalebar=False,
#                         vlimround=vlimround,
#                         T=T, ylim=[-1550, 50],
#                         color='r', label='$reconstr$',
#                         rasterized=False)
#         ax.set_title('LFP & population \n rate predictor')
#         if i > 0:
#             ax.set_yticklabels([])
#
#
#
#         ###########################################################################
#         # Part D,G: Correlation coefficient
#         ############################################################################
#
#         ax = fig.add_subplot(gsb[:, i*4+2:i*4+3])
#         phlp.remove_axis_junk(ax)
#         phlp.annotate_subplot(ax, ncols=8./1, nrows=4, letter=alphabet[i*3+3],
#                               linear_offset=0.02)
#
#         cc = np.zeros(len(zvec))
#         for ch in np.arange(len(zvec)):
#             cc[ch] = np.corrcoef(data[ch], LFP_reconst[ch])[1, 0]
#
#         ax.barh(zvec, cc, height=90, align='center', color='1', linewidth=0.5)
#         ax.set_ylim([-1550, 50])
#         ax.set_yticklabels([])
#         ax.set_yticks(zvec)
#         ax.set_xlim([0.0, 1.])
#         ax.set_xticks([0.0, 0.5, 1])
#         ax.yaxis.tick_left()
#         ax.set_xlabel('$cc$ (-)', labelpad=0.1)
#         ax.set_title('corr. \n coef.')
#
#         print('correlation coefficients:')
#         print(cc)
#
#
#         ###########################################################################
#         # Part E,H: Power spectra
#         ############################################################################
#
#
#         #compute PSDs ratio between ground truth and estimate
#         freqs, PSD_data = calc_signal_power(params, fname=data,
#                                             transient=transient, Df=None, mlab=True,
#                                             NFFT=256, noverlap=128,
#                                             window=plt.mlab.window_hanning)
#         freqs, PSD_LFP_reconst = calc_signal_power(params, fname=LFP_reconst,
#                                             transient=transient, Df=None, mlab=True,
#                                             NFFT=256, noverlap=128,
#                                             window=plt.mlab.window_hanning)
#
#         zv = np.r_[params.electrodeParams['z']]
#         zv = np.r_[zv, zv[-1] + np.diff(zv)[-1]]
#         inds = freqs >= 1  # frequencies greater than 1 Hz
#
#         for j, ch in enumerate(channels):
#
#             ax = fig.add_subplot(gsb[j, (i*4+3):(i*4+4)])
#             if j == 0:
#                 phlp.annotate_subplot(ax, ncols=8./1, nrows=4.5*len(channels),
#                                       letter=alphabet[i*3+4], linear_offset=0.02)
#                 ax.set_title('PSD')
#
#
#             phlp.remove_axis_junk(ax)
#             ax.loglog(freqs[inds], PSD_data[ch, inds], 'k', label='LFP', clip_on=True)
#             ax.loglog(freqs[inds], PSD_LFP_reconst[ch, inds], 'r', label='predictor', clip_on=True)
#             ax.set_xlim([4E0,4E2])
#             ax.set_ylim([1E-8, 1E-4])
#             ax.tick_params(axis='y', which='major', pad=0)
#             ax.set_yticks([1E-8,1E-6,1E-4])
#             ax.yaxis.set_minor_locator(plt.NullLocator())
#             ax.text(0.8, 0.9, 'ch. %i' % (ch+1),
#                     horizontalalignment='left',
#                     verticalalignment='center',
#                     fontsize=6,
#                     transform=ax.transAxes)
#
#             if j == 0:
#                 ax.set_ylabel('(mV$^2$/Hz)', labelpad=0.)
#             if j > 0:
#                 ax.set_yticklabels([])
#             if j == len(channels)-1:
#                 ax.set_xlabel(r'$f$ (Hz)', labelpad=0.)
#             else:
#                 ax.set_xticklabels([])
#
#
#
#     return fig, PSD_LFP_reconst, PSD_data


if __name__ == '__main__':

    params = multicompartment_params()

    savefolders = [
        'simulation_output_modified_spontan',
        'simulation_output_modified_ac_input',
    ]
    lags = [20., 50.]

    for X in ['L5E']:  # params.X:
        fig, PSD_LFP_reconst, PSD_data = fig_kernel_lfp(
            savefolders, params, transient=100., T=[
                800., 900.], X=X, lags=lags)

        fig.savefig(
            'figure_13_{}.pdf'.format(X),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0)
        fig.savefig(
            'figure_13_{}.eps'.format(X),
            bbox_inches='tight',
            pad_inches=0.01)

    plt.show()
