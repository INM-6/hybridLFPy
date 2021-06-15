#!/usr/bin/env python
# -*- coding: utf-8 -*-
import analysis_params
from cellsim16popsParams_modified_spontan import multicompartment_params
from plot_methods import plot_signal_sum, plot_signal_sum_colorplot
import plotting_helpers as phlp
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')


def fig_exc_inh_contrib(
    fig, axes, params, savefolders, T=[
        800, 1000], transient=200, panel_labels='FGHIJ', show_xlabels=True):
    '''
    plot time series LFPs and CSDs with signal variances as function of depth
    for the cases with all synapses intact, or knocking out excitatory
    input or inhibitory input to the postsynaptic target region

    args:
    ::
        fig :
        axes :
        savefolders : list of simulation output folders
        T : list of ints, first and last time sample
        transient : int, duration of transient period

    returns:
    ::

        matplotlib.figure.Figure object

    '''
    # params = multicompartment_params()
    # ana_params = analysis_params.params()

    # file name types
    file_names = [
        'LaminarCurrentSourceDensity_sum.h5',
        'RecExtElectrode_sum.h5']

    # CSD # unit nA um^-3 -> muA mm-3
    scaling_factors = [1E6, 1.]

    # panel titles
    panel_titles = [
        'LFP&CSD\nexc. syn.',
        'LFP&CSD\ninh. syn.',
        'LFP&CSD\ncompound',
        'CSD variance',
        'LFP variance', ]

    # labels
    labels = [
        'exc. syn.',
        'inh. syn.',
        'SUM']

    # some colors for traces
    if analysis_params.bw:
        colors = ['k', 'gray', 'k']
        # lws = [0.75, 0.75, 1.5]
        lws = [1.25, 1.25, 1.25]
    else:
        colors = [analysis_params.colorE, analysis_params.colorI, 'k']
        # colors = 'rbk'
        # lws = [0.75, 0.75, 1.5]
        lws = [1.25, 1.25, 1.25]

    # scalebar labels
    units = ['$\\mu$A mm$^{-3}$', 'mV']

    # depth of each contact site
    depth = params.electrodeParams['z']

    # #set up figure
    # #figure aspect
    # ana_params.set_PLOS_2column_fig_style(ratio=0.5)
    # fig, axes = plt.subplots(1,5)
    # fig.subplots_adjust(left=0.06, right=0.96, wspace=0.4, hspace=0.2)

    # clean up
    for ax in axes.flatten():
        phlp.remove_axis_junk(ax)

    for i, (scaling_factor, file_name) in enumerate(
            zip(scaling_factors, file_names)):
        # get the global data scaling bar range for use in latter plots
        # TODO: find nicer solution without creating figure
        dum_fig, dum_ax = plt.subplots(1)
        vlim_LFP = 0
        vlim_CSD = 0
        for savefolder in savefolders:
            vlimround0 = plot_signal_sum(
                dum_ax,
                params,
                os.path.join(
                    os.path.split(
                        params.savefolder)[0],
                    savefolder,
                    file_name),
                rasterized=False,
                scaling_factor=scaling_factor)
            if vlimround0 > vlim_LFP:
                vlim_LFP = vlimround0
            im = plot_signal_sum_colorplot(
                dum_ax, params, os.path.join(
                    os.path.split(
                        params.savefolder)[0], savefolder, file_name), cmap=plt.get_cmap(
                    'gray', 21) if analysis_params.bw else plt.get_cmap(
                    'bwr_r', 21), rasterized=False, scaling_factor=scaling_factor)
            if abs(im.get_array()).max() > vlim_CSD:
                vlim_CSD = abs(im.get_array()).max()

        plt.close(dum_fig)

        for j, savefolder in enumerate(savefolders):
            ax = axes[j]
            if i == 1:
                plot_signal_sum(ax, params, os.path.join(os.path.split(params.savefolder)[0], savefolder, file_name),
                                unit=units[i],
                                scaling_factor=scaling_factor,
                                T=T,
                                color='k',
                                # color='k' if analysis_params.bw else colors[j],
                                vlimround=vlim_LFP, rasterized=False)
            elif i == 0:
                im = plot_signal_sum_colorplot(
                    ax,
                    params,
                    os.path.join(
                        os.path.split(
                            params.savefolder)[0],
                        savefolder,
                        file_name),
                    unit=r'($\mu$Amm$^{-3}$)',
                    T=T,
                    ylabels=True,
                    colorbar=False,
                    fancy=False,
                    cmap=plt.get_cmap(
                        'gray',
                        21) if analysis_params.bw else plt.get_cmap(
                        'bwr_r',
                        21),
                    absmax=vlim_CSD,
                    rasterized=False,
                    scaling_factor=scaling_factor)

            ax.axis((T[0], T[1], -1550, 50))
            ax.set_title(panel_titles[j], va='baseline')

            if i == 0:
                phlp.annotate_subplot(
                    ax, ncols=1, nrows=1, letter=panel_labels[j])
            if j != 0:
                ax.set_yticklabels([])

            if i == 0:  # and j == 2:
                cb = phlp.colorbar(fig, ax, im,
                                   width=0.05, height=0.5,
                                   hoffset=-0.05, voffset=0.5)
                cb.set_label('($\\mu$Amm$^{-3}$)', labelpad=0.)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            if show_xlabels:
                ax.set_xlabel(r'$t$ (ms)', labelpad=0.)
            else:
                ax.set_xlabel('')

    #power in CSD
    ax = axes[3]
    datas = []
    for j, savefolder in enumerate(savefolders):
        f = h5py.File(os.path.join(os.path.split(params.savefolder)[0],
                                   savefolder,
                                   'LaminarCurrentSourceDensity_sum.h5'),
                      'r')
        data = f['data'][()] * 1E6  # unit nA um^-3 -> muA mm-3
        var = data[:, transient:].var(axis=1)
        ax.semilogx(var, depth,
                    color=colors[j], label=labels[j], lw=lws[j], clip_on=False)
        datas.append(f['data'][()][:, transient:])
        f.close()
    # control variances
    vardiff = datas[0].var(axis=1) + datas[1].var(axis=1) + np.array([2 * np.cov(x, y)[0, 1]
                                                                      for (x, y) in zip(datas[0], datas[1])]) - datas[2].var(axis=1)
    #ax.semilogx(abs(vardiff), depth, color='gray', lw=1, label='control')
    ax.axis(ax.axis('tight'))
    ax.set_ylim(-1550, 50)
    ax.set_yticks(-np.arange(16) * 100)
    if show_xlabels:
        ax.set_xlabel(r'$\sigma^2$ ($(\mu$Amm$^{-3})^2$)', labelpad=0.)
    ax.set_title(panel_titles[3], va='baseline')
    phlp.annotate_subplot(ax, ncols=1, nrows=1, letter=panel_labels[3])
    ax.set_yticklabels([])

    #power in LFP
    ax = axes[4]

    datas = []
    for j, savefolder in enumerate(savefolders):
        f = h5py.File(os.path.join(os.path.split(params.savefolder)[0],
                                   savefolder,
                                   'RecExtElectrode_sum.h5'),
                      'r')
        var = f['data'][()][:, transient:].var(axis=1)
        ax.semilogx(var, depth,
                    color=colors[j], label=labels[j], lw=lws[j], clip_on=False)
        datas.append(f['data'][()][:, transient:])
        f.close()
    # control variances
    vardiff = datas[0].var(axis=1) + datas[1].var(axis=1) + np.array([2 * np.cov(x, y)[0, 1]
                                                                      for (x, y) in zip(datas[0], datas[1])]) - datas[2].var(axis=1)
    ax.axis(ax.axis('tight'))
    ax.set_ylim(-1550, 50)
    ax.set_yticks(-np.arange(16) * 100)
    if show_xlabels:
        ax.set_xlabel(r'$\sigma^2$ (mV$^2$)', labelpad=0.)
    ax.set_title(panel_titles[4], va='baseline')
    phlp.annotate_subplot(ax, ncols=1, nrows=1, letter=panel_labels[4])

    ax.legend(bbox_to_anchor=(1.3, 1.0), frameon=False)
    ax.set_yticklabels([])

    # return fig


if __name__ == '__main__':

    plt.close('all')

    params = multicompartment_params()
    ana_params = analysis_params.params()

    ana_params.set_PLOS_2column_fig_style(ratio=1)
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        wspace=0.4,
        hspace=0.16,
        bottom=0.05,
        top=0.95)

    fig_exc_inh_contrib(fig, axes[0], params,
                        savefolders=['simulation_output_modified_ac_exc',
                                     'simulation_output_modified_ac_inh',
                                     'simulation_output_modified_ac_input'],
                        T=[800, 1000], transient=200, panel_labels='ABCDE',
                        show_xlabels=False)
    fig_exc_inh_contrib(fig,
                        axes[1],
                        params,
                        savefolders=['simulation_output_modified_regular_exc',
                                     'simulation_output_modified_regular_inh',
                                     'simulation_output_modified_regular_input'],
                        T=[890,
                            920],
                        transient=200,
                        panel_labels='FGHIJ')
    fig.savefig('figure_10.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig('figure_10.eps', bbox_inches='tight', pad_inches=0.01)

    plt.show()
