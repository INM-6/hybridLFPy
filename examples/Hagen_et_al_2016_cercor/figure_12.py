#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import h5py

from hybridLFPy import CachedNetwork, GDF
import plotting_helpers as phlp
import analysis_params

######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

from cellsim16popsParams_modified_spontan import multicompartment_params 
params = multicompartment_params()

#
ana_params = analysis_params.params()

######################################
### IMPORT PANELS                  ###
######################################

from plot_methods import plot_signal_sum, calc_signal_power, plotting_correlation

######################################
### FIGURE                         ###
######################################


'''
Run analysis.py first to generate data files.
'''

def fig_lfp_scaling(fig, params, bottom=0.55, top=0.95, channels=[0,3,7,11,13], T=[800.,1000.], Df=None,
                    mlab=True, NFFT=256, noverlap=128,
                    window=plt.mlab.window_hanning, letters='ABCD',
                    lag=20, show_titles=True, show_xlabels=True):

   
    fname_fullscale=os.path.join(params.savefolder, 'LFPsum.h5')
    fname_downscaled=os.path.join(params.savefolder, 'populations','subsamples', 'LFPsum_10_0.h5')

    # ana_params.set_PLOS_2column_fig_style(ratio=0.5)
    
    gs = gridspec.GridSpec(len(channels), 8, bottom=bottom, top=top)
    
    # fig = plt.figure()
    # fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, wspace=0.8, hspace=0.1)
  
    scaling_factor = np.sqrt(10)

    ##################################
    ###  LFP traces                ###
    ##################################

    ax = fig.add_subplot(gs[:, :3])

    phlp.annotate_subplot(ax, ncols=8/3., nrows=1, letter=letters[0], linear_offset=0.065)
    plot_signal_sum(ax, params, fname=os.path.join(params.savefolder, 'LFPsum.h5'),
                    unit='mV', scaling_factor= 1., scalebar=True,
                    vlimround=None,
                    T=T, ylim=[-1600, 50] ,color='k',label='$\Phi$',
                    rasterized=False,
                    zorder=1,)
    plot_signal_sum(ax, params, fname=os.path.join(params.savefolder, 'populations',
                                            'subsamples', 'LFPsum_10_0.h5'),
                    unit='mV', scaling_factor= scaling_factor,scalebar=False,
                    vlimround=None,
                    T=T, ylim=[-1600, 50],
                    color='gray' if analysis_params.bw else analysis_params.colorP,
                    label='$\hat{\Phi}^{\prime}$',
                    rasterized=False,
                    lw=1, zorder=0)

    
    if show_titles:
        ax.set_title('LFP & low-density predictor')
    if show_xlabels:
        ax.set_xlabel('$t$ (ms)', labelpad=0.)
    else:
        ax.set_xlabel('')


    #################################
    ### Correlations              ###
    #################################
    
    ax = fig.add_subplot(gs[:, 3])
    phlp.annotate_subplot(ax, ncols=8, nrows=1, letter=letters[1], linear_offset=0.065)
    phlp.remove_axis_junk(ax)
    
    datas = []
    files = [os.path.join(params.savefolder, 'LFPsum.h5'),
             os.path.join(params.savefolder, 'populations',
                                            'subsamples', 'LFPsum_10_0.h5')]
    for fil in files:
        f = h5py.File(fil)
        datas.append(f['data'].value[:, 200:])
        f.close()
    

    zvec = np.r_[params.electrodeParams['z']]
    cc = np.zeros(len(zvec))
    for ch in np.arange(len(zvec)):
        x0 = datas[0][ch]
        x0 -= x0.mean()
        x1 = datas[1][ch]
        x1 -= x1.mean()
        cc[ch] = np.corrcoef(x0, x1)[1, 0]
        
    ax.barh(zvec, cc, height=80, align='center', color='0.5', linewidth=0.5)
    
    # superimpose the chance level, obtained by mixing one input vector N times
    # while keeping the other fixed. We show boxes drawn left to right where
    # these denote mean +/- two standard deviations.
    N = 1000
    method = 'randphase' #or 'permute'
    chance = np.zeros((cc.size, N))
    for ch in np.arange(len(zvec)):
        x1 = datas[1][ch]
        x1 -= x1.mean()
        if method == 'randphase':
            x0 = datas[0][ch]
            x0 -= x0.mean()
            X00 = np.fft.fft(x0)
        for n in range(N):
            if method == 'permute':
                x0 = np.random.permutation(datas[0][ch])
            elif method == 'randphase':
                X0 = np.copy(X00)
                #random phase information such that spectra is preserved
                theta = np.random.uniform(0, 2*np.pi, size=X0.size // 2)
                #half-sided real and imaginary component 
                real = abs(X0[1:X0.size // 2 + 1])*np.cos(theta)
                imag = abs(X0[1:X0.size // 2 + 1])*np.sin(theta)
                
                #account for the antisymmetric phase values
                X0.imag[1:imag.size+1] = imag
                X0.imag[imag.size+1:] = -imag[::-1]
                X0.real[1:real.size+1] = real
                X0.real[real.size+1:] = real[::-1]
                x0 = np.fft.ifft(X0).real

            chance[ch, n] = np.corrcoef(x0, x1)[1, 0]
    
    # p-values, compute the fraction of chance correlations > cc at each channel
    p = []
    for i, x in enumerate(cc):
        p += [(chance[i, ] >= x).sum() / float(N)]
    
    print(('p-values:', p))
    
    
    #compute the 99% percentile of the chance data
    right = np.percentile(chance, 99, axis=-1)
    
    ax.plot(right, zvec, ':', color='k', lw=1.)    
    ax.set_ylim([-1550, 50])
    ax.set_yticklabels([])
    ax.set_yticks(zvec)
    ax.set_xlim([0., 1.])
    ax.set_xticks([0.0, 0.5, 1])
    ax.yaxis.tick_left()

    if show_titles:
        ax.set_title('corr.\ncoef.')
    if show_xlabels:
        ax.set_xlabel('$cc$ (-)', labelpad=0.)
    



    ##################################
    ###  Single channel PSDs       ###
    ##################################

    freqs, PSD_fullscale = calc_signal_power(params, fname=fname_fullscale,
                                             transient=200, Df=Df, mlab=mlab,
                                             NFFT=NFFT, noverlap=noverlap,
                                             window=window)
    freqs, PSD_downscaled = calc_signal_power(params, fname=fname_downscaled,
                                              transient=200, Df=Df, mlab=mlab,
                                              NFFT=NFFT, noverlap=noverlap,
                                              window=window)
    inds = freqs >= 1  # frequencies greater than 4 Hz  

    for i, ch in enumerate(channels):
    
        ax = fig.add_subplot(gs[i, 4:6])
        if i == 0:
            phlp.annotate_subplot(ax, ncols=8/2., nrows=len(channels), letter=letters[2], linear_offset=0.065)
        phlp.remove_axis_junk(ax)
        ax.loglog(freqs[inds],PSD_fullscale[ch][inds],
                  color='k', label='$\gamma=1.0$',
                  zorder=1,)
        ax.loglog(freqs[inds],PSD_downscaled[ch][inds]*scaling_factor**2,
                  lw=1,
                  color='gray' if analysis_params.bw else analysis_params.colorP,
                  label='$\gamma=0.1, \zeta=\sqrt{10}$',
                  zorder=0,)
        ax.loglog(freqs[inds],PSD_downscaled[ch][inds]*scaling_factor**4,
                  lw=1,
                  color='0.75', label='$\gamma=0.1, \zeta=10$',
                  zorder=0)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.text(0.8,0.9,'ch. %i' %(ch+1),horizontalalignment='left',
                     verticalalignment='center',
                     fontsize=6, 
                     transform=ax.transAxes)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        if i < len(channels)-1:
            #ax.set_xticks([])
            ax.set_xticklabels([])
        ax.tick_params(axis='y',which='minor',bottom='off')
        ax.set_xlim([4E0,4E2])
        ax.set_ylim([3E-8,1E-4])
        if i == 0:
            ax.tick_params(axis='y', which='major', pad=0)
            ax.set_ylabel('(mV$^2$/Hz)', labelpad=0.)
            if show_titles:
                ax.set_title('power spectra')
        #ax.set_yticks([1E-9,1E-7,1E-5])
        if i > 0:
            ax.set_yticklabels([])
            
    if show_xlabels:
        ax.set_xlabel(r'$f$ (Hz)', labelpad=0.)     


    ##################################
    ###  PSD ratios                ###
    ##################################


    ax = fig.add_subplot(gs[:, 6:8])
    phlp.annotate_subplot(ax, ncols=8./2, nrows=1, letter=letters[3], linear_offset=0.065)

    
    PSD_ratio = PSD_fullscale/(PSD_downscaled*scaling_factor**2)
    zvec = np.r_[params.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    inds = freqs >= 1  # frequencies greater than 4 Hz  

    im = ax.pcolormesh(freqs[inds], zvec+40, PSD_ratio[:, inds],
                        rasterized=False,
                        cmap=plt.get_cmap('gray_r', 18) if analysis_params.bw else plt.cm.get_cmap('Reds', 18),
                        vmin=1E0,vmax=1.E1)
    ax.set_xlim([4E0,4E2])
    ax.set_xscale('log')
    ax.set_yticks(zvec)
    yticklabels = ['ch. %i' %i for i in np.arange(len(zvec))+1]
    ax.set_yticklabels(yticklabels)
    plt.axis('tight')
    cb = phlp.colorbar(fig, ax, im,
                       width=0.05, height=0.5,
                       hoffset=-0.05, voffset=0.0)
    cb.set_label('(-)', labelpad=0.)
    phlp.remove_axis_junk(ax)

    if show_titles:
        ax.set_title('power ratio')
    if show_xlabels:
        ax.set_xlabel(r'$f$ (Hz)', labelpad=0.)


  
    return fig


if __name__ == '__main__':
    
    params = multicompartment_params()

    savefolders = [
        'simulation_output_modified_spontan',
        'simulation_output_modified_ac_input'
    ]
    letterslist = [
        'ABCD',
        'EFGH',
    ]
    lags = [20, 100]
    
    show_titles = [True, False]
    show_xlabels = [False, True]
    
    bottoms = [0.535, 0.04]
    tops = [0.96, 0.465]


    ana_params.set_PLOS_2column_fig_style(ratio=1)
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.075, wspace=0.8, hspace=0.1)

    for i in range(2):
        savefolder = savefolders[i]
        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                           'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path
    
        fig_lfp_scaling(fig, params=params, bottom=bottoms[i], top=tops[i], channels=[0,3,7,11,13],
                        T=[800., 950], Df=None,
                        mlab=True, NFFT=256, noverlap=128,
                        window=plt.mlab.window_hanning,letters=letterslist[i],
                        lag=lags[i], show_titles=show_titles[i],
                        show_xlabels=show_xlabels[i])

    fig.savefig('figure_12.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig('figure_12.eps', bbox_inches='tight', pad_inches=0.01)

    plt.show()
