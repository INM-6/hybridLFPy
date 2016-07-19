#!/usr/bin/env python

#from introfig import params
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from cellsim16popsParams import multicompartment_params 
import analysis_params

#parameters
params = multicompartment_params()
ana_params = analysis_params.params()


#def tree(ax):
#
#    #ana_params.set_PLOS_2column_fig_style(ratio=0.5)
#    ax.axison = False
#
#    # tree
#    ax.plot([0,5],[5,2],color='k',linewidth = 1.)
#    ax.plot([0,5],[4,2],color='k',linewidth = 1.)
#    ax.plot([0,5],[3,2],color='k',linewidth = 1.)
#    ax.plot([0,5],[2,2],color='k',linewidth = 1.)
#    ax.plot([0,5],[1,2],color='k',linewidth = 1.)
#
#    ax.plot([0,5],[-5,-2],color='k',linewidth = 1.)
#    ax.plot([0,5],[-4,-2],color='k',linewidth = 1.)
#    ax.plot([0,5],[-3,-2],color='k',linewidth = 1.)
#    ax.plot([0,5],[-2,-2],color='k',linewidth = 1.)
#    ax.plot([0,5],[-1,-2],color='k',linewidth = 1.)
#    
#    ax.plot([5,10],[2,0],color='k',linewidth = 3.)
#    ax.plot([5,10],[2,0],color='k',linewidth = 3.)
#    
#    ax.plot([5,10],[-2,0],color='k',linewidth = 3.)
#    ax.plot([5,10],[-2,0],color='k',linewidth = 3.)
#    
#    ax.plot([10,15],[0,0],color='k',linewidth = 5.)
#    
#    # tree labels
#    ax.text(15.2,-1.5,'$X$',size=7)
#    ax.text(10.2,-1.5,'$Y$',size=7)
#    ax.text(5.,-1.5,'$y^{\prime}$',size=7)
#    ax.text(5.,0.95,'$y$',size=7)
#    ax.text(-1.,0.5,'$L6$',size=7)
#    ax.text(-1.,1.5,'$L5$',size=7)
#    ax.text(-1.,2.5,'$L4$',size=7)
#    ax.text(-1.,3.5,'$L23$',size=7)
#    ax.text(-1.,4.5,'$L1$',size=7)
#    ax.text(-1.,-1.5,'$L1$',size=7)
#    ax.text(-1.,-2.5,'$L23$',size=7)
#    ax.text(-1.,-3.5,'$L4$',size=7)
#    ax.text(-1.,-4.5,'$L5$',size=7)
#    ax.text(-1.,-5.5,'$L6$',size=7)
#    
#    # top arrow
#    ax.arrow(0.,6.,14.5,0., head_width=0.3, head_length=0.5, fc='k', ec='k')
#    # bottom arrow
#    ax.arrow(15.,-6.,-14.5,0., head_width=0.3, head_length=0.5, fc='k', ec='k')
#    
#    # formulas for top arrow
#    ax.text(-2.,7.1,'$K_{yXL}$',size=7)
#    ax.text(6.,7.1,'$K_{yX}=\sum_{L} K_{yXL}$',size=7)
#    ax.text(15.,7.1,'$K_{YX}=\sum_{y} K_{yX}$',size=7)
#    
#    # formulas for bottom arrow
#    ax.text(-2.,-7.5,'$K_{yXL}=\mathcal{L}_{yXL}\cdot K_{yX} $',size=7)
#    ax.text(6.,-7.5,'$K_{yX}=\mathcal{T}_{yX} \cdot K_{YX}$',size=7)
#    ax.text(15.,-7.5,'$K_{YX}$',size=7)
#    
#    # double arrows
#    ax.arrow(-1.5,9.,0.,1., head_width=0.3, head_length=0.5, fc='k', ec='k')
#    ax.arrow(-1.5,10.,0.,-1., head_width=0.3, head_length=0.5, fc='k', ec='k')
#    ax.arrow(15.5,9.,0.,1., head_width=0.3, head_length=0.5, fc='k', ec='k')
#    ax.arrow(15.5,10.,0.,-1., head_width=0.3, head_length=0.5, fc='k', ec='k')
#    
#    ax.set_ylim([-10,12])
#    ax.set_xlim([-2,17])
#    
#    return ax


if analysis_params.bw:
    cmap = plt.get_cmap('gray_r', 30)
    cmap.set_bad('w', 1.)
else:
    # cmap = plt.get_cmap('gray_r', 30)
    # cmap.set_bad('w', 1.)
    cmap = plt.get_cmap('hot', 20)
    # cmap = plt.get_cmap(plt.cm.hot, 20)
    # cmap.set_bad('k', 0.5)
    cmap.set_bad('0.5')

def connectivity(ax):

    '''make an imshow of the intranetwork connectivity'''
    masked_array = np.ma.array(params.C_YX, mask=params.C_YX==0)
    # if analysis_params.bw:
    #     cmap = plt.get_cmap(gray, 20)
    #     cmap.set_bad('k', 1.)
    # else:
    #     cmap = plt.get_cmap('hot', 20)
    #     cmap.set_bad('k', 0.5)
    # im = ax.imshow(masked_array, cmap=cmap, vmin=0, interpolation='nearest')
    im = ax.pcolormesh(masked_array, cmap=cmap, vmin=0, ) #interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(9)+0.5)
    ax.set_yticks(np.arange(8)+0.5)
    ax.set_xticklabels(params.X, rotation=270)
    ax.set_yticklabels(params.Y, )
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(r'$X$', labelpad=-1,fontsize=8)
    ax.set_ylabel(r'$Y$', labelpad=0, rotation=0,fontsize=8)

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.01
    fig = plt.gcf()
    cax = fig.add_axes(rect)

    cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label(r'$C_{YX}$', ha='center')
    cbar.set_label(r'$C_{YX}$', labelpad=0)


def cell_type_specificity(ax):
    '''make an imshow of the intranetwork connectivity'''
    masked_array = np.ma.array(params.T_yX, mask=params.T_yX==0)
    # cmap = plt.get_cmap('hot', 20)
    # cmap.set_bad('k', 0.5)
    # im = ax.imshow(masked_array, cmap=cmap, vmin=0, interpolation='nearest')
    im = ax.pcolormesh(masked_array, cmap=cmap, vmin=0, ) #interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(9)+0.5)
    ax.set_yticks(np.arange(16)+0.5)
    ax.set_xticklabels(params.X, rotation=270)
    ax.set_yticklabels(params.y, )
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(r'$X$', labelpad=-1,fontsize=8)
    ax.set_ylabel(r'$y$', labelpad=0, rotation=0,fontsize=8)

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.01
    fig = plt.gcf()
    cax = fig.add_axes(rect)

    cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label(r'$\mathcal{T}_{yX}$', ha='center')
    cbar.set_label(r'$\mathcal{T}_{yX}$', labelpad=0)


def quantity_yXL(fig, left, bottom, top, quantity=params.L_yXL, label=r'$\mathcal{L}_{yXL}$'):
                            
    '''make a bunch of image plots, each showing the spatial normalized
    connectivity of synapses'''
    
 
    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6']
    ncols = len(params.y) / 4
    
    #assess vlims
    vmin = 0
    vmax = 0
    for y in params.y:
        if quantity[y].max() > vmax:
            vmax = quantity[y].max()
    
    gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, top=top)
    
    for i, y in enumerate(params.y):
        ax = fig.add_subplot(gs[i/4, i%4])

        masked_array = np.ma.array(quantity[y], mask=quantity[y]==0)
        # cmap = plt.get_cmap('hot', 20)
        # cmap.set_bad('k', 0.5)
        
        # im = ax.imshow(masked_array,
        im = ax.pcolormesh(masked_array,
                            vmin=vmin, vmax=vmax,
                            cmap=cmap,
                            #interpolation='nearest',
                            )
        ax.invert_yaxis()

        ax.axis(ax.axis('tight'))
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(np.arange(9)+0.5)
        ax.set_yticks(np.arange(5)+0.5)
        
        #if divmod(i, 4)[1] == 0:
        if i % 4 == 0:
            ax.set_yticklabels(layers, )
            ax.set_ylabel('$L$', labelpad=0.)
        else:
            ax.set_yticklabels([])
        if i < 4:
            ax.set_xlabel(r'$X$', labelpad=-1,fontsize=8)
            ax.set_xticklabels(params.X, rotation=270)
        else:
            ax.set_xticklabels([])
        ax.xaxis.set_label_position('top')
        
        ax.text(0.5, -0.13, r'$y=$'+y,
            horizontalalignment='center',
            verticalalignment='center',
            #
                transform=ax.transAxes,fontsize=5.5)
    
    #colorbar
    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[1] = bottom
    rect[2] = 0.01
    rect[3] = top-bottom
    cax = fig.add_axes(rect)
    cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label(label, ha='center')
    cbar.set_label(label, labelpad=0)


def fig_connectivity():
    fig = plt.figure()  
    fig.subplots_adjust(left=0.07, right=0.92,top=0.90, bottom=0.05, hspace=0.25)

    # C_{YX}
    ax1 = plt.subplot2grid((9, 8), (0, 0), colspan=3, rowspan=4)
    connectivity(ax1)

    # C_{yXL}
    quantity_yXL(fig, left=0.5, bottom=0.53, top=0.9, quantity=params.C_yXL, label=r'$C_{yXL}$')
    
    # T_{yX}
    ax3 = plt.subplot2grid((9, 8), (5,0), colspan=3,rowspan=4)
    cell_type_specificity(ax3)

    # L_{yXL}
    quantity_yXL(fig, left=0.5, bottom=0.05, top=0.42, label=r'$\mathcal{L}_{yXL}$')
    
    plt.figtext(0.03, 0.975, 'A', fontweight='demibold', fontsize=8)
    plt.figtext(0.47, 0.975, 'B', fontweight='demibold', fontsize=8)
    plt.figtext(0.03, 0.49, 'C', fontweight='demibold', fontsize=8)
    plt.figtext(0.47, 0.49, 'D', fontweight='demibold', fontsize=8)

    plt.figtext(0.15, 0.965, 'network connectivity', fontsize=8)
    plt.figtext(0.63, 0.965, 'spatial connectivity', fontsize=8)
    plt.figtext(0.15, 0.48, 'cell-type specificity', fontsize=8)
    plt.figtext(0.64, 0.48, 'layer specificity', fontsize=8)
    
    return fig


def fig_connectivity_wo_C_yXL():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.85)
    
    gs = gridspec.GridSpec(3, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    connectivity(ax0)
    
    gs = gridspec.GridSpec(3, 3)
    ax1 = fig.add_subplot(gs[1:, 0])
    #ax1.set_xticks([])
    cell_type_specificity(ax1)
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    
    quantity_yXL(fig, left=0.5, bottom=0.1, top=0.85, label=r'$\mathcal{L}_{yXL}$') 

    plt.figtext(0.075, 0.95, 'A', fontweight='demibold', fontsize=8)
    plt.figtext(0.075, 0.6, 'B', fontweight='demibold', fontsize=8)
    plt.figtext(0.47, 0.95, 'C', fontweight='demibold', fontsize=8)
    
    plt.figtext(0.16, 0.95, 'network connectivity', fontsize=8)
    plt.figtext(0.16, 0.6, 'cell-type specificity', fontsize=8)
    plt.figtext(0.64, 0.95, 'layer specificity', fontsize=8)


    return fig


if __name__ == '__main__':
    ana_params.set_PLOS_2column_fig_style(ratio=0.8)
    
    plt.rcParams.update({
         'xtick.labelsize' : 5,
         'ytick.labelsize' : 5,
         })

    fig = fig_connectivity()
    fig.savefig(os.path.join(params.figures_path,
                             'figure_05.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig(os.path.join(params.figures_path,
                             'figure_05.eps'),
                bbox_inches='tight', pad_inches=0.01)


    # ana_params.set_PLOS_2column_fig_style(ratio=0.6)
    # fig = fig_connectivity_wo_C_yXL()
    # fig.savefig(os.path.join(params.figures_path,
    #                          'fig_connectivity_wo_C_yXL.pdf'),
    #             bbox_inches='tight', pad_inches=0)
   
    plt.show()
  


