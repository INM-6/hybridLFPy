#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import analysis_params

#######################################
### PLOTTING                        ###
#######################################

def add_gdf_to_dot_display(ax, gdf, color= '0.', alpha= 1.):
    plt.sca(ax)
    gids,times = list(zip(*gdf))
    ax.plot(times,gids,'k o', ms = 0.5, mfc=color , mec=color, alpha=alpha)
    plt.xlabel('t in ms')
    plt.ylabel('neuron id')
    return ax


def add_sst_to_dot_display(ax, sst, color= '0.',alpha= 1.):
    '''
    suitable for plotting fraction of neurons
    '''
    plt.sca(ax)
    N = len(sst)
    current_ymax = 0
    counter = 0
    while True:
        if len(ax.get_lines()) !=0:
            data = ax.get_lines()[-1-counter].get_data()[1]
            if np.sum(data) != 0: # if not empty array 
                current_ymax = np.max(data)
                break
            counter +=1
        else: 
            break
    for i in np.arange(N):
        plt.plot(sst[i],np.ones_like(sst[i])+i+current_ymax -1, 'k o',ms=0.5, mfc=color,mec=color, alpha=alpha)
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'neuron id')
    return ax


def empty_bar_plot(ax):
    ''' Delete all axis ticks and labels '''
    plt.sca(ax)
    plt.setp(plt.gca(),xticks=[],xticklabels=[]) 
    return ax


def add_to_bar_plot(ax, x, number, name = '', color = '0.'):
    ''' This function takes an axes and adds one bar to it '''
    plt.sca(ax)
    plt.setp(ax,xticks=np.append(ax.get_xticks(),np.array([x]))\
             ,xticklabels=[item.get_text() for item in ax.get_xticklabels()] +[name])
    plt.bar([x],number , color = color, width = 1.)
    return ax
     

def add_to_line_plot(ax, x, y, color = '0.' , label = ''):
    ''' This function takes an axes and adds one line to it '''
    plt.sca(ax)
    plt.plot(x,y, color = color, label = label)
    return ax


def colorbar(fig, ax, im,
                 width=0.05,
                 height=1.0,
                 hoffset=0.01,
                 voffset=0.0,
                 orientation='vertical'):
    '''
    draw colorbar without resizing the axes object to make room

    kwargs:
    ::
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.AxesSubplot
        im : matplotlib.image.AxesImage
        width : float, colorbar width in fraction of ax width
        height : float, colorbar height in fraction of ax height
        hoffset : float, horizontal spacing to main axes in fraction of width
        voffset : float, vertical spacing to main axis in fraction of height
        orientation : str, 'horizontal' or 'vertical'

    return:
    ::
        object : colorbar handle

    '''
    rect = np.array(ax.get_position().bounds)
    
    rect = np.array(ax.get_position().bounds)
    caxrect = [0]*4
    caxrect[0] = rect[0] + rect[2] + hoffset*rect[2]
    caxrect[1] = rect[1] + voffset*rect[3]
    caxrect[2] = rect[2]*width
    caxrect[3] = rect[3]*height
    
    cax = fig.add_axes(caxrect)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    
    return cb


#######################################
### FIGURE STYLES                   ###
#######################################


def frontiers_style():
    '''
    Figure styles for frontiers
    '''
    
    inchpercm = 2.54
    frontierswidth=8.5 
    textsize = 5
    titlesize = 7
    plt.rcdefaults()
    plt.rcParams.update({
        'figure.figsize' : [frontierswidth/inchpercm, frontierswidth/inchpercm],
        'figure.dpi' : 160,
        'xtick.labelsize' : textsize,
        'ytick.labelsize' : textsize,
        'font.size' : textsize,
        'axes.labelsize' : textsize,
        'axes.titlesize' : titlesize,
        'axes.linewidth': 0.75,
        'lines.linewidth': 0.75,
        'legend.fontsize' : textsize,
    })
    return None


def PLOS_style():
    '''
    Figure styles for PLOS CB
    '''
    return None


def large_fig_style():
    ''' Adapted from frontiers style, but larger figures '''
    inchpercm = 2.54
    frontierswidth=18.5 
    textsize = 5
    titlesize = 7
    plt.rcdefaults()
    plt.rcParams.update({
        'figure.figsize' : [frontierswidth/inchpercm, frontierswidth/inchpercm],
        'figure.dpi' : 160,
        'xtick.labelsize' : textsize,
        'ytick.labelsize' : textsize,
        'font.size' : textsize,
        'axes.labelsize' : textsize,
        'axes.titlesize' : titlesize,
        'axes.linewidth': 0.75,
        'lines.linewidth': 0.75,
        'legend.fontsize' : textsize,
    })
    return None



####################################################################


def remove_axis_junk(ax, which=['right', 'top']):
    '''remove upper and right axis'''
    for loc, spine in ax.spines.items():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def annotate_subplot(ax, ncols=1, nrows=1, letter='a',
                     linear_offset=0.075, fontsize=8):
    '''add a subplot annotation number'''
    ax.text(-ncols*linear_offset, 1+nrows*linear_offset, letter,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=fontsize, fontweight='demibold',
        transform=ax.transAxes)


def get_colors(num=16, cmap=plt.cm.Dark2):
    '''return a list of color tuples to use in plots'''
    colors = []
    for i in range(num):
        if analysis_params.bw:
            colors.append('k' if i % 2 == 0 else 'gray')
        else:
            i *= float(cmap.N)
            if num > 1:
                i /= num - 1.
            else:
                i /= num
            colors.append(cmap(int(i)))
    return colors



if __name__ == '__main__':
    #test annotate_subplot
    
    plt.rcdefaults()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    annotate_subplot(ax, 1, 1, 'a')
    

    fig = plt.figure()
    ax = fig.add_subplot(221)
    annotate_subplot(ax, 2, 2, 'a')

    fig = plt.figure()
    ax = fig.add_subplot(331)
    annotate_subplot(ax, 3, 3, 'a')


    fig = plt.figure()
    ax = fig.add_subplot(441)
    annotate_subplot(ax, 4, 4, 'a')


    fig = plt.figure()
    ax = fig.add_subplot(551)
    annotate_subplot(ax, 5, 5, 'a')
    
    
    plt.show()

