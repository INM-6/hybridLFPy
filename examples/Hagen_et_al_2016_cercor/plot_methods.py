#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Plotting function helper script
'''
import os
import numpy as np
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.style
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm #, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import h5py
from hybridLFPy import helpers
import LFPy
import neuron
import plotting_helpers as phlp
import analysis_params
from mpi4py import MPI


###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################


######################################
### FUNCTIONS FOR FILLING AXES     ###
######################################

def network_sketch(ax, highlight=None, labels=True, yscaling=1.):    
    '''
    highlight : None or string
         if string, then only the label of this population is set and the box is highlighted
    '''

    name_to_id_mapping={'L6E':(0,0),
                        'L6I':(0,1),
                        'L5E':(1,0),
                        'L5I':(1,1),
                        'L4E':(2,0),
                        'L4I':(2,1),
                        'L23E':(3,0),
                        'L23I':(3,1)
    } 
    
    showgrid=False  ## switch on/off grid

    ## sketch parameters
    layer_x=0.1                            ## x position of left boundary of cortex layers
    layer6_y=0.2*yscaling                           ## y position of lower boundary of layer 6
    layer_width=0.65                       ## width of cortex layers
    layer_height=0.21*yscaling                      ## height of cortex layers
    layer_colors=['0.9','0.8','0.9','0.8'] ## layer colors
    c_pop_size=0.15                        ## cortex population size
    c_pop_dist=0.17                        ## distance between cortex populations
    t_pop_size=0.15                        ## thalamus population size
    t_pop_y=0.0                         ## y position of lower thalamus boundary
    axon_cell_sep=0.04                     ## distance between axons and popualations
    cc_input_y=0.6*yscaling                        ## y position of cortico-cortical synapses (relative to cortex population)
    tc_input_y=0.4*yscaling                         ## y position of thalamo-cortical synapses (relative to cortex population)

    exc_clr = 'k' if analysis_params.bw else analysis_params.colorE                            ## color of excitatory axons/synapses
    inh_clr = 'gray' if analysis_params.bw else analysis_params.colorI                            ## color of inhibitory axons/synapses

    lw_pop=0.5                              ## linewidth for populations
    lw_axons=0.4                           ## linewidth for axons

    arrow_size=0.013                       ## arrow size
    conn_radius=0.005                      ## radius of connector marker

    legend_length=0.07                     ## length of legend arrows
    
    colors = phlp.get_colors(8)[::-1]            ## colors of each population

    fontdict1={'fontsize': 6, ## population name 
              'weight':'normal',
              'horizontalalignment':'center',
              'verticalalignment':'center'}

    fontdict2={'fontsize': 6, ## cortico-cortical input
              'weight':'normal',
              'horizontalalignment':'center',
              'verticalalignment':'center'}

    fontdict3={'fontsize': 6, ## legend 
              'weight':'normal',
              'horizontalalignment':'left',
              'verticalalignment':'center'}

    ######################################################################################

    def draw_box(ax,pos,lw=1.,ls='solid',eclr='k',fclr='w',zorder=0, clip_on=False,
                 boxstyle=patches.BoxStyle("Round", pad=0.0), padadjust=0.):
        '''Draws a rectangle.'''    
        rect = patches.FancyBboxPatch((pos[0]+padadjust, pos[1]+padadjust),
            pos[2]-2*padadjust, pos[3]-2*padadjust, ec=eclr, fc=fclr,
            lw=lw, ls=ls, zorder=zorder, clip_on=clip_on,
            boxstyle=boxstyle)
        ax.add_patch(rect)

    def draw_circle(ax,xy,radius,lw=1.,ls='solid',eclr='k',fclr='w',zorder=0):
        '''Draws a circle.'''    
        circ = plt.Circle((xy[0],xy[1]),radius=radius, ec=eclr,fc=fclr,lw=lw,ls=ls,zorder=zorder)
        ax.add_patch(circ)

    def put_text(ax,xy,txt,clr,fontdict,zorder=10):
        '''Puts text to a specific position.'''
        ax.text(xy[0],xy[1],txt,fontdict=fontdict,color=clr,zorder=zorder)

    def draw_line(ax,path,lw=1.,ls='solid',lclr='k',zorder=0):
        '''Draws a path.'''
        #pth = path.Path(np.array(path))
        pth = Path(np.array(path))        
        patch = patches.PathPatch(pth, fill=False, lw=lw,ls=ls,ec=lclr,fc=lclr,zorder=zorder)
        ax.add_patch(patch)

    def draw_arrow(ax,path,lw=1.0,ls='solid',lclr='k',arrow_size=0.025,zorder=0):
        '''Draws a path with an arrow at the end. '''
        x=path[-2][0]
        y=path[-2][1]
        dx=path[-1][0]-path[-2][0]
        dy=path[-1][1]-path[-2][1]

        D=np.array([dx,dy])
        D=D/np.sqrt(D[0]**2+D[1]**2)
        path2=np.array(path).copy()
        path2[-1,:]=path2[-1,:]-arrow_size*D

        pth = Path(np.array(path2))
        patch = patches.PathPatch(pth, fill=False, lw=lw,ls=ls,ec=lclr,fc=lclr,zorder=zorder)
        ax.add_patch(patch)

        arr=patches.FancyArrow(\
            x,y,dx,dy,\
                length_includes_head=True,width=0.0,head_width=arrow_size,\
                overhang=0.2,ec=lclr,fc=lclr,linewidth=0)
        ax.add_patch(arr)


    ##################################################
    ## populations

    ## cortex
    layer_pos=[]
    c_pop_pos=[]
    for i in range(4):

        ## cortex layers
        layer_pos+=[[layer_x,layer6_y+i*layer_height*yscaling,layer_width,layer_height]] ## layer positions
        draw_box(ax,layer_pos[i],lw=0.,fclr=layer_colors[i],zorder=0)

        ## cortex populations
        l_margin=(layer_width-2.*c_pop_size-c_pop_dist)/2.
        b_margin=(layer_height-c_pop_size)/2.

        ## positions of cortex populations
        c_pop_pos+=[[ [layer_pos[i][0] + l_margin,
                       layer_pos[i][1] + b_margin,
                       c_pop_size, c_pop_size], ## E
                      [layer_pos[i][0] + l_margin + c_pop_size + c_pop_dist,
                       layer_pos[i][1] + b_margin,
                       c_pop_size, c_pop_size]  ]] ## I

        #draw_box(ax,c_pop_pos[i][0],lw=lw_pop,eclr='k',fclr='w',zorder=2) ## E
        #draw_box(ax,c_pop_pos[i][1],lw=lw_pop,eclr='k',fclr='w',zorder=2) ## I    
        draw_box(ax,c_pop_pos[i][0],lw=lw_pop,eclr='k',fclr=colors[i*2+1],zorder=2,
                 boxstyle=patches.BoxStyle("Round", pad=0.02), padadjust=0.02) ## E
        draw_box(ax,c_pop_pos[i][1],lw=lw_pop,eclr='k',fclr=colors[i*2],zorder=2,
                 boxstyle=patches.BoxStyle("Round", pad=0.02), padadjust=0.02) ## I    


    ## thalamus
    c_center_x=layer_x+layer_width/2. ## x position of cortex center
    t_pos=[c_center_x-t_pop_size/2.,t_pop_y*yscaling,t_pop_size,t_pop_size] ## thalamus position
    #draw_box(ax,t_pos,lw=lw_pop,eclr='k',fclr='w',zorder=2) ## Th
    draw_box(ax,t_pos,lw=lw_pop,eclr='k',fclr='k',zorder=2,
             boxstyle=patches.BoxStyle("Round", pad=0.02), padadjust=0.02) ## Th

    ##################################################
    ## intracortical axons

    axon_x_dist=(c_pop_dist-2.*axon_cell_sep)/7.
    assert(axon_x_dist>0.)
    axon_y_dist=c_pop_size/9.#*yscaling
    c_axon_x=[]
    c_axon_y=[]

    # x positions of vertical intracortical axons
    for i in range(4): # pre layer
        exc=c_pop_pos[i][0][0]+c_pop_size+axon_cell_sep+i*axon_x_dist ## E
        inh=exc+4.*axon_x_dist ## I
        c_axon_x+=[[exc,inh]]

    # y positions of horizontal intracortical axons
    for i in range(4): ## post layer
        c_axon_y+=[[]]
        for j in range(4): ## pre layer
            exc=c_pop_pos[i][0][1]+(j+1.)*axon_y_dist ## E
            inh=c_pop_pos[i][0][1]+c_pop_size-(j+1.)*axon_y_dist ## I        
            c_axon_y[i]+=[[exc,inh]]

    ## vertical intracortical axons
    for i in range(4):
        draw_line(ax,[[c_axon_x[i][0],c_axon_y[0][i][0]],[c_axon_x[i][0],c_axon_y[-1][i][0]]],lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1)
        draw_line(ax,[[c_axon_x[i][1],c_axon_y[0][i][1]],[c_axon_x[i][1],c_axon_y[-1][i][1]]],lw=lw_axons,ls='solid',lclr=inh_clr,zorder=0)

    ## horizontal intracortical axons
    for i in range(4): ## post layer
        for j in range(4): ## pre layer
            path=[[c_axon_x[j][0],c_axon_y[i][j][0]],[c_pop_pos[i][0][0]+c_pop_size,c_axon_y[i][j][0]]]        
            draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=1)

            path=[[c_axon_x[j][0],c_axon_y[i][j][0]],[c_pop_pos[i][1][0],c_axon_y[i][j][0]]]        
            draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=1)

            path=[[c_axon_x[j][1],c_axon_y[i][j][1]],[c_pop_pos[i][1][0],c_axon_y[i][j][1]]]
            draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=inh_clr,arrow_size=arrow_size,zorder=0)

            path=[[c_axon_x[j][1],c_axon_y[i][j][1]],[c_pop_pos[i][0][0]+c_pop_size,c_axon_y[i][j][1]]]
            draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=inh_clr,arrow_size=arrow_size,zorder=0)

            ## connector markers
            draw_circle(ax,[c_axon_x[j][0],c_axon_y[i][j][0]],conn_radius,lw=0,fclr=exc_clr,zorder=0)
            draw_circle(ax,[c_axon_x[j][1],c_axon_y[i][j][1]],conn_radius,lw=0,fclr=inh_clr,zorder=0)        

    ## cell outputs
    for i in range(4):

        path=[[c_pop_pos[i][0][0]+c_pop_size/2.,c_pop_pos[i][0][1]],
              [c_pop_pos[i][0][0]+c_pop_size/2.,c_pop_pos[i][0][1]-axon_y_dist],
              [c_axon_x[i][0],c_pop_pos[i][0][1]-axon_y_dist]]
        draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1)  ## excitatory
        draw_circle(ax,path[-1],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

        path=[[c_pop_pos[i][1][0]+c_pop_size/2.,c_pop_pos[i][1][1]],
              [c_pop_pos[i][1][0]+c_pop_size/2.,c_pop_pos[i][1][1]-axon_y_dist],
              [c_axon_x[-1-i][1],c_pop_pos[i][1][1]-axon_y_dist]]
        draw_line(ax,path,lw=lw_axons,ls='solid',lclr=inh_clr,zorder=1)  ## inhibitory
        draw_circle(ax,path[-1],conn_radius,lw=0,fclr=inh_clr,zorder=0) ## connector

    ## remaining first segments for L6
    path=[[c_axon_x[0][0],c_pop_pos[0][0][1]-axon_y_dist],[c_axon_x[0][0],c_axon_y[0][0][0]]]        
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=0)        
    path=[[c_axon_x[-1][1],c_pop_pos[0][1][1]-axon_y_dist],[c_axon_x[-1][1],c_axon_y[0][0][1]]]        
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=inh_clr,zorder=0)        

    ##################################################
    ## cortico-cortical axons

    ## horizontal branch in L1
    path=[[0.,c_pop_pos[-1][0][1]+c_pop_size+axon_cell_sep],
          [c_pop_pos[-1][1][0]+c_pop_size+axon_cell_sep,c_pop_pos[-1][0][1]+c_pop_size+axon_cell_sep]]
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1)

    ## vertical branches
    path=[[c_pop_pos[-1][0][0]-axon_cell_sep,c_pop_pos[-1][0][1]+c_pop_size+axon_cell_sep],
          [c_pop_pos[-1][0][0]-axon_cell_sep,c_pop_pos[0][0][1]+cc_input_y*c_pop_size]]
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1) ## cc input to exc pop
    draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector
    path=[[c_pop_pos[-1][1][0]+c_pop_size+axon_cell_sep,c_pop_pos[-1][0][1]+c_pop_size+axon_cell_sep],
          [c_pop_pos[-1][1][0]+c_pop_size+axon_cell_sep,c_pop_pos[0][0][1]+cc_input_y*c_pop_size]]
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1) ## cc input to inh pop
    draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

    ## horizontal branches (arrows)
    for i in range(4):
        ## cc input to excitatory populations
        path=[[c_pop_pos[-1][0][0]-axon_cell_sep,c_pop_pos[i][0][1]+cc_input_y*c_pop_size],
              [c_pop_pos[-1][0][0],c_pop_pos[i][0][1]+cc_input_y*c_pop_size],]
        draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=0)
        draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

        ## cc input to inhibitory populations
        path=[[c_pop_pos[-1][1][0]+c_pop_size+axon_cell_sep,c_pop_pos[i][0][1]+cc_input_y*c_pop_size],
              [c_pop_pos[-1][1][0]+c_pop_size,c_pop_pos[i][0][1]+cc_input_y*c_pop_size]]
        draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=0)
        draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

    ##################################################
    ## thalamo-cortical axons

    path=[[t_pos[0]+t_pop_size/2.,t_pos[1]+t_pop_size],
          [t_pos[0]+t_pop_size/2.,t_pos[1]+t_pop_size+axon_y_dist]]
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1) ## thalamic output
    draw_circle(ax,path[-1],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

    path=[[c_pop_pos[0][0][0]-(axon_cell_sep+axon_y_dist),t_pos[1]+t_pop_size+axon_y_dist],
          [c_pop_pos[0][1][0]+c_pop_size+(axon_cell_sep+axon_y_dist),t_pos[1]+t_pop_size+axon_y_dist]]
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1) ## horizontal branch

    path=[[c_pop_pos[0][0][0]-(axon_cell_sep+axon_y_dist),t_pos[1]+t_pop_size+axon_y_dist],
          [c_pop_pos[0][0][0]-(axon_cell_sep+axon_y_dist),c_pop_pos[2][0][1]+tc_input_y*c_pop_size]]
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1) ## left vertical branch

    path=[[c_pop_pos[0][1][0]+c_pop_size+(axon_cell_sep+axon_y_dist),t_pos[1]+t_pop_size+axon_y_dist],
          [c_pop_pos[0][1][0]+c_pop_size+(axon_cell_sep+axon_y_dist),c_pop_pos[2][0][1]+tc_input_y*c_pop_size]]
    draw_line(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,zorder=1) ## right vertical branch

    path=[[c_pop_pos[0][0][0]-(axon_cell_sep+axon_y_dist),c_pop_pos[2][0][1]+tc_input_y*c_pop_size],
          [c_pop_pos[0][0][0],c_pop_pos[2][0][1]+tc_input_y*c_pop_size],]
    draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=1) ## Th -> L4E synapses (arrows)
    draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

    path=[[c_pop_pos[0][0][0]-(axon_cell_sep+axon_y_dist),c_pop_pos[0][0][1]+tc_input_y*c_pop_size],
          [c_pop_pos[0][0][0],c_pop_pos[0][0][1]+tc_input_y*c_pop_size],]
    draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=1) ## Th -> L6E synapses (arrows)
    draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

    path=[[c_pop_pos[0][1][0]+c_pop_size+(axon_cell_sep+axon_y_dist),c_pop_pos[2][0][1]+tc_input_y*c_pop_size],
          [c_pop_pos[0][1][0]+c_pop_size,c_pop_pos[2][0][1]+tc_input_y*c_pop_size],]
    draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=1) ## Th -> L4I synapses (arrows)
    draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

    path=[[c_pop_pos[0][1][0]+c_pop_size+(axon_cell_sep+axon_y_dist),c_pop_pos[0][0][1]+tc_input_y*c_pop_size],
          [c_pop_pos[0][1][0]+c_pop_size,c_pop_pos[0][0][1]+tc_input_y*c_pop_size],]
    draw_arrow(ax,path,lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=1) ## Th -> L6I synapses (arrows)
    draw_circle(ax,path[0],conn_radius,lw=0,fclr=exc_clr,zorder=0) ## connector

    if labels:
        ##################################################
        ## legend
        legend_x=[t_pos[0]+t_pop_size+axon_cell_sep,t_pos[0]+t_pop_size+axon_cell_sep+legend_length]
        legend_y=[t_pos[1],(t_pos[1]+2*t_pop_size/3)]
        draw_arrow(ax,[[legend_x[0],legend_y[1]],[legend_x[1],legend_y[1]]],lw=lw_axons,ls='solid',lclr=exc_clr,arrow_size=arrow_size,zorder=1)
        draw_arrow(ax,[[legend_x[0],legend_y[0]],[legend_x[1],legend_y[0]]],lw=lw_axons,ls='solid',lclr=inh_clr,arrow_size=arrow_size,zorder=1)

        ##################################################
        ## population names

        put_text(ax,[t_pos[0]+t_pop_size/2.,(t_pos[1]+t_pop_size/2.)],r'TC','w',fontdict1)

        put_text(ax,[c_pop_pos[0][0][0]+c_pop_size/2.,c_pop_pos[0][0][1]+c_pop_size/2.],r'L6E','w' if analysis_params.bw else 'k',fontdict1)
        put_text(ax,[c_pop_pos[0][1][0]+c_pop_size/2.,c_pop_pos[0][1][1]+c_pop_size/2.],r'L6I','w' if analysis_params.bw else 'k',fontdict1)
        
        put_text(ax,[c_pop_pos[1][0][0]+c_pop_size/2.,c_pop_pos[1][0][1]+c_pop_size/2.],r'L5E','w' if analysis_params.bw else 'k',fontdict1)
        put_text(ax,[c_pop_pos[1][1][0]+c_pop_size/2.,c_pop_pos[1][1][1]+c_pop_size/2.],r'L5I','w' if analysis_params.bw else 'k',fontdict1)
        
        put_text(ax,[c_pop_pos[2][0][0]+c_pop_size/2.,c_pop_pos[2][0][1]+c_pop_size/2.],r'L4E','w' if analysis_params.bw else 'k',fontdict1)
        put_text(ax,[c_pop_pos[2][1][0]+c_pop_size/2.,c_pop_pos[2][1][1]+c_pop_size/2.],r'L4I','w' if analysis_params.bw else 'k',fontdict1)
    
        put_text(ax,[c_pop_pos[3][0][0]+c_pop_size/2.,c_pop_pos[3][0][1]+c_pop_size/2.],r'L23E','w' if analysis_params.bw else 'k',fontdict1)
        put_text(ax,[c_pop_pos[3][1][0]+c_pop_size/2.,c_pop_pos[3][1][1]+c_pop_size/2.],r'L23I','w' if analysis_params.bw else 'k',fontdict1)
        
        put_text(ax,[c_pop_pos[-1][0][0],
                 c_pop_pos[-1][0][1]+c_pop_size+1.7*axon_cell_sep + 0.01],
                 r'cortico-cortical input','k',fontdict2)

        put_text(ax,[legend_x[1]+axon_y_dist,legend_y[1]],r'excitatory','k',fontdict3)
        put_text(ax,[legend_x[1]+axon_y_dist,legend_y[0]],r'inhibitory','k',fontdict3)

        ##################################################
        ## layer names

        put_text(ax,[0.2*c_pop_pos[0][0][0],c_pop_pos[0][1][1]+c_pop_size/2.],r'L6','k',fontdict1)
        put_text(ax,[0.2*c_pop_pos[1][0][0],c_pop_pos[1][1][1]+c_pop_size/2.],r'L5','k',fontdict1)
        put_text(ax,[0.2*c_pop_pos[2][0][0],c_pop_pos[2][1][1]+c_pop_size/2.],r'L4','k',fontdict1)
        put_text(ax,[0.2*c_pop_pos[3][0][0],c_pop_pos[3][1][1]+c_pop_size/2.],r'L2/3','k',fontdict1)
        
                
    if highlight is not None:
        ids = name_to_id_mapping[highlight]
        fontdict1['fontsize']=4
        put_text(ax,[c_pop_pos[ids[0]][ids[1]][0]+c_pop_size/2.,c_pop_pos[ids[0]][ids[1]][1]+c_pop_size/2.],highlight,'k',fontdict1)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.axis(ax.axis('equal'))
    
    return ax



def plot_population(ax,
                    params,
                    aspect='tight',
                    isometricangle=0,
                    plot_somas = True, plot_morphos = False,
                    num_unitsE = 1, num_unitsI=1,
                    clip_dendrites=False,
                    main_pops=True, 
                    Y = None,
                    big=True,
                    title='cell positions',
                    rasterized=True):
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
        Y : None, or string
            if not None, plot only soma locations of Y
        big : bool
            if False: leave out labels and reduce marker size
    
    return:
    ::
        axis : list
            the plt.axis() corresponding to input aspect
    '''

    name_to_id_mapping={'L6E':(3,0),
                        'L6I':(3,1),
                        'L5E':(2,0),
                        'L5I':(2,1),
                        'L4E':(1,0),
                        'L4I':(1,1),
                        'L23E':(0,0),
                        'L23I':(0,1)
    } 
    
    
    # DRAW OUTLINE OF POPULATIONS 
    
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    #contact points
    if big:
        ax.plot(params.electrodeParams['x'],
            params.electrodeParams['z'],
            '.', marker='o', markersize=2, color='k', zorder=0)
    else: 
        ax.plot(params.electrodeParams['x'],
            params.electrodeParams['z'],
            '.', marker='o', markersize=0.5, color='k', zorder=0)

    #outline of electrode       
    x_0 = params.electrodeParams['r_z'][1, 1:-1]
    z_0 = params.electrodeParams['r_z'][0, 1:-1]
    x = np.r_[x_0[-1], x_0[::-1], -x_0[1:], -x_0[-1]]
    z = np.r_[100, z_0[::-1], z_0[1:], 100]
    ax.fill(x, z, fc='w', lw=0.1, ec='k', zorder=-0.1, clip_on=False)

    #outline of populations:
    #fetch the population radius from some population
    r = params.populationParams['p23']['radius']

    theta0 = np.linspace(0, np.pi, 20)
    theta1 = np.linspace(np.pi, 2*np.pi, 20)
    
    zpos = np.r_[params.layerBoundaries[:, 0],
                 params.layerBoundaries[-1, 1]]
    
    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6']
    for i, z in enumerate(params.layerBoundaries.mean(axis=1)):
        if big:
            ax.text(r, z, ' %s' % layers[i], va='center', ha='left')

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
    
    if big:
        #plot a horizontal radius scalebar
        ax.plot([0, r], [z_0.min()]*2, 'k', lw=1, zorder=0, clip_on=False)
        ax.text(r / 2., z_0.min()-100, '$r$ = %i $\mu$m' % int(r), ha='center')
    
        #plot a vertical depth scalebar
        ax.plot([-r]*2, [z_0.min()+50, z_0.min()-50],
                'k', lw=1, zorder=0, clip_on=False)
        ax.text(-r, z_0.min(), r'100 $\mu$m', va='center', ha='right')
    
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    #fake ticks:
    if big:
        for pos in zpos:
            ax.text(-r, pos, '$z$=%i-' % int(pos), ha='right', va='center')
        ax.set_title(title, va='bottom')
   
    axis = ax.axis(ax.axis(aspect))


    def plot_pop_scatter(somapos, marker, colors, i):
        #scatter plot setting appropriate zorder for each datapoint by binning
        pitch = 100
        for lower in np.arange(-600, 601, pitch):
            upper = lower + pitch
            inds = (somapos[:, 1] >= lower) & (somapos[:, 1] < upper)
            if np.any(inds):
                if big:
                    ax.scatter(somapos[inds, 0],
                               somapos[inds, 2] - somapos[inds, 1] * np.sin(isometricangle),
                               s=10, facecolors=colors[i], edgecolors='gray', linewidths=0.1, zorder=lower,
                               marker = marker, clip_on=False, rasterized=rasterized)
                else:
                    ax.scatter(somapos[inds, 0],
                               somapos[inds, 2] - somapos[inds, 1] * np.sin(isometricangle),
                               s=3, facecolors=colors[i], edgecolors='gray', linewidths=0.1, zorder=lower,
                               marker = marker, clip_on=False, rasterized=rasterized)
                    



    # DRAW UNITS
    pop = next(zip(*params.mapping_Yy))
    
    #plot a symbol in each location with a unit 
    if plot_somas:
        if main_pops:
            colors = phlp.get_colors(np.unique(pop).size)

            #restructure 
            E, I = list(zip(*params.y_in_Y))
            pops_ = []

            if Y is None:
                for i in range(len(E)):
                    pops_.append(E[i])
                    pops_.append(I[i])
            else:
                ids = name_to_id_mapping[Y]
                if ids[1] == 0:
                    pops_.append(E[ids[0]])
                if ids[1] == 1:
                    pops_.append(I[ids[0]])

            for i, pops in enumerate(pops_):
                layer = np.unique(pop)[i]

                if layer.rfind('p') >= 0 or layer.rfind('E') >= 0:
                    marker = '^'
                elif layer.rfind('b') >= 0 or layer.rfind('I') >= 0:
                    marker = '*'
                elif layer.rfind('ss') >= 0:
                    marker = 'o'
                else:
                    raise Exception
                
                #get the somapos
                somapos = []
                for j, lname in enumerate(pops):
                    fname = glob.glob(os.path.join(params.populations_path, '%s*somapos.gdf' % lname))[0]
                    if j == 0:
                        somapos = np.loadtxt(fname).reshape((-1, 3))
                    else:
                        somapos = np.r_['0, 2', somapos, np.loadtxt(fname).reshape((-1, 3))]
                somapos = somapos[::5, :]
                if Y is None:
                    plot_pop_scatter(somapos, marker, colors, i)
                else:
                    plot_pop_scatter(somapos, marker, colors, ids[0]*2+ids[1])
            
        else:
            colors = phlp.get_colors(len(pop))
            i = 0
            for layer, _, _, _ in params.y_zip_list:
                
                #assign symbol
                if layer.rfind('p') >= 0 or layer.rfind('E') >= 0:
                    marker = '^'
                elif layer.rfind('b') >= 0 or layer.rfind('I') >= 0:
                    marker = '*'
                elif layer.rfind('ss') >= 0:
                    marker = 'x'
                else:
                    raise Exception
                    
                #get the somapos
                fname = glob.glob(os.path.join(params.populations_path, '%s*somapos.gdf' % layer))[0]
                somapos = np.loadtxt(fname).reshape((-1, 3))
            
                plot_pop_scatter(somapos, marker, colors, i)
                
                i += 1
    

    #plot morphologies in their appropriate locations
    if plot_morphos:
        if main_pops:
            colors = phlp.get_colors(np.unique(pop).size)

            #restructure 
            E, I = list(zip(*params.y_in_Y))
            pops_ = []
            for i in range(len(E)):
                pops_.append(E[i])
                pops_.append(I[i])

            for i, pops in enumerate(pops_):
                layer = np.unique(pop)[i]
                
                #get the somapos and morphos
                somapos = []
                for j, lname in enumerate(pops):
                    fname = glob.glob(os.path.join(params.populations_path, '%s*somapos.gdf' % lname))[0]
                    if j == 0:
                        somapos = np.loadtxt(fname).reshape((-1, 3))
                    else:
                        somapos = np.r_['0, 2', somapos, np.loadtxt(fname).reshape((-1, 3))]


                #add num_units morphologies per population with a random z-rotation
                if layer.rfind('p') >= 0 or layer.rfind('ss') >= 0 or layer.rfind('E') >= 0:
                    num_units = num_unitsE
                else:
                    num_units = num_unitsI            
                
                if num_units > somapos.shape[0]:
                    n = somapos.shape[0]
                else:
                    n = num_units

                #find some morphos for this population:
                morphos = []
                for fname in params.m_y:
                    if fname.rfind(layer) >= 0:
                        morphos.append(fname)
                                
                #plot some units
                for j in range(n):
                    cell = LFPy.Cell(morphology=os.path.join(params.PATH_m_y,
                                            np.random.permutation(morphos)[0]),
                                     nsegs_method='lambda_f',
                                     lambda_f=10,
                                     extracellular=False
                                    )
                    cell.set_pos(somapos[j, 0], somapos[j, 1], somapos[j, 2])
                    cell.set_rotation(z=np.random.rand()*np.pi*2)
    
                    #set up a polycollection
                    zips = []
                    for x, z in cell.get_idx_polygons():
                        zips.append(list(zip(x, z-somapos[j, 1] * np.sin(isometricangle))))
                    
                    polycol = PolyCollection(zips,
                                             edgecolors=colors[i],
                                             facecolors=colors[i],
                                             linewidths=(0.5),
                                             zorder=somapos[j, 1],
                                             clip_on=clip_dendrites,
                                             rasterized=rasterized)
    
                    ax.add_collection(polycol)
                
                i += 1


        else:
            colors = phlp.get_colors(len(pop))
            i = 0
            for layer, morpho, depth, size in params.y_zip_list:
    
    
                #get the somapos
                fname = glob.glob(os.path.join(params.populations_path, '%s*somapos.gdf' % layer))[0]
                somapos = np.loadtxt(fname).reshape((-1, 3))
    
                
                #add num_units morphologies per population with a random z-rotation
                if layer.rfind('p') >= 0 or layer.rfind('ss') >= 0 or layer.rfind('E') >= 0:
                    num_units = num_unitsE
                else:
                    num_units = num_unitsI            
                
                if num_units > somapos.shape[0]:
                    n = somapos.shape[0]
                else:
                    n = num_units
                
                #plot some units
                for j in range(n):
                    cell = LFPy.Cell(morphology=os.path.join(params.PATH_m_y, morpho),
                                     nsegs_method='lambda_f',
                                     lambda_f=10,
                                     extracellular=False
                                    )
                    cell.set_pos(somapos[j, 0], somapos[j, 1], somapos[j, 2])
                    cell.set_rotation(z=np.random.rand()*np.pi*2)
    
                    #set up a polycollection
                    zips = []
                    for x, z in cell.get_idx_polygons():
                        zips.append(list(zip(x, z-somapos[j, 1] * np.sin(isometricangle))))
                    
                    polycol = PolyCollection(zips,
                                             edgecolors=colors[i],
                                             facecolors=colors[i],
                                             linewidths=(0.5),
                                             zorder=somapos[j, 1],
                                             clip_on=clip_dendrites,
                                             rasterized=rasterized)
    
                    ax.add_collection(polycol)
                
                i += 1
    
    return axis


def plot_signal_sum(ax, params, fname='LFPsum.h5', unit='mV', scaling_factor=1.,
                    ylabels=True, scalebar=True, vlimround=None,
                    T=[800, 1000], ylim=[-1500, 0], color='k', fancy=False,
                    label='', transient=200, clip_on=False, rasterized=True,
                    **kwargs):
    '''
    on axes plot the summed LFP contributions
    
    args:
    ::
        
        ax : matplotlib.axes.AxesSubplot object
        fname : str/np.ndarray, path to h5 file or ndim=2 numpy.ndarray 
        unit : str, scalebar unit
        scaling_factor : float, scaling factor (e.g. to scale 10% data set up)
        ylabels : bool, show labels on y-axis
        scalebar : bool, show scalebar in plot
        vlimround : None/float, override autoscaling of data and scalebar
        T : list, [tstart, tstop], which timeinterval
        ylim : list of floats, see plt.gca().set_ylim
        color : str/colorspec tuple, color of shown lines
        fancy : bool,
        label : str, line labels
        rasterized : bool, rasterize line plots if true
        kwargs : additional keyword arguments passed to ax.plot()
    
    
    returns:
    ::
        
        vlimround : float, scalebar scaling factor, i.e., to match up plots
    
    '''

    if type(fname) == str and os.path.isfile(fname):
        f = h5py.File(fname)
        #load data
        data = f['data'].value
            
        tvec = np.arange(data.shape[1]) * 1000. / f['srate'].value    

        #for mean subtraction
        datameanaxis1 = f['data'].value[:, tvec >= transient].mean(axis=1)
        
        #close dataset
        f.close()
    elif type(fname) == np.ndarray and fname.ndim==2:
        data = fname
        tvec = np.arange(data.shape[1]) * params.dt_output
        datameanaxis1 = data[:, tvec >= transient].mean(axis=1)
    else:
        raise Exception('type(fname)={} not str or numpy.ndarray'.format(type(fname)))

    # slice
    slica = (tvec <= T[1]) & (tvec >= T[0])
    data = data[:,slica]
    
    #subtract mean in each channel
    #dataT = data.T - data.mean(axis=1)
    dataT = data.T - datameanaxis1
    data = dataT.T

    # normalize
    data = data*scaling_factor
    
    zvec = np.r_[params.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    vlim = abs(data).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim))
    else:
        pass
    yticklabels=[]
    yticks = []
    
    if fancy:
        colors=phlp.get_colors(data.shape[0])
    else:
        colors = [color]*data.shape[0]
    
    for i, z in enumerate(params.electrodeParams['z']):
        if i == 0:
            ax.plot(tvec[slica], data[i] * 100 / vlimround + z,
                    color=colors[i], rasterized=rasterized, label=label,
                    clip_on=clip_on, **kwargs)
        else: 
            ax.plot(tvec[slica], data[i] * 100 / vlimround + z,
                    color=colors[i], rasterized=rasterized, clip_on=clip_on,
                    **kwargs)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)
     
    if scalebar:
        ax.plot([tvec[slica][-1], tvec[slica][-1]],
                [-1300, -1400], lw=2, color='k', clip_on=False)
        ax.text(tvec[slica][-1]+np.diff(T)*0.02, -1350,
                r'%g %s' % (vlimround, unit),
                color='k', rotation='vertical',
                va='center')

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
    ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
    ax.set_ylim(ylim)
    ax.set_xlim(T)
    
    return vlimround


def plotConnectivity(ax):
    '''make an imshow of the intranetwork connectivity'''
    im = ax.pcolor(params.C_YX, cmap='hot')
    ax.axis(ax.axis('tight'))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(9)+0.5)
    ax.set_yticks(np.arange(8)+0.5)
    ax.set_xticklabels(params.X, rotation=270)
    ax.set_yticklabels(params.Y, )
    #ax.set_ylabel(r'to ($Y$)', ha='center')
    #ax.set_xlabel(r'from ($X$)', va='center')
    ax.xaxis.set_label_position('top')

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.01
    fig = plt.gcf()
    cax = fig.add_axes(rect)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('connectivity', ha='center')


def plotMorphologyTable(fig, params, rasterized=True):
    ax = fig.add_axes([0.075, 0.05, 0.925, 0.9])

    #using colors assosiated with each main postsyn population
    colors = phlp.get_colors(len(params.Y))

    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6']
    
    #cell composition
    composition = params.N_y.astype(float) / params.N_y.sum() * 100

    morphotypes = [
        'p23',
        'i23',
        'i23',
        'p4',
        'ss4',
        'ss4',
        'i4',
        'i4',
        'p5v1',
        'p5v2',
        'i5',
        'i5',
        'p6',
        'p5v1',
        'i5',
        'i5',
    ]

    y_zip_list = list(zip(params.y,
                   params.m_y,
                   params.depths,
                   params.N_y,
                   composition,
                   morphotypes))

    xpos = 300
    xvec = [xpos]
    COUNTER = 0
    COUNTER_J = 0
    prevpop = None
    totnsegs = []
    
    for layerind, morpho, depth, size, relsize, mtype in y_zip_list:
        fil = os.path.join(params.PATH_m_y, morpho)
        neuron.h('forall delete_section()')
        cell = LFPy.Cell(fil,
                         #nsegs_method='lambda_f',
                         #lambda_f = 10,
                         pt3d=False,
                         **params.cellParams)
        
        cell.set_pos(xpos, 0, depth)
        upperbound = params.layerBoundaries[0, 0]
        
        totnsegs.append(cell.totnsegs)
    
        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z)))

        if COUNTER > 0 and prevpop != morpho.split('_')[0]:
            COUNTER_J += 1
        prevpop = morpho.split('_')[0]


        polycol = PolyCollection(zips,
                                 linewidths=0.5,
                                 edgecolors=colors[COUNTER_J],
                                 facecolors=colors[COUNTER_J],
                                 rasterized=rasterized)
        ax.add_collection(polycol)
        
        
        xpos += 300
    
        xvec = np.r_[xvec, xpos]
        
        COUNTER += 1
        
    xvec = xvec[:-1]
    
            
    ax.hlines(params.layerBoundaries[:, 0], 0, xpos-100, linestyles='dotted')
    ax.hlines(params.layerBoundaries[-1, -1], 0, xpos-100, linestyles='dotted')
    ax.set_ylabel(r'depth ($\mu$m)')
    ax.set_yticks(np.r_[params.layerBoundaries[:, 0], params.layerBoundaries[-1, -1]])
    ax.set_xticks([])
    
    for i, z in enumerate(params.layerBoundaries.mean(axis=1)):
        ax.text(-50, z, layers[i], verticalalignment='center')
    
    for loc, spine in ax.spines.items():
        spine.set_color('none') # don't draw spine
    ax.yaxis.set_ticks_position('left')
            
    ax.axis(ax.axis('equal'))

    #plot annotations
    i = 0
    j = 0
    xpos = 150
    #prevcolor = None
    prevpop = None
    for layerind, morpho, depth, size, relsize, mtype in y_zip_list:
        pop = morpho.split('_')[0]
        
        ax.text(xpos+30, 300, '{:.1f}%'.format(relsize), ha='left')

        if i > 0 and prevpop != pop:            
                ax.vlines(xpos, -1800, 900,
                    clip_on=False)
                j += 1
                
        if j > 7: #HACK
            j = 7

        bigsize = np.array(params.full_scale_num_neurons).flatten()[j]
        if prevpop != pop:
            ax.text(xpos+30, 800, pop, ha='left', clip_on=False,)
            ax.text(xpos+30, 700, bigsize, ha='left', clip_on=False)
        
        ax.text(xpos+30, 100, size, ha='left', clip_on=False)        

        ax.text(xpos+30, 200,
                '{:.1f}%'.format(100*float(size)/bigsize),
                ha='left')
        #ax.text(xpos+30, 400, morpho.split('_', 1)[1].split('.hoc')[0][:8],
        #        ha='left', clip_on=False)
        ax.text(xpos+30, 400, '{}'.format(totnsegs[i]))
        ax.text(xpos+30, 500, mtype,
                ha='left', clip_on=False)
        ax.text(xpos+30, 600, layerind, ha='left', clip_on=False)

        prevpop = pop
        xpos += 300
    
        i += 1


    ax.text(90, 800, r'Population $Y$:', ha='right', clip_on=False)
    ax.text(90, 700, r'Pop. size $N_Y$:', ha='right', clip_on=False)
    ax.text(90, 600, r'Cell type $y$:', ha='right', clip_on=False)
    ax.text(90, 500, r'Morphology $M_y$:', ha='right', clip_on=False)
    ax.text(90, 400, r'Segments $n_\mathrm{comp}$:', ha='right', clip_on=False)
    ax.text(90, 300, r'Occurrence $F_y$:', ha='right', clip_on=False)
    ax.text(90, 200, r'Rel. Occurr. $F_{yY}$:', ha='right', clip_on=False)
    ax.text(90, 100, r'Cell count $N_y$:', ha='right', clip_on=False)
    
    ax.axis(ax.axis('equal'))
    
    return fig


def getMeanInpCurrents(params, numunits=100,
                       filepattern=os.path.join('simulation_output_default',
                                                'population_input_spikes*')):
    '''return a dict with the per population mean and std synaptic current,
    averaging over numcells recorded units from each population in the
    network
    
    Returned currents are in unit of nA.
    
    '''
    #convolution kernels
    x = np.arange(100) * params.dt
    kernel = np.exp(-x / params.model_params['tau_syn_ex'])


    #number of external inputs:
    K_bg = np.array(sum(params.K_bg, []))

    #compensate for DC CC connections if we're using that
    iDC =  K_bg * params.dc_amplitude * 1E-3 # unit ????
    
    data = {}
    
    #loop over network-populations
    for i, Y in enumerate(params.Y):
        if i % SIZE == RANK:
            #file to open
            fname = glob.glob(filepattern+'*' + Y + '*')[0]
            print(fname)
            #read in read data and assess units, up to numunits
            rawdata = np.array(helpers.read_gdf(fname))
            units = np.unique(rawdata[:, 0])
            if numunits > units.size:
                numcells = units.size
            else:
                numcells = numunits
            units = units[:numcells]
            
            #churn through data and extract the input currents per cell
            for j, unit in enumerate(units):
                slc = rawdata[:, 0] == unit
                #just the spikes:
                if j == 0:
                    dataslc = rawdata[slc, 2:]
                else:
                    dataslc = np.r_['0,3', dataslc, rawdata[slc, 2:]]
            
            #fix the datatype, it may be object
            dataslc = dataslc.astype(float)
                   
            #fill in data-structure
            data.update({
                Y : {
                    'E' : np.convolve(dataslc[:, :, 0].mean(axis=0),
                                      kernel, 'same')*1E-3 + float(iDC[i]),
                    'I' : np.convolve(dataslc[:, :, 1].mean(axis=0),
                                      kernel, 'same')*1E-3,
                    'tvec' : rawdata[slc, 1],
                    'numunits' : numunits,
                }
            })
    
    
    data = COMM.allgather(data)
    
    return {k: v for d in data for k, v in list(d.items())}


def getMeanVoltages(params, numunits=100,
                   filepattern=os.path.join('simulation_output_default',
                                            'voltages')):
    '''return a dict with the per population mean and std synaptic current,
    averaging over numcells recorded units from each population in the
    network
    
    Returned currents are in unit of nA.
    
    '''    
    data = {}
    
    #loop over network-populations
    for i, Y in enumerate(params.Y):
        if i % SIZE == RANK:
            #read in read data and assess units, up to numunits
            fname = glob.glob(filepattern+'*' + Y + '*')[0]
            print(fname)
            rawdata = np.array(helpers.read_gdf(fname))
            units = np.unique(rawdata[:, 0])
            if numunits > units.size:
                numcells = units.size
            else:
                numcells = numunits
            units = units[:numcells]
            
            #churn through data and extract the per cell voltages
            for j, unit in enumerate(units):
                slc = rawdata[:, 0] == unit
                #just the spikes:
                if j == 0:
                    dataslc = rawdata[slc, 2:]
                else:
                    dataslc = np.r_['0,3', dataslc, rawdata[slc, 2:]]
            
            #fix the datatype, it may be object
            dataslc = dataslc.astype(float)
                   
            #fill in data-structure
            data.update({
                Y : {
                    'data' : dataslc[:, :, 0].mean(axis=0),
                    'std' : dataslc[:, :, 0].std(axis=0),
                    'sample' : dataslc[0, :, 0],
                    'tvec' : rawdata[slc, 1],
                    'numunits' : numunits,
                }
            })
            
    
    data = COMM.allgather(data)
    
    return {k: v for d in data for k, v in list(d.items())}


def plot_signal_sum_colorplot(ax, params, fname='LFPsum.h5', unit='mV', N=1, ylabels = True,
                              T=[800, 1000], ylim=[-1500, 0], fancy=False, colorbar=True,
                              cmap='spectral_r', absmax=None, transient=200, rasterized=True):
    '''
    on colorplot and as background plot the summed CSD contributions
    
    args:
    ::
        ax : matplotlib.axes.AxesSubplot object
        T : list, [tstart, tstop], which timeinterval
        ylims : list, set range of yaxis to scale with other plots
        fancy : bool, 
        N : integer, set to number of LFP generators in order to get the normalized signal
    '''
    f = h5py.File(fname)
    data = f['data'].value
    tvec = np.arange(data.shape[1]) * 1000. / f['srate'].value
    
    #for mean subtraction
    datameanaxis1 = f['data'].value[:, tvec >= transient].mean(axis=1)
    
    # slice
    slica = (tvec <= T[1]) & (tvec >= T[0])
    data = data[:,slica]

    # subtract mean
    #dataT = data.T - data.mean(axis=1)
    dataT = data.T - datameanaxis1
    data = dataT.T

    # normalize
    data = data/N
    zvec = params.electrodeParams['z']
    
    if fancy:
        colors = phlp.get_colors(data.shape[0])
    else:
        colors = ['k']*data.shape[0]
    
    if absmax == None:
        absmax=abs(np.array([data.max(), data.min()])).max()  
    im = ax.pcolormesh(tvec[slica], np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]] + 50, data,
                           rasterized=rasterized, vmax=absmax, vmin=-absmax, cmap=cmap)
    ax.set_yticks(params.electrodeParams['z'])
    if ylabels:
        yticklabels = ['ch. %i' %(i+1) for i in np.arange(len(params.electrodeParams['z']))]
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels([])

    if colorbar:
        #colorbar
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.1)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(unit,labelpad=0.1)
        
    plt.axis('tight')

    ax.set_ylim(ylim)
    ax.set_xlim(T)

    
    f.close()
    
    return im


def calc_signal_power(params, fname, transient=200, Df=None, mlab=True, NFFT=1000,
                      noverlap=0, window=plt.mlab.window_hanning):
    '''
    calculates power spectrum of sum signal for all channels

    '''


    if type(fname) is str and os.path.isfile(fname):
        #open file
        f = h5py.File(fname)
        data = f['data'].value
        srate = f['srate'].value 
        tvec = np.arange(data.shape[1]) * 1000. / srate
        f.close()
    elif type(fname) is np.ndarray:
        data = fname
        srate = 1000.
        tvec = np.arange(data.shape[1]) * 1000. / srate
    else:
        raise Exception('{} not a file or array'.format(fname))
    
    # slice
    slica = (tvec >= transient)
    data = data[:,slica]

    # subtract mean
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    
    #extract PSD
    PSD=[]
    for i in np.arange(len(params.electrodeParams['z'])):
        if mlab:
            Pxx, freqs=plt.mlab.psd(data[i], NFFT=NFFT,
                                    Fs=srate, noverlap=noverlap, window=window)
        else:
            [freqs, Pxx] = helpers.powerspec([data[i,]], tbin= 1.,
                                        Df=Df, pointProcess=False)
            mask = np.where(freqs >= 0.)
            freqs = freqs[mask]
            Pxx = Pxx.flatten()
            Pxx = Pxx[mask]
            Pxx = Pxx/tvec[tvec >= transient].size**2
        PSD +=[Pxx.flatten()]
        
    PSD=np.array(PSD)

    return freqs, PSD


def plot_signal_power_colorplot(ax, params, fname, transient=200, Df=None,
                                mlab=True, NFFT=1000,
                                window=plt.mlab.window_hanning,
                                noverlap=0,
                                cmap = plt.cm.get_cmap('jet', 21),
                                vmin=None,
                                vmax=None):
    '''
    on axes plot the LFP power spectral density  
    The whole signal duration is used.
    args:
    ::
        ax : matplotlib.axes.AxesSubplot object
        fancy : bool, 
    '''
  
    zvec = np.r_[params.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    
    #labels
    yticklabels=[]
    yticks = []

    for i, kk in enumerate(params.electrodeParams['z']):
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(kk)

    
    freqs, PSD = calc_signal_power(params, fname=fname, transient=transient,Df=Df,
                                   mlab=mlab, NFFT=NFFT,
                                   window=window, noverlap=noverlap)

    #plot only above 1 Hz
    inds = freqs >= 1  # frequencies greater than 4 Hz  
    im = ax.pcolormesh(freqs[inds], zvec+50, PSD[:, inds],
                       rasterized=True, norm=LogNorm(),
                       vmin=vmin,vmax=vmax,
                       cmap=cmap, )
    
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)
    ax.semilogx()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(r'$f$ (Hz)', labelpad=0.1)
    ax.axis(ax.axis('tight'))


    return im


def plotPowers(ax, params, popkeys, dataset, linestyles, linewidths, transient=200, SCALING_POSTFIX='', markerstyles=None):
    '''plot power (variance) as function of depth for total and separate
    contributors

    Plot variance of sum signal
    '''
    
    colors = phlp.get_colors(len(popkeys))
    
    depth = params.electrodeParams['z']
    zpos = np.r_[params.layerBoundaries[:, 0],
                 params.layerBoundaries[-1, 1]]

    for i, layer in enumerate(popkeys):
        f = h5py.File(os.path.join(params.populations_path,
                                   '%s_population_%s' % (layer, dataset) + SCALING_POSTFIX + '.h5' ))
        ax.semilogx(f['data'].value[:, transient:].var(axis=1), depth,
                 color=colors[i],
                 ls=linestyles[i],
                 lw=linewidths[i],
                 marker=None if markerstyles is None else markerstyles[i],
                 markersize=2.5,
                 markerfacecolor=colors[i],
                 markeredgecolor=colors[i],
                 label=layer,
                 clip_on=True
                 )
    
        f.close()
    
    f = h5py.File(os.path.join(params.savefolder, '%ssum' % dataset + SCALING_POSTFIX + '.h5' ))
    ax.plot(f['data'].value[:, transient:].var(axis=1), depth,
                 'k', label='SUM', lw=1.25, clip_on=False)
    
    f.close()

    ax.set_yticks(zpos)
    ax.set_yticklabels([])
    #ax.set_xscale('log')
    try: # numticks arg only exists for latest matplotlib version
        ax.xaxis.set_major_locator(plt.LogLocator(base=10,
                                    subs=np.linspace(-10, 10, 2), numticks=6))
    except:
        ax.xaxis.set_major_locator(plt.LogLocator(base=10,
                                    subs=np.linspace(-10, 10, 2)))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=[1.]))
    ax.axis('tight')


def plotting_correlation(params, x0, x1, ax, lag=20., scaling=None, normalize=True,
                         color='k', unit=r'$cc=%.3f$' , title='firing_rate vs LFP',
                         scalebar=True, **kwargs):
    ''' mls
    on axes plot the correlation between x0 and x1
    
    args:
    ::
        x0 : first dataset
        x1 : second dataset - the LFP usually here
        ax : matplotlib.axes.AxesSubplot object
        title : text to be used as current axis object title
        normalize : if True, signals are z-scored before applying np.correlate
        unit : unit for scalebar
    '''
    zvec = np.r_[params.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    
    xcorr_all=np.zeros((params.electrodeParams['z'].size, x0.shape[-1]), dtype=float)
    
    if normalize:
        for i, z in enumerate(params.electrodeParams['z']):
            if x0.ndim == 1:
                x2 = x1[i, ]
                xcorr1 = np.correlate(helpers.normalize(x0),
                                      helpers.normalize(x2), 'same') / x0.size
            elif x0.ndim == 2:
                xcorr1 = np.correlate(helpers.normalize(x0[i, ]),
                                      helpers.normalize(x1[i, ]), 'same') / x0.shape[-1]
            
            xcorr_all[i,:]=xcorr1
    else:
        for i, z in enumerate(params.electrodeParams['z']):
            if x0.ndim == 1:
                x2 = x1[i, ]
                xcorr1 = np.correlate(x0,x2, 'same')
            elif x0.ndim == 2:
                xcorr1 = np.correlate(x0[i, ],x1[i, ], 'same')
                
            xcorr_all[i,:]=xcorr1
    

    # Find limits for the plot
    if scaling is None:
        vlim = abs(xcorr_all).max()
        vlimround = 2.**np.round(np.log2(vlim))
    else:
        vlimround = scaling
    
    yticklabels=[]
    yticks = []
    
    #temporal slicing
    lagvector = np.arange(-lag, lag+1).astype(int)
    inds = lagvector + x0.shape[-1] // 2
    
    
    for i, z in enumerate(params.electrodeParams['z']):
        ax.plot(lagvector, xcorr_all[i,inds[::-1]] * 100. / vlimround + z, 'k',
                clip_on=True, rasterized=False, color=color, **kwargs)
        yticklabels.append('ch. %i' %(i+1))
        yticks.append(z)    

    phlp.remove_axis_junk(ax)
    ax.set_title(title, va='center')
    ax.set_xlabel(r'$\tau$ (ms)', labelpad=0.1)

    ax.set_xlim(-lag, lag)
    ax.set_ylim(z-100, 100)
    
    axis = ax.axis()
    ax.vlines(0, axis[2], axis[3], 'k' if analysis_params.bw else 'k', 'dotted', lw=0.25)

    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ## Create a scaling bar
    if scalebar:
        ax.plot([lag, lag],
            [-1500, -1400], lw=2, color='k', clip_on=False)
        ax.text(lag*1.04, -1450, unit % vlimround,
                    rotation='vertical', va='center')
    
    return xcorr_all[:, inds[::-1]], vlimround
