#!/usr/bin/env python
"""
Function true_lam_csd specification for calculation of true laminar CSD
from the current-distribution on `LFPy.cell.Cell` objects, assuming line
sources for each individual compartment, including the soma.
"""
import numpy as np


def _PrPz(r0, z0, r1, z1, r2, z2, r3, z3):
    """
    Intersection point for infinite lines.
    
    Parameters
    ----------
    r0 : float
    z0 : float
    r1 : float
    z1 : float
    r2 : float
    z2 : float
    r3 : float
    z3 : float

    Returns
    ----------
    Pr : float    
    Pz : float
    hit : bool

    """
    Pr = ((r0*z1 - z0*r1)*(r2 - r3) - (r0 - r1)*(r2*z3 - r3*z2)) / \
                        ((r0 - r1)*(z2 - z3) - (z0 - z1)*(r2-r3))
    Pz = ((r0*z1 - z0*r1)*(z2 - z3) - (z0 - z1)*(r2*z3 - r3*z2)) / \
                        ((r0 - r1)*(z2 - z3) - (z0 - z1)*(r2-r3))
    
    if Pr >= r0 and Pr <= r1 and Pz >= z0 and Pz <= z1:
        hit = True
    elif Pr <= r0 and Pr >= r1 and Pz >= z0 and Pz <= z1:
        hit = True
    elif Pr >= r0 and Pr <= r1 and Pz <= z0 and Pz >= z1:
        hit = True
    elif Pr <= r0 and Pr >= r1 and Pz <= z0 and Pz >= z1:
        hit = True
    else:
        hit = False
        
    return [Pr, Pz, hit]


def true_lam_csd(cell, dr=100, z=None):
    """
    Return CSD from membrane currents as function along the coordinates
    of the electrode along z-axis. 


    Parameters
    ----------
    cell : `LFPy.cell.Cell` or `LFPy.cell.TemplateCell` object.
        Cell.
    dr : float
        Radius of the cylindrical CSD volume.
    z : numpy.ndarray
        Z-coordinates of electrode.


    Returns
    ----------
    CSD : numpy.ndarray
        Current-source density (in pA * mum^-3).
    
    """
    if type(z) != type(np.ndarray(shape=0)):
        raise ValueError('type(z) should be a numpy.ndarray')

    dz = abs(z[1] - z[0])
    CSD = np.zeros((z.size, cell.tvec.size,))
    r_end = np.sqrt(cell.xend**2 + cell.yend**2)
    r_start = np.sqrt(cell.xstart**2 + cell.ystart**2)
    volume = dz * np.pi * dr**2

    for i in range(len(z)):
        aa0 = cell.zstart < z[i] + dz/2
        aa1 = cell.zend < z[i] + dz/2
        bb0 = cell.zstart >= z[i] - dz/2
        bb1 = cell.zend >= z[i] - dz/2
        cc0 = r_start < dr
        cc1 = r_end < dr
        ii = aa0 & bb0 & cc0 # startpoint inside volume
        jj = aa1 & bb1 & cc1 # endpoint inside volume

        for j in range(cell.zstart.size):
            # Calc fraction of source being inside control volume from 0-1
            # both start and endpoint in volume
            if ii[j] and jj[j]:
                CSD[i,] = CSD[i, ] + cell.imem[j, ] / volume
            # Startpoint in volume:
            elif ii[j] and not jj[j]: 
                z0 = cell.zstart[j]
                r0 = r_start[j]
                
                z1 = cell.zend[j]
                r1 = r_end[j]

                if r0 == r1:
                    # Segment is parallel with z-axis
                    frac = -(z0 - z[i]-dz/2) / (z1 - z0)
                else:
                    # Not parallel with z-axis                    
                    L2 = (r1 - r0)**2 + (z1 - z0)**2
    
                    z2 = [z[i]-dz/2, z[i]+dz/2, z[i]-dz/2]
                    r2 = [0, 0, dr]
                    
                    z3 = [z[i]-dz/2, z[i]+dz/2, z[i]+dz/2]
                    r3 = [dr, dr, dr]
    
                    P = []
                    for k in range(3):
                        P.append(_PrPz(r0, z0, r1, z1, r2[k], z2[k], r3[k], z3[k]))
                        if P[k][2]:
                            vL2 = (P[k][0] - r0)**2 + (P[k][1] -z0)**2
                            frac = np.sqrt(vL2 / L2)
                CSD[i,] = CSD[i, ] + frac * cell.imem[j, ] / volume
            # Endpoint in volume:
            elif jj[j] and not ii[j]: 
                z0 = cell.zstart[j]
                r0 = r_start[j]
                
                z1 = cell.zend[j]
                r1 = r_end[j]
                
                if r0 == r1:
                    # Segment is parallel with z-axis
                    frac = (z1 - z[i]+dz/2) / (z1 - z0)
                else:
                    # Not parallel with z-axis
                    L2 = (r1 - r0)**2 + (z1 - z0)**2
    
                    z2 = [z[i]-dz/2, z[i]+dz/2, z[i]-dz/2]
                    r2 = [0, 0, dr]
                    
                    z3 = [z[i]-dz/2, z[i]+dz/2, z[i]+dz/2]
                    r3 = [dr, dr, dr]
    
                    P = []
                    for k in range(3):
                        P.append(_PrPz(r0, z0, r1, z1, r2[k], z2[k], r3[k], z3[k]))
                        if P[k][2]:
                            vL2 = (r1 - P[k][0])**2 + (z1 - P[k][1])**2
                            frac = np.sqrt(vL2 / L2)
                CSD[i,] = CSD[i, ] + frac * cell.imem[j, ] / volume
            else:
                pass

    return CSD


if __name__ == "__main__":
    import doctest
    doctest.testmod()
