#!/usr/bin/env python
# -*- coding: utf-8 -*-
from plot_methods import os, plt, plotMorphologyTable
from cellsim16popsParams_default import multicompartment_params
import analysis_params

if __name__ == '__main__':
    params = multicompartment_params()
    ana_params = analysis_params.params()
    ana_params.set_PLOS_2column_fig_style(ratio=0.52)
    fig = plt.figure()
    fig = plotMorphologyTable(fig, params, rasterized=False)
    fig.savefig('figure_02.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    fig.savefig('figure_02.eps', bbox_inches='tight', pad_inches=0)

    plt.show()
