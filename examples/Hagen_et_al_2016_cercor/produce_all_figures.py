#!/usr/bin/env python
import os
from glob import glob

# run analysis script to precompute certain plotted values such as power spectra
os.system('python analysis.py')

# iterate over figure generating scripts to create figures
for f in glob('figure_*.py'):
    os.system('python {}'.format(f))

