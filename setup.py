#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''setup file for the hybridLFPy python module'''
import os
from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name = 'hybridLFPy',
    version = '0.1.3',
    maintainer = 'Espen Hagen',
    maintainer_email = 'espe.hage@fys.uio.no',
    url = 'https://github.com/INM-6/hybridLFPy',
    download_url = 'https://github.com/INM-6/hybridLFPy/tarball/v0.1.3',
    packages = ['hybridLFPy'],
    provides = ['hybridLFPy'],
    package_data = {'hybridLFPy' : [os.path.join('testing', 'testing-X-0.gdf')]},
    description = 'methods to calculate LFPs with spike events from network sim',
    long_description = long_description,
    license='LICENSE',
        classifiers=[
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Cython',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Development Status :: 4 - Beta',
            ],
    install_requires = [
        'numpy', 'scipy', 'matplotlib', 'LFPy', 'mpi4py',
        'NeuroTools',
        ],

)
