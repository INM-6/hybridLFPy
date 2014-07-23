#!/usr/bin/env python

from sys import version
if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None
    
    
from distutils.core import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name = 'hybridLFPy',
    version = '0.1',
    maintainer = 'Espen Hagen',
    maintainer_email = 'e.hagen@fz-juelich.de',
    url = 'http://www.fz-juelich.de/inm/inm-6',
    packages = ['hybridLFPy'],
    provides = ['hybridLFPy'],
    description = 'methods to calculate LFPs with spike events from network sim',
    long_description = long_description,
    license='LICENSE',
        classifiers=[
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Cython',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Development Status :: 4 - Beta',
            ],
    requires = [
        'numpy', 'scipy', 'matplotlib', 'neuron', 'LFPy', 'sqlite3', 'mpi4py',
        'nest', 'NeuroTools',
        ],

)
