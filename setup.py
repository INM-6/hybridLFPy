#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''setup file for the hybridLFPy Python module'''
import os
from setuptools import setup, Extension
import numpy
from Cython.Distutils import build_ext

with open('README.md') as file:
    long_description = file.read()

# read version file
d = {}
exec(open(os.path.join('hybridLFPy', 'version.py')).read(), None, d)
version = d['version']

# Cython extension
cmdclass = {'build_ext': build_ext}
ext_modules = [Extension('hybridLFPy.helperfun',
                         [os.path.join('hybridLFPy', 'helperfun.pyx')],
                         include_dirs=[numpy.get_include()])]

setup(
    name='hybridLFPy',
    version=version,
    maintainer='Espen Hagen',
    maintainer_email='espenhgn@users.noreply.github.com',
    url='https://github.com/INM-6/hybridLFPy',
    download_url='https://github.com/INM-6/hybridLFPy/tarball/v{}'.format(
        version),
    packages=['hybridLFPy'],
    provides=['hybridLFPy'],
    package_data={'hybridLFPy': ['*.pyx',
                                 os.path.join('test', '*.py'),
                                 os.path.join('test', 'testing-X-0.gdf')]
                  },
    include_package_data=True,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    description=('methods to calculate extracellular signals of neural '
                 'activity from spike events from spiking neuron networks'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='LICENSE',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
    ],
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.14',
        'Cython>=0.20',
        'h5py>=2.5',
        'mpi4py>=1.2',
        'LFPy>=2.2'],
    extras_require={'tests': ['pytest'],
                    'docs': ['sphinx', 'numpydoc', 'sphinx_rtd_theme'],
                    },
    dependency_links=[],
)
