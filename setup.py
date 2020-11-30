#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''setup file for the hybridLFPy python module'''
import os
from setuptools import setup
import numpy
from Cython.Build import cythonize

with open('README.md') as file:
    long_description = file.read()

# read version file
d = {}
exec(open(os.path.join('hybridLFPy', 'version.py')).read(), None, d)
version = d['version']

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
    package_data={'hybridLFPy': [os.path.join('testing', 'testing-X-0.gdf')]},
    # cmdclass=cmdclass,
    # ext_modules=ext_modules,
    ext_modules=cythonize(os.path.join('hybridLFPy', 'helperfun.pyx'),
                          language_level='3',
                          ),
    include_dirs=[numpy.get_include()],
    description='methods to calculate LFPs with spike events from network sim',
    long_description=long_description,
    license='LICENSE',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
    ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'LFPy',
                      'mpi4py', 'Cython'],
)
