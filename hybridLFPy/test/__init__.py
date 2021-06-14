#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Documentation:

Tests for hybridLFPy classes and methods
"""


def _test(verbosity=2):
    '''
    Run unittests for hybridLFPy


    Arguments
    ---------
    verbosity : int
        verbosity level


    Returns
    -------
    None
    '''
    import unittest
    from .test_module import TestHybridLFPy
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridLFPy)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
