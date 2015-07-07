#!/usr/bin/env python
'''Tests for hybridLFPy classes and methods'''
import os
import sys
import numpy as np
import numpy.testing as nt
import unittest
import hybridLFPy


class TestHybridLFPy(unittest.TestCase):
    '''
    hybridLFPy test class
    
    To run tests, simply issue
    >>>import hybridLFPy
    >>>hybridLFPy.test()
    
    '''
    
    def __init__(self, *args, **kwargs):
        super(TestHybridLFPy, self).__init__(*args, **kwargs)
    
        self.networkSim = hybridLFPy.CachedNetwork(
            simtime=1000.,
            dt=0.1,
            spike_output_path=os.path.join(hybridLFPy.__path__[0], 'testing'),
            label='testing',
            ext='gdf',
            GIDs={'X' : [1, 100]}   
        )

    def test_CachedNetwork_01(self):
        '''test CachedNetwork 01'''
        assert(self.networkSim.X == ['X'])

    def test_CachedNetwork_02(self):
        '''test CachedNetwork 02'''
        nt.assert_array_equal(self.networkSim.N_X, np.array([100]))

    def test_CachedNetwork_03(self):
        '''test CachedNetwork 01'''
        nt.assert_array_equal(self.networkSim.nodes['X'], np.arange(100)+1)

    def test_CachedNetwork_04(self):
        '''test CachedNetwork 04''' 
        nt.assert_array_equal(self.networkSim.dbs['X'].neurons(), np.array([1, 50]))

    def test_CachedNetwork_05(self):
        '''test CachedNetwork 05'''
        nt.assert_array_equal(self.networkSim.dbs['X'].interval((0, 1000)), [(50, 100.0), (1, 200.0)])

    def test_CachedNetwork_06(self):
        '''test CachedNetwork 06'''
        nt.assert_array_equal(self.networkSim.dbs['X'].interval((0, 100)), [(50, 100.0)])

    def test_CachedNetwork_07(self):
        '''test CachedNetwork 07'''
        nt.assert_array_equal(self.networkSim.dbs['X'].interval((200, 1000)), [(1, 200.0)])
        
    def test_CachedNetwork_08(self):
        '''test CachedNetwork 08'''
        nt.assert_array_equal(self.networkSim.dbs['X'].select_neurons_interval(self.networkSim.dbs['X'].neurons(), (0,1000)), [np.array([ 200.]), np.array([ 100.])])


def test(verbosity=2):
    '''
    Run unittests for the CSD toolbox
    
    
    Arguments
    ---------
    verbosity : int
        verbosity level
    
    
    Returns
    -------
    None
    '''
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridLFPy)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
    


if __name__ == '__main__':
    test()