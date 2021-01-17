# Unit tests for rrng.py
#
# To run these tests use the following command from the
# root directory:
#
# python -m unittest
#

import unittest
import numpy as np
from rrng import is_power_of_two, xgcd_x
from rrng import ReversibleLCG, GeneratorLCGReversible


class TestRRNG(unittest.TestCase):
    
    def test_is_power_of_two(self):
        self.assertTrue(all(is_power_of_two(x) for x in [0, 1, 2, 8, 256]))
        self.assertFalse(any(is_power_of_two(x) for x in [-1, 3, 9, 257]))

    def test_xgcd_x(self):
        self.assertEqual(xgcd_x(46, 240), 47)

    def test_ReversibleLCG(self):

        # Values produced by rlcg.hpp with seed = 42
        test_values = [0, 293047021, 968358053, 1773127077, 560055359, 773728940]

        # Example use: as an iterator
        rng = ReversibleLCG(42)
        for i, x in enumerate(iter(rng)):
            self.assertEqual(x, test_values[i + 1])
            if i == 4:
                break
        
        # Reverse and go back
        rng.reverse()
        self.assertFalse(rng.forward)
        for i, x in enumerate(iter(rng)):
            assert x == test_values[4 - i]
            if i == 4:
                break
        
        # Using the class methods directly
        rng = ReversibleLCG(42)
        x = [rng.next() for i in range(5)]
        self.assertEqual(x, test_values[1:])
        x = [rng.prev() for i in range(5)]
        self.assertEqual(x, list(reversed(test_values[:-1])))
        self.assertTrue(rng.forward)

        # Class for producing arrays of random numbers
        rng = GeneratorLCGReversible(42)
        x = rng.random()
        self.assertTrue(np.array_equal(x, test_values[1]))

        rng = GeneratorLCGReversible(42)
        x = rng.random(size=2)
        self.assertTrue(np.array_equal(x, test_values[1:3]))

        x = rng.random(size=3)
        self.assertTrue(np.array_equal(x, test_values[3:]))

        rng.reverse()
        x = rng.random()
        self.assertEqual(x, test_values[4])

        rng.reverse()
        x = rng.random()
        self.assertEqual(x, test_values[5])

        rng.reverse()
        x = rng.random(size=5, update=False)
        self.assertTrue(np.array_equal(x, test_values[-2::-1]))

        x = rng.random(size=5)
        self.assertTrue(np.array_equal(x, test_values[-2::-1]))

        x = rng.random(size=5, increment=1)
        self.assertTrue(np.array_equal(x, test_values[1:]))

        x = rng.random(size=5, increment=-1)
        self.assertTrue(np.array_equal(x, test_values[-2::-1]))
