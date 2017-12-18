import unittest

import manipulations as mp
from wavesets import get_wavesets


class ManipulationTests(unittest.TestCase):

    def testMaxJump(self):
        result = mp.max_jump([1, 0, -1, 4])
        self.failUnless(result == 5)

    def testMost(self):
        result = mp.most([1, 1, 2, 3, 1])
        self.failUnless(result == 1)

    def testMostWavesets(self):
        xs = [[2, 4, 2], [1, 1, 2, 3]]
        result = mp.most_wavesets(xs)
        self.failUnless(result == [[1, 1, 2, 3], [2, 4, 2]])

    def testMaxWavesets(self):
        xs = [[2, 4, 2], [1, 1, 2, 3]]
        result = mp.max_wavesets(xs)
        self.failUnless(result == [[1, 1, 2, 3], [2, 4, 2]])

    def testAvgWavesets(self):
        xs = [[2, 4, 2], [1, 1, 2, 3]]
        result = mp.avg_wavesets(xs)
        self.failUnless(result == [[1, 1, 2, 3], [2, 4, 2]])

    def testInterpolate(self):
        xs = [1, 2, 3]
        ys = [2, 2, 4]
        one = mp.interpolate(xs, ys, 1)
        self.failUnless(one == xs)
        two = mp.interpolate(xs, ys, 0.5)
        self.failUnless(two == [1.5, 2, 3.5])

    def testStretches(self):
        xs = [1, 3, 5, 7]
        one = mp.stretches(xs, [4])
        self.failUnless(list(one[0]) == xs)
        two = mp.stretches(xs, [7])
        self.failUnless(list(two[0]) == [1, 2, 3, 4, 5, 6, 7])

    def testLinspace(self):
        result = mp.my_linspace(0, 1, 1)
        self.failUnless(list(result) == [0.5])

    def testLensN(self):
        result = mp.lens_n(0, 1, 3)
        self.failUnless(list(result) == [0.25, 0.5, 0.75])

    def testLensStep(self):
        result = mp.lens_step(0, 1, 0.25)
        self.failUnless(list(result) == [0.25, 0.5, 0.75])

    def testLensRel(self):
        result = mp.lens_rel(1, 2, 2)
        self.failUnless(list(result) == [1.5])

    def testQuantize(self):
        result = mp.quantize(6, [8, 5])
        self.failUnless(result == 5)

    def testLensQuant(self):
        result = mp.lens_quant(1, 1.9, lambda l1, l2: mp.lens_n(l1, l2, 1), [3 / 2, 5 / 4])
        self.failUnless(result == [1.5])

    def testGetHarmony(self):
        result = mp.get_harmony([1, 1, 0, 0], 2)
        self.failUnless(result == [1, 0, 1, 0])

    def testHarmonize(self):
        result = mp.harmonize([1, 1, 0, 0], [1, 2])
        self.failUnless(result == [1, 0.5, 0.5, 0])

    def testHarmonizeWavesets(self):
        result = mp.harmonize_wavesets([[1, 1, 0, 0]], [1, 2])
        self.failUnless(result == [[1, 0.5, 0.5, 0]])


class WavesetTests(unittest.TestCase):

    def testGetWavesets(self):
        xs = [1, 1, 1, 1, -1, 1, 1, 1, -1, -1]
        first, result, last = get_wavesets(xs, 0, 1)
        self.failUnless(first == [1, 1, 1, 1])
        self.failUnless(result == [[-1], [1, 1, 1]])
        self.failUnless(last == [-1, -1])
