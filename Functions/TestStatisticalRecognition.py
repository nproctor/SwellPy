import unittest
import numpy as np
from ParticleSuspension import *
import StatisticalRecognition as sr

class TestStatisticalRecognition(unittest.TestCase):

    def test_load_save(self):
        a = [0,0,0,0,0]
        b = [1,1,1,1,1]
        sr.save("test.txt", [a,b], header="This is a test file")
        (A, B) = sr.load("test.txt")
        self.assertTrue( np.array_equal(A, a) ) 
        self.assertTrue( np.array_equal(B, b) )

    # Created the params in genCurveNoiseFit.py
    def test_curveNoiseFit(self):
        x = ParticleSuspension(100, 0.2)
        (swells, curve) = x.tagCurve(0, 2.0, 0.01)
        Max = max(curve)
        meanParams, sdParams = sr.load("curveNoiseParams.txt")
        mean, sd = sr.expectedNoise(100, meanParams, sdParams)
        self.assertTrue( ((mean - sd*1.5) < Max < (mean + sd*1.5)) )

    def test_rateNoiseFit(self):
        x = ParticleSuspension(100, 0.2)
        (swells, rate) = x.tagRate(0, 2.0, 0.01)
        Max = max(rate)
        meanParams, sdParams = sr.load("rateNoiseParams.txt")
        mean, sd = sr.expectedNoise(100, meanParams, sdParams)
        self.assertTrue( ((mean - sd*1.5) < Max < (mean + sd*1.5)) )




if __name__ == "__main__":
    unittest.main()