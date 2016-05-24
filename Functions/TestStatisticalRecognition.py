import unittest
import numpy as np
from ParticleSuspension import *
from StatisticalRecognition import *

class TestStatisticalRecognition(unittest.TestCase):

    def test_load_save(self):
        x = StatisticalRecognition()
        a = [0,0,0,0,0]
        b = [1,1,1,1,1]
        x.save("test.txt", [a,b], header="This is a test file")
        (A, B) = x.load("test.txt")
        self.assertTrue( np.array_equal(A, a) ) 
        self.assertTrue( np.array_equal(B, b) )

    # Created the params in genCurveNoiseFit.py
    def test_curveNoiseFit(self):
        x = ParticleSuspension(100, 0.2)
        (swells, curve) = x.tagCurvature(0, 2.0, 0.01)
        Max = max(curve)
        sr = StatisticalRecognition()
        meanParams, sdParams = sr.load("curveNoiseParams.txt")
        mean, sd = sr.expectedNoise(100, meanParams, sdParams)
        self.assertTrue( ((mean - sd*1.5) < Max < (mean + sd*1.5)) )

    def test_rateNoiseFit(self):
        x = ParticleSuspension(100, 0.2)
        (swells, rate) = x.tagRate(0, 2.0, 0.01)
        Max = max(rate)
        sr = StatisticalRecognition()
        meanParams, sdParams = sr.load("rateNoiseParams.txt")
        mean, sd = sr.expectedNoise(100, meanParams, sdParams)
        self.assertTrue( ((mean - sd*1.5) < Max < (mean + sd*1.5)) )




if __name__ == "__main__":
    unittest.main()