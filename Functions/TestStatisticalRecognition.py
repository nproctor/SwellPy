import unittest
from ParticleSuspension import *
from MemoryRecognition import *
from StatisticalRecognition import *

class TestStatisticalRecognition(unittest.TestCase):

	def test_tagNoise(self):
		x = StatisticalRecognition()
		x.newSystem(100, 0.2)
		x.tagCurveNoise(-1.0, 1.0, 0.2, 2) # Just make sure these run 
		x.tagRateNoise(-1.0, 1.0, 0.2, 5) # Just make sure these run 
		# Think of way to do actual test (needed?)


if __name__ == "__main__":
	unittest.main()