import unittest
import Analysis
from ParticleSuspension import *

class TestAnalysis(unittest.TestCase):

	def test_binarySearch(self):
		x = ParticleSuspension(2, 0.2)
		x.setCenters([[0,0],[0.5, 0]])
		self.assertEqual(Analysis.searchForFrac(x, 1.0, 0.01), 0.5)

	def test_binarySearch2(self):
		x = ParticleSuspension(2, 0.2)
		x.setCenters([[0,0],[0.1, 0]])
		self.assertEqual(Analysis.searchForFrac(x, 1.0, 0.01), 0.1)

	def test_binarySearch3(self):
		x = ParticleSuspension(3, 0.2)
		x.setCenters([[0,0],[0.5, 0], [1.3, 0]])
		self.assertEqual(Analysis.searchForFrac(x, 1.0, 0.01), 1.3-0.5)

	def test_binarySearch4(self):
		x = ParticleSuspension(4, 0.2)
		x.setCenters([[0,0],[0.5, 0], [1.3, 0], [1.3, 1.3]])
		self.assertEqual(Analysis.searchForFrac(x, 1.0, 0.01), 1.3)

	def test_binarySearch5(self):
		x = ParticleSuspension(4, 0.2)
		x.setCenters([[0,0],[0.5, 0], [1.3, 0], [1.3, 1.3]])
		self.assertEqual(Analysis.searchForFrac(x, 0.20, 0.01), 0.5)




	


if __name__ == "__main__":
	unittest.main() 