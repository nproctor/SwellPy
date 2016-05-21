import unittest
from ParticleSuspension import *

class TestParticleSuspension(unittest.TestCase):


	def test_randomCentersShape(self):
		x = ParticleSuspension(10, 0.2)
		self.assertEqual(x.centers.shape, (10, 2))

	def test_randomCentersBounds(self):
		x = ParticleSuspension(10, 0.2)
		self.assertTrue( (x.centers > 0).all() )
		self.assertTrue( (x.centers < x.boxsize).all() )

	def test_setBadFormatCenters(self):
		x = ParticleSuspension(3, 0.2)
		before = x.centers
		x.setCenters([0,2,3,4,4,5,6]) # Attempt bad center reassignment (message will print)
		after = x.centers
		self.assertTrue( before is after )

	def test_setOutOfBoundsCenters(self):
		x = ParticleSuspension(2, 0.2)
		before = x.centers
		x.setCenters([[0,1],[-10,12]])
		after = x.centers
		self.assertTrue( before is after ) # Attemp bad center reassignment (message will print)

	def test_setCenters(self):
		x = ParticleSuspension(2, 0.2)
		new = [[0,1],[2.0, 1.89]]
		x.setCenters(new)
		self.assertTrue( (x.centers == new).all() )

	def test_reset(self):
		x = ParticleSuspension(10, 0.2, 10) # Feeding seed for randomization
		before = x.centers
		x.reset(10) # feeding same seed
		after = x.centers
		self.assertTrue( (before == after).all() ) 


if __name__ == "__main__":
	unittest.main()