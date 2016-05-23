import unittest
from ParticleSuspension import *
from MemoryRecognition import *

class TestMemoryRecognition(unittest.TestCase):

	def test_fracTagAt(self):
		x = MemoryRecognition()
		x.newSystem(2, 0.2)
		x.system.setCenters([[0,1],[0, 1.25]])
		self.assertEqual(x.fracTagAt(1.0), 1.0)
		self.assertEqual(x.fracTagAt(0.24), 0)

	def test_fracTag(self):
		x = MemoryRecognition()
		x.newSystem(3, 0.2)
		x.system.setCenters([[0,1],[0, 1.25], [0, 0.25]])
		(swell, tagged) = x.fracTag(0, 1.0, 0.2)
		self.assertTrue( (tagged == [0.0, 0.0, 2.0/3, 2.0/3, 1.0, 1.0]).all() )
		self.assertTrue( np.allclose(swell, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) )

	def test_fracTagNegative(self):
		x = MemoryRecognition()
		x.newSystem(3, 0.2)
		x.system.setCenters([[0,1],[0, 1.25], [0, 0.25]])
		(swell, tagged) = x.fracTag(0, -1.0, -0.2)
		self.assertTrue( (tagged == [0.0, 0.0, 2.0/3, 2.0/3, 1.0, 1.0]).all() )

	def test_tagRate(self):
		x = MemoryRecognition()
		x.newSystem(3, 0.2)
		x.system.setCenters([[0,1],[0, 1.25], [0, 0.25]])
		(swells, rate) = x.tagRate(0, 1.0, 0.2)
		self.assertTrue( np.allclose(rate, [0, (2.0/3)/0.2, 0, 0, (1-2/3)/0.2, 0]) )

	def test_tagCurve(self):
		x = MemoryRecognition()
		x.newSystem(3, 0.2)
		x.system.setCenters([[0,1],[0, 1.25], [0, 0.25]])
		(swells, curve) = x.tagCurvature(0, 1.0, 0.2)
		self.assertTrue( np.allclose(curve, [0, (2.0/3)/0.04, -(2.0/3)/0.04, (1/3)/0.04, -(1/3)/0.04, 0]) )


if __name__ == "__main__":
	unittest.main()