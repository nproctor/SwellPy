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



if __name__ == "__main__":
	unittest.main()