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

	def test_fracTagFor(self):
		x = MemoryRecognition()
		x.newSystem(3, 0.2)
		x.system.setCenters([[0,1],[0, 1.25], [0, 0.25]])
		self.assertTrue( (x.fracTagFor(0, 1.0, 0.2) == [0, 0, 2.0/3, 2.0/3, 1.0, 1.0]).all() )

if __name__ == "__main__":
	unittest.main()