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

if __name__ == "__main__":
	unittest.main()