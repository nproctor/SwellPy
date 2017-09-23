import unittest
from .monodisperse import *

"""
Test cases for particle suspension classes

NOTE:
Three warning messages are expected to print. 
"""

class Test_Monodisperse(unittest.TestCase):


    def test_randomCentersShape(self):
        x = Monodisperse(10, 0.2)
        self.assertEqual(x.centers.shape, (10, 2))

    def test_randomCentersBounds(self):
        x = Monodisperse(100, 0.2)
        self.assertTrue( (x.centers > 0).all() )
        self.assertTrue( (x.centers < x.boxsize).all() )

    def test_setBadFormatCenters(self):
        x = Monodisperse(3, 0.2)
        before = x.centers
        with self.assertRaises(TypeError):
            x.set_centers([0,2,3,4,4,5,6]) # Attempt bad center reassignment (message will print)
        after = x.centers
        self.assertTrue( before is after )

    def test_setOutOfBoundsCenters(self):
        x = Monodisperse(2, 0.2)
        x.set_centers([[0,1],[-10,12]])
        self.assertTrue( (x.centers == [[0,1],[-10,12]]).all() ) # Out of bounds reassignment (message will print)

    def test_set_centers(self):
        x = Monodisperse(2, 0.2)
        new = [[0,1],[2.0, 1.89]]
        x.set_centers(new)
        self.assertTrue( (x.centers == new).all() )

    def test_reset(self):
        x = Monodisperse(10, 0.2, 10) # Feeding seed for randomization
        before = x.centers
        x.reset(10) # feeding same seed
        after = x.centers
        self.assertTrue( (before == after).all() ) 

    def test_directTag(self):
        x = Monodisperse(2, 0.2)
        x.set_centers([[1,1], [1,1.5]])
        tagged = x.tag(1.0)
        self.assertTrue( (tagged == [[0,1]]).all() )

    def test_directTag2(self):
        x = Monodisperse(4, 0.2)
        x.boxsize = 1
        x.set_centers([[0,0], [0,0.9], [0.9, 0], [0.9,0.9]])
        tagged = x.tag(0.1)
        self.assertTrue( len(tagged) == 4 )
        tagged = x.tag(np.sqrt(2)*0.1)
        self.assertTrue( len(tagged) == 6 )  

    def test_boundaryTag(self):
        x = Monodisperse(2, 0.2)
        x.set_centers([[0,0],[0, 2]])
        tagged = x.tag(1.0)
        self.assertTrue( (tagged == [[0,1]]).all() ) 

    def test_wrap(self):
        x = Monodisperse(2, 0.2)
        x.set_centers([[-1, 0], [0, 1]]) # Will print warning message
        x.wrap()
        self.assertTrue( (x.centers == [[x.boxsize-1, 0], [0,1]]).all() )

    def test_normalRepel(self):
        x = Monodisperse(2, 0.1)
        x.set_centers([[1,1],[1.5,1.5]])
        x.repel(np.array([[0,1]]), 1.0, 1.0)
        self.assertEqual(x.centers[0][0], x.centers[0][1])
        self.assertEqual(x.centers[1][0], x.centers[1][1])
        np.testing.assert_almost_equal(1-x.centers[0][0], x.centers[1][0]-1.5, decimal=12)
        np.testing.assert_almost_equal(1-x.centers[0][1], x.centers[1][1]-1.5, decimal=12) 
        self.assertTrue((x.centers[0][0] < 1) and (x.centers[0][1] < 1))
        self.assertTrue((x.centers[1][0] > 1.5) and (x.centers[1][1] > 1.5))

    def test_boundaryRepelVertical(self):
        x = Monodisperse(2, 1)
        x.boxsize=100
        x.set_centers([[0,0],[0, 99]])
        x.repel([[0,1]], 1.0, 1.0)
        self.assertEqual( x.centers[0][0], 0)
        self.assertEqual( x.centers[1][0], 0)
        np.testing.assert_almost_equal( x.centers[0][1], 99-x.centers[1][1], decimal=12)
        self.assertTrue( x.centers[0][1] < 100.0)

    def test_boundaryRepelHorizontal(self):
        x = Monodisperse(2, 0.1)
        x.set_centers([[0,0],[2, 0]])
        x.repel([[0,1]], 1.0, 1.0)
        self.assertEqual( x.centers[0][1], 0)
        self.assertEqual( x.centers[1][1], 0)
        np.testing.assert_almost_equal( x.centers[0][0], 2-x.centers[1][0], decimal=12)
        self.assertTrue(x.centers[0][0] < 1.0)

    def test_boundaryRepelDiagonal(self):
        x = Monodisperse(2, 0.1)
        x.set_centers([[0,0],[2, 2]])
        x.repel([[0,1]], 1.0, 1.0)
        np.testing.assert_almost_equal( x.centers[0][0], 2-x.centers[1][0], decimal=12)
        np.testing.assert_almost_equal( x.centers[0][1], 2-x.centers[1][1], decimal=12)

    def test_train(self):
        x = Monodisperse(2, 0.1)
        before = [[1, 0],[1.25, 0]]
        x.set_centers(before)
        x.train(1.0, 0.1)
        after = x.centers
        self.assertEqual( before[0][1], after[0][1] )
        self.assertEqual( before[1][1], after[1][1] ) 
        self.assertTrue( abs(after[1,0] - after[1][1]) > 1.0 )

    def test_tagCountAt(self):
        x = Monodisperse(2, 0.2)
        x.set_centers([[0,1],[0, 1.25]])
        self.assertEqual(x.tag_count_at(1.0), 1.0)
        self.assertEqual(x.tag_count_at(0.24), 0)

    def test_tagCount(self):
        x = Monodisperse(3, 0.2)
        x.set_centers([[0,1],[0, 1.25], [0, 0.25]])
        (swell, tagged) = x.tag_count(0, 1.0, 0.2)
        self.assertTrue( np.array_equal(tagged, [0.0, 0.0, 2.0/3, 2.0/3, 1.0, 1.0]))
        np.testing.assert_array_almost_equal(swell, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], decimal = 5 )


    def test_tagRate(self):
        x = Monodisperse(3, 0.2)
        x.set_centers([[0,1],[0, 1.25], [0, 0.25]])
        (swells, rate) = x.tag_rate(0, 1.0, 0.2)
        np.testing.assert_array_almost_equal(rate, [0, (2.0/3)/0.2, 0, 0, (1-2/3)/0.2, 0], decimal = 5 )

    def test_tagCurve(self):
        x = Monodisperse(3, 0.2)
        x.set_centers([[0,1],[0, 1.25], [0, 0.25]])
        (swells, curve) = x.tag_curve(0, 1.0, 0.2)
        np.testing.assert_array_almost_equal(curve, [0, (2.0/3)/0.04, -(2.0/3)/0.04, (1/3)/0.04, -(1/3)/0.04, 0], decimal = 5 )


if __name__ == "__main__":
    unittest.main()