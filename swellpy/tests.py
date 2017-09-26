import unittest
import numpy.testing as npt
from .monodisperse import *

"""
Test cases for particle suspension classes

NOTE:
Three warning messages are expected to print. 
"""

class Test_Monodisperse(unittest.TestCase):

    def test0_boxsize(self):
        two = Monodisperse(2)
        three = Monodisperse(3)
        many = Monodisperse(100)
        set_boxsize = Monodisperse(20, boxsize=1.0)
        self.assertTrue(two.boxsize < three.boxsize)
        self.assertTrue(set_boxsize.boxsize == 1.0)
        self.assertAlmostEqual( (many.N * np.pi * (1/2)**2)/(many.boxsize**2), 0.2)
        
    def test1_afToSwell(self):
        N = 100
        af = 1.0
        radius = 1.0
        boxsize = np.sqrt(N * np.pi * radius**2 / af)
        x = Monodisperse(N, boxsize)
        self.assertEqual(x.af_to_swell(af), 2.0)
        

    def test2_randomCentersShape(self):
        x = Monodisperse(10)
        self.assertEqual(x.centers.shape, (10, 2))

    def test3_randomCentersBounds(self):
        x = Monodisperse(100)
        self.assertTrue( (x.centers > 0).all() )
        self.assertTrue( (x.centers < x.boxsize).all() )

    def test4_setBadFormatCenters(self):
        x = Monodisperse(3, boxsize = 1.0)
        before = x.centers
        with self.assertRaises(TypeError):
            x.set_centers([0,2,3,4,4,5,6]) # Attempt bad center reassignment (message will print)
        after = x.centers
        self.assertTrue( before is after )

    def test5_setOutOfBoundsCenters(self):
        x = Monodisperse(2)
        x.set_centers([[0,1],[-10,12]])
        self.assertTrue( (x.centers == [[0,1],[-10,12]]).all() ) # Out of bounds reassignment (message will print)

    def test6_setCenters(self):
        x = Monodisperse(2)
        new = [[0,1],[2.0, 1.89]]
        x.set_centers(new)
        self.assertTrue( (x.centers == new).all() )

    def test7_reset(self):
        x = Monodisperse(10, seed = 10)
        before = x.centers
        x.reset(10) # feeding same seed
        after = x.centers
        self.assertTrue( (before == after).all() ) 

    def test8_directTag(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[0.5, 0.5], [0.7, 0.5]])
        tagged = x.tag(0.2)
        self.assertTrue( (tagged == [[0,1]]).all() )

    def test9_directTag2(self):
        x = Monodisperse(4, boxsize = 1.0)
        x.set_centers([[0,0], [0,0.9], [0.9, 0], [0.9,0.9]])
        tagged = x.tag(0.1)
        self.assertTrue( len(tagged) == 4 )
        tagged = x.tag(np.sqrt(2)*0.1)
        self.assertTrue( len(tagged) == 6 )  

    def test10_boundaryTag(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[0, 0], [0, 0.9]])
        tagged = x.tag(0.2)
        self.assertTrue( (tagged == [[0,1]]).all() ) 

    def test11_wrap(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[-0.1, 0], [0, x.boxsize + 0.1]]) # Will print warning message
        x.wrap()
        npt.assert_array_almost_equal(x.centers[0], [x.boxsize - 0.1, 0])
        npt.assert_array_almost_equal(x.centers[1], [0, 0.1])

    def test12_normalRepel(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[0.5, 0.5],[0.7, 0.7]])
        x.repel(np.array([[0,1]]), 0.3, 0.1)
        self.assertEqual(x.centers[0][0], x.centers[0][1])
        self.assertEqual(x.centers[1][0], x.centers[1][1])
        np.testing.assert_almost_equal(0.5 - x.centers[0][0], x.centers[1][0] - 0.7)
        np.testing.assert_almost_equal(0.5 - x.centers[0][1], x.centers[1][1] - 0.7) 

    def test13_boundaryRepelVertical(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[0, 0],[0, 0.9]])
        x.repel([[0,1]], 0.2, 0.1)
        self.assertEqual( x.centers[0][0], 0)
        self.assertEqual( x.centers[1][0], 0)
        np.testing.assert_almost_equal( x.centers[0][1], 0.9 - x.centers[1][1])

    def test14_boundaryRepelHorizontal(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[0,0],[0.9, 0]])
        x.repel([[0,1]], 0.2, 0.1)
        self.assertEqual( x.centers[0][1], 0)
        self.assertEqual( x.centers[1][1], 0)
        np.testing.assert_almost_equal( x.centers[0][0], 0.9 - x.centers[1][0])

    def test15_boundaryRepelDiagonal(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[0, 0], [0.9, 0.9]])
        x.repel([[0,1]], 0.2, 0.1)
        np.testing.assert_almost_equal( x.centers[0][0], 0.9 - x.centers[1][0])
        np.testing.assert_almost_equal( x.centers[0][1], 0.9 - x.centers[1][1])

    def test16_train(self):
        x = Monodisperse(100)
        af = 0.6
        swell = x.af_to_swell(af)
        x.train(af, swell/10)
        self.assertEqual( x.tag_count_at(swell), 0 )

    def test17_tagCountAt(self):
        x = Monodisperse(2, boxsize = 1.0)
        x.set_centers([[0, 0.2],[0, 0.4]])
        self.assertEqual(x.tag_count_at(0.2), 1)
        self.assertEqual(x.tag_count_at(0.1), 0)

    def test18_tagCount(self):
        x = Monodisperse(3, boxsize = 10.0)
        x.set_centers([[0, 1.0],[0, 1.25], [0, 0.25]])
        (swell, tagged) = x.tag_count(0, 1.0, 0.2)
        npt.assert_array_almost_equal(tagged, [0.0, 0.0, 2.0/3, 2.0/3, 1.0, 1.0])
        npt.assert_array_almost_equal(swell, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


    def test19_tagRate(self):
        x = Monodisperse(3)
        x.set_centers([[0, 1.0],[0, 1.25], [0, 0.25]])
        (swells, rate) = x.tag_rate(0, 1.0, 0.2)
        npt.assert_array_almost_equal(rate, [0, (2.0/3)/0.2, 0, 0, (1-2/3)/0.2, 0])
        npt.assert_array_almost_equal(swells, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test20_tagCurve(self):
        x = Monodisperse(3)
        x.set_centers([[0,1],[0, 1.25], [0, 0.25]])
        (swells, curve) = x.tag_curve(0, 1.0, 0.2)
        np.testing.assert_array_almost_equal(curve, [0, (2.0/3)/0.04, -(2.0/3)/0.04, (1/3)/0.04, -(1/3)/0.04, 0], decimal = 5 )

if __name__ == "__main__":
    unittest.main()