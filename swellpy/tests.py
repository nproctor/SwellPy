import unittest
import numpy.testing as npt
from .monodisperse import *

"""
Test cases for particle suspension classes
"""

class Test_Monodisperse(unittest.TestCase):

    def test0_boxsize(self):
        two = Monodisperse(2)
        three = Monodisperse(3)
        many = Monodisperse(100)
        set_boxsize = Monodisperse(20, boxsize=1.0)
        self.assertTrue(two.boxsize < three.boxsize)
        self.assertTrue(set_boxsize.boxsize == 1.0)
        area_frac = (many.N * np.pi * (1/2)**2)/(many.boxsize**2)
        self.assertAlmostEqual(area_frac, 0.2)
        
    def test1_equivSwell(self):
        N = 100
        af = 1.0
        radius = 1.0
        boxsize = np.sqrt(N * np.pi * radius**2 / af)
        x = Monodisperse(N, boxsize)
        npt.assert_array_almost_equal(x.equiv_swell(af), [2.0])
    
    def test2_equivAreaFrac(self):
        N = 100
        x = Monodisperse(N)
        af = 1.0
        swell = x.equiv_swell(af)
        npt.assert_array_almost_equal([af], x.equiv_area_frac(swell))
    
    def test3_afSwellConversion(self):
        n = 5
        N = 10000
        sm = Monodisperse(n)
        l = Monodisperse(N)
        swells = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        npt.assert_array_almost_equal(swells, sm.equiv_swell(sm.equiv_area_frac(swells)))
        npt.assert_array_almost_equal(swells, l.equiv_swell(l.equiv_area_frac(swells)))


    def test3_randomCenters(self):
        N = 100
        x = Monodisperse(N)
        self.assertTrue( (x.centers > 0).all() )
        self.assertTrue( (x.centers < x.boxsize).all() )
        self.assertEqual(x.centers.shape, (N, 2))

    def test4_setBadFormatCenters(self):
        N = 3
        boxsize = 1.0
        x = Monodisperse(N, boxsize = boxsize)
        before = x.centers
        with self.assertRaises(TypeError):
            x.set_centers([0,2,3,4,4,5,6]) # Attempt bad center type reassignment
        after = x.centers
        self.assertTrue( before is after )

    def test5_setOutOfBoundsCenters(self):
        N = 2
        boxsize = 1.0
        x = Monodisperse(N, boxsize = boxsize)
        with self.assertRaises(RuntimeError):
            x.set_centers([[0, boxsize],[boxsize/2, -boxsize * 10]]) # Out of bounds reassignment

    def test6_setCenters(self):
        N = 2
        boxsize = 10
        x = Monodisperse(N, boxsize = boxsize)
        new = [[0,1],[2.0, 1.89]]
        x.set_centers(new)
        self.assertTrue( (x.centers == new).all() )

    def test7_reset(self):
        N = 10
        seed = 10
        x = Monodisperse(N, seed = seed)
        before = x.centers
        x.reset(seed) # feeding same seed
        after = x.centers
        self.assertTrue( (before == after).all() ) 

    def test8_directTag(self):
        N = 2
        boxsize = 1.0
        x = Monodisperse(N, boxsize = boxsize)
        af = x.equiv_area_frac(0.2)
        x.set_centers([[0.5, 0.5], [0.7, 0.5]])
        tagged = x.tag(af)
        self.assertTrue( (tagged == [[0,1]]).all() )

    def test9_directTagMany(self):
        N = 4
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        af = x.equiv_area_frac(0.1)
        x.set_centers([[0,0], [0,0.9], [0.9, 0], [0.9,0.9]])
        tagged = x.tag(af)
        self.assertTrue( len(tagged) == 4 )
        tagged = x.tag(np.sqrt(2)*0.1)
        self.assertTrue( len(tagged) == 6 )  

    def test10_boundaryTag(self):
        N = 2
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        af = x.equiv_area_frac(0.2)
        x.set_centers([[0, 0], [0, 0.9]])
        tagged = x.tag(af)
        self.assertTrue( (tagged == [[0,1]]).all() ) 

    def test11_wrap(self):
        N = 2
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        x.set_centers([[0, 0], [0, boxsize - 0.1]])
        x.centers[0][0] = -0.1
        x.centers[1][1] = boxsize + 0.1
        x.wrap()
        npt.assert_array_almost_equal(x.centers[0], [boxsize - 0.1, 0])
        npt.assert_array_almost_equal(x.centers[1], [0, 0.1])

    def test12_wrapLarge(self):
        N = 2
        boxsize = 0.8
        x = Monodisperse(N, boxsize = boxsize)
        x.centers[0][0] = -2.9
        x.centers[1][1] = 2.9
        x.wrap()
        self.assertAlmostEqual(x.centers[0][0], 0.3)
        self.assertAlmostEqual(x.centers[1][1], 0.5)

    def test13_normalRepel(self):
        N = 2
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        x.set_centers([[0.5, 0.5],[0.7, 0.7]])
        x.repel(np.array([[0,1]]), 0.3, 0.1)
        self.assertEqual(x.centers[0][0], x.centers[0][1])
        self.assertEqual(x.centers[1][0], x.centers[1][1])
        np.testing.assert_almost_equal(0.5 - x.centers[0][0], x.centers[1][0] - 0.7)
        np.testing.assert_almost_equal(0.5 - x.centers[0][1], x.centers[1][1] - 0.7) 

    def test14_boundaryRepelVertical(self):
        N = 2
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        af = x.equiv_area_frac(0.2)
        x.set_centers([[0, 0],[0, 0.9]])
        x.repel([[0,1]], af, 0.1)
        self.assertEqual( x.centers[0][0], 0)
        self.assertEqual( x.centers[1][0], 0)
        np.testing.assert_almost_equal( x.centers[0][1], 0.9 - x.centers[1][1])

    def test15_boundaryRepelHorizontal(self):
        N = 2
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        af = x.equiv_area_frac(0.2)
        x.set_centers([[0,0],[0.9, 0]])
        x.repel([[0,1]], af, 0.1)
        self.assertEqual( x.centers[0][1], 0)
        self.assertEqual( x.centers[1][1], 0)
        np.testing.assert_almost_equal( x.centers[0][0], 0.9 - x.centers[1][0])

    def test16_boundaryRepelDiagonal(self):
        N = 2
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        af = x.equiv_area_frac(0.2)
        x.set_centers([[0, 0], [0.9, 0.9]])
        x.repel([[0,1]], af, 0.1)
        self.assertAlmostEqual( x.centers[0][0], 0.9 - x.centers[1][0])
        self.assertAlmostEqual( x.centers[0][1], 0.9 - x.centers[1][1])

    def test17_train(self):
        N = 100
        af = 0.7
        x = Monodisperse(N)
        x.train(af, 0.001)
        npt.assert_array_almost_equal(x.tag_count(af), [0])

    def test18_tagCount(self):
        N = 2
        boxsize = 1
        x = Monodisperse(N, boxsize = boxsize)
        x.set_centers([[0, 0.2],[0, 0.4]])
        af0 = x.equiv_area_frac(0.1)
        af1 = x.equiv_area_frac(0.2)
        npt.assert_array_almost_equal(x.tag_count([af0, af1]), [0, 1])

    def test19_tagCount(self):
        N = 3
        boxsize = 2
        x = Monodisperse(N, boxsize = boxsize)
        x.set_centers([[0, 1.0],[0, 1.25], [0, 0.25]])
        tagged = x.tag_count([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        npt.assert_array_almost_equal(tagged, [0.0, 2.0/3, 2.0/3, 2.0/3, 1.0, 1.0])

    def test20_tagRate(self):
        N = 3
        boxsize = 3
        x = Monodisperse(N, boxsize = boxsize)
        x.set_centers([[0, 0.4], [0, 1.0], [0, 1.2]])
        af = x.equiv_area_frac([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        rate = x.tag_rate(af)
        expected = np.array([2.0/3, 2.0/3, (1-2.0/3), (1-2.0/3), 0.0, 0.0])
        npt.assert_array_almost_equal(rate, expected)

    def test21_tagCurve(self):
        N = 3
        boxsize = 3
        x = Monodisperse(N, boxsize = boxsize)
        x.set_centers([[0, 0.4], [0, 1.0], [0, 1.2]])
        af = x.equiv_area_frac([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        curve = x.tag_curve(af)
        expected = np.array([2.0/3, (1-2.0/3) - 2.0/3, (1-2.0/3) - 2.0/3, -(1-2.0/3), -(1-2.0/3), 0.0])
        np.testing.assert_array_almost_equal(curve, expected)
    
    def test22_detection(self):
        N = 2000
        x = Monodisperse(N)
        area_frac = np.arange(0.2, 0.7, 0.1)
        for af in area_frac:
            x.train(af, 0.001)
            a = x.detect_memory(0.1, 1.0, 0.1)
            print(af, end=" ")
            print(a)
            self.assertEqual(a.size, 1)
            npt.assert_almost_equal(af, a[0])

if __name__ == "__main__":
    unittest.main()