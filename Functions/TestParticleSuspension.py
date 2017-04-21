import unittest
from ParticleSuspension import *

class TestParticleSuspension(unittest.TestCase):


    def test_randomCentersShape(self):
        x = ParticleSuspension(10, 0.2)
        self.assertEqual(x.centers.shape, (10, 2))

    def test_randomCentersBounds(self):
        x = ParticleSuspension(100, 0.2)
        self.assertTrue( (x.centers > 0).all() )
        self.assertTrue( (x.centers < x.boxsize).all() )

    def test_setBadFormatCenters(self):
        x = ParticleSuspension(3, 0.2)
        before = x.centers
        with self.assertRaises(TypeError):
            x.setCenters([0,2,3,4,4,5,6]) # Attempt bad center reassignment (message will print)
        after = x.centers
        self.assertTrue( before is after )

    def test_setOutOfBoundsCenters(self):
        x = ParticleSuspension(2, 0.2)
        x.setCenters([[0,1],[-10,12]])
        self.assertTrue( (x.centers == [[0,1],[-10,12]]).all() ) # Out of bounds reassignment (message will print)

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

    def test_directTag(self):
        x = ParticleSuspension(2, 0.2)
        x.setCenters([[1,1], [1,1.5]])
        tagged = x.tag(1.0)
        self.assertTrue( (tagged == [[0,1]]).all() )

    def test_directTag2(self):
        x = ParticleSuspension(4, 0.2)
        x.boxsize = 1
        x.setCenters([[0,0], [0,0.9], [0.9, 0], [0.9,0.9]])
        tagged = x.tag(0.1)
        self.assertTrue( len(tagged) == 4 )
        tagged = x.tag(np.sqrt(2)*0.1)
        self.assertTrue( len(tagged) == 6 )  

    def test_boundaryTag(self):
        x = ParticleSuspension(2, 0.2)
        x.setCenters([[0,0],[0, 2]])
        tagged = x.tag(1.0)
        self.assertTrue( (tagged == [[0,1]]).all() ) 

    def test_wrap(self):
        x = ParticleSuspension(2, 0.2)
        x.setCenters([[-1, 0], [0, 1]]) # Will print warning message
        x.wrap()
        self.assertTrue( (x.centers == [[x.boxsize-1, 0], [0,1]]).all() )

    def test_normalRepel(self):
        x = ParticleSuspension(2, 0.1)
        x.setCenters([[1,1],[1.5,1.5]])
        x.repel(np.array([[0,1]]), 1.0, 1.0)
        self.assertEqual(x.centers[0][0], x.centers[0][1])
        self.assertEqual(x.centers[1][0], x.centers[1][1])
        np.testing.assert_almost_equal(1-x.centers[0][0], x.centers[1][0]-1.5, decimal=12)
        np.testing.assert_almost_equal(1-x.centers[0][1], x.centers[1][1]-1.5, decimal=12) 
        self.assertTrue((x.centers[0][0] < 1) and (x.centers[0][1] < 1))
        self.assertTrue((x.centers[1][0] > 1.5) and (x.centers[1][1] > 1.5))

    def test_boundaryRepelVertical(self):
        x = ParticleSuspension(2, 1)
        x.boxsize=100
        x.setCenters([[0,0],[0, 99]])
        x.repel([[0,1]], 1.0, 1.0)
        self.assertEqual( x.centers[0][0], 0)
        self.assertEqual( x.centers[1][0], 0)
        np.testing.assert_almost_equal( x.centers[0][1], 99-x.centers[1][1], decimal=12)
        self.assertTrue( x.centers[0][1] < 100.0)

    def test_boundaryRepelHorizontal(self):
        x = ParticleSuspension(2, 0.1)
        x.setCenters([[0,0],[2, 0]])
        x.repel([[0,1]], 1.0, 1.0)
        self.assertEqual( x.centers[0][1], 0)
        self.assertEqual( x.centers[1][1], 0)
        np.testing.assert_almost_equal( x.centers[0][0], 2-x.centers[1][0], decimal=12)
        self.assertTrue(x.centers[0][0] < 1.0)

    def test_boundaryRepelDiagonal(self):
        x = ParticleSuspension(2, 0.1)
        x.setCenters([[0,0],[2, 2]])
        x.repel([[0,1]], 1.0, 1.0)
        np.testing.assert_almost_equal( x.centers[0][0], 2-x.centers[1][0], decimal=12)
        np.testing.assert_almost_equal( x.centers[0][1], 2-x.centers[1][1], decimal=12)

    def test_train(self):
        x = ParticleSuspension(2, 0.1)
        before = [[1, 0],[1.25, 0]]
        x.setCenters(before)
        x.train(1.0, 0.1)
        after = x.centers
        self.assertEqual( before[0][1], after[0][1] )
        self.assertEqual( before[1][1], after[1][1] ) 
        self.assertTrue( abs(after[1,0] - after[1][1]) > 1.0 )

    def test_tagFracAt(self):
        x = ParticleSuspension(2, 0.2)
        x.setCenters([[0,1],[0, 1.25]])
        self.assertEqual(x.tagFracAt(1.0), 1.0)
        self.assertEqual(x.tagFracAt(0.24), 0)

    def test_tagFrac(self):
        x = ParticleSuspension(3, 0.2)
        x.setCenters([[0,1],[0, 1.25], [0, 0.25]])
        (swell, tagged) = x.tagFrac(0, 1.0, 0.2)
        self.assertTrue( (tagged == [0.0, 0.0, 2.0/3, 2.0/3, 1.0, 1.0]).all() )
        np.testing.assert_array_almost_equal(swell, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], decimal = 5 )

    def test_fracTagNegative(self):
        x = ParticleSuspension(3, 0.2)
        x.setCenters([[0,1],[0, 1.25], [0, 0.25]])
        (swell, tagged) = x.tagFrac(0, -1.0, -0.2)
        self.assertTrue( (tagged == [0.0, 0.0, 2.0/3, 2.0/3, 1.0, 1.0]).all() )

    def test_tagRate(self):
        x = ParticleSuspension(3, 0.2)
        x.setCenters([[0,1],[0, 1.25], [0, 0.25]])
        (swells, rate) = x.tagRate(0, 1.0, 0.2)
        np.testing.assert_array_almost_equal(rate, [0, (2.0/3)/0.2, 0, 0, (1-2/3)/0.2, 0], decimal = 5 )

    def test_tagCurve(self):
        x = ParticleSuspension(3, 0.2)
        x.setCenters([[0,1],[0, 1.25], [0, 0.25]])
        (swells, curve) = x.tagCurve(0, 1.0, 0.2)
        np.testing.assert_array_almost_equal(curve, [0, (2.0/3)/0.04, -(2.0/3)/0.04, (1/3)/0.04, -(1/3)/0.04, 0], decimal = 5 )


if __name__ == "__main__":
    unittest.main()