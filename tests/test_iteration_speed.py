import unittest
import numpy as np
from lightsem.experiments.iteration_speed import relative_comparison


class NoninferiorityTests(unittest.TestCase):
    def test_known_quality_degradations(self):
        baseline=np.arange(1,41,dtype=float)
        for lower in (True,False):
            sign=1 if lower else -1
            self.assertTrue(relative_comparison(baseline,baseline*(1+sign*.02),lower)['passed'])
            self.assertFalse(relative_comparison(baseline,baseline*(1+sign*.04),lower)['passed'])
            self.assertTrue(relative_comparison(baseline,baseline*(1-sign*.04),lower)['passed'])

    def test_uncertainty_is_not_a_pass(self):
        baseline=np.ones(40);candidate=baseline.copy();candidate[::2]+=.5;candidate[1::2]-=.5
        r=relative_comparison(baseline,candidate,True)
        self.assertEqual(r['status'],'inconclusive')
        self.assertGreater(r['ci95'][1],.03)

    def test_invalid_pairs_fail(self):
        for a,b in [([0],[1]),([1,2],[1]),([1],[np.nan]),([],[])]:
            with self.assertRaises(ValueError):relative_comparison(a,b,True)


if __name__=='__main__':unittest.main()
