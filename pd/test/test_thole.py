import unittest
import numpy as np
from ..thole import TholeList
from data_thole import *
from nose.plugins.attrib import attr


DECIMALS=1

@attr(speed='fast')
class TholeTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_H2_iso(self):
        h2 = TholeList.from_string(H2["POTFILE"])
        print(h2)
        print('additive:', h2[0]._a0.trace()/3 * 2)
        print('ref:', H2["ALPHA_ISO"])
        print('got alpha iso:', h2.alpha_iso())
        self.assertAlmostEqual( h2.alpha_iso(), H2["ALPHA_ISO"], places=DECIMALS )

    def test_N2_iso(self):
        n2 = TholeList.from_string(N2["POTFILE"])
        print(n2)
        print('additive:', n2[0]._a0.trace()/3 * 2)
        print('ref:', N2["ALPHA_ISO"])
        print('got alpha iso:', n2.alpha_iso())
        self.assertAlmostEqual( n2.alpha_iso(), N2["ALPHA_ISO"], places=DECIMALS )
