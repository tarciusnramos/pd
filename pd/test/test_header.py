import unittest
from ..particles import *
from nose.plugins.attrib import attr




@attr(speed='fast')
class  TestHeader(unittest.TestCase):
    def test_header_number_of_atoms(self):
        header_dict = header_to_dict("2 0 0")
        self.assertEqual(header_dict["#atoms"], 2)

    def test_header_max_ang_mom(self):
        header_dict = header_to_dict("2 1 0")
        self.assertEqual(header_dict["max_angmom"], 1)

    def test_header_nopol_isotropic_polarizability_false(self):
        header_dict = header_to_dict("2 1")
        self.assertEqual(header_dict["iso_pol"], False)

    def test_header_0_isotropic_polarizability_false(self):
        header_dict = header_to_dict("2 1 0")
        self.assertEqual(header_dict["iso_pol"], False)

    def test_header_1_isotropic_polarizability_true(self):
        header_dict = header_to_dict("2 1 1")
        self.assertEqual(header_dict["iso_pol"], True)

    def test_header_2_isotropic_polarizability_false(self):
        header_dict = header_to_dict("2 1 2")
        self.assertEqual(header_dict["iso_pol"], False)

    def test_header_3_unvalid(self):
        self.assertRaises(TypeError, header_to_dict, "2 1 3")

    def test_header_nopol_upper_triangular_polarizability_false(self):
        header_dict = header_to_dict("2 1")
        self.assertEqual(header_dict["ut_pol"], False)

    def test_header_1_upper_triangular_polarizability_false(self):
        header_dict = header_to_dict("2 1 1")
        self.assertEqual(header_dict["ut_pol"], False)

    def test_header_2_upper_triangular_polarizability_true(self):
        header_dict = header_to_dict("2 1 2")
        self.assertEqual(header_dict["ut_pol"], True)

    def test_header_0_ut_hyperpolarizability_false(self):
        header_dict = header_to_dict("2 1 01 1")
        self.assertEqual(header_dict["ut_hyppol"], False)

    def test_header_1_ut_hyperpolarizability_False(self):
        header_dict = header_to_dict("2 1 11 1")
        self.assertEqual(header_dict["ut_hyppol"], False)

    def test_header_2_ut_hyperpolarizability_true(self):
        header_dict = header_to_dict("2 1 21 1")
        self.assertEqual(header_dict["ut_hyppol"], True)

    def test_line_to_dict_charges(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 0}
        pot_line = "1 0 0 0 1.5"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['charge'], 1.5)

    def test_line_to_dict_dipole(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 1}
        pot_line = "1 0 0 0 1.5 1 2 3"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['dipole'], [1.0, 2.0, 3.0])

    def test_line_to_dict_quadrupole(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 2}
        pot_line = "1 0 0 0 1.5 1 2 3 .1 .2 .3 .4 .5 .6"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['quadrupole'], [.1, .2, .3, .4, .5, .6])

    def test_line_to_dict_isopol(self):
        header_dict = {"iso_pol": True}
        pot_line = "1 0 0 0 0 7"
        line_dict = line_to_dict(header_dict, pot_line)
        print("line_dict", line_dict)
        self.assertEqual(line_dict['iso_alpha'], 7.0)

    def test_line_to_dict_ut_pol(self):
        header_dict = {"ut_pol": True}
        pot_line = "1 0 0 0 0 1 2 3 4 5 6"
        line_dict = line_to_dict(header_dict, pot_line)
        print("line_dict", line_dict)
        self.assertEqual(line_dict['ut_alpha'], range(1,7))

    def test_line_to_dict_ut_hyppol(self):
        header_dict = {"ut_hyppol": True}
        pot_line = "1 0 0 0 0  0 1 2 3 4 5 6 7 8 9"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['ut_beta'], range(10))

    def test_line_to_dict_ut_hyppol2(self):
        header_dict = {"ut_pol": True, "ut_hyppol": True}
        pot_line = "1 0 0 0  0  1 2 3 4 5 6   0 1 2 3 4 5 6 7 8 9"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['ut_beta'], range(10))

    def test_dict_isopol_to_PointDipole(self):
        pd = PointDipole(iso_alpha=7.0)
        np.testing.assert_equal(pd._a0, 7.0*I_3)

    def test_dict_utpol_to_PointDipole(self):
        pd = PointDipole(ut_alpha=(1, 2, 3, 4, 5, 6))
        np.testing.assert_equal(pd._a0, ((1, 2, 3), (2, 4, 5), (3, 5, 6)))

    def test_dict_ut_hyppol_to_PointDipole(self):
        pd = PointDipole(ut_beta=range(10))
        np.testing.assert_equal(
            pd._b0, (
                (
                    (0, 1, 2),
                    (1, 3, 4), 
                    (2, 4, 5),
                ),
                (
                    (1, 3, 4),
                    (3, 6, 7),
                    (4, 7, 8),
                ),
                (
                    (2, 4, 5),
                    (4, 7, 8),
                    (5, 8, 9)
                )
            )
        )
