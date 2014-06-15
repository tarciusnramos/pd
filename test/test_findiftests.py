import unittest
import numpy as np
from ..particles import *
from util import field_gradient, field_hessian

class PointDipoleFiniteFieldTests(unittest.TestCase):

    def setUp(self):
        self.particle = PointDipole()

    def test_finite_difference_energy(self):
#zero
        gradE = field_gradient(self.particle.total_energy)
        dipole = self.particle._p0 + \
           self.particle.induced_dipole_moment()

        np.testing.assert_almost_equal(-gradE, dipole)


    def test_finite_difference_permanent_dipole_energy(self):
        self.particle._p0 = random_vector()
        gradE = field_gradient(self.particle.permanent_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.permanent_dipole_moment())

    def test_finite_difference_alpha_induced_dipole_energy(self):
        self.particle._a0 = random_tensor()
        gradE = field_gradient(self.particle.alpha_induced_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.alpha_induced_dipole_moment())

    def test_finite_difference_induced_dipole_energy(self):
        self.particle._a0 = random_tensor()
        gradE = field_gradient(self.particle.alpha_induced_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.induced_dipole_moment())

    def test_finite_difference_total_dipole_energy(self):
        self.particle._p0 = random_vector()
        self.particle._a0 = random_tensor()
        gradE = field_gradient(self.particle.dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.dipole_moment())

    def test_finite_difference_permanent_dipole_moment(self):
        self.particle._p0 = np.random.random(3)
        gradE = field_gradient(self.particle.permanent_dipole_moment)
        np.testing.assert_almost_equal(-gradE, np.zeros((3, 3)))

    def test_finite_difference_alpha_induced_dipole_moment(self):
        self.particle._a0 = random_tensor()
        gradp = field_gradient(self.particle.alpha_induced_dipole_moment)
        np.testing.assert_almost_equal(gradp, self.particle._a0)

    def test_finite_difference_beta_induced_dipole_moment(self):
        self.particle._b0 = random_tensor2()
        self.particle.set_local_field(random_vector())
        gradp = field_gradient(self.particle.beta_induced_dipole_moment)
        ref = np.dot(self.particle._b0, self.particle.local_field())
        np.testing.assert_almost_equal(gradp, ref)

    def test_finite_difference_total_dipole_moment(self):
        self.particle._p0 = random_vector()
        self.particle._a0 = random_tensor()
        gradp = field_gradient(self.particle.dipole_moment)
        np.testing.assert_almost_equal(gradp, self.particle._a0)

    def test_finite_difference_hessian_dipole_moment(self):
        self.particle._p0 = random_vector()
        self.particle._a0 = random_tensor()
        self.particle._b0 = random_tensor2()
        hess_p = field_hessian(self.particle.dipole_moment)
        np.testing.assert_almost_equal(hess_p, self.particle._b0)

class PointDipoleListFiniteFieldTests(unittest.TestCase):
    
    def setUp(self):
        self.charges = PointDipoleList.from_string("""AU
2 0 0
1 0 0 0 1.0
2 0 0 1 1.0
"""
)

        self.h2o_dimer=PointDipoleList.from_string("""AU
2 1 1 0
1 0.00000  0.00000  0.48861 0.0 0.00000 -0.00000 -0.76539  6.61822
1 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
""")

        self.h2o_monomer=PointDipoleList.from_string("""AU
1 1 1 0
1 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
""")

        self.h2o_monomer_hyp=PointDipoleList()

        self.h2o_monomer_hyp.append(
            PointDipole(
                coordinates=(0, 0, 0.48861),
                charge=0,
                dipole=(0, 0, -0.81457755),
                alpha=(
                    (7.21103625278, 0, 0), 
                    (0, 3.03446384360, 0),
                    (0, 0, 5.22710462524)
                ),
                beta=(
                    (
                        (0, 0, -18.48299918),
                        (0, 0, 0),
                        (-18.48299918,0,0)
                    ),
                    (
                        (0, 0, 0),
                        (0, 0, -2.33649395),
                        (0, -2.33649395, 0)
                    ),
                    (
                        (-18.48299798, 0, 0),
                        (0, -2.33649400, 0),
                        (0, 0, -11.17349291)
                    )
                )
            )
        )

        self.h2o_dimer_hyp=PointDipoleList()

        self.h2o_dimer_hyp.append(
            PointDipole(
                coordinates=(0, 0, 0),
                charge=0,
                dipole=(0, 0, -0.81457755),
                alpha=(
                    (7.21103625278, 0, 0), 
                    (0, 3.03446384360, 0),
                    (0, 0, 5.22710462524)
                ),
                beta=(
                    (
                        (0, 0, -18.48299918),
                        (0, 0, 0),
                        (-18.48299918,0,0)
                    ),
                    (
                        (0, 0, 0),
                        (0, 0, -2.33649395),
                        (0, -2.33649395, 0)
                    ),
                    (
                        (-18.48299798, 0, 0),
                        (0, -2.33649400, 0),
                        (0, 0, -11.17349291)
                    )
                )
            )
        )

        self.h2o_dimer_hyp.append(
            PointDipole(
                coordinates=(0, 0, 10),
                charge=0,
                dipole=(0, 0, -0.81457755),
                alpha=(
                    (7.21103625278, 0, 0), 
                    (0, 3.03446384360, 0),
                    (0, 0, 5.22710462524)
                ),
                beta=(
                    (
                        (0, 0, -18.48299918),
                        (0, 0, 0),
                        (-18.48299918,0,0)
                    ),
                    (
                        (0, 0, 0),
                        (0, 0, -2.33649395),
                        (0, -2.33649395, 0)
                    ),
                    (
                        (-18.48299798, 0, 0),
                        (0, -2.33649400, 0),
                        (0, 0, -11.17349291)
                    )
                )
            )
        )

    def test_finite_difference_polarizable_monomer_z(self):
        alphas = self.h2o_monomer.solve_Applequist_equation()
        dp0_dF = self.h2o_monomer.field_gradient_of_method(self.h2o_monomer.induced_dipole_moment)
        self.assertAlmostEqual(dp0_dF[0, 2, 2], alphas[0][2, 2], places=3)

    def test_finite_difference_polarizable_dimer_z(self):
        alphas = self.h2o_dimer.solve_Applequist_equation()
        dp0_dF = self.h2o_dimer.field_gradient_of_method(self.h2o_dimer.induced_dipole_moment)
        self.assertAlmostEqual(dp0_dF[0, 2, 2], alphas[0][2, 2], places=3)

    def test_finite_difference_polarizable_dimer_x(self):
        alphas = self.h2o_dimer.solve_Applequist_equation()
        dp0_dF = self.h2o_dimer.field_gradient_of_method(self.h2o_dimer.induced_dipole_moment)
        self.assertAlmostEqual(dp0_dF[0, 0, 0], alphas[0][0, 0], places=3)

    def test_finite_difference_polarizable_dimer_y(self):
        alphas = self.h2o_dimer.solve_Applequist_equation()
        dp0_dF = self.h2o_dimer.field_gradient_of_method(self.h2o_dimer.induced_dipole_moment)
        self.assertAlmostEqual(dp0_dF[0, 1, 1], alphas[0][1, 1], places=3)

    def test_finite_difference_polarizable_dimer_z(self):
        alphas = self.h2o_dimer.solve_Applequist_equation()
        dp0_dF = self.h2o_dimer.field_gradient_of_method(self.h2o_dimer.induced_dipole_moment)
        self.assertAlmostEqual(dp0_dF[0, 2, 2], alphas[0][2, 2], places=3)

    def test__field_vs_external_field(self):
        pass
        
    def test_finite_difference_hyperpolarizable_monomer_z(self):
        print self.h2o_monomer_hyp[0]._a0
        print self.h2o_monomer_hyp[0]._b0
        alphas = self.h2o_monomer_hyp.solve_Applequist_equation()
        dp0_dF = self.h2o_monomer_hyp.field_gradient_of_method(self.h2o_monomer_hyp.induced_dipole_moment)
        self.assertAlmostEqual(dp0_dF[0, 2, 2], alphas[0][2, 2], places=3)

    def test_finite_difference_hyperpolarizable_dimer(self):
        dimer = self.h2o_dimer_hyp
        alphas = dimer.solve_Applequist_equation()
        dp_dF = dimer.field_gradient_of_method(dimer.induced_dipole_moment)
        np.testing.assert_almost_equal(dp_dF, alphas, decimal=3)

    def test_finite_difference_hessian_dipole_moment(self):
        monomer = PointDipoleList()
        monomer.append(PointDipole(
            dipole=random_vector(),
            alpha=random_tensor(),
            beta=random_tensor2()
            ))
        d2p_dF2 = monomer.field_hessian_of_method(monomer.induced_dipole_moment)
        betas = monomer.solve_second_Applequist_equation()
        np.testing.assert_almost_equal(d2p_dF2, betas, decimal=3)

    def test_second_finite_difference_hyperpolarizable_monomer(self):
        monomer = self.h2o_monomer_hyp
        betas = monomer.solve_second_Applequist_equation()
        d2p_dF2 = monomer.field_hessian_of_method(monomer.induced_dipole_moment)
        np.testing.assert_almost_equal(d2p_dF2, betas, decimal=3)

    def test_second_finite_difference_hyperpolarizable_dimer(self):
        dimer = self.h2o_dimer_hyp
        betas = dimer.solve_second_Applequist_equation()
        d2p_dF2 = dimer.field_hessian_of_method(dimer.induced_dipole_moment)
        np.testing.assert_allclose(d2p_dF2, betas, rtol=1e-3)

    def test_finite_difference_local_fields(self):
        molecule = self.h2o_dimer_hyp
        molecule.solve_scf_for_external(ORIGO)
        test_this =  molecule._dEi_dF_indirect()
        method = molecule.evaluate_field_at_atoms
        dEi_dF = molecule.field_gradient_of_method(method)
        np.testing.assert_almost_equal(test_this, dEi_dF, decimal=3)

def random_vector():
    return np.random.random(3)

def random_tensor():
    a = np.random.random((3, 3))
    a = 0.5*(a + a.T)
    return  a

def random_tensor2():
    b = np.random.random((3, 3, 3))
    b = b + b.transpose((1, 2, 0)) +  b.transpose((2, 0, 1)) +\
        b.transpose((1, 0, 2)) + b.transpose((2, 1, 0)) + b.transpose((0, 2, 1))
    return  b
