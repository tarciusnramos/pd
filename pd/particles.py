#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module defining polarizable point dipole class""" 

import numpy as np 
from numpy.linalg import norm 
from numpy import outer, dot, array, zeros
import pd.ut as ut
from pd.constants import I_3, ZERO_VECTOR, ZERO_TENSOR, ZERO_RANK_3_TENSOR
import pd.cell as cell
import pd.optimized_func as optimized_func

class PointDipoleList(list):
    """A list class of ``PointDipole`` objects"""

    def __init__(self, pf=None):
        """Class constructor 
        pf: potential file object (or iterator)
        """
        a0 = 0.52917721092
        self._Cell = None
        if pf is not None:
            units = next(pf)
            self.header_dict = header_to_dict(next(pf))
            for i, line in enumerate(pf):
                if i == self.header_dict["#atoms"]: break
                line_dict = line_to_dict(self.header_dict, line)
                if line_dict:
                    self.append( PointDipole(**line_dict ) )
            if units == 'AA':
                for p in self:
                    p._r /= a0

    @classmethod
    def from_string(cls, potential):
        """Used to build the ``PointDipoleList`` object when the
           potential file is given as a triple quoted string with newlines
        """
        return cls(iter(potential.split("\n")))

    @classmethod
    def cell_from_string(cls, potential, co = 25.0 ):
        """Used to build the ``PointDipoleList`` object when the
           potential file is given as a triple quoted string with newlines

           For very large clusters usage of co will create a 3d cell structure with particles within cutoff = co interacting via the dipole coupling tensor, default is 25 {\AA}
        """
        co = float( co )
        pdl = cls(iter(potential.split("\n")))
        pdl._Cell = cell.Cell.from_PointDipoleList( pdl, co = co )
        return pdl
        
    def __str__(self):
        """String representation of class - delgated to list members"""
        return "\n\n".join([str(p) for p in self])

    def append(self, arg):
        """Overriding superclass list append: check if arg is PointDipole"""
        if not isinstance(arg,  PointDipole):
            print("PointDipoleList.append called with object of type", type(arg))
            raise TypeError
        super(PointDipoleList, self).append(arg)

    def total_charge(self):
        """Returns sum of charges of particles in the list"""
        return sum([p.charge() for p in self])

    def set_groups(self, groups):
        for p, group in zip(self, groups):
            p.group = group

    def groups(self):
        return tuple([p.group for p in self])

    def set_charges(self, charges):
        for p, q in zip(self, charges):
            p.set_charge(q)

    def set_dipoles(self, dipoles):
        for p, p0 in zip(self, dipoles):
            p.set_dipole(p0)

    def total_static_dipole_moment(self):
       return sum([p.permanent_dipole_moment() for p in self])

    def total_induced_dipole_moment(self):
       return sum([p.induced_dipole_moment() for p in self])

    def total_dipole_moment(self, cython = True, coc = [0, 0, 0] ):
        self.solve_scf( cython = cython )
        coc = np.array( coc )
        return sum([ (p.dipole_moment() + (p._r - coc ) * p._q) for p in self] )
        
    def dipole_coupling_tensor(self, cython = True, num_threads = 1):
        """Calculates the dipole coupling, tensor, describing the
        electric field strength at a given particle due to
        another electric dipole:

        .. math::
            \mathbf{T}_{ij} = (3\mathbf{r}_{ij}\mathbf{r}_{ij}-r_{ij}^2\mathbf{1})/r_{ij}^5

        """

        n = len(self)
        if cython:
            #import optimized_func
            _T = optimized_func.dipole_coupling_tensor_pointdipole_cython( 
                    particles = array([p.group for p in self], dtype = 'long') ,
                    _r = array([p._r for p in self], dtype = float),
                    num_threads = num_threads  )
        else:
            _T = zeros((n, 3, n,  3))
            for i in range(n):
                ri = self[i]._r
                for j in range(i):
                    if self[i].in_group_of( self[j] ):
                        continue
                    rj = self[j]._r
                    rij = ri - rj
                    rij2 = dot(rij, rij)
                    Tij = (3*outer(rij, rij) - rij2*I_3)/rij2**2.5
                    _T[i, :, j, :] = Tij
                    _T[j, :, i, :] = Tij
        return _T


    def alpha_iso(self):
        r"""Return the isotropic polarizability

        .. math::
            \alpha_{iso} = \sum_k \alpha_{kk}/3
        """

        return np.trace(self.alpha())/3

    def alpha(self, cython = True, threshold = 1e-8, num_threads = 1):
        try:
            dpdF = self.solve_Applequist_equation( 
                    threshold = threshold,
                    cython=cython,
                    num_threads = num_threads,
                    )
        except SCFNotConverged:
            return np.zeros((3, 3))
        return dpdF.sum(axis=0)

    def beta(self, cython = True, threshold = 1e-8,
            num_threads = 1):
        try:
            d2p_dF2 = self.solve_second_Applequist_equation( cython = cython,
                    threshold = threshold,
                    num_threads = num_threads )
        except SCFNotConverged:
            return np.zeros((3, 3, 3))
        return d2p_dF2.sum(axis=0)

    def solve_Applequist_equation(self, cython = True, threshold = 1e-8,
            num_threads = 1):
        # Solve the linear response equaitons
        n = len(self)
        try:
            self.solve_scf_for_external(ZERO_VECTOR,
                    cython = cython,
                    threshold = threshold,
                    num_threads = num_threads )
        except SCFNotConverged as e:
            print("SCF Not converged: residual=%f, threshold=%f"% (
                float(e.residual), float(e.threshold)
                ))
            raise SystemExit
            
        dE = self.form_Applequist_rhs()
        L = self.form_Applequist_coefficient_matrix( cython = cython,
                num_threads = num_threads)
        dpdE = np.linalg.solve(L, dE).reshape((n, 3, 3))
        return dpdE

    def solve_second_Applequist_equation(self, cython = True, threshold = 1e-8,
            num_threads = 1):
        # Solve the quadratic response equaitons
        n = len(self)
        self.solve_scf_for_external(ZERO_VECTOR,
                cython = cython, 
                threshold = threshold,
                num_threads = num_threads )
        dF2 = self.form_second_Applequist_rhs( cython = cython,
                threshold = threshold,
                num_threads = num_threads
                )
        L = self.form_Applequist_coefficient_matrix( cython = cython,
                num_threads = num_threads,
                )
        d2p_dF2 = np.linalg.solve(L, dF2).reshape((n, 3, 3, 3))
        return d2p_dF2

    def form_Applequist_rhs(self):
        n = len(self)
        alphas = [pd.a for pd in self]
        dE = array(alphas).reshape((n*3, 3))
        return dE

    def form_second_Applequist_rhs(self, cython = True,
            threshold = 1e-8,
            num_threads = 1):
        n = len(self)
        betas = [pd._b0 for pd in self]  #(n, 3, 3, 3)
        C = self._dEi_dF( cython = cython,
                threshold = threshold,
                num_threads = num_threads
                )                 #(n, 3; 3)

        dF2 = [np.einsum('ijk,jl,km', b, c, c) for b, c in zip(betas, C)]   # b( i, j, k) c(k; l) -> b(i, j, l) 
        return array(dF2).reshape((n*3, 9))

    def form_Applequist_coefficient_matrix(self, cython = True,
            num_threads = 1 ):
        n = len(self)

        aT = self.dipole_coupling_tensor( cython = cython,
                num_threads = num_threads
                ).reshape((n, 3, 3*n))

        # evaluate alphai*Tij
        alphas = [pd.a for pd in self]
        for i, a in enumerate(alphas):
            aT[i, :, :] = dot(a, aT[i, :, :])
        #matrix (1 - alpha*T)
        L = np.identity(3*n) - aT.reshape((3*n, 3*n))
        return L

    def solve_scf(self, E = ZERO_VECTOR, max_it=100, cython = True, threshold=1e-6,
            num_threads = 1):
        self.solve_scf_for_external( E = E, max_it = max_it,
                cython = cython,
                threshold = threshold,
                num_threads = num_threads, )

    def solve_scf_for_external(self, E, max_it=100, cython = True,
            threshold=1e-8,
            num_threads = 1 ):

        E_p0 = np.zeros((len(self), 3))
        if cython:
            #import pyximport
            #pyximport.install()
            #import optimized_func 

            E_at_p, i, residual = optimized_func.solve_scf_for_external_pointdipole_cython(
                    particles = array([p.group for p in self]) ,
                    E = array(E),
                    _r = array([p._r for p in self], dtype = float ),
                    _q = array([p._q for p in self], dtype = float ),
                    _p0 = array([p._p0 for p in self], dtype = float ),
                    _a0 = array([p._a0 for p in self], dtype = float ),
                    _b0 = array([p._b0 for p in self], dtype = float ),
                    _field = array([p._field for p in self], dtype = float ),
                    max_it = max_it ,
                    threshold = threshold,
                    num_threads = num_threads )
            for p, Ep in zip(self, E_at_p):
                p.set_local_field( Ep )
            return i, residual
        else:
            for i in range(max_it):
                E_at_p = self.evaluate_field_at_atoms(
                        external=E
                        )
                #print i, E_at_p
                for p, Ep in zip(self, E_at_p):
                    p.set_local_field(Ep)
                residual = norm(E_p0 - E_at_p)
                #print residual, threshold
                if residual < threshold:
                    return i, residual
                E_p0[:, :] = E_at_p

        raise SCFNotConverged(residual, threshold)

    def evaluate_field_at_atoms(self, external=None):

        if self._Cell is not None:
            E_at_p =  [
                array(
                    [o.field_at(p._r) for o in self._Cell.get_closest(p) if not o.in_group_of(p)]
                    ).sum(axis=0) 
                for p in self
                ]
        else:
            E_at_p =  [
                array(
                    [o.field_at(p._r) for o in self if not o.in_group_of(p)]
                    ).sum(axis=0) 
                for p in self
                ]

        if external is not None:
            E_at_p = [external + p for p in E_at_p]

        return array(E_at_p)

    def evaluate_potential_at_atoms(self, external=None):
        V_at_p =  [
            array(
                [o.potential_at(p._r) for o in self if not o.in_group_of(p)]
                ).sum(axis=0)
            for p in self
            ]
        if external is not None:
            V_at_p = [external + p for p in V_at_p]

        return V_at_p


    def _dEi_dF(self, cython = False, threshold = 1e-8, num_threads = 1 ):
        """Represents change of local field due to change in external"""
        n = len(self)
        TR = self._dEi_dF_indirect( cython = cython, 
                threshold = threshold,
                num_threads = num_threads
                )
        return np.array([I_3 + tr for tr in TR])
            
    def _dEi_dF_indirect(self, cython = False, num_threads = 1, threshold = 1e-8 ):
        """Change in local field due to change from other dipoles"""
        n = len(self)
        T = self.dipole_coupling_tensor( cython = cython, 
                num_threads = num_threads
                ).reshape(n, 3, n*3)
        R = self.solve_Applequist_equation( cython = cython,
                threshold = threshold,
                num_threads = num_threads
                ).reshape(n*3, 3)
        TR = dot(T, R)
        return TR

    def alphas_as_matrix(self):
        n = len(self)
        M = np.zeros((n, 3, n, 3))
        for i, p in enumerate(self):
            M[i, :, i, :] = p.a[:, :]
        return M.reshape((n*3, n*3))

    def update_local_fields(self):
        _potentials = self.evaluate_potential_at_atoms()
        _fields = self.evaluate_field_at_atoms()
        for p, E, V in zip(self, _fields, _potentials):
            p.set_local_field(E)
            p.set_local_potential(V)

    def total_energy(self):
        """Energy of induced/static dipoles in local field"""
        energy = 0
        for p in self:
            energy += p.total_energy()
        energy *= 0.5
        return energy

    def induced_dipole_moment(self):
        return np.array([p.induced_dipole_moment() for p in self])

    def field_gradient_of_method(self, method):
        from pd.test.util import ex, ey, ez, EPSILON

        fx = self.finite_difference(method, ex)
        fy = self.finite_difference(method, ey)
        fz = self.finite_difference(method, ez)

        #transpose to put field index last
        inrank = len(fx.shape)
        cycle = tuple(range(1, inrank+1)) + (0,)
        return np.array((fx, fy, fz)).transpose(cycle)

    def finite_difference(self, method, dF):
        self.clear_fields()
        self.solve_scf_for_external(dF)
        dm = method()
        self.clear_fields()
        self.solve_scf_for_external(-dF)
        dm -= method()
        dm /= 2*norm(dF)
        return dm
        
    def field_hessian_of_method(self, method):
        from pd.test.util import ex, ey, ez, EPSILON

        fxx = self.finite_difference2(method, ex, ex)
        fxy = self.finite_difference2(method, ex, ey)
        fxz = self.finite_difference2(method, ex, ez)
        fyy = self.finite_difference2(method, ey, ey)
        fyz = self.finite_difference2(method, ey, ez)
        fzz = self.finite_difference2(method, ez, ez)

        inrank = len(fxx.shape)
        cycle_twice_left = tuple(range(2, inrank+2)) + (0, 1)
        return np.array(
            ((fxx, fxy, fxz), 
             (fxy, fyy, fyz),
             (fxz, fyz, fzz))
            ).transpose(cycle_twice_left)

    def finite_difference2(self, method, dF1, dF2):
        self.clear_fields()
        self.solve_scf_for_external(dF1+dF2)
        d2m = method()
        self.clear_fields()
        self.solve_scf_for_external(dF1-dF2)
        d2m -= method()
        self.clear_fields()
        self.solve_scf_for_external(-dF1+dF2)
        d2m -= method()
        self.clear_fields()
        self.solve_scf_for_external(-dF1-dF2)
        d2m += method()
        d2m /= 4*norm(dF1)*norm(dF2)
        return d2m


    def clear_fields(self):
        for p in self:
            p.set_local_field(ZERO_VECTOR)
            p.set_local_potential(0)

class PointDipole(object):
    r""" 
    A hyperpolarizable dipole object with charge :math:`q` and dipole moment

    .. math::
        \bar{p} = \bar{p}^0 + \bar{\bar\alpha}\cdot\bar{E} + \frac12 \bar{\bar\beta}:\bar{E}\bar{E}
            
    """
    def __init__(self, *args, **kwargs):
        """
        fixed quantities: 
           _r: coordinates
           _q: charge
           _p0: permanent dipole
            _a0: polarizability tensor
            _b0: hyperpolarizability tensor

        variable:
            _field
            _potential
        
        derived quantities 
        
           p:  total dipole moment
           a:  effective polarizability
        
        """

        self.group = kwargs.get('group', 1)
        self._r = array(kwargs.get('coordinates', np.zeros(3)))

        if 'charge' in kwargs:
            self._q = float(kwargs['charge'])
        else:
            self._q = 0.0

        if "dipole" in kwargs:
            self._p0 = array(kwargs["dipole"])
        else:
            self._p0 = ZERO_VECTOR

        if "iso_alpha" in kwargs:
            self._a0 = float(kwargs["iso_alpha"])*I_3
        elif "ut_alpha" in kwargs:
            upper_triangular_pol = array(kwargs["ut_alpha"])
            assert upper_triangular_pol.shape == (6,)
            self._a0 = np.zeros((3,3))
            for ij, (i, j) in enumerate(ut.upper_triangular(2)):
                aij = upper_triangular_pol[ij]
                self._a0[i, j] = aij
                self._a0[j, i] = aij
        elif "alpha" in kwargs:
            self._a0 = array(kwargs["alpha"])
        else:
            self._a0 = ZERO_TENSOR
            

        if "ut_beta" in kwargs:
            upper_triangular_hyppol = array(kwargs["ut_beta"])
            assert upper_triangular_hyppol.shape == (10,)
            self._b0 = np.zeros((3,3,3))
            for ijk, (i, j, k) in enumerate(ut.upper_triangular(3)):
                bijk = upper_triangular_hyppol[ijk]
                self._b0[i, j, k] = bijk
                self._b0[k, i, j] = bijk
                self._b0[j, k, i] = bijk
                self._b0[i, k, j] = bijk
                self._b0[j, i, k] = bijk
                self._b0[k, j, i] = bijk
        elif "beta" in kwargs:
            self._b0 = np.array(kwargs["beta"])
        else:
            self._b0 = ZERO_RANK_3_TENSOR

        self.args = args

        self.fmt = kwargs.get('fmt', "%10.5f")
        self.set_local_field(kwargs.get('local_field', np.zeros(3)))
        self._potential = kwargs.get('local_potential', 0)

    def in_group_of(self, other):
        return self.group == other.group

    @property
    def p(self):
        return self.dipole_moment()

    @property
    def a(self):
        return self._a0 + self.induced_polarizability()

    def coordinates(self):
        r"""
        get Cartesian coordinates of particle

        .. math::
            \bar{r} = (x, y, z)

        :returns: ``np.array((x, y, z))``
        """
        return self._r

    def set_coordinates(self, r):
        r"""Set coordinates

        Arguments:

        :param r: coordinates :math:`(x, y, z)`
        :type r: `arraylike`

        :returns: ``None``
        """
        
        self._r = np.array(r)

    def local_field(self):
        return self._field

    def set_local_field(self, args):
        PointDipole._assert_vector_like(args)
        self._field = np.array(args, dtype=float)

    @staticmethod
    def _assert_vector_like(args):
        if len(args) != 3: raise ValueError

    def set_local_potential(self, args):
        self._potential = float(args)

    def charge(self):
        """get particle charge
        
        :returns: charge :math:`q`
        :rtype: `float`
        """

        return self._q

    def set_charge(self, q):
        """set particle charge

        :param q: charge
        :type q: float
        """

        self._q = float(q)

    def set_dipole(self, p0):
        self._p0 = np.array(p0)


    def dipole_moment(self):
        return self.permanent_dipole_moment() + self.induced_dipole_moment( )

    def permanent_dipole_moment(self):
        return self._p0

    def induced_dipole_moment(self):
        return self.alpha_induced_dipole_moment() + self.beta_induced_dipole_moment()

    def alpha_induced_dipole_moment(self):
        return dot(self._a0, self.local_field()) 

    def beta_induced_dipole_moment(self):
        return 0.5*dot(dot(self._b0, self.local_field()), self.local_field())

    def induced_polarizability(self):
        return dot(self._b0, self.local_field())
            

    def __str__(self):
        """The output simulate the line of a potential input file"""
        #for isotropic alpha
        value_line = list(self._r) + [self._q]
        if self._p0 is not None:
            value_line += list(self._p0)
        if self._a0 is not None:
            value_line +=  [self._a0[0, 0]]
        return "%d" % self.group + self.fmt*len(value_line) % tuple(value_line)

    def charge_energy(self):
        r"""Electrostatic energy of charge in local potential

        :returns: energy :math:`E=qV(\bar{r})`
        :rtype: float
        """

        return self._q*self._potential

    def dipole_energy(self):
        """Return total dipole energy in local field

        :returns: :math:`E_{dip} = E_{perm} + E_{ind}`
        """

        return self.permanent_dipole_energy() + \
            self.induced_dipole_energy()

    def permanent_dipole_energy(self):
        r"""Returns permanent dipole energy in local field

        :returns: :math:`E_{perm} = -\bar{p}\cdot\bar{E}(\bar{r})`
        :rtype: float
        """
        return -dot(self._p0, self.local_field())

    def induced_dipole_energy(self):
        r"""Total induced dipole energy
        
        :returns: :math:`E_\alpha + E_\beta`
        :rtype: float
        """

        return self.alpha_induced_dipole_energy() + \
               self.beta_induced_dipole_energy()

    def alpha_induced_dipole_energy(self):
        r""":math:`\alpha`-induced dipole energy
        
        :returns: :math:`E_\alpha = -\frac 12 \bar{E}\cdot\bar{\bar\alpha}\cdot\bar{E}`
        :rtype: float
        """
        return -0.5*dot(self.local_field(), dot(self._a0, self.local_field()))

    def beta_induced_dipole_energy(self):
        r""":math:`\beta`-induced dipole energy

        :returns: :math:`E_\beta = -\frac16 \bar{E}\cdot(\bar{\bar{\bar\beta}}\cdot\bar{E})\cdot\bar{E}
        """
        e_field = self.local_field()
        return -dot(e_field, dot(dot(self._b0, e_field), e_field))/6

    def total_energy(self):
        return \
            self.charge_energy() + \
            self.permanent_dipole_energy() + \
            self.alpha_induced_dipole_energy() + \
            self.beta_induced_dipole_energy()

    def potential_at(self, r):
        return self.monopole_potential_at(r) + self.dipole_potential_at(r)

    def monopole_potential_at(self, r):
        dr = norm(r - self._r)
        return self._q/dr

    def dipole_potential_at(self, r):
        dr = (r - self._r)
        return dot(self.p, dr)/norm(dr)**3

    def field_at(self, r):
        return self.monopole_field_at(r) + self.dipole_field_at(r)

    def monopole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")
        return self._q*dr/dr2**1.5

    def dipole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")
        p = self.dipole_moment()
        return (3*dr*dot(dr, p) - dr2*p)/dr2**2.5




def header_to_dict(header):
    """Transfer header string in pot file to dict data"""

    header_dict = {
        "#atoms": 0, 
        "max_angmom": 0, 
        "iso_pol": False, 
        "ut_pol": False, 
        "ut_hyppol": False, 
        }

    header_data = list(map(int, header.split()))
    header_dict["#atoms"] = header_data[0]
    header_dict["max_angmom"] = header_data[1]
    header_dict["iso_pol"] = len(header_data) > 2 and header_data[2] % 10 == 1
    header_dict["ut_pol"] = len(header_data) > 2 and header_data[2] % 10 == 2
    if len(header_data) > 2 and header_data[2] % 10 > 2: 
        raise TypeError
    header_dict["ut_hyppol"] = len(header_data) > 2 and header_data[2] // 10 == 2
    return header_dict

def line_to_dict(header_dict, line):
    """Transfer line data to dictonary"""

    line_dict = {}
    line_items = line.split()

    line_dict['group'] = int(line_items[0])
    line_data = list(map(float, line_items[1:]))

    line_dict['coordinates'] = line_data[:3]
    nextstart = 3

    max_angmom = header_dict.get('max_angmom', 0)
    iso_pol = header_dict.get('iso_pol', False)
    ut_pol = header_dict.get('ut_pol', False)
    hyp_pol = header_dict.get('ut_hyppol', False)

    if max_angmom >= 0: 
        nextend = nextstart + 1
        line_dict['charge'] = line_data[nextstart]
        nextstart = nextend
    if max_angmom >= 1: 
        nextend = nextstart + 3
        line_dict['dipole'] = line_data[nextstart: nextend]
        nextstart = nextend
    if max_angmom >= 2: 
        nextend = nextstart + 6
        line_dict['quadrupole'] = line_data[nextstart: nextend]
        nextstart = nextend

    if iso_pol:
        nextend = nextstart + 1
        line_dict['iso_alpha'] = line_data[nextstart]
    elif ut_pol:
        nextend = nextstart + 6
        line_dict['ut_alpha'] = line_data[nextstart: nextend]
    nextstart = nextend

    if hyp_pol:
        nextend = nextstart + 10
        line_dict['ut_beta'] = line_data[nextstart: nextend]
    else:
        #line_dict['beta'] = np.zeros(27)
        pass

#skip particles with zero charge, zero multipoles, zero alpha, zero beta
    backup = line_dict.copy()
    del backup['coordinates']
    del backup['group']

    valid = False
    for val in backup.values():
        if np.allclose( np.zeros( np.array((val,)).shape ), np.array((val,)) , atol = 1e-14):
#Temporary hack since one test has zero charge only see test_particles.py line 176, due to DIMER_TEMPLATE
            if len(backup) == 1:
                valid = True
        else:
            valid = True
    if valid:
        return line_dict
    return None

class SCFNotConverged(Exception):
    def __init__(self, residual, threshold):
        self.residual = residual
        self.threshold = threshold

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('potfile')
    args = parser.parse_args()
    pdl = PointDipoleList(open(args.potfile))
