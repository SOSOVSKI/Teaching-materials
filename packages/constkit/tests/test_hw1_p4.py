"""HW1 Problem 4 — Kinematics.

Tests: deformation gradient for simple shear, C, b, E, e,
comparison at small and large shear.
"""

import sympy as sp

from constkit.tensor2 import Tensor2
from constkit.kinematics import (
    deformation_gradient,
    right_cauchy_green,
    left_cauchy_green,
    green_lagrange,
    almansi,
    infinitesimal_strain,
    jacobian,
    polar_decomposition,
    principal_stretches,
)


class TestProblem4a:
    """(a) Compute F for simple shear x = X + k X₂ e₁."""

    def test_F_symbolic(self, k_sym):
        F = deformation_gradient("simple_shear", k=k_sym)
        expected = sp.Matrix([[1, k_sym, 0], [0, 1, 0], [0, 0, 1]])
        assert F.matrix == expected

    def test_J_equals_one(self, k_sym):
        F = deformation_gradient("simple_shear", k=k_sym)
        assert sp.simplify(jacobian(F) - 1) == 0


class TestProblem4b:
    """(b) Compute C = F^T F and b = F F^T."""

    def test_C_symbolic(self, k_sym):
        F = deformation_gradient("simple_shear", k=k_sym)
        C = right_cauchy_green(F)
        expected = sp.Matrix([
            [1, k_sym, 0],
            [k_sym, 1 + k_sym**2, 0],
            [0, 0, 1],
        ])
        assert sp.simplify(C.matrix - expected) == sp.zeros(3, 3)

    def test_b_symbolic(self, k_sym):
        F = deformation_gradient("simple_shear", k=k_sym)
        b = left_cauchy_green(F)
        expected = sp.Matrix([
            [1 + k_sym**2, k_sym, 0],
            [k_sym, 1, 0],
            [0, 0, 1],
        ])
        assert sp.simplify(b.matrix - expected) == sp.zeros(3, 3)

    def test_C_numeric_k1(self):
        F = deformation_gradient("simple_shear", k=sp.Integer(1))
        C = right_cauchy_green(F)
        assert C.matrix == sp.Matrix([[1, 1, 0], [1, 2, 0], [0, 0, 1]])

    def test_b_numeric_k1(self):
        F = deformation_gradient("simple_shear", k=sp.Integer(1))
        b = left_cauchy_green(F)
        assert b.matrix == sp.Matrix([[2, 1, 0], [1, 1, 0], [0, 0, 1]])


class TestProblem4c:
    """(c) Compute Green-Lagrange E and Almansi e."""

    def test_E_symbolic(self, k_sym):
        F = deformation_gradient("simple_shear", k=k_sym)
        E = green_lagrange(F)
        expected = sp.Matrix([
            [0, k_sym / 2, 0],
            [k_sym / 2, k_sym**2 / 2, 0],
            [0, 0, 0],
        ])
        assert sp.simplify(E.matrix - expected) == sp.zeros(3, 3)

    def test_e_numeric_k1(self):
        F = deformation_gradient("simple_shear", k=sp.Integer(1))
        e = almansi(F)
        expected = sp.Matrix([
            [0, sp.Rational(1, 2), 0],
            [sp.Rational(1, 2), sp.Rational(-1, 2), 0],
            [0, 0, 0],
        ])
        assert sp.simplify(e.matrix - expected) == sp.zeros(3, 3)

    def test_E_vanishes_for_rigid_rotation(self, theta_sym):
        """Green-Lagrange strain should be zero for pure rotation."""
        F = deformation_gradient("rotation", theta=theta_sym, axis=3)
        E = green_lagrange(F)
        assert sp.simplify(E.matrix) == sp.zeros(3, 3)


class TestProblem4d:
    """(d) Compare E and e at small and large shear."""

    def test_small_shear_convergence(self):
        """For k=1/100, E ≈ e ≈ ε (small-strain limit)."""
        k = sp.Rational(1, 100)
        F = deformation_gradient("simple_shear", k=k)
        E = green_lagrange(F)
        e = almansi(F)
        eps = infinitesimal_strain(F)

        # E_12 ≈ e_12 ≈ ε_12 = k/2 = 1/200
        assert E[0, 1] == sp.Rational(1, 200)
        assert eps[0, 1] == sp.Rational(1, 200)
        # Almansi e_12 should also be close to k/2
        assert sp.simplify(e[0, 1] - sp.Rational(1, 200)) == 0

    def test_large_shear_divergence(self):
        """For k=1, E_22 and e_22 differ significantly."""
        k = sp.Integer(1)
        F = deformation_gradient("simple_shear", k=k)
        E = green_lagrange(F)
        e = almansi(F)

        # E_22 = k²/2 = 1/2 (positive)
        assert E[1, 1] == sp.Rational(1, 2)
        # e_22 = -k²/2 / (1+k²) ... actually e_22 = -1/2 for simple shear
        assert e[1, 1] == sp.Rational(-1, 2)

    def test_eigenvalues_of_C(self):
        """Principal stretches squared: eigenvalues of C for k=1."""
        F = deformation_gradient("simple_shear", k=sp.Integer(1))
        C = right_cauchy_green(F)
        eigs = sorted(C.matrix.eigenvals().keys())
        # Expected: (3±√5)/2 and 1
        expected = sorted([
            sp.Rational(3, 2) - sp.sqrt(5) / 2,
            sp.Integer(1),
            sp.Rational(3, 2) + sp.sqrt(5) / 2,
        ])
        for a, b in zip(eigs, expected):
            assert sp.simplify(a - b) == 0
