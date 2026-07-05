"""HW1 Problem 3 — Curvilinear Coordinates.

Tests: cylindrical coordinate basis vectors, metric tensor,
contravariant basis, biorthogonality verification.
"""

import sympy as sp

from constkit.coordinates import (
    cylindrical_basis,
    contravariant_basis,
    metric_tensor,
    verify_biorthogonality,
)


class TestProblem3a:
    """(a) Derive covariant basis vectors g_i for cylindrical coords."""

    def test_covariant_basis_general(self, r_sym, phi_sym):
        g = cylindrical_basis(r_sym, phi_sym)
        # g_r = [cos φ, sin φ, 0]
        assert g[0] == sp.Matrix([sp.cos(phi_sym), sp.sin(phi_sym), 0])
        # g_φ = [-r sin φ, r cos φ, 0]
        assert g[1] == sp.Matrix(
            [-r_sym * sp.sin(phi_sym), r_sym * sp.cos(phi_sym), 0]
        )
        # g_z = [0, 0, 1]
        assert g[2] == sp.Matrix([0, 0, 1])

    def test_metric_tensor(self, r_sym, phi_sym):
        g_cov = cylindrical_basis(r_sym, phi_sym)
        g = metric_tensor(g_cov)
        expected = sp.Matrix([
            [1, 0, 0],
            [0, r_sym**2, 0],
            [0, 0, 1],
        ])
        assert sp.simplify(g - expected) == sp.zeros(3, 3)

    def test_metric_at_specific_point(self):
        """Evaluate at r=2, φ=π/6."""
        r, phi = sp.Integer(2), sp.pi / 6
        g_cov = cylindrical_basis(r, phi)
        g = metric_tensor(g_cov)
        assert g[0, 0] == 1
        assert g[1, 1] == 4  # r² = 4
        assert g[2, 2] == 1
        assert g[0, 1] == 0


class TestProblem3b:
    """(b) Derive contravariant basis vectors g^i."""

    def test_contravariant_basis(self, r_sym, phi_sym):
        g_cov = cylindrical_basis(r_sym, phi_sym)
        g_contra = contravariant_basis(g_cov)
        # g^r should be same as g_r for orthogonal system: [cos φ, sin φ, 0]
        expected_gr = sp.Matrix([sp.cos(phi_sym), sp.sin(phi_sym), 0])
        assert sp.trigsimp(g_contra[0] - expected_gr) == sp.zeros(3, 1)
        # g^z = [0, 0, 1]
        assert sp.trigsimp(g_contra[2] - sp.Matrix([0, 0, 1])) == sp.zeros(3, 1)


class TestProblem3c:
    """(c) Verify g_i · g^j = δ^j_i."""

    def test_biorthogonality_symbolic(self, r_sym, phi_sym):
        g_cov = cylindrical_basis(r_sym, phi_sym)
        g_contra = contravariant_basis(g_cov)
        assert verify_biorthogonality(g_cov, g_contra) is True

    def test_biorthogonality_numeric(self):
        """Verify at a specific point r=2, φ=π/6."""
        r, phi = sp.Integer(2), sp.pi / 6
        g_cov = cylindrical_basis(r, phi)
        g_contra = contravariant_basis(g_cov)
        assert verify_biorthogonality(g_cov, g_contra) is True
