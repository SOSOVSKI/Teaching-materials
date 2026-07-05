"""HW1 Problem 1 — Tensor Operations.

Tests: outer product, single contraction, double contraction,
principal invariants, eigendecomposition, coordinate transformation.
"""

import sympy as sp

from constkit.tensor2 import Tensor2
from constkit.vector import outer_product, dot, cross
from constkit.transforms import rotation_matrix


class TestProblem1a:
    """(a) Compute u⊗v, S·v, and S:T."""

    def test_outer_product(self, u_vec, v_vec):
        result = outer_product(u_vec, v_vec)
        expected = sp.Matrix([
            [4, 5, 6],
            [8, 10, 12],
            [12, 15, 18],
        ])
        assert result == expected

    def test_single_contraction(self, S_tensor, v_vec):
        result = S_tensor.dot(v_vec)
        expected = sp.Matrix([13, 25, 29])
        assert result == expected

    def test_double_contraction(self, S_tensor, T_tensor):
        result = S_tensor.double_contract(T_tensor)
        assert result == 17


class TestProblem1b:
    """(b) Compute invariants I₁, I₂, I₃ of S."""

    def test_I1(self, S_tensor):
        assert S_tensor.I1() == 9

    def test_I2(self, S_tensor):
        assert S_tensor.I2() == 24

    def test_I3(self, S_tensor):
        assert S_tensor.I3() == 18


class TestProblem1c:
    """(c) Principal values and directions of S."""

    def test_eigenvalues(self, S_tensor):
        eig = S_tensor.eigendecomposition()
        vals = sorted([sp.simplify(v) for v in eig["eigenvalues"]])
        # Expected: 1 + √3, 3, 3 + √3... actually let's compute
        # The eigenvalues of [[2,1,0],[1,3,1],[0,1,4]] are 3, 3±√3
        assert sp.simplify(vals[0] - (3 - sp.sqrt(3))) == 0
        assert sp.simplify(vals[1] - 3) == 0
        assert sp.simplify(vals[2] - (3 + sp.sqrt(3))) == 0

    def test_characteristic_equation(self, S_tensor):
        """Verify eigenvalues satisfy -λ³ + I₁λ² - I₂λ + I₃ = 0."""
        i1 = S_tensor.I1()
        i2 = S_tensor.I2()
        i3 = S_tensor.I3()
        eig = S_tensor.eigendecomposition()
        for lam in eig["eigenvalues"]:
            residual = -lam**3 + i1 * lam**2 - i2 * lam + i3
            assert sp.simplify(residual) == 0


class TestProblem1d:
    """(d) Transform S to a coordinate system rotated 45° about x₃."""

    def test_invariant_preservation(self, S_tensor):
        Q = rotation_matrix(sp.pi / 4, axis=3)
        S_rot = S_tensor.rotate(Q)
        assert sp.simplify(S_rot.I1() - S_tensor.I1()) == 0
        assert sp.simplify(S_rot.I2() - S_tensor.I2()) == 0
        assert sp.simplify(S_rot.I3() - S_tensor.I3()) == 0

    def test_rotated_components(self, S_tensor):
        Q = rotation_matrix(sp.pi / 4, axis=3)
        S_rot = S_tensor.rotate(Q)
        # S'_11 should be (S11 + S22 + 2*S12)/2 = (2+3+2)/2 = 7/2
        assert sp.simplify(S_rot[0, 0] - sp.Rational(7, 2)) == 0
        # S'_22 should be (S11 + S22 - 2*S12)/2 = (2+3-2)/2 = 3/2
        assert sp.simplify(S_rot[1, 1] - sp.Rational(3, 2)) == 0
        # S'_33 unchanged
        assert sp.simplify(S_rot[2, 2] - 4) == 0
