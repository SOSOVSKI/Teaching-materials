"""HW1 Problem 2 — Tensor Functions.

Tests: spectral decomposition, ln(A), √A, ∂A/∂A for symmetric
and non-symmetric tensors.
"""

import pytest
import sympy as sp

from constkit.tensor2 import Tensor2
from constkit.calculus import (
    identity_4th,
    symmetric_identity_4th,
    tensor_derivative,
)
from constkit.invariants import invariant_derivative


class TestProblem2abc:
    """(a)-(c) Eigenvalue problem, ln(A), √A."""

    def test_sqrt_squares_to_original(self, S_tensor):
        """√A @ √A should equal A."""
        sqrt_S = S_tensor.sqrt()
        reconstructed = sqrt_S.single_contract(sqrt_S)
        diff = sp.simplify(reconstructed.matrix - S_tensor.matrix)
        assert diff == sp.zeros(3, 3)

    def test_log_via_eigenvalues(self, S_tensor):
        """ln(A) eigenvalues should be ln(eigenvalues of A)."""
        ln_S = S_tensor.log()
        # Check trace: tr(ln A) = ln(det A)
        tr_ln = sp.simplify(ln_S.trace())
        ln_det = sp.simplify(sp.log(S_tensor.det()))
        assert sp.simplify(tr_ln - ln_det) == 0

    def test_spectral_function_rejects_nonsymmetric_tensor(self):
        """The orthonormal spectral formula is only valid for symmetric A."""
        A = Tensor2([[1, 1, 0], [0, 2, 0], [0, 0, 3]], name="A")
        with pytest.raises(ValueError, match="symmetric tensor"):
            A.sqrt()


class TestProblem2de:
    """(d)-(e) ∂A/∂A for symmetric and non-symmetric A."""

    def test_symmetric_identity(self):
        """For symmetric A, ∂A/∂A = ½(δ_ik δ_jl + δ_il δ_jk)."""
        I_sym = symmetric_identity_4th()
        # Check: I_sym_{1122} = 0 (no coupling of 11 and 22)
        assert I_sym[0, 0, 1, 1] == 0
        # Check: I_sym_{1212} = 1/2
        assert I_sym[0, 1, 0, 1] == sp.Rational(1, 2)
        # Check: I_sym_{1111} = 1
        assert I_sym[0, 0, 0, 0] == 1

    def test_nonsymmetric_identity(self):
        """For non-symmetric A, ∂A/∂A = δ_ik δ_jl."""
        I_full = identity_4th()
        # I_{1111} = 1
        assert I_full[0, 0, 0, 0] == 1
        # I_{1212} = 1 (unlike symmetric case where it's 1/2)
        assert I_full[0, 1, 0, 1] == 1
        # I_{1221} = 0
        assert I_full[0, 1, 1, 0] == 0

    def test_symmetric_identity_contracts_correctly(self, S_tensor):
        """Verify that I^s : A = A for a symmetric tensor."""
        I_sym = symmetric_identity_4th()
        # Contract: (I_sym)_{ijkl} A_{kl} = A_{ij}
        A = S_tensor.matrix
        for i in range(3):
            for j in range(3):
                contracted = sum(
                    I_sym[i, j, k, l] * A[k, l]
                    for k in range(3)
                    for l in range(3)
                )
                assert sp.simplify(contracted - A[i, j]) == 0

    def test_I2_derivative_matches_componentwise_gradient(self):
        """For a general tensor, ∂I₂/∂A = I₁I − Aᵀ."""
        a11, a12, a13 = sp.symbols("a11 a12 a13")
        a21, a22, a23 = sp.symbols("a21 a22 a23")
        a31, a32, a33 = sp.symbols("a31 a32 a33")
        A = Tensor2(
            [
                [a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33],
            ],
            name="A",
        )

        closed_form = invariant_derivative(2, A)
        componentwise = tensor_derivative(A.I2(), A)

        assert sp.simplify(closed_form.matrix - componentwise) == sp.zeros(3, 3)
