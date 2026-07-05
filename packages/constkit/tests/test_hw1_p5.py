"""HW1 Problem 5 — Stress Measures.

Tests: PK1 and PK2 stress computation, stress transformation laws,
Jaumann rate analysis.
"""

import pytest
import sympy as sp

from constkit.tensor2 import Tensor2
from constkit.kinematics import deformation_gradient
from constkit.stress import (
    cauchy_to_pk1,
    cauchy_to_pk2,
    check_objectivity,
    verify_stress_transformations,
)
from constkit.rates import jaumann_rate
from constkit.transforms import rotation_matrix


class TestProblem5a:
    """(a) Compute P = J σ F^{-T} for simple shear with σ = σ₀ e₁⊗e₁."""

    def test_pk1_simple_shear(self):
        sigma_0 = sp.Symbol("sigma_0", positive=True)
        F = deformation_gradient("simple_shear", k=sp.Integer(1))
        sigma = Tensor2(sigma_0 * sp.Matrix([
            [1, 0, 0], [0, 0, 0], [0, 0, 0]
        ]))
        P = cauchy_to_pk1(sigma, F)
        # For simple shear with J=1, F^{-T} = [[1,0,0],[-k,1,0],[0,0,1]]^T
        # P = σ F^{-T} = σ₀ [[1,0,0],[0,0,0],[0,0,0]] @ [[1,-k,0],[0,1,0],[0,0,1]]
        # = σ₀ [[1, -1, 0],[0,0,0],[0,0,0]]
        # Wait — F^{-1} for [[1,1,0],[0,1,0],[0,0,1]] is [[1,-1,0],[0,1,0],[0,0,1]]
        # F^{-T} = [[1,0,0],[-1,1,0],[0,0,1]]
        # P = J σ F^{-T} = σ₀ [[1,0,0],[0,0,0],[0,0,0]] @ [[1,0,0],[-1,1,0],[0,0,1]]
        # = σ₀ [[1,0,0],[0,0,0],[0,0,0]]
        expected = sigma_0 * sp.Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert sp.simplify(P.matrix - expected) == sp.zeros(3, 3)


class TestProblem5b:
    """(b) Compute S = J F^{-1} σ F^{-T}."""

    def test_pk2_simple_shear(self):
        sigma_0 = sp.Symbol("sigma_0", positive=True)
        F = deformation_gradient("simple_shear", k=sp.Integer(1))
        sigma = Tensor2(sigma_0 * sp.Matrix([
            [1, 0, 0], [0, 0, 0], [0, 0, 0]
        ]))
        S = cauchy_to_pk2(sigma, F)
        # S = F^{-1} σ F^{-T} (J=1)
        # F^{-1} = [[1,-1,0],[0,1,0],[0,0,1]]
        # F^{-T} = [[1,0,0],[-1,1,0],[0,0,1]]
        # F^{-1} σ = σ₀ [[1,0,0],[0,0,0],[0,0,0]]
        # F^{-1} σ F^{-T} = σ₀ [[1,0,0],[0,0,0],[0,0,0]]
        expected = sigma_0 * sp.Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert sp.simplify(S.matrix - expected) == sp.zeros(3, 3)


class TestProblem5c:
    """(c) Assess stress transformations under superposed rotations."""

    def test_S_invariant_under_rotation(self):
        """S* = S under superposed rotation (S is invariant)."""
        sigma_0 = sp.Symbol("sigma_0", positive=True)
        k = sp.Symbol("k", positive=True)
        F = deformation_gradient("simple_shear", k=k)
        sigma = Tensor2(sigma_0 * sp.Matrix([
            [1, 0, 0], [0, 0, 0], [0, 0, 0]
        ]))
        result = verify_stress_transformations(sigma, F)
        assert result["sigma_matches_rotation"] is True
        assert result["tau_matches_rotation"] is True
        assert result["S_invariant"] is True
        assert result["P_two_point"] is True

    def test_nonobjective_sigma_star_is_detected(self):
        """Supplying an inconsistent rotated stress should fail the check."""
        sigma_0 = sp.Symbol("sigma_0", positive=True)
        F = deformation_gradient("simple_shear", k=sp.Rational(1, 2))
        sigma = Tensor2(sigma_0 * sp.Matrix([
            [1, 0, 0], [0, 0, 0], [0, 0, 0]
        ]))
        Q = rotation_matrix(sp.pi / 4, axis=3)

        result = verify_stress_transformations(
            sigma,
            F,
            Q=Q,
            sigma_star=sigma,
        )

        assert result["sigma_matches_rotation"] is False

    def test_check_objectivity_is_deprecated_alias(self):
        """Legacy name still works, but points callers to the new API."""
        sigma_0 = sp.Symbol("sigma_0", positive=True)
        F = deformation_gradient("simple_shear", k=sp.Rational(1, 2))
        sigma = Tensor2(sigma_0 * sp.Matrix([
            [1, 0, 0], [0, 0, 0], [0, 0, 0]
        ]))

        with pytest.deprecated_call(match="verify_stress_transformations"):
            result = check_objectivity(sigma, F)

        assert result["sigma_objective"] is True
        assert result["tau_objective"] is True


class TestProblem5d:
    """(d) Rate forms and objectivity analysis."""

    def test_jaumann_rate_simple_shear(self):
        """With σ_dot = 0 and steady simple shear, Jaumann rate ≠ 0."""
        sigma_0 = sp.Symbol("sigma_0", positive=True)
        sigma = Tensor2(sigma_0 * sp.Matrix([
            [1, 0, 0], [0, 0, 0], [0, 0, 0]
        ]))
        sigma_dot = Tensor2(sp.zeros(3, 3))

        # W for simple shear with k_dot = 1
        k_dot = sp.Integer(1)
        W = Tensor2(sp.Matrix([
            [0, k_dot / 2, 0],
            [-k_dot / 2, 0, 0],
            [0, 0, 0],
        ]))

        sigma_J = jaumann_rate(sigma, sigma_dot, W)

        # σ̊_12 should be σ₀/2 (= 50 if σ₀=100)
        assert sp.simplify(sigma_J[0, 1] - sigma_0 / 2) == 0
        # σ̊_21 should also be σ₀/2 (symmetric)
        assert sp.simplify(sigma_J[1, 0] - sigma_0 / 2) == 0
        # σ̊_11 should be 0 (diagonal elements unaffected for this case)
        assert sp.simplify(sigma_J[0, 0]) == 0

    def test_S_is_rotation_invariant(self):
        """S is invariant under superposed spatial rotations.

        This is a conceptual test: S lives in the reference config
        which is unaffected by spatial rotations.
        """
        # S* = S for any Q, therefore Ṡ* = Ṡ — no correction needed.
        # We verify by checking S at two different rotations.
        sigma_0 = sp.Symbol("sigma_0", positive=True)
        F = deformation_gradient("simple_shear", k=sp.Rational(1, 2))
        sigma = Tensor2(sigma_0 * sp.Matrix([
            [1, 0, 0], [0, 0, 0], [0, 0, 0]
        ]))

        for angle_val in [sp.pi / 6, sp.pi / 3]:
            Q = rotation_matrix(angle_val, axis=3)
            result = verify_stress_transformations(sigma, F, Q)
            assert result["S_invariant"] is True
