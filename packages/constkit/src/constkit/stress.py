"""Symbolic stress measure conversions and stress transformation checks.

Covers the four stress measures from L03:
    σ (Cauchy), P (1st Piola-Kirchhoff), S (2nd Piola-Kirchhoff), τ (Kirchhoff)
and their inter-conversion formulas.
"""

from __future__ import annotations

import warnings

import sympy as sp

from constkit.tensor2 import Tensor2


def _mat(T):
    """Extract sp.Matrix from Tensor2 or pass through."""
    return T.matrix if isinstance(T, Tensor2) else sp.Matrix(T)


# ------------------------------------------------------------------
# Stress conversions
# ------------------------------------------------------------------

def cauchy_to_pk1(sigma: Tensor2, F: Tensor2) -> Tensor2:
    """First Piola-Kirchhoff stress: P = J σ F^{-T}.

    Parameters
    ----------
    sigma : Tensor2
        Cauchy stress tensor.
    F : Tensor2
        Deformation gradient.
    """
    Fm = _mat(F)
    J = Fm.det()
    sig = _mat(sigma)
    return Tensor2(sp.simplify(J * sig * Fm.inv().T), name="P")


def cauchy_to_pk2(sigma: Tensor2, F: Tensor2) -> Tensor2:
    """Second Piola-Kirchhoff stress: S = J F^{-1} σ F^{-T}.

    Parameters
    ----------
    sigma : Tensor2
        Cauchy stress tensor.
    F : Tensor2
        Deformation gradient.
    """
    Fm = _mat(F)
    J = Fm.det()
    sig = _mat(sigma)
    F_inv = Fm.inv()
    return Tensor2(sp.simplify(J * F_inv * sig * F_inv.T), name="S")


def cauchy_to_kirchhoff(sigma: Tensor2, F: Tensor2) -> Tensor2:
    """Kirchhoff stress: τ = J σ.

    Parameters
    ----------
    sigma : Tensor2
        Cauchy stress tensor.
    F : Tensor2
        Deformation gradient (used only for J = det F).
    """
    Fm = _mat(F)
    J = Fm.det()
    sig = _mat(sigma)
    return Tensor2(sp.simplify(J * sig), name="τ")


def pk2_to_cauchy(S: Tensor2, F: Tensor2) -> Tensor2:
    """Cauchy stress from 2nd Piola-Kirchhoff: σ = J⁻¹ F S F^T.

    Parameters
    ----------
    S : Tensor2
        Second Piola-Kirchhoff stress.
    F : Tensor2
        Deformation gradient.
    """
    Fm = _mat(F)
    Sm = _mat(S)
    J = Fm.det()
    return Tensor2(sp.simplify(Fm * Sm * Fm.T / J), name="σ")


def pk1_to_cauchy(P: Tensor2, F: Tensor2) -> Tensor2:
    """Cauchy stress from 1st Piola-Kirchhoff: σ = J⁻¹ P F^T.

    Parameters
    ----------
    P : Tensor2
        First Piola-Kirchhoff stress.
    F : Tensor2
        Deformation gradient.
    """
    Fm = _mat(F)
    Pm = _mat(P)
    J = Fm.det()
    return Tensor2(sp.simplify(Pm * Fm.T / J), name="σ")


# ------------------------------------------------------------------
# Stress transformation checks
# ------------------------------------------------------------------

def verify_stress_transformations(
    sigma: Tensor2,
    F: Tensor2,
    Q: sp.Matrix | None = None,
    sigma_star: Tensor2 | sp.Matrix | None = None,
) -> dict:
    """Verify stress transformation laws under a superposed rotation.

    Applies the superposed rigid rotation ``Q`` to the motion
    ``F* = Q F`` and checks the associated stress transformations:

        σ* = Q σ Q^T
        τ* = Q τ Q^T
        S* = S
        P* = Q P

    If ``sigma_star`` is supplied, it is treated as the actual rotated
    Cauchy stress and compared against ``Q σ Q^T``. If omitted, the
    theoretical transform ``Q σ Q^T`` is used, so the routine checks the
    internal consistency of the stress-conversion formulas.

    Parameters
    ----------
    sigma : Tensor2
        Cauchy stress.
    F : Tensor2
        Deformation gradient.
    Q : sp.Matrix, optional
        Rotation matrix. If None, uses a symbolic rotation by angle θ
        about the z-axis.
    sigma_star : Tensor2, sp.Matrix, or None, optional
        Rotated Cauchy stress to compare against ``Q σ Q^T``. Supplying
        this makes the ``sigma_matches_rotation`` check a genuine test of
        constitutive objectivity.

    Returns
    -------
    dict with keys:
        'sigma_matches_rotation' : bool
        'tau_matches_rotation' : bool
        'S_invariant' : bool
        'P_two_point' : bool
        'details' : dict with the differences (should be zero matrices).
    """
    if Q is None:
        from constkit.transforms import rotation_matrix
        theta = sp.Symbol("θ_obj")
        Q = rotation_matrix(theta, axis=3)

    Fm = _mat(F)
    sig = _mat(sigma)

    F_star = Q * Fm
    sigma_expected = sp.simplify(Q * sig * Q.T)
    sigma_star_mat = sigma_expected if sigma_star is None else _mat(sigma_star)

    sigma_tensor = Tensor2(sig, name="σ")
    sigma_star_tensor = Tensor2(sigma_star_mat, name="σ*")
    F_tensor = Tensor2(Fm, name="F")
    F_star_tensor = Tensor2(F_star, name="F*")

    tau_orig = cauchy_to_kirchhoff(sigma_tensor, F_tensor).matrix
    tau_star = cauchy_to_kirchhoff(sigma_star_tensor, F_star_tensor).matrix
    S_orig = cauchy_to_pk2(sigma_tensor, F_tensor).matrix
    S_star = cauchy_to_pk2(sigma_star_tensor, F_star_tensor).matrix
    P_orig = cauchy_to_pk1(sigma_tensor, F_tensor).matrix
    P_star = cauchy_to_pk1(sigma_star_tensor, F_star_tensor).matrix

    diff_sigma = sp.simplify(sigma_star_mat - sigma_expected)
    diff_tau = sp.simplify(tau_star - Q * tau_orig * Q.T)
    diff_S = sp.simplify(S_star - S_orig)
    diff_P = sp.simplify(P_star - Q * P_orig)

    return {
        "sigma_matches_rotation": diff_sigma == sp.zeros(3, 3),
        "tau_matches_rotation": diff_tau == sp.zeros(3, 3),
        "S_invariant": diff_S == sp.zeros(3, 3),
        "P_two_point": diff_P == sp.zeros(3, 3),
        "used_supplied_sigma_star": sigma_star is not None,
        "details": {
            "sigma_star - Q σ Q^T": diff_sigma,
            "tau_star - Q τ Q^T": diff_tau,
            "S_star - S": diff_S,
            "P_star - Q P": diff_P,
        },
    }


def check_objectivity(
    sigma: Tensor2,
    F: Tensor2,
    Q: sp.Matrix | None = None,
    sigma_star: Tensor2 | sp.Matrix | None = None,
) -> dict:
    """Deprecated alias for :func:`verify_stress_transformations`.

    The older name suggested a direct objectivity test even when only the
    theoretical transformation ``σ* = Q σ Q^T`` was being constructed.
    Prefer ``verify_stress_transformations(..., sigma_star=...)``.
    """
    warnings.warn(
        "check_objectivity() is deprecated; use "
        "verify_stress_transformations() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = verify_stress_transformations(
        sigma,
        F,
        Q=Q,
        sigma_star=sigma_star,
    )
    legacy_result = dict(result)
    legacy_result["sigma_objective"] = result["sigma_matches_rotation"]
    legacy_result["tau_objective"] = result["tau_matches_rotation"]
    return legacy_result
