"""Symbolic kinematics: deformation gradient, strain measures, polar decomposition.

Covers all kinematic quantities from L02–L03:
    F, C, b, E, e, ε, R, U, V, L, D, W, J, isochoric split.

All functions accept and return SymPy matrices or Tensor2 objects.
"""

from __future__ import annotations

import sympy as sp

from constkit.tensor2 import Tensor2


# ------------------------------------------------------------------
# Deformation gradient presets
# ------------------------------------------------------------------

def deformation_gradient(mode: str, **params) -> Tensor2:
    """Construct a symbolic deformation gradient for a named deformation.

    Parameters
    ----------
    mode : str
        One of: ``'simple_shear'``, ``'uniaxial'``, ``'biaxial'``,
        ``'rotation'``, ``'custom'``.
    **params
        Mode-specific parameters (may be SymPy symbols or floats).

        - ``simple_shear``: ``k`` — shear parameter.
        - ``uniaxial``: ``lam`` — principal stretch.
        - ``biaxial``: ``lam1``, ``lam2`` — principal stretches.
        - ``rotation``: ``theta``, ``axis`` (default 3).
        - ``custom``: ``F`` — 3×3 matrix.

    Returns
    -------
    Tensor2
        The deformation gradient F.
    """
    if mode == "simple_shear":
        k = params["k"]
        return Tensor2(
            sp.Matrix([[1, k, 0], [0, 1, 0], [0, 0, 1]]),
            name="F",
        )
    elif mode == "uniaxial":
        lam = params["lam"]
        # Incompressible: lateral stretch = 1/√λ
        lat = 1 / sp.sqrt(lam)
        return Tensor2(
            sp.Matrix([[lam, 0, 0], [0, lat, 0], [0, 0, lat]]),
            name="F",
        )
    elif mode == "biaxial":
        l1 = params["lam1"]
        l2 = params["lam2"]
        l3 = 1 / (l1 * l2)
        return Tensor2(
            sp.Matrix([[l1, 0, 0], [0, l2, 0], [0, 0, l3]]),
            name="F",
        )
    elif mode == "rotation":
        from constkit.transforms import rotation_matrix
        theta = params["theta"]
        axis = params.get("axis", 3)
        return Tensor2(rotation_matrix(theta, axis), name="F")
    elif mode == "custom":
        return Tensor2(sp.Matrix(params["F"]), name="F")
    else:
        raise ValueError(
            f"Unknown deformation mode '{mode}'. "
            "Choose from: simple_shear, uniaxial, biaxial, rotation, custom."
        )


# ------------------------------------------------------------------
# Cauchy-Green tensors
# ------------------------------------------------------------------

def right_cauchy_green(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Right Cauchy-Green tensor: C = F^T F (reference configuration)."""
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    return Tensor2(sp.simplify(M.T * M), name="C")


def left_cauchy_green(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Left Cauchy-Green (Finger) tensor: b = F F^T (current configuration)."""
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    return Tensor2(sp.simplify(M * M.T), name="b")


# ------------------------------------------------------------------
# Strain measures
# ------------------------------------------------------------------

def green_lagrange(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Green-Lagrange strain: E = ½(C − I) = ½(F^T F − I).

    Reference-configuration (material) strain measure.
    """
    C = right_cauchy_green(F)
    return Tensor2(
        sp.Rational(1, 2) * (C.matrix - sp.eye(3)),
        name="E",
    )


def almansi(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Euler-Almansi strain: e = ½(I − b⁻¹) = ½(I − (FF^T)⁻¹).

    Current-configuration (spatial) strain measure.
    """
    b = left_cauchy_green(F)
    b_inv = b.inv()
    return Tensor2(
        sp.simplify(sp.Rational(1, 2) * (sp.eye(3) - b_inv.matrix)),
        name="e",
    )


def infinitesimal_strain(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Infinitesimal (small) strain: ε = sym(F − I) = ½(∇u + ∇u^T).

    Valid only when |∇u| ≪ 1.
    """
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    grad_u = M - sp.eye(3)
    return Tensor2(
        sp.Rational(1, 2) * (grad_u + grad_u.T),
        name="ε",
    )


# ------------------------------------------------------------------
# Jacobian
# ------------------------------------------------------------------

def jacobian(F: Tensor2 | sp.Matrix) -> sp.Expr:
    """Jacobian determinant: J = det(F) > 0."""
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    return sp.simplify(M.det())


# ------------------------------------------------------------------
# Polar decomposition
# ------------------------------------------------------------------

def polar_decomposition(F: Tensor2 | sp.Matrix) -> dict:
    """Polar decomposition: F = R U = V R.

    Computes U = √C, R = F U⁻¹, V = R U R^T.

    Returns
    -------
    dict with keys:
        'R' : Tensor2 — rotation tensor
        'U' : Tensor2 — right stretch tensor
        'V' : Tensor2 — left stretch tensor
        'stretches' : list — principal stretches (eigenvalues of U)

    Note: This uses SymPy's eigendecomposition to compute √C, which
    works well for structured matrices (e.g. simple shear) but may
    produce unwieldy expressions for fully symbolic F.
    """
    C = right_cauchy_green(F)
    U = C.sqrt()
    U._m = sp.simplify(U._m)

    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    R = Tensor2(sp.simplify(M * U.inv().matrix), name="R")
    V = Tensor2(sp.simplify(R.matrix * U.matrix * R.T.matrix), name="V")

    # Principal stretches: eigenvalues of U
    stretches = [sp.simplify(v) for v in U.matrix.eigenvals().keys()]

    return {"R": R, "U": U, "V": V, "stretches": stretches}


# ------------------------------------------------------------------
# Isochoric (volume-preserving) decomposition
# ------------------------------------------------------------------

def isochoric_split(F: Tensor2 | sp.Matrix) -> dict:
    """Isochoric decomposition: F = J^{1/3} F̄, det(F̄) = 1.

    Returns
    -------
    dict with keys:
        'F_bar' : Tensor2 — isochoric deformation gradient
        'J' : sp.Expr — Jacobian determinant
        'C_bar' : Tensor2 — isochoric right Cauchy-Green
    """
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    J = sp.simplify(M.det())
    F_bar = Tensor2(
        sp.simplify(J ** sp.Rational(-1, 3) * M),
        name="F̄",
    )
    C_bar = right_cauchy_green(F_bar)
    C_bar.name = "C̄"
    return {"F_bar": F_bar, "J": J, "C_bar": C_bar}


# ------------------------------------------------------------------
# Principal stretches
# ------------------------------------------------------------------

def principal_stretches(F: Tensor2 | sp.Matrix) -> list:
    """Principal stretches λ_i from eigenvalues of C = F^T F.

    λ_i = √(eigenvalue_i of C).
    """
    C = right_cauchy_green(F)
    eigenvalues = list(C.matrix.eigenvals().keys())
    return [sp.simplify(sp.sqrt(v)) for v in eigenvalues]


# ------------------------------------------------------------------
# Metric tensors from deformation gradient
# ------------------------------------------------------------------

def metric_from_F(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Reference-configuration metric tensor: G = Fᵀ F.

    The pull-back of the Euclidean metric to the reference configuration.
    Identical to the right Cauchy-Green tensor C, but named and returned
    as a metric for use in curvilinear reference-frame computations.

    Parameters
    ----------
    F : Tensor2 or sp.Matrix
        Deformation gradient.

    Returns
    -------
    Tensor2
        Symmetric positive-definite metric G = Fᵀ F, named ``'G'``.
    """
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    return Tensor2(sp.simplify(M.T * M), name="G")


def spatial_metric_from_F(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Current-configuration (spatial) metric tensor: g = F Fᵀ.

    The push-forward of the Euclidean metric to the current configuration.
    Identical to the left Cauchy-Green tensor b, but named and returned
    as a spatial metric.

    Parameters
    ----------
    F : Tensor2 or sp.Matrix
        Deformation gradient.

    Returns
    -------
    Tensor2
        Symmetric positive-definite metric g = F Fᵀ, named ``'g'``.
    """
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    return Tensor2(sp.simplify(M * M.T), name="g")


def metric_from_F_inv(F: Tensor2 | sp.Matrix) -> Tensor2:
    """Inverse reference metric: G⁻¹ = F⁻¹ F⁻ᵀ.

    The contravariant metric tensor (dual to G = Fᵀ F) in the reference
    configuration.  Satisfies G · G⁻¹ = I.

    Note: (Fᵀ F)⁻¹ = F⁻¹ F⁻ᵀ ≠ F⁻ᵀ F⁻¹ (which is the inverse of g = FFᵀ).

    Parameters
    ----------
    F : Tensor2 or sp.Matrix
        Deformation gradient.

    Returns
    -------
    Tensor2
        Symmetric positive-definite tensor F⁻¹ F⁻ᵀ, named ``'G_inv'``.
    """
    M = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    F_inv = M.inv()
    return Tensor2(sp.simplify(F_inv * F_inv.T), name="G_inv")


# ------------------------------------------------------------------
# Velocity gradient and rates
# ------------------------------------------------------------------

def velocity_gradient(F: Tensor2 | sp.Matrix, F_dot: Tensor2 | sp.Matrix) -> dict:
    """Velocity gradient and its decomposition.

    L = Ḟ F⁻¹,  D = sym(L),  W = skew(L).

    Parameters
    ----------
    F : Tensor2
        Deformation gradient.
    F_dot : Tensor2
        Time derivative of F.

    Returns
    -------
    dict with keys:
        'L' : Tensor2 — velocity gradient
        'D' : Tensor2 — rate of deformation (symmetric)
        'W' : Tensor2 — spin tensor (skew-symmetric)
    """
    M_F = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    M_Fd = F_dot.matrix if isinstance(F_dot, Tensor2) else sp.Matrix(F_dot)

    L = Tensor2(sp.simplify(M_Fd * M_F.inv()), name="L")
    D = L.sym()
    D.name = "D"
    W = L.skew()
    W.name = "W"

    return {"L": L, "D": D, "W": W}
