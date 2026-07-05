"""Principal invariants, deviatoric invariants, and invariant derivatives.

Provides symbolic expressions for I₁, I₂, I₃ and their derivatives
with respect to the tensor — the building blocks for invariant-based
constitutive functions (L02).
"""

from __future__ import annotations

import sympy as sp

from constkit.tensor2 import Tensor2


def I1(A: Tensor2 | sp.Matrix) -> sp.Expr:
    """First principal invariant: I₁ = tr(A)."""
    if isinstance(A, Tensor2):
        return A.trace()
    return sp.Matrix(A).trace()


def I2(A: Tensor2 | sp.Matrix) -> sp.Expr:
    """Second principal invariant: I₂ = ½[(tr A)² − tr(A²)]."""
    if isinstance(A, Tensor2):
        return A.I2()
    M = sp.Matrix(A)
    return sp.Rational(1, 2) * (M.trace() ** 2 - (M * M).trace())


def I3(A: Tensor2 | sp.Matrix) -> sp.Expr:
    """Third principal invariant: I₃ = det(A)."""
    if isinstance(A, Tensor2):
        return A.det()
    return sp.Matrix(A).det()


def J2_invariant(A: Tensor2) -> sp.Expr:
    """Second deviatoric invariant: J₂ = ½ A':A' where A' = dev(A).

    Used in J2 (von Mises) plasticity yield criteria.
    """
    A_dev = A.dev()
    return sp.Rational(1, 2) * A_dev.double_contract(A_dev)


def J3_invariant(A: Tensor2) -> sp.Expr:
    """Third deviatoric invariant: J₃ = det(dev(A))."""
    return A.dev().det()


def invariant_derivative(n: int, A: Tensor2) -> Tensor2:
    """Derivative of the n-th principal invariant with respect to A.

    ∂I₁/∂A = I
    ∂I₂/∂A = I₁ I − Aᵀ
    ∂I₃/∂A = I₃ A⁻ᵀ = cof(A)

    Parameters
    ----------
    n : int
        Invariant number (1, 2, or 3).
    A : Tensor2
        The tensor to differentiate with respect to.

    Returns
    -------
    Tensor2
        The gradient ∂Iₙ/∂A.
    """
    eye = sp.eye(3)
    if n == 1:
        return Tensor2(eye, name="∂I₁/∂A")
    elif n == 2:
        return Tensor2(
            A.trace() * eye - A.matrix.T,
            name="∂I₂/∂A",
        )
    elif n == 3:
        return Tensor2(
            A.det() * A.inv().T.matrix,
            name="∂I₃/∂A",
        )
    else:
        raise ValueError(f"n must be 1, 2, or 3, got {n}")
