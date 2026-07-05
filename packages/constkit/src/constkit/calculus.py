"""Tensor calculus: derivatives, push-forward, pull-back.

Provides symbolic tensor derivatives and configuration-mapping
operations from L02.
"""

from __future__ import annotations

import sympy as sp

from constkit.tensor2 import Tensor2


# ------------------------------------------------------------------
# Tensor derivatives
# ------------------------------------------------------------------

def tensor2_derivative(A: Tensor2, B: Tensor2) -> sp.Array:
    """Derivative of a rank-2 tensor with respect to another rank-2 tensor.

    Computes ∂A_ij/∂B_kl, producing a 4th-order tensor.

    Parameters
    ----------
    A : Tensor2
        The tensor being differentiated (numerator).
    B : Tensor2
        The tensor to differentiate with respect to (denominator).

    Returns
    -------
    sp.Array
        Shape (3, 3, 3, 3) where ``result[i, j, k, l]`` = ∂A_ij/∂B_kl.

    Notes
    -----
    If A is a function of B (e.g. A = B^T or A = f(B)), the result is
    meaningful symbolically. For independent tensors, the result is zero.
    The special case A = B gives the 4th-order identity ``identity_4th()``.
    """
    components = [
        [
            [
                [sp.diff(A.matrix[i, j], B.matrix[k, l]) for l in range(3)]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)


def scalar_tensor_derivative(f: sp.Expr, A: Tensor2) -> Tensor2:
    """Derivative of a scalar expression with respect to a tensor.

    Computes (∂f/∂A)_ij = ∂f/∂A_ij.

    Parameters
    ----------
    f : sp.Expr
        Scalar symbolic expression depending on components of A.
    A : Tensor2
        The tensor to differentiate with respect to.

    Returns
    -------
    Tensor2
        Rank-2 tensor of partial derivatives.

    Notes
    -----
    Equivalent to ``tensor_derivative`` but returns a ``Tensor2`` instead
    of a plain ``sp.Matrix``, enabling chained tensor operations.
    """
    result = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            result[i, j] = sp.diff(f, A.matrix[i, j])
    return Tensor2(result)


def scalar_tensor2_derivative(f: sp.Expr, A: Tensor2, B: Tensor2) -> sp.Array:
    """Second mixed derivative of a scalar w.r.t. two rank-2 tensors.

    Computes ∂²f/(∂A_ij ∂B_kl), producing a 4th-order tensor.

    Parameters
    ----------
    f : sp.Expr
        Scalar symbolic expression depending on both A and B.
    A : Tensor2
        First differentiation tensor (row indices i, j).
    B : Tensor2
        Second differentiation tensor (column indices k, l).

    Returns
    -------
    sp.Array
        Shape (3, 3, 3, 3) where ``result[i, j, k, l]`` = ∂²f/(∂A_ij ∂B_kl).

    Notes
    -----
    When A and B share symbolic components (e.g. A = B), the mixed Hessian
    captures the second-order sensitivity of f to those components.
    """
    components = [
        [
            [
                [
                    sp.diff(sp.diff(f, A.matrix[i, j]), B.matrix[k, l])
                    for l in range(3)
                ]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)


def tensor_derivative(expr: sp.Expr, A: Tensor2) -> sp.Matrix:
    """Derivative of a scalar expression with respect to a tensor.

    Computes (∂f/∂A)_ij = ∂f/∂A_ij.

    Parameters
    ----------
    expr : sp.Expr
        Scalar symbolic expression that depends on elements of A.
    A : Tensor2
        The tensor to differentiate with respect to.

    Returns
    -------
    sp.Matrix
        3×3 matrix of partial derivatives.
    """
    result = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            result[i, j] = sp.diff(expr, A.matrix[i, j])
    return result


def invariant_gradient(n: int, A: Tensor2) -> Tensor2:
    """Derivative of the n-th principal invariant with respect to A.

    Uses the closed-form expressions from L02:
        ∂I₁/∂A = I
        ∂I₂/∂A = I₁ I − Aᵀ
        ∂I₃/∂A = I₃ A⁻ᵀ

    This is an alias for ``constkit.invariants.invariant_derivative``.
    """
    from constkit.invariants import invariant_derivative
    return invariant_derivative(n, A)


def symmetric_identity_4th() -> sp.Array:
    """Fourth-order symmetric identity tensor.

    I^s_{ijkl} = ½(δ_ik δ_jl + δ_il δ_jk)

    This is ∂A/∂A for a symmetric tensor A.
    """
    components = [
        [
            [
                [
                    sp.Rational(1, 2) * (
                        (1 if i == k else 0) * (1 if j == l else 0)
                        + (1 if i == l else 0) * (1 if j == k else 0)
                    )
                    for l in range(3)
                ]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)


def identity_4th() -> sp.Array:
    """Fourth-order identity tensor (non-symmetric).

    I_{ijkl} = δ_ik δ_jl

    This is ∂A/∂A for a general (non-symmetric) tensor A.
    """
    components = [
        [
            [
                [(1 if i == k and j == l else 0) for l in range(3)]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)


# ------------------------------------------------------------------
# Push-forward and pull-back
# ------------------------------------------------------------------

def push_forward(S: Tensor2, F: Tensor2) -> Tensor2:
    """Push-forward of a covariant 2-tensor from reference to current config.

    φ_*(S) = F^{-T} S F^{-1}

    This is the generic component transformation for a covariant
    second-order tensor. It is not the PK2-to-Kirchhoff stress mapping,
    which is τ = F S F^T.
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    Sm = S.matrix if isinstance(S, Tensor2) else sp.Matrix(S)
    F_inv = Fm.inv()
    return Tensor2(sp.simplify(F_inv.T * Sm * F_inv))


def pull_back(t: Tensor2, F: Tensor2) -> Tensor2:
    """Pull-back of a contravariant 2-tensor from current to reference config.

    φ*(t) = F^T t F

    This is the generic component transformation for a contravariant
    second-order tensor. It is not the Kirchhoff-to-PK2 stress mapping,
    which is S = F⁻¹ τ F⁻ᵀ.
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    tm = t.matrix if isinstance(t, Tensor2) else sp.Matrix(t)
    return Tensor2(sp.simplify(Fm.T * tm * Fm))


# ------------------------------------------------------------------
# Rank-1 push-forward / pull-back (vectors)
# ------------------------------------------------------------------

def push_forward_vector(v: sp.Matrix, F: Tensor2) -> sp.Matrix:
    """Push-forward of a contravariant vector (reference → current).

    v_spatial = F · v_material
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    vm = sp.Matrix(v).reshape(3, 1)
    return sp.simplify(Fm * vm)


def pull_back_vector(v: sp.Matrix, F: Tensor2) -> sp.Matrix:
    """Pull-back of a contravariant vector (current → reference).

    v_material = F⁻¹ · v_spatial
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    vm = sp.Matrix(v).reshape(3, 1)
    return sp.simplify(Fm.inv() * vm)


# ------------------------------------------------------------------
# Rank-2 push-forward / pull-back (contravariant and covariant)
# ------------------------------------------------------------------

def push_forward_tensor2(T: Tensor2, F: Tensor2) -> Tensor2:
    """Contravariant push-forward of a rank-2 tensor (reference → current).

    t = F · T · Fᵀ

    Maps a reference-config contravariant tensor to the current configuration.
    Used e.g. to map the 2nd PK stress S to the Kirchhoff stress τ = F S Fᵀ.
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    Tm = T.matrix if isinstance(T, Tensor2) else sp.Matrix(T)
    return Tensor2(sp.simplify(Fm * Tm * Fm.T))


def pull_back_tensor2(t: Tensor2, F: Tensor2) -> Tensor2:
    """Contravariant pull-back of a rank-2 tensor (current → reference).

    T = F⁻¹ · t · F⁻ᵀ

    Inverse of ``push_forward_tensor2``.
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    tm = t.matrix if isinstance(t, Tensor2) else sp.Matrix(t)
    F_inv = Fm.inv()
    return Tensor2(sp.simplify(F_inv * tm * F_inv.T))


def push_forward_covariant(T: Tensor2, F: Tensor2) -> Tensor2:
    """Covariant push-forward of a rank-2 tensor (reference → current).

    t = F⁻ᵀ · T · F⁻¹

    Alias for ``push_forward``.  Maps covariant components from reference
    to current configuration.
    """
    return push_forward(T, F)


def pull_back_covariant(t: Tensor2, F: Tensor2) -> Tensor2:
    """Covariant pull-back of a rank-2 tensor (current → reference).

    T = Fᵀ · t · F

    Alias for ``pull_back``.  Maps covariant components from current
    to reference configuration.
    """
    return pull_back(t, F)


# ------------------------------------------------------------------
# Rank-4 push-forward / pull-back (tangent moduli)
# ------------------------------------------------------------------

def push_forward_tensor4(C4: sp.Array, F: Tensor2) -> sp.Array:
    """Piola push-forward of a 4th-order tensor (reference → current).

    c_ijkl = (1/J) F_iI F_jJ F_kK F_lL C_IJKL

    Maps the reference-config material tangent moduli C to the spatial
    (Eulerian) tangent moduli c used in the linearised spatial BVP.

    Parameters
    ----------
    C4 : sp.Array
        Shape (3,3,3,3) — reference-config 4th-order tensor.
    F : Tensor2
        Deformation gradient.

    Returns
    -------
    sp.Array
        Shape (3,3,3,3) — spatial tangent moduli.
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    J = Fm.det()
    components = [
        [
            [
                [
                    1 / J * sum(
                        Fm[i, I] * Fm[j, J] * Fm[k, K] * Fm[l, L] * C4[I, J, K, L]
                        for I in range(3) for J in range(3)
                        for K in range(3) for L in range(3)
                    )
                    for l in range(3)
                ]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)


def pull_back_tensor4(c4: sp.Array, F: Tensor2) -> sp.Array:
    """Piola pull-back of a 4th-order tensor (current → reference).

    C_IJKL = J (F⁻¹)_Ii (F⁻¹)_Jj (F⁻¹)_Kk (F⁻¹)_Ll c_ijkl

    Inverse of ``push_forward_tensor4``.

    Parameters
    ----------
    c4 : sp.Array
        Shape (3,3,3,3) — spatial 4th-order tensor.
    F : Tensor2
        Deformation gradient.

    Returns
    -------
    sp.Array
        Shape (3,3,3,3) — reference-config tangent moduli.
    """
    Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
    jac = Fm.det()
    F_inv = Fm.inv()
    components = [
        [
            [
                [
                    jac * sum(
                        F_inv[I, i] * F_inv[J, j] * F_inv[K, k] * F_inv[L, l] * c4[i, j, k, l]
                        for i in range(3) for j in range(3)
                        for k in range(3) for l in range(3)
                    )
                    for L in range(3)
                ]
                for K in range(3)
            ]
            for J in range(3)
        ]
        for I in range(3)
    ]
    return sp.Array(components)


# ------------------------------------------------------------------
# Higher-order contractions
# ------------------------------------------------------------------

def double_contract_4_2(C4: sp.Array, A: Tensor2) -> Tensor2:
    r"""Double contraction of a 4th-order tensor with a rank-2 tensor.

    Contracts the last two indices of C4 with A:

    $$B_{ij} = C_{ijkl}\,A_{kl}$$

    This is the primary constitutive-law operation: given the material
    tangent moduli $\mathbb{C}$ (rank-4) and a strain-like tensor
    $\boldsymbol{\varepsilon}$ (rank-2), compute the stress
    $\boldsymbol{\sigma} = \mathbb{C} : \boldsymbol{\varepsilon}$.

    Parameters
    ----------
    C4 : sp.Array
        Shape (3, 3, 3, 3).
    A : Tensor2
        Rank-2 tensor to contract with.

    Returns
    -------
    Tensor2
        Result B where B_ij = Σ_{kl} C4[i,j,k,l] * A[k,l].
    """
    Am = A.matrix if isinstance(A, Tensor2) else sp.Matrix(A)
    result = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            val = sp.Integer(0)
            for k in range(3):
                for l in range(3):
                    val += C4[i, j, k, l] * Am[k, l]
            result[i, j] = val
    return Tensor2(result)


def double_contract_2_4(A: Tensor2, C4: sp.Array) -> Tensor2:
    r"""Double contraction of a rank-2 tensor with a 4th-order tensor.

    Contracts the first two indices of C4 with A:

    $$B_{kl} = A_{ij}\,C_{ijkl}$$

    This is the transpose-side constitutive operation. For a
    major-symmetric tangent moduli ($C_{ijkl} = C_{klij}$) this equals
    ``double_contract_4_2(C4, A)``.

    Parameters
    ----------
    A : Tensor2
        Rank-2 tensor to contract with.
    C4 : sp.Array
        Shape (3, 3, 3, 3).

    Returns
    -------
    Tensor2
        Result B where B_kl = Σ_{ij} A[i,j] * C4[i,j,k,l].
    """
    Am = A.matrix if isinstance(A, Tensor2) else sp.Matrix(A)
    result = sp.zeros(3, 3)
    for k in range(3):
        for l in range(3):
            val = sp.Integer(0)
            for i in range(3):
                for j in range(3):
                    val += Am[i, j] * C4[i, j, k, l]
            result[k, l] = val
    return Tensor2(result)


def double_contract_4_4(C4: sp.Array, D4: sp.Array) -> sp.Expr:
    r"""Full double contraction of two 4th-order tensors (→ scalar).

    Contracts all four shared indices:

    $$s = C_{ijkl}\,D_{ijkl}$$

    Generalises the rank-2 Frobenius product to rank-4 tensors.
    Useful for computing norms of tangent moduli or comparing
    stiffness tensors.

    Parameters
    ----------
    C4 : sp.Array
        Shape (3, 3, 3, 3).
    D4 : sp.Array
        Shape (3, 3, 3, 3).

    Returns
    -------
    sp.Expr
        Scalar result.
    """
    val = sp.Integer(0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val += C4[i, j, k, l] * D4[i, j, k, l]
    return val


# ------------------------------------------------------------------
# Index raising / lowering via metric tensor
# ------------------------------------------------------------------

def raise_index_vector(v: sp.Matrix, g_contra: sp.Matrix) -> sp.Matrix:
    """Raise the index of a covariant vector: v^i = g^ij v_j."""
    gm = sp.Matrix(g_contra)
    vm = sp.Matrix(v).reshape(3, 1)
    return sp.simplify(gm * vm)


def lower_index_vector(v: sp.Matrix, g_cov: sp.Matrix) -> sp.Matrix:
    """Lower the index of a contravariant vector: v_i = g_ij v^j."""
    gm = sp.Matrix(g_cov)
    vm = sp.Matrix(v).reshape(3, 1)
    return sp.simplify(gm * vm)


def raise_index(T_cov: Tensor2, g_contra: sp.Matrix) -> Tensor2:
    """Raise both indices of a covariant rank-2 tensor.

    T^ij = g^ik g^jl T_kl

    Equivalent to g_contra @ T_cov @ g_contra (since g is symmetric).

    Parameters
    ----------
    T_cov : Tensor2
        Covariant rank-2 tensor with both lower indices.
    g_contra : sp.Matrix
        Contravariant (inverse) metric tensor g^ij.

    Returns
    -------
    Tensor2
        Contravariant tensor T^ij.
    """
    gm = sp.Matrix(g_contra)
    Tm = T_cov.matrix if isinstance(T_cov, Tensor2) else sp.Matrix(T_cov)
    return Tensor2(sp.simplify(gm * Tm * gm))


def lower_index(T_cont: Tensor2, g_cov: sp.Matrix) -> Tensor2:
    """Lower both indices of a contravariant rank-2 tensor.

    T_ij = g_ik g_jl T^kl

    Equivalent to g_cov @ T_cont @ g_cov (since g is symmetric).

    Parameters
    ----------
    T_cont : Tensor2
        Contravariant rank-2 tensor with both upper indices.
    g_cov : sp.Matrix
        Covariant metric tensor g_ij.

    Returns
    -------
    Tensor2
        Covariant tensor T_ij.
    """
    gm = sp.Matrix(g_cov)
    Tm = T_cont.matrix if isinstance(T_cont, Tensor2) else sp.Matrix(T_cont)
    return Tensor2(sp.simplify(gm * Tm * gm))


def raise_one_index(T: Tensor2, g_contra: sp.Matrix, which: str = "first") -> Tensor2:
    """Raise a single index of a mixed rank-2 tensor.

    Parameters
    ----------
    T : Tensor2
        Rank-2 tensor.
    g_contra : sp.Matrix
        Contravariant metric tensor g^ij.
    which : str
        ``'first'``  — raise the first index: T^i_j = g^ik T_kj
        ``'second'`` — raise the second index: T_i^j = T_ik g^kj

    Returns
    -------
    Tensor2
        Mixed tensor with one raised index.
    """
    gm = sp.Matrix(g_contra)
    Tm = T.matrix if isinstance(T, Tensor2) else sp.Matrix(T)
    if which == "first":
        return Tensor2(sp.simplify(gm * Tm))
    elif which == "second":
        return Tensor2(sp.simplify(Tm * gm))
    else:
        raise ValueError(f"which must be 'first' or 'second', got '{which}'")


def raise_index_tensor4(C4: sp.Array, g_contra: sp.Matrix) -> sp.Array:
    """Raise all four indices of a covariant 4th-order tensor.

    C^{ijkl} = g^{iI} g^{jJ} g^{kK} g^{lL} C_{IJKL}

    Parameters
    ----------
    C4 : sp.Array
        Shape (3,3,3,3) — fully covariant 4th-order tensor.
    g_contra : sp.Matrix
        Contravariant (inverse) metric tensor g^ij.

    Returns
    -------
    sp.Array
        Shape (3,3,3,3) — fully contravariant 4th-order tensor.
    """
    gm = sp.Matrix(g_contra)
    components = [
        [
            [
                [
                    sum(
                        gm[i, I] * gm[j, J] * gm[k, K] * gm[l, L] * C4[I, J, K, L]
                        for I in range(3) for J in range(3)
                        for K in range(3) for L in range(3)
                    )
                    for l in range(3)
                ]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)


def lower_index_tensor4(c4: sp.Array, g_cov: sp.Matrix) -> sp.Array:
    """Lower all four indices of a contravariant 4th-order tensor.

    C_{ijkl} = g_{iI} g_{jJ} g_{kK} g_{lL} C^{IJKL}

    Parameters
    ----------
    c4 : sp.Array
        Shape (3,3,3,3) — fully contravariant 4th-order tensor.
    g_cov : sp.Matrix
        Covariant metric tensor g_ij.

    Returns
    -------
    sp.Array
        Shape (3,3,3,3) — fully covariant 4th-order tensor.
    """
    gm = sp.Matrix(g_cov)
    components = [
        [
            [
                [
                    sum(
                        gm[i, I] * gm[j, J] * gm[k, K] * gm[l, L] * c4[I, J, K, L]
                        for I in range(3) for J in range(3)
                        for K in range(3) for L in range(3)
                    )
                    for l in range(3)
                ]
                for k in range(3)
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)
