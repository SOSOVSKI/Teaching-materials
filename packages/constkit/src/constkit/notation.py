"""Voigt and Mandel notation for symmetric tensors and elasticity tensors.

Ordering convention (used by both Voigt and Mandel):
    index 0 → (1,1),  index 1 → (2,2),  index 2 → (3,3)
    index 3 → (2,3),  index 4 → (1,3),  index 5 → (1,2)

    i.e. rows/cols of the 6-vector: 11 22 33 23 13 12

Voigt notation:
    - Rank-2: v_α = T_ij   (no scaling)
    - Rank-4: M_αβ = C_ijkl  (no scaling)
    - Does NOT preserve the inner product T:S = vᵀ w in general.

Mandel notation:
    - Rank-2: m_α = T_ij  for α ∈ {0,1,2} (diagonal)
               m_α = √2 T_ij  for α ∈ {3,4,5} (off-diagonal)
    - Rank-4: M_αβ = C_ijkl scaled by appropriate √2 or 2 factors
    - Preserves: T:S = mᵀ n  (Frobenius inner product preserved).
"""

from __future__ import annotations

import sympy as sp

from constkit.tensor2 import Tensor2


# Voigt / Mandel index map: α → (i, j)   (0-based)
_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

# √2 scale factor for off-diagonal Mandel components
_SQRT2 = sp.sqrt(2)


def _scale_mandel(alpha: int) -> sp.Expr:
    """Return the Mandel scaling factor for index α: 1 for α<3, √2 for α≥3."""
    return sp.Integer(1) if alpha < 3 else _SQRT2


# ------------------------------------------------------------------
# Rank-2 tensor ↔ Voigt vector
# ------------------------------------------------------------------

def to_voigt(T: Tensor2) -> sp.Matrix:
    """Convert a symmetric rank-2 tensor to a 6-component Voigt vector.

    v = [T_11, T_22, T_33, T_23, T_13, T_12]^T

    Parameters
    ----------
    T : Tensor2
        Symmetric 3×3 tensor. Off-diagonal entries are taken from the
        upper triangle; symmetry is assumed but not enforced.

    Returns
    -------
    sp.Matrix
        Shape (6, 1) column vector.
    """
    m = T.matrix if isinstance(T, Tensor2) else sp.Matrix(T)
    return sp.Matrix([m[i, j] for (i, j) in _PAIRS])


def from_voigt(v: sp.Matrix) -> Tensor2:
    """Reconstruct a symmetric Tensor2 from a 6-component Voigt vector.

    Parameters
    ----------
    v : sp.Matrix
        Shape (6,) or (6,1) vector in Voigt ordering.

    Returns
    -------
    Tensor2
        Symmetric 3×3 tensor.
    """
    v = sp.Matrix(v).reshape(6, 1)
    m = sp.zeros(3, 3)
    for alpha, (i, j) in enumerate(_PAIRS):
        m[i, j] = v[alpha]
        m[j, i] = v[alpha]
    return Tensor2(m)


# ------------------------------------------------------------------
# Rank-4 tensor ↔ Voigt matrix
# ------------------------------------------------------------------

def to_voigt_matrix(C4: sp.Array) -> sp.Matrix:
    """Convert a symmetric rank-4 tensor to a 6×6 Voigt matrix.

    M_αβ = C_{ijkl}  where (i,j) ↔ α and (k,l) ↔ β.

    Parameters
    ----------
    C4 : sp.Array
        Shape (3,3,3,3) rank-4 tensor.

    Returns
    -------
    sp.Matrix
        Shape (6,6) Voigt stiffness matrix.
    """
    M = sp.zeros(6, 6)
    for alpha, (i, j) in enumerate(_PAIRS):
        for beta, (k, l) in enumerate(_PAIRS):
            M[alpha, beta] = C4[i, j, k, l]
    return M


def from_voigt_matrix(M: sp.Matrix) -> sp.Array:
    """Reconstruct a symmetric rank-4 tensor from a 6×6 Voigt matrix.

    Assumes major and minor symmetries: C_ijkl = C_jikl = C_ijlk = C_klij.

    Parameters
    ----------
    M : sp.Matrix
        Shape (6,6) Voigt stiffness matrix.

    Returns
    -------
    sp.Array
        Shape (3,3,3,3) fourth-order tensor.
    """
    M = sp.Matrix(M)
    components = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
    for alpha, (i, j) in enumerate(_PAIRS):
        for beta, (k, l) in enumerate(_PAIRS):
            val = M[alpha, beta]
            for ii, jj in [(i, j), (j, i)]:
                for kk, ll in [(k, l), (l, k)]:
                    components[ii, jj, kk, ll] = val
    return sp.Array(components)


# ------------------------------------------------------------------
# Rank-2 tensor ↔ Mandel vector
# ------------------------------------------------------------------

def to_mandel(T: Tensor2) -> sp.Matrix:
    """Convert a symmetric rank-2 tensor to a 6-component Mandel vector.

    m_α = T_ij  for α ∈ {0,1,2}  (diagonal)
    m_α = √2 T_ij  for α ∈ {3,4,5}  (off-diagonal)

    This scaling ensures T:S = mᵀ n (inner product preserved).

    Parameters
    ----------
    T : Tensor2
        Symmetric 3×3 tensor.

    Returns
    -------
    sp.Matrix
        Shape (6, 1) column vector.
    """
    m = T.matrix if isinstance(T, Tensor2) else sp.Matrix(T)
    return sp.Matrix([_scale_mandel(alpha) * m[i, j]
                      for alpha, (i, j) in enumerate(_PAIRS)])


def from_mandel(v: sp.Matrix) -> Tensor2:
    """Reconstruct a symmetric Tensor2 from a 6-component Mandel vector.

    Parameters
    ----------
    v : sp.Matrix
        Shape (6,) or (6,1) vector in Mandel ordering.

    Returns
    -------
    Tensor2
        Symmetric 3×3 tensor.
    """
    v = sp.Matrix(v).reshape(6, 1)
    m = sp.zeros(3, 3)
    for alpha, (i, j) in enumerate(_PAIRS):
        val = v[alpha] / _scale_mandel(alpha)
        m[i, j] = val
        m[j, i] = val
    return Tensor2(m)


# ------------------------------------------------------------------
# Rank-4 tensor ↔ Mandel matrix
# ------------------------------------------------------------------

def to_mandel_matrix(C4: sp.Array) -> sp.Matrix:
    """Convert a symmetric rank-4 tensor to a 6×6 Mandel matrix.

    M_αβ = s_α s_β C_{ijkl}

    where s_α = 1 for α < 3 (diagonal) and s_α = √2 for α ≥ 3 (off-diagonal).

    This preserves the double contraction: A:C:B = m_A^T M m_B.

    Parameters
    ----------
    C4 : sp.Array
        Shape (3,3,3,3) rank-4 tensor with at least minor symmetry.

    Returns
    -------
    sp.Matrix
        Shape (6,6) Mandel stiffness matrix.
    """
    M = sp.zeros(6, 6)
    for alpha, (i, j) in enumerate(_PAIRS):
        for beta, (k, l) in enumerate(_PAIRS):
            M[alpha, beta] = _scale_mandel(alpha) * _scale_mandel(beta) * C4[i, j, k, l]
    return M


def from_mandel_matrix(M: sp.Matrix) -> sp.Array:
    """Reconstruct a symmetric rank-4 tensor from a 6×6 Mandel matrix.

    Assumes major and minor symmetries.

    Parameters
    ----------
    M : sp.Matrix
        Shape (6,6) Mandel stiffness matrix.

    Returns
    -------
    sp.Array
        Shape (3,3,3,3) fourth-order tensor.
    """
    M = sp.Matrix(M)
    components = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
    for alpha, (i, j) in enumerate(_PAIRS):
        for beta, (k, l) in enumerate(_PAIRS):
            val = M[alpha, beta] / (_scale_mandel(alpha) * _scale_mandel(beta))
            for ii, jj in [(i, j), (j, i)]:
                for kk, ll in [(k, l), (l, k)]:
                    components[ii, jj, kk, ll] = val
    return sp.Array(components)


# ------------------------------------------------------------------
# Voigt ↔ Mandel conversions (vector)
# ------------------------------------------------------------------

def voigt_to_mandel(v: sp.Matrix) -> sp.Matrix:
    """Convert a Voigt 6-vector to Mandel notation.

    Multiplies off-diagonal components (indices 3,4,5) by √2.

    Parameters
    ----------
    v : sp.Matrix
        Shape (6,) or (6,1) Voigt vector.

    Returns
    -------
    sp.Matrix
        Shape (6, 1) Mandel vector.
    """
    v = sp.Matrix(v).reshape(6, 1)
    return sp.Matrix([_scale_mandel(alpha) * v[alpha] for alpha in range(6)])


def mandel_to_voigt(v: sp.Matrix) -> sp.Matrix:
    """Convert a Mandel 6-vector to Voigt notation.

    Divides off-diagonal components (indices 3,4,5) by √2.

    Parameters
    ----------
    v : sp.Matrix
        Shape (6,) or (6,1) Mandel vector.

    Returns
    -------
    sp.Matrix
        Shape (6, 1) Voigt vector.
    """
    v = sp.Matrix(v).reshape(6, 1)
    return sp.Matrix([v[alpha] / _scale_mandel(alpha) for alpha in range(6)])


# ------------------------------------------------------------------
# Voigt ↔ Mandel conversions (matrix)
# ------------------------------------------------------------------

def voigt_matrix_to_mandel(C: sp.Matrix) -> sp.Matrix:
    """Convert a 6×6 Voigt stiffness matrix to Mandel notation.

    M_αβ = s_α s_β V_αβ  where s_α = 1 (α<3) or √2 (α≥3).

    Parameters
    ----------
    C : sp.Matrix
        Shape (6,6) Voigt stiffness matrix.

    Returns
    -------
    sp.Matrix
        Shape (6,6) Mandel stiffness matrix.
    """
    C = sp.Matrix(C)
    M = sp.zeros(6, 6)
    for alpha in range(6):
        for beta in range(6):
            M[alpha, beta] = _scale_mandel(alpha) * _scale_mandel(beta) * C[alpha, beta]
    return M


def mandel_matrix_to_voigt(M: sp.Matrix) -> sp.Matrix:
    """Convert a 6×6 Mandel stiffness matrix to Voigt notation.

    V_αβ = M_αβ / (s_α s_β).

    Parameters
    ----------
    M : sp.Matrix
        Shape (6,6) Mandel stiffness matrix.

    Returns
    -------
    sp.Matrix
        Shape (6,6) Voigt stiffness matrix.
    """
    M = sp.Matrix(M)
    V = sp.zeros(6, 6)
    for alpha in range(6):
        for beta in range(6):
            V[alpha, beta] = M[alpha, beta] / (_scale_mandel(alpha) * _scale_mandel(beta))
    return V
