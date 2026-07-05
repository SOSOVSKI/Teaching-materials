"""Symbolic vector operations in 3D.

All operations work with SymPy Matrix column vectors (3×1).
"""

from __future__ import annotations

import sympy as sp


def _as_vec(v) -> sp.Matrix:
    """Coerce input to a 3×1 SymPy column vector."""
    if isinstance(v, sp.Matrix):
        if v.shape == (3, 1):
            return v
        if v.shape == (1, 3):
            return v.T
        if v.shape == (3,):
            return v.reshape(3, 1)
    return sp.Matrix(v).reshape(3, 1)


def dot(u, v) -> sp.Expr:
    """Inner (dot) product: u · v = u_i v_i."""
    u, v = _as_vec(u), _as_vec(v)
    return (u.T * v)[0, 0]


def cross(u, v) -> sp.Matrix:
    """Cross product: u × v.

    Returns a 3×1 column vector.
    """
    u, v = _as_vec(u), _as_vec(v)
    return u.cross(v)


def norm(v) -> sp.Expr:
    """Euclidean norm: |v| = √(v · v)."""
    v = _as_vec(v)
    return sp.sqrt(dot(v, v))


def outer_product(u, v) -> sp.Matrix:
    """Outer (dyadic) product: (u ⊗ v)_ij = u_i v_j.

    Returns a 3×3 matrix.
    """
    u, v = _as_vec(u), _as_vec(v)
    return u * v.T


def outer_vec_tensor2(v, A) -> sp.Array:
    """Outer product of a vector with a rank-2 tensor: (v ⊗ A)_ijk = v_i A_jk.

    Parameters
    ----------
    v : array-like
        3-component vector.
    A : Tensor2 or sp.Matrix
        3×3 rank-2 tensor.

    Returns
    -------
    sp.Array
        Shape (3, 3, 3) where ``result[i, j, k]`` = v_i A_jk.
    """
    from constkit.tensor2 import Tensor2
    v = _as_vec(v)
    Am = A.matrix if isinstance(A, Tensor2) else sp.Matrix(A)
    components = [
        [[v[i] * Am[j, k] for k in range(3)] for j in range(3)]
        for i in range(3)
    ]
    return sp.Array(components)


def triple_product(u, v, w) -> sp.Expr:
    """Scalar triple product: [u, v, w] = u · (v × w).

    Geometrically, this is the signed volume of the parallelepiped
    spanned by u, v, w.
    """
    u, v, w = _as_vec(u), _as_vec(v), _as_vec(w)
    return dot(u, cross(v, w))


def angle(u, v) -> sp.Expr:
    """Angle between two vectors: θ = arccos(u·v / |u||v|)."""
    return sp.acos(dot(u, v) / (norm(u) * norm(v)))
