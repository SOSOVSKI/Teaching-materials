"""Rotation matrices and coordinate transformation utilities."""

from __future__ import annotations

import sympy as sp


def rotation_matrix(angle, axis: int = 3) -> sp.Matrix:
    """Rotation matrix about a coordinate axis.

    Parameters
    ----------
    angle : sp.Expr or float
        Rotation angle (counterclockwise, in radians).
    axis : int
        Axis of rotation: 1 (x), 2 (y), or 3 (z).

    Returns
    -------
    sp.Matrix
        3×3 orthogonal rotation matrix Q with det(Q) = +1.
    """
    c = sp.cos(angle)
    s = sp.sin(angle)
    if axis == 3:
        return sp.Matrix([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1],
        ])
    elif axis == 1:
        return sp.Matrix([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c],
        ])
    elif axis == 2:
        return sp.Matrix([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c],
        ])
    else:
        raise ValueError(f"axis must be 1, 2, or 3, got {axis}")


def rotation_matrix_axis_angle(axis_vec: sp.Matrix, angle) -> sp.Matrix:
    """Rotation matrix from axis-angle (Rodrigues' formula).

    Parameters
    ----------
    axis_vec : sp.Matrix
        3×1 unit vector defining the rotation axis.
    angle : sp.Expr
        Rotation angle in radians.

    Returns
    -------
    sp.Matrix
        3×3 rotation matrix.
    """
    n = sp.Matrix(axis_vec).reshape(3, 1)
    # Skew-symmetric matrix of n
    N = sp.Matrix([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0],
    ])
    I3 = sp.eye(3)
    return I3 + sp.sin(angle) * N + (1 - sp.cos(angle)) * N * N
