"""Index notation helpers: Kronecker delta and Levi-Civita symbol.

These are pedagogical tools for understanding Einstein summation
and index-notation proofs.
"""

from __future__ import annotations

import sympy as sp


def kronecker(i: int, j: int) -> int:
    r"""Kronecker delta: δ_ij.

    Returns 1 if i == j, else 0.

    Parameters
    ----------
    i, j : int
        Indices (0-based or 1-based — the function only checks equality).
    """
    return 1 if i == j else 0


def levi_civita(i: int, j: int, k: int) -> int:
    r"""Levi-Civita permutation symbol: ε_ijk.

    Returns +1 for even permutations of (1,2,3), −1 for odd, 0 if
    any two indices are equal.

    Parameters
    ----------
    i, j, k : int
        Indices (1-based: values in {1, 2, 3}).

    Examples
    --------
    >>> levi_civita(1, 2, 3)
    1
    >>> levi_civita(2, 1, 3)
    -1
    >>> levi_civita(1, 1, 2)
    0
    """
    if i == j or j == k or i == k:
        return 0
    # Even permutations of (1,2,3): (1,2,3), (2,3,1), (3,1,2)
    if (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
        return 1
    return -1


def epsilon_delta_identity(j: int, k: int, m: int, n: int) -> int:
    r"""Epsilon-delta identity: ε_ijk ε_imn = δ_jm δ_kn − δ_jn δ_km.

    This contracts the Levi-Civita symbol over one shared index.

    Parameters
    ----------
    j, k, m, n : int
        Free indices (1-based).

    Returns
    -------
    int
        Value of δ_jm δ_kn − δ_jn δ_km.
    """
    return kronecker(j, m) * kronecker(k, n) - kronecker(j, n) * kronecker(k, m)


def kronecker_matrix() -> sp.Matrix:
    r"""Return the 3×3 Kronecker delta as a SymPy identity matrix."""
    return sp.eye(3)


def levi_civita_tensor() -> sp.Array:
    r"""Return the full 3×3×3 Levi-Civita tensor as a SymPy Array.

    Components use 0-based indexing internally but values correspond
    to the standard 1-based definition.
    """
    components = [
        [
            [levi_civita(i + 1, j + 1, k + 1) for k in range(3)]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return sp.Array(components)
