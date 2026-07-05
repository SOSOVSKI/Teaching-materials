"""Curvilinear coordinate systems: basis vectors, metric tensor, Christoffel symbols.

Provides symbolic machinery for cylindrical and spherical coordinates
as covered in L01 (curvilinear coordinates section) and HW1 Problem 3.
"""

from __future__ import annotations

import sympy as sp


# ------------------------------------------------------------------
# Cylindrical coordinates (r, φ, z)
# ------------------------------------------------------------------

def cylindrical_basis(r, phi) -> list[sp.Matrix]:
    """Covariant basis vectors for cylindrical coordinates (r, φ, z).

    The mapping is x¹ = r cos φ, x² = r sin φ, x³ = z.

    Parameters
    ----------
    r, phi : sp.Expr or float
        Coordinate values (may be symbolic).

    Returns
    -------
    list of sp.Matrix
        [g_r, g_φ, g_z] — three 3×1 column vectors.
    """
    g_r = sp.Matrix([sp.cos(phi), sp.sin(phi), 0])
    g_phi = sp.Matrix([-r * sp.sin(phi), r * sp.cos(phi), 0])
    g_z = sp.Matrix([0, 0, 1])
    return [g_r, g_phi, g_z]


# ------------------------------------------------------------------
# Spherical coordinates (R, θ, φ)
# ------------------------------------------------------------------

def spherical_basis(R, theta, phi) -> list[sp.Matrix]:
    """Covariant basis vectors for spherical coordinates (R, θ, φ).

    The mapping is x¹ = R sin θ cos φ, x² = R sin θ sin φ, x³ = R cos θ.

    Parameters
    ----------
    R, theta, phi : sp.Expr or float
        Coordinate values.

    Returns
    -------
    list of sp.Matrix
        [g_R, g_θ, g_φ] — three 3×1 column vectors.
    """
    st, ct = sp.sin(theta), sp.cos(theta)
    sp_, cp = sp.sin(phi), sp.cos(phi)

    g_R = sp.Matrix([st * cp, st * sp_, ct])
    g_theta = sp.Matrix([R * ct * cp, R * ct * sp_, -R * st])
    g_phi = sp.Matrix([-R * st * sp_, R * st * cp, 0])
    return [g_R, g_theta, g_phi]


# ------------------------------------------------------------------
# General tools
# ------------------------------------------------------------------

def metric_tensor(g_cov: list[sp.Matrix]) -> sp.Matrix:
    """Metric tensor g_ij = g_i · g_j from covariant basis vectors.

    Parameters
    ----------
    g_cov : list of sp.Matrix
        Three covariant basis vectors [g_1, g_2, g_3].

    Returns
    -------
    sp.Matrix
        3×3 metric tensor (symmetric, positive-definite).
    """
    g = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            g[i, j] = sp.trigsimp(g_cov[i].dot(g_cov[j]))
    return g


def contravariant_basis(g_cov: list[sp.Matrix]) -> list[sp.Matrix]:
    """Contravariant (reciprocal) basis vectors satisfying g_i · g^j = δ^j_i.

    Computed via: g^i = (g_j × g_k) / (g_1 · (g_2 × g_3))
    with cyclic permutations.

    Parameters
    ----------
    g_cov : list of sp.Matrix
        Three covariant basis vectors.

    Returns
    -------
    list of sp.Matrix
        [g^1, g^2, g^3] — three 3×1 column vectors.
    """
    g1, g2, g3 = g_cov
    triple = sp.trigsimp(g1.dot(g2.cross(g3)))
    if triple == 0:
        raise ValueError("Basis vectors are linearly dependent (triple product = 0).")

    g1_up = sp.trigsimp(g2.cross(g3) / triple)
    g2_up = sp.trigsimp(g3.cross(g1) / triple)
    g3_up = sp.trigsimp(g1.cross(g2) / triple)
    return [g1_up, g2_up, g3_up]


def verify_biorthogonality(g_cov: list[sp.Matrix], g_contra: list[sp.Matrix]) -> bool:
    """Verify that g_i · g^j = δ^j_i for all i, j.

    Returns True if all 9 dot products match the Kronecker delta.
    Raises AssertionError with details if any product is wrong.
    """
    for i in range(3):
        for j in range(3):
            expected = 1 if i == j else 0
            actual = sp.trigsimp(g_cov[i].dot(g_contra[j]))
            if actual != expected:
                raise AssertionError(
                    f"Biorthogonality failed: g_{i+1} · g^{j+1} = {actual}, "
                    f"expected {expected}"
                )
    return True


def covariant_derivative_vector(
    v: sp.Matrix,
    christoffel: sp.Array,
    coord_symbols: list[sp.Symbol],
    variant: str = "contravariant",
) -> sp.Array:
    r"""Covariant derivative of a vector field.

    Parameters
    ----------
    v : sp.Matrix
        3×1 column vector of components.
    christoffel : sp.Array
        Christoffel symbols of the second kind, shape (3, 3, 3).
        ``christoffel[k, i, j]`` = Γ^k_ij  (as returned by
        ``christoffel_symbols``).
    coord_symbols : list of sp.Symbol
        Coordinate symbols [θ¹, θ², θ³].
    variant : str
        ``'contravariant'`` (default) for an upper-index vector v^j:
            (∇_i v)^j = ∂v^j/∂θ^i + Γ^j_ik v^k
        ``'covariant'`` for a lower-index covector v_j:
            (∇_i v)_j = ∂v_j/∂θ^i − Γ^k_ij v_k

    Returns
    -------
    sp.Array
        Shape (3, 3) where ``result[i, j]`` = (∇_i v)[j].
    """
    v = sp.Matrix(v).reshape(3, 1)
    result = [[sp.Integer(0)] * 3 for _ in range(3)]

    if variant == "contravariant":
        for i in range(3):
            for j in range(3):
                val = sp.diff(v[j], coord_symbols[i])
                for k in range(3):
                    val += christoffel[j, i, k] * v[k]
                result[i][j] = val
    elif variant == "covariant":
        for i in range(3):
            for j in range(3):
                val = sp.diff(v[j], coord_symbols[i])
                for k in range(3):
                    val -= christoffel[k, i, j] * v[k]
                result[i][j] = val
    else:
        raise ValueError(f"variant must be 'contravariant' or 'covariant', got '{variant}'")

    return sp.Array(result)


def covariant_derivative_tensor2(
    T: sp.Matrix,
    christoffel: sp.Array,
    coord_symbols: list[sp.Symbol],
    variant: str = "contravariant",
) -> sp.Array:
    r"""Covariant derivative of a rank-2 tensor field.

    Parameters
    ----------
    T : sp.Matrix or Tensor2
        3×3 matrix of components.
    christoffel : sp.Array
        Christoffel symbols, shape (3, 3, 3). ``christoffel[k, i, j]`` = Γ^k_ij.
    coord_symbols : list of sp.Symbol
        Coordinate symbols [θ¹, θ², θ³].
    variant : str
        Index type of T:

        - ``'contravariant'`` (default) — T^ij upper indices:
            (∇_k T)^ij = ∂T^ij/∂θ^k + Γ^i_km T^mj + Γ^j_km T^im
        - ``'covariant'`` — T_ij lower indices:
            (∇_k T)_ij = ∂T_ij/∂θ^k − Γ^m_ki T_mj − Γ^m_kj T_im
        - ``'mixed'`` — T^i_j mixed indices:
            (∇_k T)^i_j = ∂T^i_j/∂θ^k + Γ^i_km T^m_j − Γ^m_kj T^i_m

    Returns
    -------
    sp.Array
        Shape (3, 3, 3) where ``result[k, i, j]`` = (∇_k T)[i, j].
    """
    from constkit.tensor2 import Tensor2 as _Tensor2
    M = T.matrix if isinstance(T, _Tensor2) else sp.Matrix(T)

    result = [[[sp.Integer(0)] * 3 for _ in range(3)] for _ in range(3)]

    if variant == "contravariant":
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    val = sp.diff(M[i, j], coord_symbols[k])
                    for m in range(3):
                        val += christoffel[i, k, m] * M[m, j]
                        val += christoffel[j, k, m] * M[i, m]
                    result[k][i][j] = val
    elif variant == "covariant":
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    val = sp.diff(M[i, j], coord_symbols[k])
                    for m in range(3):
                        val -= christoffel[m, k, i] * M[m, j]
                        val -= christoffel[m, k, j] * M[i, m]
                    result[k][i][j] = val
    elif variant == "mixed":
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    val = sp.diff(M[i, j], coord_symbols[k])
                    for m in range(3):
                        val += christoffel[i, k, m] * M[m, j]
                        val -= christoffel[m, k, j] * M[i, m]
                    result[k][i][j] = val
    else:
        raise ValueError(
            f"variant must be 'contravariant', 'covariant', or 'mixed', got '{variant}'"
        )

    return sp.Array(result)


def christoffel_symbols(g_cov: list[sp.Matrix], coords: list[sp.Symbol]) -> sp.Array:
    r"""Christoffel symbols of the second kind: Γ^k_{ij}.

    Computed from the metric tensor:
        Γ^k_{ij} = ½ g^{km} (∂g_{mj}/∂θ^i + ∂g_{im}/∂θ^j − ∂g_{ij}/∂θ^m)

    Parameters
    ----------
    g_cov : list of sp.Matrix
        Covariant basis vectors.
    coords : list of sp.Symbol
        Coordinate symbols [θ¹, θ², θ³].

    Returns
    -------
    sp.Array
        3×3×3 array where ``result[k, i, j]`` = Γ^k_{ij}.
    """
    g = metric_tensor(g_cov)
    g_inv = g.inv()

    gamma = [[[sp.Integer(0) for _ in range(3)] for _ in range(3)] for _ in range(3)]

    for k in range(3):
        for i in range(3):
            for j in range(3):
                val = sp.Integer(0)
                for m in range(3):
                    dg_mj = sp.diff(g[m, j], coords[i])
                    dg_im = sp.diff(g[i, m], coords[j])
                    dg_ij = sp.diff(g[i, j], coords[m])
                    val += sp.Rational(1, 2) * g_inv[k, m] * (dg_mj + dg_im - dg_ij)
                gamma[k][i][j] = sp.trigsimp(val)

    return sp.Array(gamma)
