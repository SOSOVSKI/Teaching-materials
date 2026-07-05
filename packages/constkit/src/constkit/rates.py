"""Objective stress rates: Jaumann, Oldroyd, Truesdell.

These rates correct the material time derivative of an objective tensor
so that the result remains objective under superposed rigid motions (L02).
"""

from __future__ import annotations

import sympy as sp

from constkit.tensor2 import Tensor2


def _mat(T):
    return T.matrix if isinstance(T, Tensor2) else sp.Matrix(T)


def jaumann_rate(
    T: Tensor2,
    T_dot: Tensor2,
    W: Tensor2,
) -> Tensor2:
    r"""Jaumann (corotational) rate.

    .. math::
        \overset{\triangledown}{\mathbf{T}} = \dot{\mathbf{T}}
        - \mathbf{W}\mathbf{T} + \mathbf{T}\mathbf{W}

    Parameters
    ----------
    T : Tensor2
        The objective tensor (e.g., Cauchy stress σ).
    T_dot : Tensor2
        Material time derivative of T.
    W : Tensor2
        Spin tensor (skew-symmetric part of velocity gradient).
    """
    Tm, Td, Wm = _mat(T), _mat(T_dot), _mat(W)
    result = Td - Wm * Tm + Tm * Wm
    return Tensor2(sp.simplify(result), name="T̊_J")


def oldroyd_rate(
    T: Tensor2,
    T_dot: Tensor2,
    L: Tensor2,
) -> Tensor2:
    r"""Oldroyd (upper-convected) rate.

    .. math::
        \overset{\triangle}{\mathbf{T}} = \dot{\mathbf{T}}
        - \mathbf{L}\mathbf{T} - \mathbf{T}\mathbf{L}^T

    Parameters
    ----------
    T : Tensor2
        The objective tensor.
    T_dot : Tensor2
        Material time derivative of T.
    L : Tensor2
        Velocity gradient.
    """
    Tm, Td, Lm = _mat(T), _mat(T_dot), _mat(L)
    result = Td - Lm * Tm - Tm * Lm.T
    return Tensor2(sp.simplify(result), name="T̊_O")


def truesdell_rate(
    T: Tensor2,
    T_dot: Tensor2,
    L: Tensor2,
) -> Tensor2:
    r"""Truesdell rate.

    .. math::
        \overset{\circ}{\mathbf{T}} = \dot{\mathbf{T}}
        - \mathbf{L}\mathbf{T} - \mathbf{T}\mathbf{L}^T
        + (\mathrm{tr}\,\mathbf{L})\,\mathbf{T}

    Parameters
    ----------
    T : Tensor2
        The objective tensor.
    T_dot : Tensor2
        Material time derivative of T.
    L : Tensor2
        Velocity gradient.
    """
    Tm, Td, Lm = _mat(T), _mat(T_dot), _mat(L)
    result = Td - Lm * Tm - Tm * Lm.T + Lm.trace() * Tm
    return Tensor2(sp.simplify(result), name="T̊_Tr")
