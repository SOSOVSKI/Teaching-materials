"""Symbolic fourth-order tensor in 3D.

Wraps a (3,3,3,3) SymPy Array with methods mirroring continuum-mechanics
notation: double contraction, push-forward/pull-back, etc.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from constkit.tensor2 import Tensor2


class Tensor4:
    """A symbolic fourth-order tensor in 3D Cartesian coordinates.

    Parameters
    ----------
    array : sp.Array or array-like
        Shape (3,3,3,3) array of components.
    name : str, optional
        Display name (used in LaTeX rendering).

    Examples
    --------
    >>> import sympy as sp
    >>> from constkit import Tensor4
    >>> C = Tensor4(sp.zeros(3, 3, 3, 3), name='C')
    """

    def __init__(self, array: Any, name: str = "") -> None:
        if isinstance(array, sp.Array):
            self._a = sp.MutableDenseNDimArray(array)
        elif isinstance(array, sp.MutableDenseNDimArray):
            self._a = array
        else:
            self._a = sp.MutableDenseNDimArray(array)
        if self._a.shape != (3, 3, 3, 3):
            raise ValueError(
                f"Tensor4 requires a (3,3,3,3) array, got {self._a.shape}"
            )
        self._name = name

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Display name used in LaTeX rendering. Mutable."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def array(self) -> sp.Array:
        """Underlying (3,3,3,3) SymPy Array."""
        return sp.Array(self._a)

    def __getitem__(self, idx: tuple) -> sp.Expr:
        i, j, k, l = idx
        return self._a[i, j, k, l]

    # ------------------------------------------------------------------
    # Contractions
    # ------------------------------------------------------------------

    def double_contract(self, B: Tensor2) -> Tensor2:
        """Right double contraction with a rank-2 tensor: (C:A)_ij = C_ijkl A_kl.

        Parameters
        ----------
        B : Tensor2
            The rank-2 tensor to contract from the right.

        Returns
        -------
        Tensor2
            Result B_ij = C_ijkl A_kl.
        """
        result = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                s = sp.Integer(0)
                for k in range(3):
                    for l in range(3):
                        s += self._a[i, j, k, l] * B.matrix[k, l]
                result[i, j] = s
        return Tensor2(sp.simplify(result))

    def rdouble_contract(self, A: Tensor2) -> Tensor2:
        """Left double contraction with a rank-2 tensor: (A:C)_kl = A_ij C_ijkl.

        Parameters
        ----------
        A : Tensor2
            The rank-2 tensor to contract from the left.

        Returns
        -------
        Tensor2
            Result B_kl = A_ij C_ijkl.
        """
        result = sp.zeros(3, 3)
        for k in range(3):
            for l in range(3):
                s = sp.Integer(0)
                for i in range(3):
                    for j in range(3):
                        s += A.matrix[i, j] * self._a[i, j, k, l]
                result[k, l] = s
        return Tensor2(sp.simplify(result))

    def full_contract(self, D: "Tensor4") -> sp.Expr:
        """Full double contraction with another rank-4 tensor: C_ijkl D_ijkl → scalar.

        Parameters
        ----------
        D : Tensor4
            The other fourth-order tensor.

        Returns
        -------
        sp.Expr
            Scalar result s = C_ijkl D_ijkl.
        """
        s = sp.Integer(0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        s += self._a[i, j, k, l] * D._a[i, j, k, l]
        return sp.simplify(s)

    # ------------------------------------------------------------------
    # Configuration mappings
    # ------------------------------------------------------------------

    def push_forward(self, F: Tensor2) -> "Tensor4":
        """Piola push-forward: c_ijkl = J⁻¹ F_iI F_jJ F_kK F_lL C_IJKL.

        Maps a reference-configuration rank-4 tensor to current configuration.

        Parameters
        ----------
        F : Tensor2
            Deformation gradient.

        Returns
        -------
        Tensor4
            Current-configuration tensor c_ijkl.
        """
        Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
        jac = sp.simplify(Fm.det())
        result = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        s = sp.Integer(0)
                        for I in range(3):
                            for J in range(3):
                                for K in range(3):
                                    for L in range(3):
                                        s += (Fm[i, I] * Fm[j, J] * Fm[k, K]
                                              * Fm[l, L] * self._a[I, J, K, L])
                        result[i, j, k, l] = s / jac
        return Tensor4(result, name=self.name)

    def pull_back(self, F: Tensor2) -> "Tensor4":
        """Piola pull-back: C_IJKL = J F⁻¹_Ii F⁻¹_Jj F⁻¹_Kk F⁻¹_Ll c_ijkl.

        Maps a current-configuration rank-4 tensor to reference configuration.

        Parameters
        ----------
        F : Tensor2
            Deformation gradient.

        Returns
        -------
        Tensor4
            Reference-configuration tensor C_IJKL.
        """
        Fm = F.matrix if isinstance(F, Tensor2) else sp.Matrix(F)
        Fi = Fm.inv()
        jac = sp.simplify(Fm.det())
        result = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for I in range(3):
            for J in range(3):
                for K in range(3):
                    for L in range(3):
                        s = sp.Integer(0)
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    for l in range(3):
                                        s += (Fi[I, i] * Fi[J, j] * Fi[K, k]
                                              * Fi[L, l] * self._a[i, j, k, l])
                        result[I, J, K, L] = s * jac
        return Tensor4(result, name=self.name)

    # ------------------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------------------

    def minor_symmetrize(self) -> "Tensor4":
        """Minor symmetrization: C_{(ij)(kl)} = ¼(C_ijkl + C_jikl + C_ijlk + C_jilk)."""
        result = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val = sp.Rational(1, 4) * (
                            self._a[i, j, k, l] + self._a[j, i, k, l]
                            + self._a[i, j, l, k] + self._a[j, i, l, k]
                        )
                        result[i, j, k, l] = val
        return Tensor4(result, name=self.name)

    def major_symmetrize(self) -> "Tensor4":
        """Major symmetrization: C_{ijkl}^sym = ½(C_ijkl + C_klij)."""
        result = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val = sp.Rational(1, 2) * (
                            self._a[i, j, k, l] + self._a[k, l, i, j]
                        )
                        result[i, j, k, l] = val
        return Tensor4(result, name=self.name)

    # ------------------------------------------------------------------
    # Substitution and evaluation
    # ------------------------------------------------------------------

    def subs(self, *args, **kwargs) -> "Tensor4":
        """Substitute symbolic values (delegates to SymPy Array.subs)."""
        new_a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        new_a[i, j, k, l] = sp.sympify(self._a[i, j, k, l]).subs(*args, **kwargs)
        return Tensor4(new_a, name=self.name)

    def evaluate(self, **values: float):
        """Substitute values and return a NumPy array of shape (3,3,3,3).

        Requires numpy to be installed (optional dependency).
        """
        import numpy as np

        result = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        result[i, j, k, l] = float(
                            sp.sympify(self._a[i, j, k, l]).subs(values)
                        )
        return result

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Tensor4") -> "Tensor4":
        result = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        o = other._a if isinstance(other, Tensor4) else sp.Array(other)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        result[i, j, k, l] = self._a[i, j, k, l] + o[i, j, k, l]
        return Tensor4(result)

    def __sub__(self, other: "Tensor4") -> "Tensor4":
        result = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        o = other._a if isinstance(other, Tensor4) else sp.Array(other)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        result[i, j, k, l] = self._a[i, j, k, l] - o[i, j, k, l]
        return Tensor4(result)

    def __mul__(self, scalar) -> "Tensor4":
        result = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        result[i, j, k, l] = scalar * self._a[i, j, k, l]
        return Tensor4(result, name=self.name)

    def __rmul__(self, scalar) -> "Tensor4":
        return self.__mul__(scalar)

    def __neg__(self) -> "Tensor4":
        return self.__mul__(-1)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tensor4):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            if sp.simplify(self._a[i, j, k, l] - other._a[i, j, k, l]) != 0:
                                return False
            return True
        return NotImplemented

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def latex(self) -> str:
        """LaTeX string representation of the underlying Array."""
        return sp.latex(sp.Array(self._a))

    def _repr_latex_(self) -> str:
        """Jupyter notebook LaTeX rendering."""
        if self.name:
            return f"$${self.name} = {sp.latex(sp.Array(self._a))}$$"
        return f"$${sp.latex(sp.Array(self._a))}$$"

    def __repr__(self) -> str:
        if self.name:
            return f"Tensor4({self.name})"
        return "Tensor4"

    def __str__(self) -> str:
        return str(sp.Array(self._a))
